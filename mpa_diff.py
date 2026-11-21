"""A standalone conceptual reference for MPA-Diff, not the experimental model."""

import torch
from torch import nn
from torch.nn import functional as F


class MPADiff(nn.Module):
    """Multi-period, pathwise autoregressive, prior-centered diffusion.

    history: [batch, history_length, channels]
    target:  [batch, horizon, channels] (forward/training only)
    samples: [batch, samples, horizon, channels]

    Inputs and outputs share the caller's coordinate system. Standardizing
    channels using training statistics is recommended. No weights are supplied.
    """

    def __init__(
        self, channels=2, horizon=5, periods=(5, 10, 30),
        hidden_size=64, diffusion_steps=30,
    ):
        super().__init__()
        periods = tuple(periods)
        if min(channels, horizon, hidden_size, diffusion_steps) < 1:
            raise ValueError("Model dimensions and diffusion_steps must be positive")
        if not periods or any(p < 1 or not isinstance(p, int) for p in periods):
            raise ValueError("periods must be positive integer window lengths")
        self.channels = channels
        self.horizon = horizon
        self.periods = periods
        self.diffusion_steps = diffusion_steps
        self.variance_floor = 1e-5

        # MP: one shared encoder for statistics of differently sized windows.
        self.window_encoder = nn.Sequential(
            nn.Linear(4 * channels + 1, hidden_size), nn.Tanh()
        )
        self.mean_head = nn.Linear(hidden_size * len(periods), horizon * channels)
        self.initial_state = nn.Linear(hidden_size, hidden_size)

        # AR: each path state queries the fixed historical window features.
        self.path_cell = nn.GRUCell(channels, hidden_size)
        self.retrieval = nn.MultiheadAttention(hidden_size, 1, batch_first=True)
        self.scale_head = nn.Linear(hidden_size, channels)
        self.denoiser = nn.Sequential(
            nn.Linear(3 * channels + 2 * hidden_size + 1, hidden_size),
            nn.SiLU(), nn.Linear(hidden_size, hidden_size),
            nn.SiLU(), nn.Linear(hidden_size, channels),
        )

        beta = torch.linspace(1e-4, 0.05, diffusion_steps)
        alpha = 1 - beta
        cumulative = alpha.cumprod(0)
        previous = torch.cat((torch.ones(1), cumulative[:-1]))
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", cumulative)
        self.register_buffer("posterior_variance", beta * (1 - previous) / (1 - cumulative))

    def _check_history(self, history):
        if history.ndim != 3 or history.shape[-1] != self.channels:
            raise ValueError("history must have shape [batch, length, channels]")
        if history.shape[0] < 1 or history.shape[1] < max(self.periods):
            raise ValueError("history needs a nonempty batch and all configured windows")
        if not history.is_floating_point() or not torch.isfinite(history).all():
            raise ValueError("history must contain finite floating-point values")

    def encode_history(self, history):
        """MP outputs: window features [B,P,D], mean [B,H,C], state [B,D]."""
        self._check_history(history)
        features = []
        for period in self.periods:
            window = history[:, -period:]
            summary = torch.cat((
                window.mean(1), window.std(1, unbiased=False),
                window[:, -1], window[:, -1] - window[:, 0],
                history.new_full((history.shape[0], 1), period / max(self.periods)),
            ), dim=-1)
            features.append(self.window_encoder(summary))
        memory = torch.stack(features, dim=1)
        # A last-observation reference plus a learned deterministic forecast.
        mean = history[:, -1:] + self.mean_head(memory.flatten(1)).reshape(
            history.shape[0], self.horizon, self.channels
        )
        state = torch.tanh(self.initial_state(memory.mean(1)))
        return memory, mean, state

    def _path_step(self, path, state, memory):
        state = self.path_cell(path[:, -1], state)
        context, _ = self.retrieval(state[:, None], memory, memory, need_weights=False)
        context = context[:, 0]
        recent = path[:, -min(max(self.periods), path.shape[1]):]
        base_variance = recent.var(1, unbiased=False).clamp_min(self.variance_floor)
        variance = base_variance * torch.exp(torch.tanh(self.scale_head(context)))
        scale = variance.clamp_min(self.variance_floor).sqrt()
        return state, context, scale

    def _noise_prediction(self, noisy, state, context, prior, scale, timestep):
        time = timestep.to(noisy).unsqueeze(-1) / max(self.diffusion_steps - 1, 1)
        condition = torch.cat((noisy, state, context, prior, scale, time), dim=-1)
        return self.denoiser(condition)

    def forward(self, history, target):
        """Scalar reference loss: epsilon MSE with teacher-forced path updates.

        This conceptual objective is not the formal experiment's joint loss.
        Future targets are never needed by sample().
        """
        memory, mean, state = self.encode_history(history)
        if target.shape != (history.shape[0], self.horizon, self.channels):
            raise ValueError("target must have shape [batch, horizon, channels]")
        if target.device != history.device or target.dtype != history.dtype:
            raise ValueError("history and target must share device and dtype")
        if not torch.isfinite(target).all():
            raise ValueError("target must contain finite values")
        path = history
        losses = []
        for step in range(self.horizon):
            state, context, scale = self._path_step(path, state, memory)
            prior = mean[:, step]
            timestep = torch.randint(self.diffusion_steps, (history.shape[0],), device=history.device)
            cumulative = self.alpha_bar[timestep, None]
            noise = torch.randn_like(target[:, step])
            noisy = (prior + cumulative.sqrt() * (target[:, step] - prior)
                     + (1 - cumulative).sqrt() * scale * noise)
            prediction = self._noise_prediction(noisy, state, context, prior, scale, timestep)
            losses.append(F.mse_loss(prediction, noise))
            path = torch.cat((path, target[:, step:step + 1]), dim=1)
        return torch.stack(losses).mean()

    def _reverse(self, prior, state, context, scale):
        value = prior + scale * torch.randn_like(prior)
        for index in range(self.diffusion_steps - 1, -1, -1):
            timestep = torch.full((prior.shape[0],), index, device=prior.device, dtype=torch.long)
            noise = self._noise_prediction(value, state, context, prior, scale, timestep)
            value = prior + (
                value - prior - self.beta[index] * scale * noise
                / (1 - self.alpha_bar[index]).sqrt()
            ) / self.alpha[index].sqrt()
            if index > 0:
                value = value + self.posterior_variance[index].sqrt() * scale * torch.randn_like(value)
        return value

    @torch.no_grad()
    def sample(self, history, num_samples=100):
        """Generate independent complete paths, returned as [B,S,H,C]."""
        if not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        memory, mean, state = self.encode_history(history)
        batch = history.shape[0]
        memory = memory.repeat_interleave(num_samples, dim=0)
        mean = mean.repeat_interleave(num_samples, dim=0)
        state = state.repeat_interleave(num_samples, dim=0)
        path = history.repeat_interleave(num_samples, dim=0)
        draws = []
        for step in range(self.horizon):
            state, context, scale = self._path_step(path, state, memory)
            draw = self._reverse(mean[:, step], state, context, scale)
            draws.append(draw)
            # Each flattened batch/sample row receives only its own draw.
            path = torch.cat((path, draw[:, None]), dim=1)
        return torch.stack(draws, dim=1).reshape(
            batch, num_samples, self.horizon, self.channels
        )
