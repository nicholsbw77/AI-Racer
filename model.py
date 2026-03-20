"""
trainer/model.py

Temporal MLP with skip connections for behavior cloning of racing inputs.

Architecture rationale:
  - MLP (not LSTM/Transformer): fast inference, simple to tune,
    temporal context supplied via action history in state vector
  - Skip connections: prevent gradient vanishing in deeper layers,
    allow low-level features to pass through directly
  - Separate output heads: throttle/brake/steering have different
    activation functions and loss weightings
  - LayerNorm (not BatchNorm): works correctly with batch_size=1
    during single-sample inference at 360Hz
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """Two-layer residual block with LayerNorm and optional projection."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.05):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act = nn.GELU()  # GELU slightly better than ReLU for smooth outputs
        self.drop = nn.Dropout(dropout)

        # Skip connection projection if dimensions differ
        if in_dim != out_dim:
            self.skip_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip_proj(x) if self.skip_proj is not None else x

        out = self.act(self.norm1(self.fc1(x)))
        out = self.drop(out)
        out = self.norm2(self.fc2(out))
        out = self.act(out + identity)
        return out


class DrivingPolicyNet(nn.Module):
    """
    Main behavior cloning network.

    Input:  state vector (current features + flattened action history)
    Output: (throttle, brake, steering) tuple

    Output ranges:
      throttle: [0, 1]  via Sigmoid
      brake:    [0, 1]  via Sigmoid
      steering: [-1, 1] via Tanh
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 128, 64),
        dropout: float = 0.05,
    ):
        super().__init__()

        self.input_dim = input_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
        )

        # Residual backbone
        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(ResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout))
        self.backbone = nn.Sequential(*blocks)

        final_dim = hidden_dims[-1]

        # Separate heads for each output - allows independent tuning
        self.throttle_head = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.brake_head = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.steering_head = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) state vector

        Returns:
            throttle: (batch, 1)
            brake:    (batch, 1)
            steering: (batch, 1)
        """
        h = self.input_proj(x)
        h = self.backbone(h)

        throttle = self.throttle_head(h)
        brake = self.brake_head(h)
        steering = self.steering_head(h)

        return throttle, brake, steering

    def predict(self, x: torch.Tensor) -> Tuple[float, float, float]:
        """
        Single-sample inference. Returns Python floats directly.
        Used by inference loop at 360Hz.
        """
        with torch.no_grad():
            t, b, s = self.forward(x.unsqueeze(0))
        return t.item(), b.item(), s.item()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BehaviorCloningLoss(nn.Module):
    """
    Weighted MSE loss with smoothness penalty and track boundary awareness.

    The smoothness penalty discourages large frame-to-frame changes
    in outputs, producing smoother driving that doesn't oscillate.

    The boundary penalty increases steering loss weight when the car
    is near the track edge, teaching the model that precision matters
    most at the limits.
    """

    def __init__(
        self,
        throttle_weight: float = 1.0,
        brake_weight: float = 1.2,
        steering_weight: float = 1.5,
        smoothness_weight: float = 0.15,
        boundary_weight: float = 0.3,
    ):
        super().__init__()
        self.w_thr = throttle_weight
        self.w_brk = brake_weight
        self.w_str = steering_weight
        self.w_smooth = smoothness_weight
        self.w_boundary = boundary_weight

    def forward(
        self,
        pred_throttle: torch.Tensor,
        pred_brake: torch.Tensor,
        pred_steering: torch.Tensor,
        target: torch.Tensor,  # (batch, 3) [throttle, brake, steering]
        prev_steering: Optional[torch.Tensor] = None,  # (batch, 1) last frame steering
        track_pos: Optional[torch.Tensor] = None,  # (batch, 1) lateral position
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, loss_components_dict)
        """
        tgt_thr = target[:, 0:1]
        tgt_brk = target[:, 1:2]
        tgt_str = target[:, 2:3]

        loss_thr = torch.mean((pred_throttle - tgt_thr) ** 2)
        loss_brk = torch.mean((pred_brake - tgt_brk) ** 2)
        loss_str = torch.mean((pred_steering - tgt_str) ** 2)

        base_loss = (
            self.w_thr * loss_thr +
            self.w_brk * loss_brk +
            self.w_str * loss_str
        )

        # Smoothness penalty: penalize large steering changes
        smooth_loss = torch.tensor(0.0, device=pred_steering.device)
        if prev_steering is not None and self.w_smooth > 0:
            steering_delta = pred_steering - prev_steering
            smooth_loss = torch.mean(steering_delta ** 2)
            base_loss = base_loss + self.w_smooth * smooth_loss

        # Boundary awareness penalty: increase loss near track edges
        # When |track_pos| is high, steering errors are more costly
        boundary_loss = torch.tensor(0.0, device=pred_steering.device)
        if track_pos is not None and self.w_boundary > 0:
            # Weight multiplier: 1.0 at center, up to 3.0 at edges
            edge_proximity = torch.abs(track_pos).clamp(0.0, 1.0)
            edge_weight = 1.0 + 2.0 * edge_proximity ** 2  # quadratic ramp
            # Extra penalty on steering error when near edge
            weighted_steer_err = edge_weight * (pred_steering - tgt_str) ** 2
            boundary_loss = torch.mean(weighted_steer_err)
            base_loss = base_loss + self.w_boundary * boundary_loss

        components = {
            "throttle": loss_thr.item(),
            "brake": loss_brk.item(),
            "steering": loss_str.item(),
            "smoothness": smooth_loss.item() if isinstance(smooth_loss, torch.Tensor)
                          else smooth_loss,
            "boundary": boundary_loss.item() if isinstance(boundary_loss, torch.Tensor)
                        else boundary_loss,
        }

        return base_loss, components
