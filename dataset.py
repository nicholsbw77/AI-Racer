"""
trainer/dataset.py

PyTorch Dataset that builds (state_vector, action_vector) pairs from
processed telemetry DataFrames.

At 360Hz with sequence_history=15, each sample includes ~42ms of context.
The state vector is:
  [current_state_features] + [history_actions × history_length]

This gives the model temporal context without needing an LSTM.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

from loader import STATE_FEATURES, ACTION_FEATURES, HISTORY_ACTIONS


class TelemetryDataset(Dataset):
    """
    Dataset of (state, action) pairs built from cleaned telemetry DataFrame.

    State vector layout (per sample):
      Indices 0..N_state-1        : current frame state features
      Indices N_state..end        : flattened action history
                                    [t-1, t-2, ..., t-history] × HISTORY_ACTIONS

    Action vector:
      [throttle, brake, steering]  (all in [0,1] or [-1,1])
    """

    def __init__(self, df, sequence_history: int = 15):
        """
        Args:
            df: Cleaned, normalized DataFrame from loader.py
            sequence_history: How many previous frames of actions to include
        """
        self.sequence_history = sequence_history

        # Extract state and action arrays
        # Handle missing columns gracefully
        state_cols = [c for c in STATE_FEATURES if c in df.columns]
        action_cols = [c for c in ACTION_FEATURES if c in df.columns]
        history_cols = [c for c in HISTORY_ACTIONS if c in df.columns]

        if not action_cols:
            raise ValueError("DataFrame missing all action columns (throttle/brake/steering)")

        self.state_arr = df[state_cols].values.astype(np.float32)
        self.action_arr = df[action_cols].values.astype(np.float32)
        self.history_arr = df[history_cols].values.astype(np.float32)

        # Track lap boundaries so we don't build sequences across lap transitions
        if "lapIndex" in df.columns:
            self.lap_ids = df["lapIndex"].values
        else:
            self.lap_ids = np.zeros(len(df), dtype=np.int64)

        # Pre-compute valid sample indices (skip first `history` frames of each lap)
        self.valid_indices = self._compute_valid_indices()

        # Compute input dimension for model construction
        self.state_dim = (
            self.state_arr.shape[1] +
            len(history_cols) * sequence_history
        )
        self.action_dim = self.action_arr.shape[1]

    def _compute_valid_indices(self) -> np.ndarray:
        """
        Find all frame indices where we have a full history window
        without crossing a lap boundary.
        """
        valid = []
        n = len(self.state_arr)
        h = self.sequence_history

        for i in range(h, n):
            # Check all frames in history window are the same lap
            window_laps = self.lap_ids[i - h:i + 1]
            if np.all(window_laps == window_laps[0]):
                valid.append(i)

        return np.array(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_idx = self.valid_indices[idx]
        h = self.sequence_history

        # Current state
        current_state = self.state_arr[frame_idx]

        # Action history: [t-1, t-2, ..., t-h] flattened
        # Ordered newest-first so index 0 = most recent context
        history_window = self.history_arr[frame_idx - h:frame_idx][::-1]  # reverse
        history_flat = history_window.flatten()

        # Concatenate into full state vector
        full_state = np.concatenate([current_state, history_flat])

        # Target action
        action = self.action_arr[frame_idx]

        return (
            torch.from_numpy(full_state),
            torch.from_numpy(action),
        )

    @property
    def input_dim(self) -> int:
        return self.state_dim

    @property
    def output_dim(self) -> int:
        return self.action_dim


def split_dataset(
    dataset: TelemetryDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train/val sets.
    Splits by lap index to avoid data leakage between train and val.
    """
    # Get unique lap IDs in valid samples
    valid_lap_ids = dataset.lap_ids[dataset.valid_indices]
    unique_laps = np.unique(valid_lap_ids)

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_laps)

    n_val_laps = max(1, int(len(unique_laps) * val_fraction))
    val_laps = set(unique_laps[:n_val_laps])
    train_laps = set(unique_laps[n_val_laps:])

    # Build index subsets
    train_mask = np.array([lid in train_laps for lid in valid_lap_ids])
    val_mask = ~train_mask

    train_indices = dataset.valid_indices[train_mask]
    val_indices = dataset.valid_indices[val_mask]

    train_subset = _IndexedSubset(dataset, train_indices)
    val_subset = _IndexedSubset(dataset, val_indices)

    return train_subset, val_subset


class _IndexedSubset(Dataset):
    """Subset of a TelemetryDataset using explicit frame indices."""

    def __init__(self, dataset: TelemetryDataset, frame_indices: np.ndarray):
        self.dataset = dataset
        self.frame_indices = frame_indices
        self.sequence_history = dataset.sequence_history

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_idx = self.frame_indices[idx]
        h = self.sequence_history

        current_state = self.dataset.state_arr[frame_idx]
        history_window = self.dataset.history_arr[frame_idx - h:frame_idx][::-1]
        history_flat = history_window.flatten()
        full_state = np.concatenate([current_state, history_flat])
        action = self.dataset.action_arr[frame_idx]

        return (
            torch.from_numpy(full_state),
            torch.from_numpy(action),
        )

    @property
    def input_dim(self) -> int:
        return self.dataset.input_dim

    @property
    def output_dim(self) -> int:
        return self.dataset.output_dim
