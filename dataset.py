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
from typing import Tuple, Optional

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


def materialize_to_gpu(
    dataset: TelemetryDataset,
    device: torch.device,
) -> "GPUResidentDataset":
    """
    Pre-compute ALL (state, action) pairs and store them as contiguous
    GPU tensors.  Eliminates per-batch numpy→torch and CPU→GPU overhead.
    """
    n = len(dataset.valid_indices)
    h = dataset.sequence_history

    # Pre-allocate numpy arrays for the full materialised dataset
    state_dim = dataset.input_dim
    action_dim = dataset.output_dim
    all_states = np.empty((n, state_dim), dtype=np.float32)
    all_actions = np.empty((n, action_dim), dtype=np.float32)

    for out_idx, frame_idx in enumerate(dataset.valid_indices):
        current_state = dataset.state_arr[frame_idx]
        history_window = dataset.history_arr[frame_idx - h:frame_idx][::-1]
        history_flat = history_window.flatten()
        all_states[out_idx] = np.concatenate([current_state, history_flat])
        all_actions[out_idx] = dataset.action_arr[frame_idx]

    states_t = torch.from_numpy(all_states).to(device)
    actions_t = torch.from_numpy(all_actions).to(device)

    return GPUResidentDataset(states_t, actions_t, state_dim, action_dim)


class GPUResidentDataset(Dataset):
    """All data lives on GPU — zero transfer overhead per batch."""

    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        input_dim: int,
        output_dim: int,
    ):
        self.states = states
        self.actions = actions
        self._input_dim = input_dim
        self._output_dim = output_dim

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def batches(self, batch_size: int, shuffle: bool = True, drop_last: bool = True):
        """
        Yield (state_batch, action_batch) slices directly from GPU tensors.
        No DataLoader needed — just index shuffling on CPU, slicing on GPU.
        """
        n = len(self)
        if shuffle:
            perm = torch.randperm(n, device="cpu")
        else:
            perm = torch.arange(n, device="cpu")

        for start in range(0, n, batch_size):
            end = start + batch_size
            if drop_last and end > n:
                break
            idx = perm[start:end]
            yield self.states[idx], self.actions[idx]


def split_gpu_dataset(
    gpu_ds: GPUResidentDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple["GPUResidentDataset", "GPUResidentDataset"]:
    """Split a GPU-resident dataset into train/val by random shuffle."""
    n = len(gpu_ds)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_fraction))

    val_idx = torch.from_numpy(indices[:n_val])
    train_idx = torch.from_numpy(indices[n_val:])

    train_ds = GPUResidentDataset(
        gpu_ds.states[train_idx],
        gpu_ds.actions[train_idx],
        gpu_ds.input_dim,
        gpu_ds.output_dim,
    )
    val_ds = GPUResidentDataset(
        gpu_ds.states[val_idx],
        gpu_ds.actions[val_idx],
        gpu_ds.input_dim,
        gpu_ds.output_dim,
    )
    return train_ds, val_ds


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
