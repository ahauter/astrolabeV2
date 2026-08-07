"""Context-aware dataset for race-condition detection with online mutation.

Each sample is a target function plus up to K depth-1 callers.  The target is
mutated at load time: with probability `mutate_frac`, all synchronization calls
are stripped and the previously-protected shared accesses are labeled as race
risks.  The original (sync-intact) version is the negative example otherwise.

To keep memory reasonable on a multi-million-unit corpus, annotation/meta/risk
files are accessed through precomputed line-offset indices rather than being
loaded into Python lists.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from astrolabe.vocab import ID_TO_TOKEN, PAD_ID


RISKY_TYPES = {"ptr", "slice", "map", "chan", "interface", "struct", "func"}


def _build_line_offsets(path: Path) -> np.ndarray:
    """Build a uint64 offset array for each newline in *path*.

    Saves the result as ``<path>.idx.npy`` and reuses it on subsequent runs.
    offsets[i] is the byte position of the start of line i; offsets[-1] is the
    file size.
    """
    idx_path = Path(str(path) + ".idx.npy")
    if idx_path.exists():
        return np.load(idx_path)

    offsets = [0]
    pos = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(64 * 1024 * 1024)
            if not chunk:
                break
            start = 0
            while True:
                nl = chunk.find(b"\n", start)
                if nl == -1:
                    break
                offsets.append(pos + nl + 1)
                start = nl + 1
            pos += len(chunk)

    arr = np.array(offsets, dtype=np.uint64)
    np.save(idx_path, arr)
    return arr


def _read_line(f, offsets: np.ndarray, idx: int) -> dict:
    start = int(offsets[idx])
    end = int(offsets[idx + 1])
    f.seek(start)
    line = f.read(end - start)
    stripped = line.strip()
    if not stripped or stripped == b"null":
        return {}
    return json.loads(stripped.decode("utf-8"))


class RaceContextDataset(Dataset):
    """Loads base corpus units and applies online sync mutation.

    Returns:
        target_ids:     (func_len,)
        caller_ids:     (max_callers, caller_len)
        target_mask:    (func_len,)
        caller_mask:    (max_callers, caller_len)
        caller_present: (max_callers,)
        race_labels:    (func_len,)
        nil_labels:     (func_len,)
        bounds_labels:  (func_len,)
    """

    def __init__(
        self,
        units_bin: Path,
        units_idx: Path,
        ann_path: Path,
        meta_path: Path,
        risk_path: Path,
        func_len: int = 256,
        caller_len: int = 128,
        max_callers: int = 8,
        mutate_frac: float = 0.5,
        seed: int = 0,
        deterministic: bool = False,
        epoch_samples: int | None = None,
    ):
        self.data = np.memmap(Path(units_bin), dtype=np.uint16, mode="r")
        self.offsets = np.load(Path(units_idx))
        self.n_units = len(self.offsets) - 1

        self.ann_path = Path(ann_path)
        self.meta_path = Path(meta_path)
        self.risk_path = Path(risk_path)

        # Precompute (or load) byte offsets for each line in the JSONL files.
        self.ann_offsets = _build_line_offsets(self.ann_path)
        self.meta_offsets = _build_line_offsets(self.meta_path)
        self.risk_offsets = _build_line_offsets(self.risk_path)

        # Keep file handles open for fast random line access.
        self._ann_f = open(self.ann_path, "rb")
        self._meta_f = open(self.meta_path, "rb")
        self._risk_f = open(self.risk_path, "rb")

        if not (len(self.ann_offsets) - 1 == self.n_units
                and len(self.meta_offsets) - 1 == self.n_units
                and len(self.risk_offsets) - 1 == self.n_units):
            raise ValueError("mismatch between units and offset files")

        self.func_len = func_len
        self.caller_len = caller_len
        self.max_callers = max_callers
        self.mutate_frac = mutate_frac
        self.seed = seed
        self.deterministic = deterministic
        self.epoch_samples = epoch_samples or self.n_units
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.epoch_samples

    def _should_mutate(self, idx: int, ann: dict) -> bool:
        if not ann.get("sync"):
            return False
        if self.deterministic:
            rng = np.random.default_rng(self.seed ^ idx)
        else:
            rng = self._rng
        return rng.random() < self.mutate_frac

    def _load_unit(self, idx: int) -> list[int]:
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        return self.data[start:end].tolist()

    def _ann(self, idx: int) -> dict:
        return _read_line(self._ann_f, self.ann_offsets, idx)

    def _meta(self, idx: int) -> dict:
        return _read_line(self._meta_f, self.meta_offsets, idx)

    def _risk(self, idx: int) -> dict:
        return _read_line(self._risk_f, self.risk_offsets, idx)

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        idx = idx % self.n_units
        ann = self._ann(idx)
        meta = self._meta(idx)
        risk = self._risk(idx)

        original_tokens = self._load_unit(idx)
        mutate = self._should_mutate(idx, ann)

        if mutate:
            sync_positions = _sync_positions(ann)
            target_tokens, pos_map = _remove_positions(original_tokens, sync_positions)
            race_labels = _race_labels(original_tokens, target_tokens, pos_map, ann)
            nil_labels = _map_positions(risk.get("nil_risks", []), pos_map)
            bounds_labels = _map_positions(risk.get("bounds_risks", []), pos_map)
        else:
            target_tokens = original_tokens
            race_labels = []
            nil_labels = risk.get("nil_risks", [])
            bounds_labels = risk.get("bounds_risks", [])

        target_ids, target_mask = _pad_sequence(target_tokens, self.func_len)

        race_vec = _positions_to_vec(race_labels, self.func_len)
        nil_vec = _positions_to_vec(nil_labels, self.func_len)
        bounds_vec = _positions_to_vec(bounds_labels, self.func_len)

        caller_ids = np.full((self.max_callers, self.caller_len), PAD_ID, dtype=np.int64)
        caller_mask = np.zeros((self.max_callers, self.caller_len), dtype=np.bool_)
        caller_present = np.zeros(self.max_callers, dtype=np.bool_)

        caller_idxs = meta.get("callers", [])[:self.max_callers]
        for k, cidx in enumerate(caller_idxs):
            c_toks = self._load_unit(cidx)
            c_ids, c_mask = _pad_sequence(c_toks, self.caller_len)
            caller_ids[k] = c_ids
            caller_mask[k] = c_mask
            caller_present[k] = True

        return (
            torch.from_numpy(target_ids),
            torch.from_numpy(caller_ids),
            torch.from_numpy(target_mask),
            torch.from_numpy(caller_mask),
            torch.from_numpy(caller_present),
            torch.from_numpy(race_vec),
            torch.from_numpy(nil_vec),
            torch.from_numpy(bounds_vec),
        )

    def __del__(self):
        for f in (self._ann_f, self._meta_f, self._risk_f):
            try:
                f.close()
            except Exception:
                pass


def _sync_positions(ann: dict) -> set[int]:
    """Return token positions occupied by synchronization calls."""
    positions: set[int] = set()
    for ev in ann.get("sync", []):
        for p in range(ev["start"], ev["end"] + 1):
            positions.add(p)
    return positions


def _remove_positions(unit: list[int], remove: set[int]) -> tuple[list[int], dict[int, int]]:
    """Return unit with positions removed and a map old->new positions."""
    mutated: list[int] = []
    pos_map: dict[int, int] = {}
    for old_pos, tok in enumerate(unit):
        if old_pos not in remove:
            pos_map[old_pos] = len(mutated)
            mutated.append(tok)
    return mutated, pos_map


def _race_labels(
    original: list[int],
    mutated: list[int],
    pos_map: dict[int, int],
    ann: dict,
) -> list[int]:
    """Label risky shared uses that survive sync removal."""
    types_map = ann.get("types", {})
    use_set = set(ann.get("use", []))
    labels: list[int] = []

    for old_pos in use_set:
        if old_pos < 0 or old_pos >= len(original):
            continue
        if old_pos not in pos_map:
            continue
        tok_name = ID_TO_TOKEN[original[old_pos]]
        if not tok_name.startswith("NAME_"):
            continue
        suffix = tok_name[len("NAME_"):]
        if not suffix.isdigit():
            continue
        slot = int(suffix)
        cat = types_map.get(str(slot), "unknown")
        if cat in RISKY_TYPES:
            labels.append(pos_map[old_pos])

    return labels


def _map_positions(positions: list[int], pos_map: dict[int, int]) -> list[int]:
    """Map original label positions to mutated positions, skipping removed ones."""
    return [pos_map[p] for p in positions if p in pos_map]


def _positions_to_vec(positions: list[int], length: int) -> np.ndarray:
    vec = np.zeros(length, dtype=np.float32)
    for p in positions:
        if 0 <= p < length:
            vec[p] = 1.0
    return vec


def _pad_sequence(tokens: list[int], length: int) -> tuple[np.ndarray, np.ndarray]:
    """Truncate or pad a token sequence and return a bool mask."""
    tokens = tokens[:length]
    ids = np.full(length, PAD_ID, dtype=np.int64)
    ids[:len(tokens)] = tokens
    mask = np.zeros(length, dtype=np.bool_)
    mask[:len(tokens)] = True
    return ids, mask
