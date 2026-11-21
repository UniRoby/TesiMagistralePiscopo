"""Low-level volume IO that avoids reading whole TIFF stacks.

The training loader used to read an entire ``slide%04d.tiff`` stack (114-324
slices of 512x512) for every single 32^3 patch, which dominated runtime. The
helpers here separate the three things that were previously fused together:

* probing a volume's shape (cheap: one slice header),
* loading a label mask (cheap: bool, no uint16 round-trip),
* loading a normalized scan (expensive: kept behind :class:`VolumeCache`).

The module deliberately stays importable without PyTorch so ``dataset.py`` and
the index builder can use it in environments where only numpy and Pillow are
installed.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable
import warnings

import numpy as np

SLICE_TEMPLATE = "slide{:04d}.tiff"


def _require_pillow():
    """Import Pillow with the same error message used across the project."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("Pillow is required to load M3Dsynth TIFF stacks") from exc
    return Image


def count_contiguous_slices(dirname: str | Path) -> int:
    """Count ``slide0000.tiff``, ``slide0001.tiff``, ... stopping at the first gap.

    This replicates the stop-at-first-missing rule of ``load_tiff_stack`` so the
    probed depth can never disagree with the depth of an actually loaded stack.
    A single ``scandir`` replaces one ``exists()`` syscall per slice.
    """

    dirname = Path(dirname)
    if not dirname.is_dir():
        raise FileNotFoundError(f"no TIFF slices found in {dirname}")
    names = {entry.name for entry in dirname.iterdir()}
    count = 0
    while SLICE_TEMPLATE.format(count) in names:
        count += 1
    return count


def probe_volume_shape(dirname: str | Path) -> tuple[int, int, int]:
    """Return ``(depth, height, width)`` without decoding the whole stack.

    Only ``slide0000.tiff`` is opened, and Pillow reads just its header for
    ``.size``. Costs milliseconds against seconds for a full stack load.
    """

    dirname = Path(dirname)
    depth = count_contiguous_slices(dirname)
    if depth == 0:
        raise FileNotFoundError(f"no TIFF slices found in {dirname}")
    Image = _require_pillow()
    with Image.open(dirname / SLICE_TEMPLATE.format(0)) as image:
        width, height = image.size  # PIL reports (W, H); volumes are (z, y, x).
    return (depth, int(height), int(width))


def load_label_mask(dirname: str | Path) -> np.ndarray:
    """Load a label stack straight to ``bool``.

    M3Dsynth label slices are Pillow mode ``1``, so ``np.asarray`` already yields
    bool. Going through the uint16 path of ``load_tiff_stack`` would double the
    memory for no benefit.
    """

    Image = _require_pillow()
    dirname = Path(dirname)
    depth = count_contiguous_slices(dirname)
    if depth == 0:
        raise FileNotFoundError(f"no TIFF slices found in {dirname}")
    slices = []
    for index in range(depth):
        with Image.open(dirname / SLICE_TEMPLATE.format(index)) as image:
            slices.append(np.asarray(image).astype(bool, copy=False))
    return np.stack(slices, axis=0)


def align_mask_to_scan(
    mask: np.ndarray,
    scan_shape: tuple[int, int, int],
    img_id: str = "",
) -> np.ndarray:
    """Align a label mask to the scan shape along z.

    All 2273 pix2pix series carry one extra empty label slice from a bug in the
    original M3Dsynth generation pipeline, and a few series are off by much more
    (rem_14 was +85). Longer masks are trimmed, shorter ones are zero-padded at
    the end. ``y`` and ``x`` must match exactly: a spatial mismatch means the
    label belongs to a different series and must not be silently reshaped.
    """

    scan_shape = tuple(int(v) for v in scan_shape)
    if mask.shape == scan_shape:
        return mask

    scan_z, scan_y, scan_x = scan_shape
    mask_z, mask_y, mask_x = mask.shape
    if scan_y != mask_y or scan_x != mask_x:
        raise ValueError(
            f"scan/mask spatial (y, x) mismatch for {img_id or '<unknown>'}: "
            f"scan {scan_shape} != mask {mask.shape}"
        )
    if mask_z > scan_z:
        return mask[:scan_z]
    padding = scan_z - mask_z
    return np.pad(mask, ((0, padding), (0, 0), (0, 0)), mode="constant", constant_values=False)


def normalize_scan(
    raw: np.ndarray,
    dirname: str | Path,
    cache_dir: str | Path | None = None,
    *, low: float = 1.0, high: float = 99.0,
) -> np.ndarray:
    """Normalize with persistent bounds, invalidated by slice metadata/parameters."""

    from .dataset import normalize_percentile, percentile_bounds

    if cache_dir is None:
        return normalize_percentile(raw, low, high)

    dirname = Path(dirname).resolve()
    slices = []
    for index in range(raw.shape[0]):
        stat = (dirname / SLICE_TEMPLATE.format(index)).stat()
        slices.append((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    # Include the physical path so real-record aliases share bounds, while
    # different generators/scans never collide. Bump version if the rule changes.
    identity = (1, str(dirname), raw.shape, raw.dtype.str, float(low), float(high), slices)
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    cache_path = Path(cache_dir) / "percentiles" / f"{key}.json"
    bounds = None
    try:
        cached = json.loads(cache_path.read_text())
        values = np.asarray(cached, dtype=np.float64)
        if values.shape == (2,) and np.isfinite(values).all() and values[1] >= values[0]:
            bounds = (float(values[0]), float(values[1]))
    except (OSError, ValueError, TypeError):
        pass  # A missing or damaged cache is rebuilt from the original scan.
    if bounds is None:
        bounds = percentile_bounds(raw, low, high)
        temporary = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Close before replace for Windows; unique temp names allow workers
            # to populate the same entry without exposing a partially written file.
            with tempfile.NamedTemporaryFile(mode="w", dir=cache_path.parent, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(bounds, handle)
            os.replace(temporary, cache_path)
        except OSError as exc:
            warnings.warn(f"Cannot save percentile cache {cache_path}: {exc}", RuntimeWarning, stacklevel=2)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return normalize_percentile(raw, low, high, bounds=bounds)


def load_normalized_scan(
    data_root: str | Path,
    record,
    real_scan_root: str | Path | None = None,
    *, cache_dir: str | Path | None = None,
) -> np.ndarray:
    """Load one scan as a percentile-normalized float32 volume.

    Imported lazily from ``dataset`` to keep this module free of a circular
    import at module load time.
    """

    from .dataset import load_tiff_stack, scan_dir

    dirname = scan_dir(data_root, record, real_scan_root)
    return normalize_scan(load_tiff_stack(dirname), dirname, cache_dir)


def load_normalized_scan_and_lung_mask(
    data_root: str | Path,
    record,
    real_scan_root: str | Path | None = None,
    *, cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one scan and derive its lung ROI before intensity normalization."""

    from .dataset import load_tiff_stack, scan_dir
    from .lung import lung_mask_from_ct

    dirname = scan_dir(data_root, record, real_scan_root)
    raw = load_tiff_stack(dirname)
    return normalize_scan(raw, dirname, cache_dir), lung_mask_from_ct(raw)


class VolumeCache:
    """Tiny LRU cache of normalized float32 scans, keyed by resolved scan dir.

    Sized in *volumes*, not bytes, because one normalized volume is 120-340 MB
    and only a handful can ever be resident. The previous design cached 1000
    entries and exhausted 32 GB of RAM before the first epoch finished.

    Effective only when consecutive dataset accesses hit the same volume, which
    is what :class:`~tesi_m3d.sampling.VolumeGroupedBatchSampler` guarantees.
    """

    def __init__(self, maxsize: int = 2) -> None:
        """Store at most ``maxsize`` volumes, evicting least recently used."""

        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self.maxsize = int(maxsize)
        self.hits = 0
        self.misses = 0
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str, loader: Callable[[], np.ndarray]) -> np.ndarray:
        """Return the cached volume for ``key``, calling ``loader`` on a miss."""

        if key in self._store:
            self.hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self.misses += 1
        value = loader()
        self._store[key] = value
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)
        return value

    def clear(self) -> None:
        """Drop every cached volume."""

        self._store.clear()

    def __len__(self) -> int:
        """Return the number of resident volumes."""

        return len(self._store)

    def __getstate__(self) -> dict:
        """Pickle without the cached arrays.

        DataLoader workers on Windows are spawned, not forked, so the dataset is
        pickled into each worker. Without this the parent's resident volumes
        (hundreds of MB) would be serialized once per worker at startup.
        """

        return {"maxsize": self.maxsize, "hits": 0, "misses": 0}

    def __setstate__(self, state: dict) -> None:
        """Restore an empty cache in the child process."""

        self.maxsize = state["maxsize"]
        self.hits = state.get("hits", 0)
        self.misses = state.get("misses", 0)
        self._store = OrderedDict()
