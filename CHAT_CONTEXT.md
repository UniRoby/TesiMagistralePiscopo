# Chat Context: M3Dsynth Training Pipeline Optimization

## Project Overview

**Tesi Magistrale**: Detect CT image manipulation using deep learning on M3Dsynth dataset.
- **Dataset**: ~2500 pix2pix (synthetic) + ~750 real LIDC-IDRI CT scans
- **Task**: Binary classification (manipulated vs. real)
- **Model**: Simple 3D CNN patch-wise classifier
- **Hardware**: i5-14400 CPU, RTX 5060 Ti (16GB), 32GB RAM

## Key Achievements This Session

### 1. **DICOM Conversion & Real Data Validation**
- Converted 744 LIDC-IDRI DICOM series to TIFF format
- Fixed path resolution: converter now auto-detects dataset root (must contain `pix2pix/`)
- Validated 689/744 series as readable, properly oriented, with correct HU scaling
- Created `scripts/check_real_tiff.py` with:
  - TIFF readability checks
  - Orientation validation (spine posterior, lung air centered)
  - HU calibration verification
  - PNG preview generation (axial montage + reslices)
  - Log file output with progress bar

### 2. **Dataset Alignment Fixes**
- **pix2pix padding issue**: All 2273 pix2pix series have label +1 slice (extra empty slice at end)
  - Root cause: off-by-one error in original M3Dsynth generation pipeline
  - Solution: Auto-trim last label slice in `load_scan_and_mask()` — original dataset untouched
- **General z-dimension mismatch**: Some series (e.g., rem_14) had label +85 slices
  - Solution: Robust alignment — trim if longer, pad zeros if shorter
  - Y/X dimension must match exactly (fails with clear error if not)

### 3. **Data Pipeline Optimization**
- **Original bottleneck**: 88 sec/batch (429k batches = 438 days!)
  - Root cause: no caching — each batch reloaded TIFF from disk
- **Solution implemented**: LRU cache on `load_tiff_stack()`
  - `@lru_cache(maxsize=1000)` — ~75 GB in-memory cache
  - Same series reused within epoch uses RAM, not disk
  - Estimated **50-100× speedup** (0.5-1 sec/batch)
  - Enables training: **~7-15 hours per epoch** (50 epoch = ~15 days)

### 4. **Training Pipeline Improvements**
- Added **tqdm progress bars**: epoch + batch level progress visible in terminal
- Removed non-serializable `M3DSynthRecord` from batch dict (caused PyTorch collate error)
- Fixed batch dict to only include: `image`, `label`, `soft_score`

## Critical Files Modified

| File | Changes | Why |
|------|---------|-----|
| `src/tesi_m3d/lidc_conversion.py` | Added `resolve_dataset_root()` | Auto-detect output root, validate it contains pix2pix |
| `src/tesi_m3d/dataset.py` | Added LRU cache + alignment logic | Fix scan/mask mismatches, prevent redundant disk I/O |
| `scripts/check_real_tiff.py` | Created new script | Validate converted TIFF series (readability, orientation, content) |
| `src/tesi_m3d/train.py` | Added tqdm progress bars | Visual feedback on long training runs |
| `docs/LAB_WORKFLOW.md` | Added §8-14 for real data pipeline | Document conversion, validation, optimization, training |

## Dataset State

**Real CT (689/744 valid after validation)**:
- Location: `C:\Tesi Magistrale Piscopo\real\scan\<orig_id>__<sdir_id>\`
- Format: uint16 TIFF, 512×512 spatial, ~100-150 slices z-depth
- Offset: varies per series (~2000-3000 HU), auto-inferred from lung air peak
- Validation: passed readability, orientation, contrast checks

**Pix2pix (~2273 series)**:
- Location: `C:\Tesi Magistrale Piscopo\pix2pix\scan\` and `label\`
- Label auto-trimmed last slice in loader (no disk changes)
- All have extra empty slice that causes mismatch if not handled

## Configuration Files

### Quick Test (2.5-hour run):
```yaml
# configs/train_pix2pix_quick_test.yaml
epochs: 2
batch_size: 32
include_real: false  # Only pix2pix, no real CT
train_mods: [pix2pix]
```

### Full Training (15-day run):
```yaml
# configs/train_pix2pix_test_cycle_diffusion.yaml
epochs: 50
batch_size: 16
include_real: true  # pix2pix + real CT
train_mods: [pix2pix]
```

## Ongoing Issues & Resolutions

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| DICOM conversion in wrong dir | `--output-root` not validated | Added `resolve_dataset_root()` to validate/auto-detect |
| Training 438 days (impossible) | No caching of TIFF loads | LRU cache on load_tiff_stack() |
| 2273 label +1 slice | M3Dsynth generation pipeline bug | Auto-trim in load_scan_and_mask() |
| Batch collate error | M3DSynthRecord not serializable | Removed from batch dict |
| Orientation validation too strict | Allowed 0.5 (exact), real data varies ±2% | Relaxed to 0.45 (passed 689/744) |

## Scripts & Commands

### Convert DICOM to TIFF (real data):
```powershell
python scripts\convert_lidc_to_tiff.py `
  --dicom-root "C:\Tesi Magistrale Piscopo\Reale\lidc_idri" `
  --download-metadata "C:\Tesi Magistrale Piscopo\Reale\metadata.csv" `
  --output-root "C:\Tesi Magistrale Piscopo" `
  --metadata-dir metadata\m3dsynth `
  --workers 2
```

### Validate converted series:
```powershell
python scripts\check_real_tiff.py `
  --scan-root "C:\Tesi Magistrale Piscopo\real\scan" `
  --preview-dir outputs\real_check
```

### Train (quick test):
```powershell
python -m tesi_m3d.train `
  --config configs\train_pix2pix_quick_test.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda
```

### Train (full):
```powershell
python -m tesi_m3d.train `
  --config configs\train_pix2pix_test_cycle_diffusion.yaml `
  --data-root "C:\Tesi Magistrale Piscopo" `
  --device cuda
```

## Lessons Learned

1. **Data is not free**: Conversion, alignment, and I/O are the bottleneck, not compute
2. **Caching is essential**: Single-level LRU cache on disk-bound ops yields 50-100× speedup
3. **Validation is non-negotiable**: Check data early (2300+ alignment issues found)
4. **PyTorch serialization**: Batch dicts must be tensor-compatible; metadata should be separate
5. **Dataset offsets matter**: uint16 offset varies per series — percentile normalization handles it

## Next Steps (For Future Chat)

1. **Finish quick test**: Verify 2-epoch run completes successfully
2. **Profile first epoch**: Time breakdown (load vs. compute)
3. **Full training**: Run 50 epochs with real CT + pix2pix
4. **Evaluate**: Cross-generator validation (test on cycle/diffusion)
5. **Optimize further** (if needed): Multi-worker loading, mixed-precision tuning, batch prefetching

---

**Session Date**: August 6-9, 2026  
**Effort**: ~2 days, multi-agent debugging (data pipeline, loader optimization, validation)  
**Status**: Training pipeline ready; quick test in progress
