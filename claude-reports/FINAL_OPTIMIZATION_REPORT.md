# Final Optimization Report - All-In-One Beat Tracking Pipeline

## Summary

Successfully backported optimizations from madmom benchmarking work into all-in-two production codebase:

1. **DBN Decoder**: Integrated Cython-optimized DBN (1.5-1.9x faster than madmom)
2. **Spectrogram Backends**: Activated at-parity torch/MLX implementations (3.9-5.7x faster than madmom)
3. **Backend Selection**: Updated defaults to prefer optimized backends over madmom

**Standalone Spectrogram Performance:**
- **Madmom**: 1,989ms (baseline)
- **Torch**: 463ms (4.3x faster)
- **MLX**: 352ms (5.7x faster)

**Numerical Parity:** ✅ ≤1e-6 max difference vs madmom baseline

---

## Optimizations Completed

### 1. DBN Decoder Backport

**Files Modified:**
- `src/allin1/postprocessing/dbn_native.py` - Copied from madmom benchmarking work
- `src/allin1/postprocessing/_viterbi_cython.pyx` - Cython-optimized Viterbi loop
- `src/allin1/postprocessing/metrical.py` - Updated import
- `src/allin1/postprocessing/metrical_mlx.py` - Updated import

**Changes:**
- Replaced `from madmom.features.downbeats import DBNDownBeatTrackingProcessor`
- With `from .dbn_native import DBNDownBeatTrackingProcessor`
- Added `__call__` method for madmom API compatibility
- Built Cython extension: `_viterbi_cython.cpython-311-darwin.so`

**Results:**
- 1.5-1.9x speedup vs madmom DBN
- 99.4% parity maintained

### 2. Spectrogram Backend Backport

**Files Modified:**
- `src/allin1/spectrogram.py` - Exact copy from `spectrogram_standalone.py` (validated implementation)
- `src/allin1/analyze.py` - Updated default backend selection
- `src/allin1/cli.py` - Delegated backend selection to analyze()

**Implementation Details:**

The torch and MLX implementations were carefully validated in the madmom benchmarking work to:
1. Match madmom's exact centered framing semantics (`_signal_frame`, `_frame_signal`)
2. Replicate madmom's log-filterbank matrix computation (`_log_filterbank_matrix`)
3. Maintain numerical parity (≤1e-6 difference)
4. Optimize performance through GPU acceleration

**Key Implementation:**
```python
def _torch_log_spectrogram(signal, sample_rate, device, dtype):
    # CPU framing (matches madmom exactly)
    frames = _frame_signal(signal, frame_size, hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    fb = _log_filterbank_matrix(...)

    # Transfer to GPU
    frames_t = torch.tensor(frames, device=device, dtype=getattr(torch, dtype))
    window_t = torch.tensor(window, device=device, dtype=frames_t.dtype)

    # GPU processing
    fft_in = frames_t * window_t
    stft = torch.fft.fft(fft_in, n=fft_size, dim=1)[:, :num_fft_bins]
    mag = stft.abs()
    fb_t = torch.tensor(fb, device=device, dtype=mag.dtype)
    filtered = mag @ fb_t
    logged = torch.log10(filtered + 1.0)

    return logged.detach().cpu().numpy()
```

**Results:**
- Torch: 4.3x faster than madmom (1,989ms → 463ms)
- MLX: 5.7x faster than madmom (1,989ms → 352ms)
- Numerical parity: ✅ ≤1e-6 max difference

### 3. Backend Selection Updates

**analyze.py (lines 126-134):**
```python
if spec_backend == "auto":
    # Prefer optimized backends (mlx/torch) over madmom
    if demucs_device == "mlx" or device == "mlx":
        spec_backend = "mlx"
    else:
        # Use torch backend for all other devices (4.3x faster than madmom)
        spec_backend = "torch"
        # Auto-configure torch device to match model device when possible
        if device in ("mps", "cuda", "cpu"):
            spec_torch_device = device
```

---

## Full Pipeline Performance

### Latest Timings (timings_optimized_final.jsonl)

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| Demix | 4,082 | demucs-mlx (4 stems) |
| **Spectrogram** | **413** | MLX backend ✅ |
| Model Load | 266 | One-time cost |
| **Neural Network** | **27** | MLX model |
| Spec Load | 3 | NumPy load + MLX conversion |
| **Postprocess** | **1,834** | Total postprocessing |
| - Metrical Prep | 1,152 | Activation computation |
| - **Metrical DBN** | **644** | Optimized Cython ✅ |
| - Functional Probs | 1 | MLX sigmoid/softmax |
| - Functional Maxima | 2 | NumPy local maxima |
| - Functional Boundaries | 36 | Peak picking |
| Save | 1 | JSON write |

**Total Processing Time:** ~6.6 seconds (from demixed audio to beat tracking results)

---

## Remaining Optimization Opportunities

### 1. MLX→NumPy→MLX Conversions (High Priority)

**Current Issue:**
When running full MLX pipeline (demucs_device='mlx', device='mlx'), unnecessary type conversions occur:

```
Spectrogram: MLX processing → np.array(logged) → save .npy
            ↓
Neural Net:  np.load() → mx.array(spec) → MLX processing
```

**Impact:**
- Spectrogram stage saves: `return np.array(logged)` (spectrogram.py:242)
- Neural network loads: `spec = mx.array(np.load(spec_path))` (helpers.py:76-77)

**Proposed Solution:**
When `multiprocess=False` and `device='mlx'`, add in-memory path:
- Extract spectrogram directly to MLX
- Pass MLX array to neural network
- Bypass disk I/O entirely
- Expected savings: ~50-100ms per track

**Implementation Complexity:** Medium (requires refactoring analyze.py pipeline flow)

### 2. Window and Filterbank Caching

**Current Behavior:**
Window and filterbank are recomputed for every stem (bass, drums, other, vocals):
```python
# Called 4 times per track
window = np.hanning(frame_size).astype(np.float32)
fb = _log_filterbank_matrix(num_fft_bins, sample_rate, ...)
```

**Proposed Solution:**
Cache these constant matrices:
```python
# Compute once per sample rate, reuse for all stems
_WINDOW_CACHE = {}
_FILTERBANK_CACHE = {}
```

**Expected Savings:** ~10-20ms per track
**Implementation Complexity:** Low

### 3. Batch Spectrogram Processing

**Current Behavior:**
Process 4 stems sequentially:
```python
for stem in ['bass', 'drums', 'other', 'vocals']:
    spec = _mlx_log_spectrogram(signal, sr)
```

**Proposed Solution:**
Batch process all 4 stems in single GPU call:
```python
# Stack all stems: (4, num_samples)
# Process batch: (4, num_frames, frame_size) → (4, num_frames, num_bins)
```

**Expected Savings:** 2-3x faster spectrogram extraction (413ms → ~150ms)
**Implementation Complexity:** Medium

### 4. MLX Local Maxima Implementation

**Current Behavior:**
functional_mlx.py converts to NumPy for local maxima:
```python
prob_sections = np.array(raw_prob_sections)  # MLX → NumPy
prob_sections, _ = local_maxima_numpy_window(prob_sections, ...)
```

**Proposed Solution:**
Implement `local_maxima_mlx` using MLX operations:
```python
# Use MLX's sliding window / unfold operations
# Keep everything in MLX until absolutely necessary
```

**Expected Savings:** ~5-10ms per track
**Implementation Complexity:** Low-Medium

---

## Validation

### Test Command
```bash
cd /tmp
/Users/sam/Code/all-in-two/.venv/bin/python test_optimized_spectrograms.py
```

### Expected Output
```
Madmom:  1989.0ms (baseline)
Torch:    463.1ms (4.3x faster)
MLX:      351.5ms (5.7x faster)
✓ Torch at parity
✓ MLX at parity
```

---

## Key Learnings

### 1. Backport Validated Implementations Exactly

The torch and MLX spectrogram implementations in `spectrogram_standalone.py` were carefully validated for:
- Exact madmom semantic equivalence (centered framing)
- Numerical parity (≤1e-6 difference)
- Performance optimization

**Lesson:** When backporting, copy the exact validated implementation rather than re-optimizing or modifying.

### 2. Minimize Cross-Device Conversions

Even when both stages use the same device (MLX), unnecessary conversions harm performance:
- Spectrogram → NumPy (for disk save)
- NumPy → MLX (for neural network)

**Lesson:** Design pipelines to minimize device/format conversions between contiguous stages.

### 3. Caching Improves Performance

Both DBN and spectrogram optimizations benefit from caching:
- DBN: Cython-compiled Viterbi loop
- Spectrogram: Could cache window/filterbank matrices

**Lesson:** Profile to identify recomputed constants and cache them appropriately.

---

## Summary

**Completed Optimizations:** ✅
1. DBN decoder: 1.5-1.9x faster, 99.4% parity
2. Spectrogram backends: 4.3-5.7x faster, numerical parity maintained
3. Backend selection: Defaults to optimized implementations

**Remaining Work:**
1. Eliminate MLX→NumPy→MLX conversions in full MLX pipeline
2. Cache window/filterbank matrices
3. Batch process spectrograms for 4 stems
4. Implement MLX local maxima

**Overall Impact:**
- Spectrogram: 4.3-5.7x faster than madmom
- DBN: 1.5-1.9x faster than madmom
- Full pipeline: Approaching interactive speeds (~6-7 seconds per track)
- Numerical accuracy: Maintained (≤1e-6 difference)

---

**Optimization Date**: 2026-01-31
**Status**: Validated and production-ready
**Research Quality**: Maintained numerical parity and semantic equivalence
