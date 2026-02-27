# MLX/Torch Separation Refactoring Plan

## Goal
Separate MLX and torch codepaths for cleaner, more maintainable code.

## Current State
- Single `analyze()` function with large if/else block (lines 287-435)
- ~150 lines of near-duplicate code for MLX vs torch inference
- All other logic is shared (setup, demix, spec, cleanup)

## Refactoring Strategy

### Option A: Extract Helper Functions (CHOSEN)
```python
def analyze(...):
    # Shared setup (demix, spec, etc.)
    if todo_paths:
        if device == "mlx":
            new_results = _run_mlx_inference(...)
        else:
            new_results = _run_torch_inference(...)
        results.extend(new_results)
    # Shared cleanup
```

**Pros:**
- Minimal code duplication
- Shared logic stays DRY
- Lower risk (smaller change)
- Easier to maintain parity

**Cons:**
- Still imports both torch and mlx

### Option B: Separate Files (NOT CHOSEN)
Create analyze_mlx.py and analyze_torch.py

**Pros:**
- Cleaner imports
- Total separation

**Cons:**
- Duplicates all shared logic
- Higher maintenance burden
- Higher risk

## Implementation Steps

### Step 1: Create _run_mlx_inference() helper
Extract lines 287-361 into new function:
- Parameters: paths, spec_paths, model_container, model_loader_thread, model_name, mlx_weights_*, include_*, timings_*, out_dir, _emit_timing
- Returns: list of AnalysisResult

### Step 2: Create _run_torch_inference() helper
Extract lines 362-435 into new function:
- Parameters: paths, spec_paths, model_container, model_loader_thread, model_name, device, include_*, timings_*, out_dir, _emit_timing
- Returns: list of AnalysisResult

### Step 3: Update analyze() to call helpers
Replace lines 287-435 with simple if/else calling the helpers

### Step 4: Validation
- Verify no logic changed
- Test both MLX and torch paths
- Check parity maintained

## Safety Checks
- [ ] No logic modified, only moved
- [ ] All parameters passed correctly
- [ ] Timing emissions preserved
- [ ] Error handling unchanged
- [ ] Return values match original

## Estimated Changes
- Add ~80 lines (two new functions)
- Remove ~150 lines (deduplicated if/else)
- Net: ~70 lines removed
- Files modified: 1 (analyze.py)
