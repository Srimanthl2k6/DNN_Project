## Debug Run — 2026-05-02

### Pass 1: Functional Issues
- **Unhandled Exception in Entropy Calculation**
  - **Location**: `src/evaluate.py`
  - **Problem**: Potential `NaN` generation during correlation calculation if entropy variance is identically zero.
  - **Cause**: SciPy's `pearsonr` and `spearmanr` throw exceptions or return `NaN`s when standard deviation drops to zero (e.g., if a model predicts perfectly uniform distributions).
  - **Fix**: Wrap correlation components in a strict `try/except` block or fallback assignment when variance equals 0.

### Pass 2: Security Issues
- **Unsafe Deserialization Vulnerability**
  - **Location**: `src/evaluate.py`
  - **Risk**: High
  - **Exploit Scenario**: The script utilizes `torch.load(ckpt_path)`, which inherently utilizes Python's `pickle`. If an attacker were to manipulate or replace the `.pth` files in your `checkpoints/` directory with a crafted payload, loading it will trigger arbitrary remote code execution on the host machine.
  - **Fix**: Update the command to strictly load tensor parameters utilizing `torch.load(ckpt_path, weights_only=True)`.

### Pass 3: Performance & Design
- **Suboptimal GPU Memory Transfer**
  - **Location**: `src/train.py` / `src/data.py` (during `DataLoader` instantiation)
  - **Problem**: Your Dataloaders are being constructed with `num_workers=2`, but critically lack `pin_memory=True`. 
  - **Impact**: PyTorch has to allocate pageable (unlocked) memory to copy tensors into the GPU, resulting in a significant bottleneck slowing down computation speed across epochs. 
  - **Optimization**: Inject `pin_memory=True` parameters into your `get_cifar10h_dataloaders()` arguments whenever your `device` detects CUDA.

## Debug Run — 2026-05-02 (Update)

### Pass 1: Execution & Numerical
- Silently Broken Target Gradients in JSDivergenceLoss (NEW)
  - File: src/losses.py
  - Lines: 43
  - Evidence: `self.kl(log_m, preds)` passes predicting probabilities as target argument. `KLDivLoss` assumes `target` is static by default mathematically blocking gradient flows for the M-Q divergence sub-component unless overridden explicitly.
  - Root Cause: PyTorch KLDivLoss computes backward passes toward input mappings, not target mappings.
  - Fix: Hardcode JS Divergence mathematically: 0.5 * (target * (log(target + eps) - log_m) + preds * (log(preds + eps) - log_m)).sum(-1).mean()

### Pass 2: ML Logic
- Naive Binning Edge-case in Expected Calibration Error (NEW)
  - File: src/evaluate_auto.py
  - Problem: `in_bin = (preds_hard > bin_boundaries[i]) & ...` misses absolute 0 confidence predictions because first floor check is strictly greater-than.
  - Why it breaks experiment validity: While softmax inherently stays above 0, clipping interventions or FP16 rounding could drop values fully into the omitted margin corrupting calibration.
  - Fix: Change the 0th bin index logic to include `=` explicitly when `i==0`.

### Pass 3: System & Performance
- Single-Threaded Looping evaluating Cosine Similarity (NEW)
  - File: src/evaluate_auto.py
  - Bottleneck: Executing `cosine_similarity([targets[i]], [preds[i]])` enclosed inside a sequential list comprehension.
  - Impact: Dramatically throttles tensor aggregation speeds.
  - Optimization: Replace with parallel Numpy operations natively `np.sum(targets * preds, axis=1) / (np.linalg.norm(targets, axis=1) * np.linalg.norm(preds, axis=1))`.


## Debug Run — 2026-05-03 (Enforcement Pass)

### Issue: FGSM Clipping Destruction on Normalized Tensors

#### Reproduction
```python
# test_bug.py triggered
```
Result: Target clipped uniformly to exactly [0.0000, 1.0000] destroying standard deviation mapping.

#### Fix Applied
```python
# robust.py modifications bridging min_val: (0.0 - mean) / std
```
#### Verification
```python
# test_bug.py re-triggered
```
Result: Target bounded correctly across [-1.9395, 2.1265]

#### Regression Checks
* Gradients valid: YES
* NaNs present: NO
* Shape consistency: YES

#### Status
FIX VERIFIED
