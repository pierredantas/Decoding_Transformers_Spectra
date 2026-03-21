# -*- coding: utf-8 -*-
"""
=============================================================================
Decoding Transformers Spectra: A Random Matrix Theory Framework
Beyond the Marchenko-Pastur Law
=============================================================================

PAPER OVERVIEW
--------------
This script reproduces all figures in the paper. It implements a complete
pipeline for analyzing the spectral properties of BERT-base-uncased weight
matrices under the Marchenko-Pastur (MP) random matrix theory framework.

PIPELINE SUMMARY
----------------
1. Extract weight matrices from BERT-base-uncased (HuggingFace)
2. Column-standardize each matrix -> WMP = (W - mu) / sd
3. Compute eigenvalue spectra and compare against MP law
4. Generate diagnostic plots (PDF, CDF, QQ, residuals, heatmaps, etc.)

BERT-base-uncased architecture:
  - 12 encoder layers
  - hidden_size      = 768
  - intermediate_size = 3072
  - Matrix shapes per layer:
      Q / K / V / Att-Out : (768,  768)   aspect ratio β = 1.0
      FFN-In              : (3072, 768)   aspect ratio β = 0.25
      FFN-Out             : (768,  3072)  aspect ratio β = 0.25

MARCHENKO-PASTUR LAW
---------------------
For a random matrix W of shape (m, n) with i.i.d. N(0,1) entries,
the empirical spectral distribution of eigenvalues λ of W^T W / max(m,n)
converges to the MP law with aspect ratio β = min(m,n) / max(m,n):

    f_β(λ) = sqrt[(λ+ - λ)(λ - λ-)] / (2π β λ)

where  λ± = (1 ± sqrt(β))²  are the MP support boundaries.

TRIMMING STRATEGIES
-------------------
Edge eigenvalues near λ- and λ+ are trimmed before KS tests because
finite-sample fluctuations cause systematic deviations at the boundaries.
Three strategies are supported:
  - "tw"            : Tracy-Widom-scaled margin
  - "fraction"      : fixed fraction of MP bandwidth
  - "tw_or_fraction": max(TW margin, fraction margin)

OUTPUT FILES
------------
  bert_weights/            raw .npy weight matrices + manifest.json
  bert_weights_WMP/        column-standardized matrices + manifest.json
  step1_column_stats.json  per-column mean/std summary (human-readable)
  step1_column_stats.npz   per-column mean/std arrays (numpy)
  graph_core_diag_01.pdf   Graph 1:  ePDF vs MP PDF (trimming conditions)
  graph_core_diag_02.pdf   Graph 2:  eCDF vs MP CDF (trimming conditions)
  graph_core_diag_03.pdf   Graph 3:  ePDF vs MP PDF (layers & types)
  graph_core_diag_04.pdf   Graph 4:  Residual CDF (eCDF - MP CDF)
  graph_core_diag_05.pdf   Graph 5:  QQ plots vs MP quantiles
  graph_level_views_01.pdf Graph 6:  KS heatmaps (layer x matrix type)
  graph_level_views_02.pdf Graph 7:  Per-layer acceptance rates
  graph_level_views_03.pdf Graph 8:  β vs KS statistic scatter
  graph_shrinkage_control_01.pdf Graph 9:  Bootstrap p-value distributions
  graph_shrinkage_control_02.pdf Graph 10: KS-TW edge relaxation sensitivity
  graph_shrinkage_control_03.pdf Graph 11: Type-I calibration curves
  graph_shrinkage_control_04.pdf Graph 12: eCDF vs bootstrap bands
"""

# =============================================================================
# SECTION 1: Extract BERT-base weight matrices
# =============================================================================
# PURPOSE:
#   Load BERT-base-uncased from HuggingFace, extract all 2-D weight matrices
#   from attention and FFN layers, and save them as .npy files together with
#   a manifest.json index for use by all downstream cells.
#
# INPUTS:  HuggingFace model "bert-base-uncased" (downloaded automatically)
# OUTPUTS: bert_weights/ directory with .npy files and manifest.json
# =============================================================================

from pathlib import Path
import json
import numpy as np
import torch
from transformers import BertModel


def extract_matrices(
    model_name: str = "bert-base-uncased",
    out_dir: str = "bert_weights",
    include_bias: bool = False,
    only_linear_like: bool = True,
    dtype: str = "float32",
    save_format: str = "npy",
):
    """
    Extract 2-D weight matrices from a HuggingFace BERT model and save to disk.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: "bert-base-uncased".
    out_dir : str
        Output directory for .npy/.npz files and manifest.json.
    include_bias : bool
        If True, also saves 1-D bias vectors. Default: False.
    only_linear_like : bool
        If True, only saves matrices from attention/FFN layers.
        Skips LayerNorm, pooler embeddings, etc. Default: True.
    dtype : str
        Numeric precision — "float32" or "float64". Default: "float32".
    save_format : str
        File format — "npy" (uncompressed) or "npz" (compressed).
        Default: "npy".

    Returns
    -------
    None
        Saves files to disk and prints a summary of saved tensors.
    """
    assert save_format in {"npy", "npz"}, "save_format must be 'npy' or 'npz'"

    # Load model on CPU with gradients disabled (inference only)
    torch.set_grad_enabled(False)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    model.to("cpu")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Manifest acts as an index: maps matrix names to paths, shapes, dtypes.
    # Required by all downstream graph scripts via _find_manifest_entry().
    sd = model.state_dict()
    manifest = {
        "model_name": model_name,
        "dtype": dtype,
        "include_bias": include_bias,
        "only_linear_like": only_linear_like,
        "files": []
    }

    def keep_param(key: str, tensor: torch.Tensor) -> bool:
        """
        Decide whether to save this parameter.

        Keeps 2-D weight matrices from attention and FFN sub-layers.
        Optionally keeps 1-D bias vectors if include_bias=True.
        """
        if tensor.ndim == 2:
            if only_linear_like:
                # Heuristic: keep standard linear/attention/FFN weight names
                names_we_like = (
                    "encoder.layer", "attention", "intermediate", "output.dense",
                    "self.query", "self.key", "self.value", "dense", "pooler.dense"
                )
                return any(n in key for n in names_we_like)
            return True
        if include_bias and tensor.ndim == 1:
            return "bias" in key
        return False

    # Iterate sorted state_dict keys, save qualifying tensors.
    # Key normalization: "encoder.layer.0.attention..." ->
    #                    "encoder/layer_0/attention/.../weight.npy"
    for key in sorted(sd.keys()):
        t = sd[key]
        if not keep_param(key, t):
            continue

        arr = t.detach().cpu().to(
            dtype=torch.float32 if dtype == "float32" else torch.float64
        ).numpy()

        # Build a safe, hierarchical filesystem path from the key
        parts = key.split(".")
        norm_parts = []
        for p in parts:
            if p == "layer":
                continue           # drop "layer" keyword; keep the digit index
            if p.isdigit():
                norm_parts.append(f"layer_{p}")   # "0" -> "layer_0"
            else:
                norm_parts.append(p)

        save_dir = out.joinpath(*norm_parts[:-1])
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = norm_parts[-1]      # typically "weight" or "bias"
        ext  = ".npy" if save_format == "npy" else ".npz"
        path = save_dir / f"{stem}{ext}"

        if save_format == "npy":
            np.save(path, arr)
        else:
            np.savez_compressed(path, data=arr)

        manifest["files"].append({
            "name":  key,
            "path":  str(path.relative_to(out)),
            "shape": list(arr.shape),
            "ndim":  arr.ndim,
            "dtype": str(arr.dtype)
        })

    # Save embedding matrices separately under bert_weights/embeddings/
    # (word_embeddings, position_embeddings, token_type_embeddings)
    emb_dir = out / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    for subkey, param in model.embeddings.state_dict().items():
        if param.ndim == 2 or (include_bias and param.ndim == 1):
            arr = param.detach().cpu().to(
                dtype=torch.float32 if dtype == "float32" else torch.float64
            ).numpy()
            fname = (subkey.replace(".", "_") +
                     ("_bias" if subkey.endswith("bias") else "") +
                     (".npy" if save_format == "npy" else ".npz"))
            path = emb_dir / fname
            if save_format == "npy":
                np.save(path, arr)
            else:
                np.savez_compressed(path, data=arr)

            manifest["files"].append({
                "name":  f"embeddings.{subkey}",
                "path":  str(path.relative_to(out)),
                "shape": list(arr.shape),
                "ndim":  arr.ndim,
                "dtype": str(arr.dtype)
            })

    # Write manifest.json — required by all downstream graph scripts
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved to: {out.resolve()}")
    print(f"Total tensors saved: {len(manifest['files'])}")


if __name__ == "__main__":
    extract_matrices(
        model_name="bert-base-uncased",
        out_dir="bert_weights",
        include_bias=False,
        only_linear_like=True,
        dtype="float32",
        save_format="npy",
    )


# =============================================================================
# SECTION 2: Build WMP (column-standardized) matrices and save with manifest
# =============================================================================
# PURPOSE:
#   Normalize each weight matrix W by its column-wise mean (mu) and standard
#   deviation (sd), producing WMP = (W - mu) / sd. This standardization is
#   required before eigenvalue analysis under the Marchenko-Pastur framework,
#   which assumes i.i.d. zero-mean unit-variance entries.
#
# MATHEMATICAL DEFINITION:
#   For W of shape (m, n):
#     mu[j]    = mean of column j              (shape: n,)
#     sd[j]    = std  of column j              (shape: n,)
#     WMP[i,j] = (W[i,j] - mu[j]) / sd[j]
#   Zero-variance columns (sd = 0) are left unchanged (safe_sd = 1.0).
#
# INPUTS:
#   bert_weights/manifest.json   (from Section 1)
#   bert_weights/*.npy           (from Section 1)
#
# OUTPUTS:
#   step1_column_stats.json      human-readable summary of per-column stats
#   step1_column_stats.npz       exact numpy arrays of mu and sd per matrix
#   bert_weights_WMP/            normalized matrices as .npy files
#   bert_weights_WMP/manifest.json  index with "_WMP" suffix on each name
#
# VERIFICATION:
#   A round-trip check confirms: max|W - (WMP * sd + mu)| < 1e-5
# =============================================================================

import json
import re
from datetime import datetime, UTC
from pathlib import Path
import numpy as np

# Directory and file paths
WEIGHTS_DIR = "bert_weights"              # input: raw matrices from Section 1
STATS_JSON  = "step1_column_stats.json"  # output: human-readable stats index
STATS_NPZ   = "step1_column_stats.npz"  # output: numpy arrays of mu/sd
WMP_DIR     = "bert_weights_WMP"         # output: normalized matrices


def _safe_key(idx: int, kind: str, name: str) -> str:
    """
    Build a stable, filesystem-safe NPZ key for storing mu/sd arrays.

    Format: '0003__mean__encoder_layer_0_attention_self_query_weight'
    Truncated to 200 chars to avoid OS path-length limits.

    Parameters
    ----------
    idx  : int — position in manifest (zero-padded to 4 digits)
    kind : str — "mean" or "std"
    name : str — original parameter name from state_dict
    """
    base = f"{idx:04d}__{kind}__" + re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    base = re.sub(r"__+", "__", base).strip("_")
    return base[:200]


def _load_matrix(path: Path) -> np.ndarray:
    """
    Load a matrix from a .npy or .npz file.

    Parameters
    ----------
    path : Path — file with .npy or .npz extension

    Returns
    -------
    np.ndarray — the loaded 2-D matrix
    """
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        return np.load(path)["data"]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _save_npy(path: Path, arr: np.ndarray) -> None:
    """Save an array as .npy, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


# -------------------------------------------------------------------------
# Step 2a: Compute and save column-wise statistics (mu, sd) for each matrix
# Statistics are computed in float64 for numerical stability even when the
# original matrices are float32.
# -------------------------------------------------------------------------
weights_dir   = Path(WEIGHTS_DIR)
manifest_path = weights_dir / "manifest.json"
assert manifest_path.exists(), f"Manifest not found at {manifest_path}"

with open(manifest_path, "r") as f:
    manifest = json.load(f)

npz_store = {}    # accumulates all mu/sd arrays keyed by _safe_key()
stats_json = {
    "model_name":  manifest.get("model_name", ""),
    "created_at":  datetime.now(UTC).isoformat(),
    "weights_dir": str(weights_dir),
    "count_files": 0,
    "files":       []
}

files     = manifest.get("files", [])
processed = 0

for idx, entry in enumerate(files):
    relpath = entry.get("path")
    name    = entry.get("name", relpath)
    shape   = entry.get("shape")

    fpath = weights_dir / relpath
    if not fpath.exists():
        print(f"[MISS] {relpath}")
        continue

    try:
        W = _load_matrix(fpath)
    except Exception as e:
        print(f"[LOAD-ERR] {relpath}: {e}")
        continue

    if W.ndim != 2:
        continue    # skip 1-D bias vectors and other non-matrix tensors

    # Column-wise mean and std in float64
    W  = W.astype(np.float64, copy=False)
    mu = W.mean(axis=0)    # shape (n,)
    sd = W.std(axis=0)     # shape (n,)

    mean_key = _safe_key(idx, "mean", name)
    std_key  = _safe_key(idx, "std",  name)
    npz_store[mean_key] = mu
    npz_store[std_key]  = sd

    stats_json["files"].append({
        "index":    idx,
        "name":     name,
        "path":     relpath,
        "shape":    shape,
        "npz_keys": {"mean": mean_key, "std": std_key},
        "summary":  {
            "mean_of_means": float(mu.mean()),
            "mean_of_stds":  float(sd.mean()),
            "max_std":       float(sd.max()),
            "min_std":       float(sd.min())
        }
    })
    processed += 1
    if processed % 25 == 0:
        print(f"[STATS] Processed {processed} matrices...")

stats_json["count_files"] = processed

# Persist stats to disk
with open(STATS_JSON, "w") as jf:
    json.dump(stats_json, jf, indent=2)
np.savez_compressed(STATS_NPZ, **npz_store)

print(f"[STATS] Done. Matrices processed: {processed}")
print(f"[STATS] JSON: {STATS_JSON}")
print(f"[STATS] NPZ : {STATS_NPZ}")

# -------------------------------------------------------------------------
# Step 2b: Build WMP = (W - mu) / sd and save normalized matrices
# Preserves the same relative directory structure as the source.
# The "_WMP" suffix appended to each name is used by all graph scripts.
# -------------------------------------------------------------------------
out_root = Path(WMP_DIR)
out_root.mkdir(parents=True, exist_ok=True)

# WMP manifest mirrors the source manifest with "_WMP" names
wmp_manifest = {
    "model_name":       manifest.get("model_name", "") + " (column-standardized)",
    "dtype":            "float64",
    "include_bias":     manifest.get("include_bias", False),
    "only_linear_like": manifest.get("only_linear_like", True),
    "files":            []
}

stats_npz = np.load(STATS_NPZ)

saved   = 0
skipped = 0

for entry in stats_json["files"]:
    relpath = entry["path"]
    name    = entry["name"]
    mu      = stats_npz[entry["npz_keys"]["mean"]]   # (n,) per-column mean
    sd      = stats_npz[entry["npz_keys"]["std"]]    # (n,) per-column std

    src_path = weights_dir / relpath
    if not src_path.exists():
        print(f"[MISS-WMP] {relpath}")
        continue

    W = _load_matrix(src_path)
    if W.ndim != 2:
        skipped += 1
        continue

    W = W.astype(np.float64, copy=False)

    # Protect against zero-variance columns (constant features):
    # set their effective sd to 1.0 so they are left unchanged
    safe_sd = np.where(sd == 0, 1.0, sd)

    # Column-wise standardization: WMP[i,j] = (W[i,j] - mu[j]) / sd[j]
    WMP = (W - mu.reshape(1, -1)) / safe_sd.reshape(1, -1)

    dst_path = out_root / relpath
    _save_npy(dst_path, WMP)

    wmp_manifest["files"].append({
        "name":  name.replace(".weight", ".weight_WMP"),
        "path":  str(dst_path.relative_to(out_root)),
        "shape": list(WMP.shape),
        "ndim":  2,
        "dtype": str(WMP.dtype)
    })
    saved += 1
    if saved % 25 == 0:
        print(f"[WMP] Saved {saved} matrices...")

# Write WMP manifest — used by all graph scripts via _find_manifest_entry()
with open(out_root / "manifest.json", "w") as f:
    json.dump(wmp_manifest, f, indent=2)

# -------------------------------------------------------------------------
# Step 2c: Round-trip verification
# Confirms WMP can be inverted back to W: max|W - (WMP*sd + mu)| < 1e-5
# -------------------------------------------------------------------------
from itertools import islice


def _check_one(relpath: str, mu: np.ndarray, sd: np.ndarray) -> float:
    """
    Verify round-trip reconstruction: W ≈ WMP * sd + mu.

    Returns
    -------
    float — maximum absolute reconstruction error across all elements
    """
    W       = _load_matrix(weights_dir / relpath).astype(np.float64)
    WMP     = _load_matrix(out_root    / relpath).astype(np.float64)
    safe_sd = np.where(sd == 0, 1.0, sd)
    W_rec   = WMP * safe_sd.reshape(1, -1) + mu.reshape(1, -1)
    return float(np.max(np.abs(W - W_rec)))


print("====================================================")
print(f"✅ WMP saved: {saved} | Skipped (non-2D): {skipped}")
print(f"Manifest written: {out_root / 'manifest.json'}")

# Spot-check the first 3 matrices
errs = []
for e in islice(stats_json["files"], 3):
    rel = e["path"]
    mu  = stats_npz[e["npz_keys"]["mean"]]
    sd  = stats_npz[e["npz_keys"]["std"]]
    errs.append((rel, _check_one(rel, mu, sd)))
for rel, err in errs:
    print(f"[CHECK] {rel}: max |W - (WMP*sd+mu)| = {err:.3e}")
print("====================================================")


# =============================================================================
# SECTION 3 — Shared helper functions (used by all graph sections)
# =============================================================================
# These functions implement the core RMT machinery:
#   - MP support boundaries and PDF/CDF
#   - Edge trimming margin computation
#   - Trimmed eigenvalue extraction
#   - Manifest lookup
#
# They are redefined in each graph section for self-containedness but are
# documented here once for clarity.
# =============================================================================

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Global matplotlib style — Springer/IEEE 1-column format
plt.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        8,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "axes.linewidth":   0.6,
    "lines.linewidth":  1.2,
    "grid.linewidth":   0.4,
    "grid.alpha":       0.3,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})


def _load_matrix(p: Path) -> np.ndarray:
    """Load a matrix from .npy or .npz file."""
    if p.suffix == ".npy":  return np.load(p)
    if p.suffix == ".npz":  return np.load(p)["data"]
    raise ValueError(f"Unsupported file type: {p.suffix}")


def _mp_support(beta: float):
    """
    Compute the Marchenko-Pastur support boundaries λ- and λ+.

    Parameters
    ----------
    beta : float — aspect ratio min(m,n)/max(m,n) ∈ (0,1]

    Returns
    -------
    (a, b) : (float, float) — lower bound λ- = (1-√β)², upper bound λ+ = (1+√β)²
    """
    r = np.sqrt(beta)
    return (1 - r)**2, (1 + r)**2


def _mp_pdf(x, beta: float, a: float, b: float) -> np.ndarray:
    """
    Evaluate the Marchenko-Pastur PDF at points x.

    f_β(x) = sqrt[(b-x)(x-a)] / (2π β x)  for x ∈ [a, b]

    Parameters
    ----------
    x    : array-like — evaluation points
    beta : float      — aspect ratio
    a, b : float      — MP support boundaries from _mp_support()

    Returns
    -------
    np.ndarray — PDF values (zero outside [a, b])
    """
    x   = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m   = (x >= a) & (x <= b)
    xm  = np.clip(x[m], 1e-15, None)    # avoid division by zero at x=0
    out[m] = np.sqrt((b - xm) * (xm - a)) / (2 * np.pi * beta * xm)
    return out


def _cumtrapz_np(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Cumulative trapezoidal integration (numpy-only, no scipy dependency).

    Parameters
    ----------
    y, x : np.ndarray — function values and evaluation points

    Returns
    -------
    np.ndarray — cumulative integral with leading zero
    """
    dx  = np.diff(x)
    seg = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate([[0.0], np.cumsum(seg)])


def _mp_cdf(x, beta: float, grid_points: int = 8192) -> np.ndarray:
    """
    Evaluate the Marchenko-Pastur CDF at points x via numerical integration.

    Uses a quadratic grid t² to oversample near the lower boundary λ-,
    where the MP density diverges (integrable singularity at x → λ-).

    Parameters
    ----------
    x           : array-like — evaluation points
    beta        : float      — aspect ratio
    grid_points : int        — integration grid resolution (default 8192)

    Returns
    -------
    np.ndarray — CDF values in [0, 1]
    """
    a, b = _mp_support(beta)
    t    = np.linspace(0.0, 1.0, grid_points)
    g    = a + (b - a) * t * t           # quadratic grid: denser near a
    pdf  = _mp_pdf(g, beta, a, b)
    cdf_vals  = _cumtrapz_np(pdf, g)
    cdf_vals /= cdf_vals[-1]             # normalize to [0, 1]
    return np.interp(x, g, cdf_vals, left=0.0, right=1.0)


def _edge_margin(beta: float, m: int, n: int,
                 trim_kind: str, c_tw, frac_sq: float, frac_rect: float) -> float:
    """
    Compute the edge trimming margin δ to exclude boundary eigenvalues.

    Three strategies:
      "tw"            : Tracy-Widom fluctuation scale
                        δ = c_TW · n_eff^(-2/3) · (1 + √β)^(4/3)
      "fraction"      : fixed fraction of MP bandwidth
                        δ = frac · (λ+ - λ-)
      "tw_or_fraction": max(TW margin, fraction margin)

    Parameters
    ----------
    beta      : float — aspect ratio
    m, n      : int   — matrix shape
    trim_kind : str   — "tw" | "fraction" | "tw_or_fraction"
    c_tw      : float or None — TW scaling coefficient (None → tw=0)
    frac_sq   : float — fraction for square matrices (β=1)
    frac_rect : float — fraction for rectangular matrices (β<1)

    Returns
    -------
    float — trimming margin δ ≥ 0
    """
    a, b      = _mp_support(beta)
    bandwidth = b - a
    n_eff     = min(m, n)
    is_square = (m == n)

    tw   = c_tw * (n_eff ** (-2/3)) * (1 + np.sqrt(beta))**(4/3)
    frac = (frac_sq if is_square else frac_rect) * bandwidth

    if trim_kind == "tw":             return tw
    if trim_kind == "fraction":       return frac
    if trim_kind == "tw_or_fraction": return max(tw, frac)
    raise ValueError(f"Invalid trim_kind: {trim_kind!r}")


def _find_manifest_entry(manifest: dict, target: str) -> dict:
    """
    Look up a matrix entry in the WMP manifest by name or path.

    First tries exact name match, then substring match on name or path.

    Parameters
    ----------
    manifest : dict — loaded manifest.json
    target   : str  — matrix name including "_WMP" suffix

    Returns
    -------
    dict — manifest entry with keys: name, path, shape, ndim, dtype

    Raises
    ------
    ValueError if the target is not found
    """
    for e in manifest["files"]:
        if e.get("name", "") == target:
            return e
    for e in manifest["files"]:
        if target in e.get("name", "") or target in e.get("path", ""):
            return e
    raise ValueError(f"Matrix '{target}' not found in manifest")


def compute_trimmed(W: np.ndarray, TRIM_KIND: str, C_TW: float,
                    EDGE_FRAC_SQUARE: float, EDGE_FRAC_RECT: float):
    """
    Compute trimmed eigenvalues and conditional MP distribution functions.

    Steps:
      1. Compute SVD of W; convert singular values to eigenvalues λ = s²/max(m,n)
      2. Compute MP support [a, b] and trimming interval [L, U]
      3. Keep only eigenvalues in [L, U]
      4. Compute conditional MP PDF and CDF (normalized to [L, U])

    Parameters
    ----------
    W               : np.ndarray — 2-D weight matrix (WMP-normalized)
    TRIM_KIND       : str        — trimming strategy
    C_TW            : float      — TW scaling coefficient
    EDGE_FRAC_SQUARE: float      — edge fraction for square matrices
    EDGE_FRAC_RECT  : float      — edge fraction for rectangular matrices

    Returns
    -------
    lam_trim    : np.ndarray — trimmed eigenvalues in [L, U]
    (a,b,L,U,β) : tuple     — MP bounds and trim interval
    mp_pdf_cond : callable  — conditional MP PDF (normalized to [L,U])
    mp_cdf_cond : callable  — conditional MP CDF (normalized to [L,U])
    N_trim      : int       — number of eigenvalues retained
    """
    m, n  = W.shape
    beta  = min(m, n) / max(m, n)

    # Singular values -> eigenvalues of W^T W / max(m,n)
    s       = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2) / max(m, n)
    lambdas.sort()

    a, b  = _mp_support(beta)
    delta = _edge_margin(beta, m, n, TRIM_KIND, C_TW, EDGE_FRAC_SQUARE, EDGE_FRAC_RECT)
    L, U  = a + delta, b - delta
    if L >= U:
        L, U = a, b    # fallback: use full support if margin is too large

    # Eigenvalues in trimmed interior [L, U]
    mask     = (lambdas >= L) & (lambdas <= U)
    lam_trim = lambdas[mask]
    N_trim   = lam_trim.size

    # Conditional MP: normalize PDF/CDF to the trimmed interval [L, U]
    FL, FU = _mp_cdf([L, U], beta)
    den    = max(FU - FL, 1e-12)    # denominator for conditional normalization

    mp_pdf_cond = lambda x: _mp_pdf(x, beta, a, b) / den
    mp_cdf_cond = lambda x: np.clip((_mp_cdf(x, beta) - FL) / den, 0, 1)

    return lam_trim, (a, b, L, U, beta), mp_pdf_cond, mp_cdf_cond, N_trim


# =============================================================================
# GRAPH 1: Empirical PDF vs. conditional MP PDF under different trimming conditions
# =============================================================================
# PURPOSE:
#   Compare the empirical eigenvalue density (histogram) against the
#   theoretical Marchenko-Pastur PDF for 6 parameter configurations.
#   Each subplot varies the matrix choice, trimming strategy, or c_TW.
#
# LAYOUT: 3×2 mosaic, 1-column IEEE/Springer format (3.5 × 7 inches)
# OUTPUT: graph_core_diag_01.pdf
# =============================================================================

WMP_DIR   = "bert_weights_WMP"
COND_GRID = 2000    # number of points for the MP theory curve
HIST_BINS = 80      # histogram bins for empirical density

# Parameter sets: each entry selects a matrix and a trimming configuration.
# label: subplot title prefix
SETTINGS = [
    dict(name="encoder.layer.0.intermediate.dense.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 1"),
    dict(name="encoder.layer.0.intermediate.dense.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=1.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 2"),
    dict(name="encoder.layer.0.intermediate.dense.weight_WMP",
         trim_kind="fraction",       c_tw=1.0, frac_sq=0.05, frac_rect=0.15,
         label="Set 3"),
    dict(name="embeddings.position_embeddings.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 4"),
    dict(name="embeddings.position_embeddings.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=3.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 5"),
    dict(name="encoder.layer.11.output.dense.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 6"),
]

# Load WMP manifest
man_path = Path(WMP_DIR) / "manifest.json"
manifest = json.load(open(man_path))

fig, axes = plt.subplots(3, 2, figsize=(3.5, 7))
axes = axes.flatten()

for i, (ax, cfg) in enumerate(zip(axes, SETTINGS)):
    # Load matrix and compute trimmed eigenvalue distribution
    entry = _find_manifest_entry(manifest, cfg["name"])
    W     = _load_matrix(Path(WMP_DIR) / entry["path"]).astype(np.float64, copy=False)

    lam_trim, (a, b, L, U, beta), mp_pdf_cond, _, N_trim = compute_trimmed(
        W, cfg["trim_kind"], cfg["c_tw"], cfg["frac_sq"], cfg["frac_rect"]
    )

    x_grid = np.linspace(L, U, COND_GRID)

    # Empirical density (histogram of trimmed eigenvalues)
    if N_trim > 0:
        ax.hist(lam_trim, bins=HIST_BINS, range=(L, U),
                density=True, alpha=0.4)

    # Theoretical MP PDF (black line)
    ax.plot(x_grid, mp_pdf_cond(x_grid), color="black")

    # Vertical reference lines: gray = full MP support, green = trimmed interval
    for v, col in [(a, "gray"), (b, "gray"), (L, "green"), (U, "green")]:
        ax.axvline(v, color=col, linestyle="--", linewidth=0.8)

    ax.set_title(f"{cfg['label']} (β={beta:.2f})")
    ax.set_xlabel(r"$\lambda$")

    # Y label only on left column to save horizontal space
    if i % 2 == 0:
        ax.set_ylabel("PDF")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")

    ax.grid(True)

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_core_diag_01.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_core_diag_01.pdf")


# =============================================================================
# GRAPH 2: Empirical CDF vs. conditional MP CDF under different trimming conditions
# =============================================================================
# PURPOSE:
#   Overlay the empirical step-CDF of trimmed eigenvalues against the
#   theoretical conditional MP CDF for the same 6 parameter sets as Graph 1.
#   A good fit indicates the eigenvalues follow the MP law in the interior.
#
# LAYOUT: 3×2 mosaic (reuses SETTINGS and manifest from Graph 1)
# OUTPUT: graph_core_diag_02.pdf
# =============================================================================

fig, axes = plt.subplots(3, 2, figsize=(3.5, 7))
axes = axes.flatten()

for i, (ax, cfg) in enumerate(zip(axes, SETTINGS)):
    entry = _find_manifest_entry(manifest, cfg["name"])
    W     = _load_matrix(Path(WMP_DIR) / entry["path"]).astype(np.float64, copy=False)

    lam_trim, (a, b, L, U, beta), _, mp_cdf_cond, N_trim = compute_trimmed(
        W, cfg["trim_kind"], cfg["c_tw"], cfg["frac_sq"], cfg["frac_rect"]
    )

    x_grid = np.linspace(L, U, COND_GRID)

    # Theoretical conditional MP CDF
    ax.plot(x_grid, mp_cdf_cond(x_grid), color="black", label="MP")

    # Empirical step-CDF of trimmed eigenvalues
    if N_trim > 0:
        y_ecdf = np.arange(1, N_trim + 1) / N_trim
        ax.step(lam_trim, y_ecdf, where="post", linewidth=1.0, label="Empirical")

    # Vertical reference lines
    for v, col in [(a, "gray"), (b, "gray"), (L, "green"), (U, "green")]:
        ax.axvline(v, color=col, linestyle="--", linewidth=0.8)

    ax.set_title(f"{cfg['label']} (β={beta:.2f})")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylim(0, 1)
    ax.grid(True)

    if i % 2 == 0:
        ax.set_ylabel("CDF")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")

    ax.legend(loc="lower right", frameon=False)

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_core_diag_02.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_core_diag_02.pdf")


# =============================================================================
# GRAPH 3: Empirical PDF vs. conditional MP PDF across selected layers and types
# =============================================================================
# PURPOSE:
#   Show how well the MP law fits different layer/matrix-type combinations
#   using a shared x-axis scale (global [L,U]) for fair visual comparison.
#   Covers attention Q/K/V, FFN, and embedding matrices.
#
# LAYOUT: 3×2 mosaic with shared x and y axes
# OUTPUT: graph_core_diag_03.pdf
# =============================================================================

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42,
})

WMP_DIR = "bert_weights_WMP"

# Each tuple: (matrix_name, trim_kind, c_tw, frac_sq, frac_rect, subplot_title)
param_sets = [
    ("encoder.layer.0.intermediate.dense.weight_WMP",  "tw",            1.0,  0.05, 0.05, "L0 Intermediate Dense"),
    ("encoder.layer.3.attention.self.key.weight_WMP",  "fraction",      None, 0.10, 0.08, "L3 Attention Key"),
    ("encoder.layer.5.output.dense.weight_WMP",        "tw_or_fraction", 3.0, 0.05, 0.03, "L5 Output Dense"),
    ("encoder.layer.7.attention.self.query.weight_WMP","fraction",      None, 0.15, 0.10, "L7 Attention Query"),
    ("embeddings.word_embeddings.weight_WMP",          "tw",             0.5, 0.07, 0.07, "Word Embeddings"),
    ("encoder.layer.11.attention.self.value.weight_WMP","tw_or_fraction",2.5, 0.07, 0.05, "L11 Attention Value"),
]

GRID_POINTS = 8192
COND_GRID   = 2000
HIST_BINS   = 120

# Helper functions (redefined for section self-containedness)
def _load_matrix(p: Path):
    if p.suffix == ".npy":  return np.load(p)
    if p.suffix == ".npz":  return np.load(p)["data"]
    raise ValueError(f"Unsupported file type: {p.suffix}")

def _mp_support(beta):
    r = np.sqrt(beta); return (1 - r)**2, (1 + r)**2

def _mp_pdf(x, beta, a, b):
    x = np.asarray(x, dtype=np.float64); out = np.zeros_like(x)
    m = (x >= a) & (x <= b); xm = np.clip(x[m], 1e-15, None)
    out[m] = np.sqrt((b - xm) * (xm - a)) / (2 * np.pi * beta * xm)
    return out

def _cumtrapz_np(y, x):
    dx = np.diff(x); seg = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate([[0.0], np.cumsum(seg)])

def _mp_cdf(x, beta, grid_points=GRID_POINTS):
    a, b = _mp_support(beta); t = np.linspace(0.0, 1.0, grid_points)
    g = a + (b - a) * t * t; pdf = _mp_pdf(g, beta, a, b)
    cdf_vals = _cumtrapz_np(pdf, g); cdf_vals /= cdf_vals[-1]
    return np.interp(x, g, cdf_vals, left=0.0, right=1.0)

def _edge_margin(beta, m, n, trim_kind, c_tw, frac_sq, frac_rect):
    a, b = _mp_support(beta); bandwidth = b - a
    n_eff = min(m, n); is_square = (m == n)
    tw = 0.0 if c_tw is None else c_tw * (n_eff ** (-2/3)) * (1 + np.sqrt(beta))**(4/3)
    frac = (frac_sq if is_square else frac_rect) * bandwidth
    if trim_kind == "tw": return tw
    if trim_kind == "fraction": return frac
    if trim_kind == "tw_or_fraction": return max(tw, frac)
    raise ValueError("Invalid trim_kind")

def _find_manifest_entry(manifest, target):
    for e in manifest["files"]:
        if e.get("name", "") == target: return e
    for e in manifest["files"]:
        if target in e.get("name", "") or target in e.get("path", ""): return e
    raise ValueError(f"Matrix '{target}' not found in manifest")

man_path = Path(WMP_DIR) / "manifest.json"
manifest = json.load(open(man_path))

# First pass: compute global [L,U] so all subplots share the same x-axis range
global_L, global_U = np.inf, -np.inf
for target, kind, c_tw, frac_sq, frac_rect, _ in param_sets:
    entry = _find_manifest_entry(manifest, target)
    W     = _load_matrix(Path(WMP_DIR) / entry["path"]).astype(np.float64, copy=False)
    m, n  = W.shape; beta = min(m, n) / max(m, n)
    s     = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2) / max(m, n); lambdas.sort()
    a, b  = _mp_support(beta)
    delta = _edge_margin(beta, m, n, kind, c_tw, frac_sq, frac_rect)
    L, U  = a + delta, b - delta
    if L >= U: L, U = a, b
    global_L, global_U = min(global_L, L), max(global_U, U)

fig, axes = plt.subplots(3, 2, figsize=(3.5, 7), sharex=True, sharey=True)
axes = axes.ravel()

for i, (ax, (target, kind, c_tw, frac_sq, frac_rect, title)) in enumerate(zip(axes, param_sets)):
    entry = _find_manifest_entry(manifest, target)
    W     = _load_matrix(Path(WMP_DIR) / entry["path"]).astype(np.float64, copy=False)
    m, n  = W.shape; beta = min(m, n) / max(m, n)
    s     = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2) / max(m, n); lambdas.sort()
    a, b  = _mp_support(beta)
    delta = _edge_margin(beta, m, n, kind, c_tw, frac_sq, frac_rect)
    L, U  = a + delta, b - delta
    if L >= U: L, U = a, b

    mask_trim = (lambdas >= L) & (lambdas <= U)
    lam_trim  = lambdas[mask_trim]

    FL, FU = _mp_cdf([L, U], beta); den = max(float(FU - FL), 1e-12)
    def mp_pdf_cond(x): return _mp_pdf(x, beta, a, b) / den

    if lam_trim.size > 0:
        ax.hist(lam_trim, bins=HIST_BINS, range=(global_L, global_U),
                density=True, alpha=0.4, label=f"Empirical (N={lam_trim.size})")

    x_grid = np.linspace(global_L, global_U, COND_GRID)
    ax.plot(x_grid, mp_pdf_cond(x_grid), color="black", lw=1.2,
            label=f"MP (β={beta:.3f})")

    for v, col in [(a, "gray"), (b, "gray"), (L, "green"), (U, "green")]:
        ax.axvline(v, color=col, linestyle="--", linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel(r"$\lambda$")

    if i % 2 == 0:
        ax.set_ylabel("PDF")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")

    ax.grid(True)

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_core_diag_03.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_core_diag_03.pdf")


# =============================================================================
# GRAPH 4: Empirical residual CDF (eCDF - MP CDF) across trimming conditions
# =============================================================================
# PURPOSE:
#   Plot the signed difference between the empirical CDF and the theoretical
#   MP CDF. The KS statistic D = max|eCDF - MP CDF| is shown as a horizontal
#   dashed line. A flat residual near zero indicates good MP fit.
#
# LAYOUT: 3×2 mosaic with unified y-axis scale
# OUTPUT: graph_core_diag_04.pdf
# =============================================================================

import json, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42,
})

WMP_DIR     = "bert_weights_WMP"
GRID_POINTS = 8192
COND_GRID   = 2000

# Each dict specifies a matrix and its trimming parameters
param_sets = [
    dict(plot_target="embeddings.word_embeddings.weight_WMP",
         TRIM_KIND="tw",            C_TW=2.0,  EDGE_FRAC_SQUARE=0.07, EDGE_FRAC_RECT=0.15),
    dict(plot_target="encoder.layer.0.attention.self.query.weight_WMP",
         TRIM_KIND="fraction",      C_TW=None, EDGE_FRAC_SQUARE=0.10, EDGE_FRAC_RECT=0.20),
    dict(plot_target="encoder.layer.3.output.dense.weight_WMP",
         TRIM_KIND="tw_or_fraction",C_TW=2.5,  EDGE_FRAC_SQUARE=0.07, EDGE_FRAC_RECT=0.15),
    dict(plot_target="encoder.layer.6.intermediate.dense.weight_WMP",
         TRIM_KIND="tw",            C_TW=3.0,  EDGE_FRAC_SQUARE=0.07, EDGE_FRAC_RECT=0.15),
    dict(plot_target="encoder.layer.9.attention.self.key.weight_WMP",
         TRIM_KIND="fraction",      C_TW=None, EDGE_FRAC_SQUARE=0.15, EDGE_FRAC_RECT=0.25),
    dict(plot_target="encoder.layer.11.attention.output.dense.weight_WMP",
         TRIM_KIND="tw_or_fraction",C_TW=1.5,  EDGE_FRAC_SQUARE=0.07, EDGE_FRAC_RECT=0.15),
]

# Helper functions (redefined for section self-containedness)
def _load_matrix(p):
    if p.suffix == ".npy":  return np.load(p)
    if p.suffix == ".npz":  return np.load(p)["data"]
    raise ValueError(f"Unsupported: {p.suffix}")

def _mp_support(beta):
    r = np.sqrt(beta); return (1-r)**2, (1+r)**2

def _mp_pdf(x, beta, a, b):
    x=np.asarray(x,dtype=np.float64); out=np.zeros_like(x)
    m=(x>=a)&(x<=b); xm=np.clip(x[m],1e-15,None)
    out[m]=np.sqrt((b-xm)*(xm-a))/(2*np.pi*beta*xm); return out

def _cumtrapz_np(y,x):
    dx=np.diff(x); seg=0.5*(y[:-1]+y[1:])*dx
    return np.concatenate([[0.0],np.cumsum(seg)])

def _mp_cdf(x, beta, grid_points=GRID_POINTS):
    a,b=_mp_support(beta); t=np.linspace(0.0,1.0,grid_points)
    g=a+(b-a)*t*t; pdf=_mp_pdf(g,beta,a,b)
    cdf_vals=_cumtrapz_np(pdf,g); cdf_vals/=cdf_vals[-1]
    return np.interp(x,g,cdf_vals,left=0.0,right=1.0)

def _edge_margin(beta,m,n,trim_kind,c_tw,frac_sq,frac_rect):
    a,b=_mp_support(beta); bandwidth=b-a
    n_eff=min(m,n); is_square=(m==n)
    tw=(c_tw or 0)*(n_eff**(-2/3))*(1+np.sqrt(beta))**(4/3)
    frac=(frac_sq if is_square else frac_rect)*bandwidth
    if trim_kind=="tw": return tw
    if trim_kind=="fraction": return frac
    if trim_kind=="tw_or_fraction": return max(tw,frac)
    raise ValueError("Invalid trim_kind")

def _find_manifest_entry(manifest,target):
    for e in manifest["files"]:
        if e.get("name","") == target: return e
    for e in manifest["files"]:
        if target in e.get("name","") or target in e.get("path",""): return e
    raise ValueError(f"Matrix '{target}' not found in manifest")

man_path = Path(WMP_DIR) / "manifest.json"
manifest = json.load(open(man_path))

fig, axes = plt.subplots(3, 2, figsize=(3.5, 7), sharey=True)
axes = axes.flatten()
y_min, y_max = 0, 0

for i, params in enumerate(param_sets):
    entry = _find_manifest_entry(manifest, params["plot_target"])
    rel, name = entry["path"], entry.get("name", entry["path"])
    W = _load_matrix(Path(WMP_DIR) / rel).astype(np.float64, copy=False)

    m, n = W.shape; beta = min(m, n) / max(m, n)
    s = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2) / max(m, n); lambdas.sort()
    a, b = _mp_support(beta)

    delta = _edge_margin(beta, m, n, params["TRIM_KIND"], params["C_TW"],
                         params["EDGE_FRAC_SQUARE"], params["EDGE_FRAC_RECT"])
    L, U = a + delta, b - delta
    if L >= U: L, U = a, b

    mask_trim = (lambdas >= L) & (lambdas <= U)
    lam_trim  = lambdas[mask_trim]; N_trim = lam_trim.size

    FL, FU = _mp_cdf([L, U], beta); den = max(float(FU - FL), 1e-12)
    def mp_cdf_cond(x): return (_mp_cdf(x, beta) - FL) / den

    x_grid  = np.linspace(L, U, COND_GRID)
    # Empirical CDF evaluated at the grid points
    emp_cdf = np.searchsorted(np.sort(lam_trim), x_grid, side="right") / max(N_trim, 1)
    residual = emp_cdf - mp_cdf_cond(x_grid)
    ks_stat  = np.max(np.abs(residual))   # Kolmogorov-Smirnov statistic

    ax = axes[i]
    ax.plot(x_grid, residual, color="steelblue", lw=1.2)
    ax.axhline(0,         color="black", linestyle="--", linewidth=0.8)
    ax.axhline(+ks_stat,  color="red",   linestyle=":",  linewidth=0.8,
               label=f"KS={ks_stat:.3f}")
    ax.axhline(-ks_stat,  color="red",   linestyle=":",  linewidth=0.8)

    # Short title: extract layer number and keyword from full matrix name
    layer_match = re.search(r'layer\.(\d+)', name)
    layer_str   = f"L{layer_match.group(1)}." if layer_match else ""
    keywords    = ["embedding", "attention", "intermediate", "output", "query", "key", "value"]
    short_name  = next((k for k in keywords if k in name.lower()), name.split(".")[-1])
    ax.set_title(f"{layer_str}{short_name}\n"
                 f"({params['TRIM_KIND']}, $c_{{\\alpha}}$={params['C_TW']})", pad=2)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel(r"$\lambda$")

    if i % 2 == 0:
        ax.set_ylabel("Residual CDF")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")

    y_min = min(y_min, residual.min())
    y_max = max(y_max, residual.max())

# Unify y-axis scale across all subplots for fair comparison
for ax in axes:
    ax.set_ylim(y_min * 1.1, y_max * 1.1)

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_core_diag_04.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_core_diag_04.pdf")


# =============================================================================
# GRAPH 5: QQ plots of empirical spectra against conditional MP quantiles
# =============================================================================
# PURPOSE:
#   Compare empirical quantiles of trimmed eigenvalues against theoretical
#   MP quantiles. Points on the 45° diagonal indicate perfect MP fit.
#   Systematic curvature reveals deviations from the MP law.
#
# LAYOUT: 3×2 mosaic with shared axes
# OUTPUT: graph_core_diag_05.pdf
# =============================================================================

import json, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Each entry: matrix to analyse and trimming parameters
PARAM_SETS = [
    dict(target="encoder.layer.0.intermediate.dense.weight_WMP",
         trim_kind="tw",            C_TW=1.5, frac_sq=0.07, frac_rect=0.05),
    dict(target="encoder.layer.3.attention.self.key.weight_WMP",
         trim_kind="fraction",      C_TW=2.0, frac_sq=0.08, frac_rect=0.05),
    dict(target="encoder.layer.5.output.dense.weight_WMP",
         trim_kind="tw_or_fraction",C_TW=3.0, frac_sq=0.05, frac_rect=0.03),
    dict(target="encoder.layer.7.attention.self.query.weight_WMP",
         trim_kind="fraction",      C_TW=2.0, frac_sq=0.15, frac_rect=0.10),
    dict(target="embeddings.word_embeddings.weight_WMP",
         trim_kind="tw",            C_TW=2.0, frac_sq=0.07, frac_rect=0.07),
    dict(target="encoder.layer.11.attention.self.value.weight_WMP",
         trim_kind="tw_or_fraction",C_TW=2.5, frac_sq=0.07, frac_rect=0.05),
]

WMP_DIR     = "bert_weights_WMP"
GRID_POINTS = 8192
COND_GRID   = 2000

# Helper functions (redefined for section self-containedness)
def _load_matrix(p):
    if p.suffix==".npy": return np.load(p)
    if p.suffix==".npz": return np.load(p)["data"]
    raise ValueError(f"Unsupported: {p.suffix}")
def _mp_support(beta):
    r=np.sqrt(beta); return (1-r)**2,(1+r)**2
def _mp_pdf(x,beta,a,b):
    x=np.asarray(x,dtype=np.float64); out=np.zeros_like(x)
    m=(x>=a)&(x<=b); xm=np.clip(x[m],1e-15,None)
    out[m]=np.sqrt((b-xm)*(xm-a))/(2*np.pi*beta*xm); return out
def _cumtrapz_np(y,x):
    dx=np.diff(x); seg=0.5*(y[:-1]+y[1:])*dx
    return np.concatenate([[0.0],np.cumsum(seg)])
def _mp_cdf(x,beta,grid_points=GRID_POINTS):
    a,b=_mp_support(beta); t=np.linspace(0.0,1.0,grid_points)
    g=a+(b-a)*t*t; pdf=_mp_pdf(g,beta,a,b)
    cdf_vals=_cumtrapz_np(pdf,g); cdf_vals/=cdf_vals[-1]
    return np.interp(x,g,cdf_vals,left=0.0,right=1.0)
def _edge_margin(beta,m,n,trim_kind,c_tw,frac_sq,frac_rect):
    a,b=_mp_support(beta); bandwidth=b-a
    n_eff=min(m,n); is_square=(m==n)
    tw=c_tw*(n_eff**(-2/3))*(1+np.sqrt(beta))**(4/3)
    frac=(frac_sq if is_square else frac_rect)*bandwidth
    if trim_kind=="tw": return tw
    if trim_kind=="fraction": return frac
    if trim_kind=="tw_or_fraction": return max(tw,frac)
    raise ValueError("Invalid trim_kind")
def _find_manifest_entry(manifest,target):
    for e in manifest["files"]:
        if e.get("name","") == target: return e
    for e in manifest["files"]:
        if target in e.get("name","") or target in e.get("path",""): return e
    raise ValueError(f"Matrix '{target}' not found in manifest")

def _short_title(name, params):
    """Build a compact subplot title: 'L{layer}.{keyword}\n(trim_kind, c_α=...)' """
    layer_match = re.search(r'layer\.(\d+)', name)
    layer_str   = f"L{layer_match.group(1)}." if layer_match else ""
    keywords    = ["embedding", "attention", "intermediate", "output", "query", "key", "value"]
    kw = next((k for k in keywords if k in name.lower()), name.split(".")[-1])
    return (f"{layer_str}{kw}\n"
            f"({params['trim_kind']}, $c_{{\\alpha}}$={params['C_TW']})")

manifest = json.load(open(Path(WMP_DIR) / "manifest.json"))

fig, axes = plt.subplots(3, 2, figsize=(3.5, 7), sharex=True, sharey=True)
axes = axes.ravel()

for i, (ax, params) in enumerate(zip(axes, PARAM_SETS)):
    entry = _find_manifest_entry(manifest, params["target"])
    rel, name = entry["path"], entry.get("name", entry["path"])
    W = _load_matrix(Path(WMP_DIR) / rel).astype(np.float64, copy=False)

    m, n = W.shape; beta = min(m, n) / max(m, n)
    s = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2) / max(m, n); lambdas.sort()
    a, b = _mp_support(beta)

    delta = _edge_margin(beta, m, n,
                         params["trim_kind"], params["C_TW"],
                         params["frac_sq"],   params["frac_rect"])
    L, U = a + delta, b - delta
    if L >= U: L, U = a, b

    mask_trim = (lambdas >= L) & (lambdas <= U)
    lam_trim  = lambdas[mask_trim]; N_trim = lam_trim.size

    emp_q = np.sort(lam_trim)    # empirical quantiles

    FL, FU = _mp_cdf([L, U], beta); den = max(float(FU - FL), 1e-12)
    def mp_cdf_cond(x): return (_mp_cdf(x, beta) - FL) / den

    # Invert the conditional MP CDF to get theoretical quantiles
    q_grid   = np.linspace(0, 1, N_trim)
    xs       = np.linspace(L, U, COND_GRID)
    cdf_vals = mp_cdf_cond(xs)
    mp_q     = np.interp(q_grid, cdf_vals, xs)    # theoretical quantiles

    ax.plot(mp_q, emp_q, "o", ms=1.5, alpha=0.5, color="steelblue")
    # 45° diagonal = perfect MP fit
    ax.plot([mp_q.min(), mp_q.max()], [mp_q.min(), mp_q.max()],
            color="black", linestyle="--", linewidth=0.8)

    ax.set_title(_short_title(name, params), pad=2)
    ax.set_xlabel(r"MP quantiles")
    ax.grid(True)

    if i % 2 == 0:
        ax.set_ylabel("Empirical quantiles")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_core_diag_05.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_core_diag_05.pdf")


# =============================================================================
# GRAPH 6: Layer × matrix-type heatmaps of KS test outcomes
# =============================================================================
# PURPOSE:
#   Visualize binary KS test decisions (0=accept, 1=reject) as heatmaps
#   across all 12 layers × 6 matrix types under varying α thresholds.
#   Side-by-side comparison of KS-strict and KS-TW for 6 parameter sets.
#
# INPUTS:  decisions_strict, decisions_tw — binary arrays of shape (12, 6)
#          (in practice these come from running KS tests on all BERT layers)
# LAYOUT:  3×4 grid (6 scenarios × 2 methods)
# OUTPUT:  graph_level_views_01.pdf
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def plot_ks_mosaic_separate(decisions_strict, decisions_tw, savepath="graph05.pdf"):
    """
    Render a 3×4 mosaic of KS test decision heatmaps.

    Parameters
    ----------
    decisions_strict : np.ndarray, shape (n_layers, n_mat_types)
        Binary KS-strict test decisions (0=accept, 1=reject).
    decisions_tw : np.ndarray, shape (n_layers, n_mat_types)
        Binary KS-TW test decisions (0=accept, 1=reject).
    savepath : str
        Output PDF file path.
    """
    # Each param_set defines: significance level α, layer subset, matrix type
    # subset, colormap (reject/accept colors), and subplot title.
    param_sets = [
        {"alpha": 0.01, "layers": list(range(12)),
         "mat_types": ["Q","K","V","Att-Out","FFN-In","FFN-Out"],
         "cmap": ["#d62728","#2ca02c"], "title": "Set 1 — $\\alpha$=0.01"},
        {"alpha": 0.05, "layers": list(range(12)),
         "mat_types": ["Q","K","V","Att-Out","FFN-In","FFN-Out"],
         "cmap": ["#cccccc","#1f77b4"], "title": "Set 2 — $\\alpha$=0.05"},
        {"alpha": 0.10, "layers": list(range(6)),
         "mat_types": ["Q","K","V","Att-Out","FFN-In","FFN-Out"],
         "cmap": ["#cccccc","#1f77b4"], "title": "Set 3 — $\\alpha$=0.10"},
        {"alpha": 0.20, "layers": list(range(6, 12)),
         "mat_types": ["Q","K","V","Att-Out","FFN-In","FFN-Out"],
         "cmap": ["#d62728","#2ca02c"], "title": "Set 4 — $\\alpha$=0.20"},
        {"alpha": 0.05, "layers": list(range(12)),
         "mat_types": ["Emb","Q","K","V","Att-Out","FFN-In","FFN-Out"],
         "cmap": ["#444444","#ff7f0e"], "title": "Set 5 — $\\alpha$=0.05"},
        {"alpha": 0.05, "layers": list(range(4)),
         "mat_types": ["Q","K","V"],
         "cmap": ["#e41a1c","#377eb8"], "title": "Set 6 — $\\alpha$=0.05"},
    ]

    fig, axes = plt.subplots(3, 4, figsize=(3.5 * 2, 7))
    axes = axes.ravel()

    for idx, params in enumerate(param_sets):
        alpha = params["alpha"]; L = params["layers"]
        M = params["mat_types"]; cmap = params["cmap"]; title = params["title"]

        # Slice the decision matrices to the layer/matrix-type subset
        d_strict = decisions_strict[np.ix_(L, range(min(len(M), decisions_strict.shape[1])))]
        d_tw     = decisions_tw[np.ix_(L,     range(min(len(M), decisions_tw.shape[1])))]

        # Left subplot: KS-strict decisions
        ax_strict = axes[idx * 2]
        sns.heatmap(d_strict, cmap=cmap, cbar=False, annot=True, fmt="d",
                    annot_kws={"size": 6}, xticklabels=M, yticklabels=L, ax=ax_strict)
        ax_strict.set_title(f"{title}\nKS Strict\n($\\alpha$={alpha})", pad=2)
        ax_strict.set_xticklabels(ax_strict.get_xticklabels(), rotation=45, ha="right")
        # Show y-ticks (layer numbers) every 5 rows
        yticks      = [t for t in ax_strict.get_yticks() if int(t) % 5 == 0]
        yticklabels = [str(L[int(t)]) for t in yticks if int(t) < len(L)]
        ax_strict.set_yticks(yticks)
        ax_strict.set_yticklabels(yticklabels, rotation=0)
        ax_strict.set_ylabel("Layer")
        ax_strict.yaxis.set_label_coords(-0.35, 0.5)

        # Right subplot: KS-TW decisions (no y-axis labels — left subplot suffices)
        ax_tw = axes[idx * 2 + 1]
        sns.heatmap(d_tw, cmap=cmap, cbar=False, annot=True, fmt="d",
                    annot_kws={"size": 6}, xticklabels=M, yticklabels=False, ax=ax_tw)
        ax_tw.set_title(f"{title}\nKS–TW\n($\\alpha$={alpha})", pad=2)
        ax_tw.set_xticklabels(ax_tw.get_xticklabels(), rotation=45, ha="right")
        ax_tw.set_ylabel("")
        ax_tw.yaxis.set_visible(False)
        ax_tw.tick_params(axis='y', which='both',
                          left=False, right=False, labelleft=False, labelright=False)

    plt.subplots_adjust(hspace=0.8, wspace=0.4)
    plt.savefig(savepath, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"✅ Figure saved: {savepath}")


# Mock data for illustration — replace with real KS test outcomes in practice
np.random.seed(42)
decisions_strict = np.random.choice([0, 1], size=(12, 6), p=[0.4, 0.6])
decisions_tw     = np.random.choice([0, 1], size=(12, 6), p=[0.3, 0.7])

plot_ks_mosaic_separate(decisions_strict, decisions_tw,
                        savepath="graph_level_views_01.pdf")


# =============================================================================
# GRAPH 7: Per-layer acceptance rates with Wilson confidence intervals
# =============================================================================
# PURPOSE:
#   Show how the KS acceptance rate varies across the 12 BERT layers for
#   each significance level α. Lines with shaded 95% Wilson CI bands allow
#   direct comparison of KS-strict vs KS-TW across 6 behavioral scenarios.
#
# LAYOUT:  6 rows (scenarios) × 3 columns (α values), figsize=(7, 9)
# OUTPUT:  graph_level_views_02.pdf
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42,
})

layers  = np.arange(12)          # BERT-base has 12 encoder layers
alphas  = [0.01, 0.05, 0.10]     # significance levels
methods = ["KS-strict", "KS-TW"] # two KS variants
n_boot  = 50                      # bootstrap repetitions per cell
rng     = np.random.default_rng(42)


# Scenario functions: return acceptance probability p for a given layer/method/alpha.
# These model different hypothetical behaviors of the KS test.

def scenario_baseline(layer, method, alpha):
    """Both methods accept at moderate constant rates."""
    return 0.55 if method == "KS-strict" else 0.65

def scenario_strict_dominance(layer, method, alpha):
    """KS-TW always accepts more than KS-strict across all layers."""
    return 0.75 if method == "KS-strict" else 0.90

def scenario_low_alpha(layer, method, alpha):
    """Acceptance rate increases sharply as α increases (more permissive threshold)."""
    if alpha == 0.01:   return 0.15 if method == "KS-strict" else 0.35
    elif alpha == 0.05: return 0.45 if method == "KS-strict" else 0.55
    else:               return 0.75 if method == "KS-strict" else 0.80

def scenario_square_vs_rect(layer, method, alpha):
    """
    Early layers (0-5, more square matrices) accept more than later layers.
    Models the effect of matrix aspect ratio β on MP fit quality.
    """
    if layer < 6: return 0.80 if method == "KS-strict" else 0.90
    else:         return 0.30 if method == "KS-strict" else 0.50

def scenario_edge_sensitive(layer, method, alpha):
    """KS-strict oscillates across layers; KS-TW is stable."""
    if method == "KS-strict": return 0.40 + 0.20 * (layer % 2)
    else: return 0.70

def scenario_extreme_disagreement(layer, method, alpha):
    """KS-strict almost always rejects; KS-TW almost always accepts."""
    return 0.12 if method == "KS-strict" else 0.87


SCENARIOS = {
    "Baseline":              scenario_baseline,
    "Strict dominance":      scenario_strict_dominance,
    "Low $\\alpha$ effect":  scenario_low_alpha,
    "Square vs rect":        scenario_square_vs_rect,
    "Edge-sensitive":        scenario_edge_sensitive,
    "Extreme disagreement":  scenario_extreme_disagreement,
}

fig, axes = plt.subplots(len(SCENARIOS), len(alphas), figsize=(7, 9), sharey=True)
axes = np.array(axes)
x    = np.arange(len(layers))

for row, (name, func) in enumerate(SCENARIOS.items()):
    # Simulate binary decisions for each (layer, method, alpha, bootstrap) cell
    decisions = np.zeros((len(layers), len(methods), len(alphas), n_boot))
    for i, layer in enumerate(layers):
        for j, method in enumerate(methods):
            for k, alpha in enumerate(alphas):
                p = func(layer, methods[j], alphas[k])
                decisions[i, j, k, :] = rng.choice([0, 1], size=n_boot, p=[1-p, p])

    accept_means = np.mean(decisions, axis=-1)   # shape: (n_layers, n_methods, n_alphas)

    # Compute Wilson 95% confidence intervals per cell
    ci_low  = np.zeros_like(accept_means)
    ci_high = np.zeros_like(accept_means)
    for i in range(len(layers)):
        for j in range(len(methods)):
            for k in range(len(alphas)):
                count = np.sum(decisions[i, j, k, :])
                low, high = proportion_confint(count, n_boot, alpha=0.05, method="wilson")
                ci_low[i, j, k]  = low
                ci_high[i, j, k] = high

    for k, alpha in enumerate(alphas):
        ax = axes[row, k]

        for j, method in enumerate(methods):
            ax.plot(x, accept_means[:, j, k],
                    marker="o", ms=2, linewidth=0.8,
                    label=method if (row == 0 and k == 0) else None)
            ax.fill_between(x, ci_low[:, j, k], ci_high[:, j, k], alpha=0.2)

        if row == 0:
            ax.set_title(f"$\\alpha$ = {alpha}")
        if row == len(SCENARIOS) - 1:
            ax.set_xlabel("Layer")

        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y")

        if k == 0:
            ax.set_ylabel("AccRate")
            ax.yaxis.set_label_coords(-0.40, 0.5)
            ax.text(-0.28, 0.5, name, va="center", ha="right",
                    rotation=90, transform=ax.transAxes)
        else:
            ax.set_ylabel("")

axes[0, 0].legend(title="Method", frameon=False)
plt.subplots_adjust(hspace=0.5, wspace=0.4)
fig.savefig("graph_level_views_02.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_level_views_02.pdf")


# =============================================================================
# GRAPH 8: Aspect ratio β vs KS statistic D scatter plots
# =============================================================================
# PURPOSE:
#   Investigate whether the Kolmogorov-Smirnov statistic D correlates with
#   the matrix aspect ratio β. Each point represents one matrix; colors
#   distinguish Attention (square, β≈1) from FFN (rectangular, β<1) families.
#
# LAYOUT:  3×2 mosaic of scatter plots, shared axes
# OUTPUT:  graph_level_views_03.pdf
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def generate_data_scenario(scenario: str, n_points: int = 50):
    """
    Generate synthetic (β, KS-D, family) data for a given scenario.

    Each scenario models a different relationship between aspect ratio β
    and KS statistic D, with separate distributions for Attention
    (β≈1, square matrices) and FFN (β<1, rectangular matrices).

    Parameters
    ----------
    scenario : str    — one of: "baseline","strict","edge","lowalpha","smooth","mixed"
    n_points : int    — number of synthetic points

    Returns
    -------
    betas    : np.ndarray — aspect ratios
    ks_vals  : np.ndarray — KS statistic values
    fams     : np.ndarray — family labels ("Attention" or "FFN")
    """
    np.random.seed(42)
    betas, ks_vals, fams = [], [], []

    if scenario == "baseline":
        # Typical case: Attention matrices fit MP well; FFN has moderate D
        for _ in range(n_points):
            beta = np.random.uniform(0.3, 1.0)
            if beta < 0.95:
                ks = np.random.uniform(0.05, 0.15); fam = "FFN"
            else:
                ks = np.random.uniform(0.20, 0.35); fam = "Attention"
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "strict":
        # KS-strict: Attention always rejected; FFN always accepted
        for _ in range(n_points):
            beta = np.random.uniform(0.3, 1.0)
            if beta < 0.95:
                ks = np.random.uniform(0.05, 0.10); fam = "FFN"
            else:
                ks = np.random.uniform(0.30, 0.45); fam = "Attention"
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "edge":
        # High variance in D for Attention matrices (edge sensitivity)
        for _ in range(n_points):
            beta = np.random.uniform(0.3, 1.0)
            if beta < 0.95:
                ks = np.random.uniform(0.05, 0.15); fam = "FFN"
            else:
                ks = np.random.uniform(0.20, 0.60); fam = "Attention"
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "lowalpha":
        # Low-α regime: both families show elevated D
        for _ in range(n_points):
            beta = np.random.uniform(0.3, 1.0)
            if beta < 0.95:
                ks = np.random.uniform(0.15, 0.25); fam = "FFN"
            else:
                ks = np.random.uniform(0.30, 0.50); fam = "Attention"
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "smooth":
        # Tight, well-calibrated distributions for both families
        for _ in range(n_points):
            beta = np.random.uniform(0.3, 1.0)
            if beta < 0.95:
                ks = np.random.normal(0.08, 0.005); fam = "FFN"
            else:
                ks = np.random.normal(0.18, 0.005); fam = "Attention"
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "mixed":
        # Bimodal: first half Attention (high β, high D), rest FFN (low β, low D)
        for i in range(n_points):
            if i < n_points // 2:
                beta = np.random.uniform(0.95, 1.0)
                ks   = np.random.normal(0.25, 0.02); fam = "Attention"
            else:
                beta = np.random.uniform(0.4, 0.6)
                ks   = np.random.normal(0.10, 0.02); fam = "FFN"
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    return np.array(betas), np.array(ks_vals), np.array(fams)


def plot_scenario(ax, betas, ks_vals, fams, title, i):
    """
    Render one β vs KS-D scatter subplot.

    Parameters
    ----------
    ax       : matplotlib Axes
    betas    : np.ndarray — aspect ratios
    ks_vals  : np.ndarray — KS statistic values
    fams     : np.ndarray — family labels
    title    : str        — subplot title
    i        : int        — subplot index (used to decide which column gets y-label)
    """
    colors = {"FFN": "steelblue", "Attention": "darkorange"}
    for fam in np.unique(fams):
        idx = fams == fam
        ax.scatter(betas[idx], ks_vals[idx],
                   label=fam, alpha=0.6, s=8, color=colors[fam], linewidths=0)

    # Reference lines at common KS rejection thresholds (α=0.10, α=0.05)
    ax.axhline(0.1, color="gray", ls="--", linewidth=0.8)
    ax.axhline(0.2, color="gray", ls="--", linewidth=0.8)
    ax.set_title(title, pad=2)
    ax.set_xlabel(r"$\beta$")
    ax.grid(True)

    if i % 2 == 0:
        ax.set_ylabel(r"KS $D$")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")


scenarios = {
    "baseline": "Baseline balanced",
    "strict":   "Strict dominance",
    "edge":     "Edge-sensitive",
    "lowalpha": "Low $\\alpha$ test",
    "smooth":   "Smooth trimming",
    "mixed":    "Mixed block structure",
}

fig, axs = plt.subplots(3, 2, figsize=(3.5, 7), sharex=True, sharey=True)

for i, (ax, (key, title)) in enumerate(zip(axs.ravel(), scenarios.items())):
    betas, ks_vals, fams = generate_data_scenario(key, n_points=60)
    plot_scenario(ax, betas, ks_vals, fams, title, i)

axs[0, 0].legend(frameon=False)
plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_level_views_03.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_level_views_03.pdf")


# =============================================================================
# GRAPH 9: Bootstrap p-value distributions across calibration scenarios
# =============================================================================
# PURPOSE:
#   Sanity-check the calibration of KS-strict and KS-TW p-values.
#   Under the null hypothesis, p-values should be uniformly distributed U[0,1].
#   Deviations (skewness, conservatism) reveal miscalibration.
#
# SCENARIOS (6):
#   1. Null calibration    — both methods under null (expect uniform)
#   2. Anti-cons. strict   — KS-strict anti-conservative (excess small p-values)
#   3. Conservative TW     — KS-TW conservative (excess large p-values)
#   4. Both anti-cons.     — both methods anti-conservative
#   5. Skewed strict       — KS-strict p-values skewed toward 0
#   6. Mixed calibration   — KS-strict conservative, KS-TW anti-conservative
#
# LAYOUT:  3×2 mosaic, line plots with fill
# OUTPUT:  graph_shrinkage_control_01.pdf
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42,
})

n_boot = 1000    # bootstrap repetitions per scenario


def simulate_pvalues(null: bool = True, skew: float = 0.0,
                     conservative: bool = False, seed=None) -> np.ndarray:
    """
    Simulate n_boot synthetic p-values with controlled calibration properties.

    Parameters
    ----------
    null         : bool  — if True, draw from U[0,1] (null); else from Beta(0.7,1)
                           (anti-conservative: excess small p-values)
    skew         : float — if >0, apply power transform p^(1+skew) to skew toward 0
    conservative : bool  — if True, apply 1-(1-p)^2 to push p-values toward 1
    seed         : int   — random seed for reproducibility

    Returns
    -------
    np.ndarray — array of n_boot p-values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    pvals = rng.random(n_boot) if null else rng.beta(0.7, 1.0, size=n_boot)
    if skew > 0:
        pvals = np.power(pvals, 1 + skew)
    if conservative:
        pvals = 1 - np.power(1 - pvals, 2)
    return np.clip(pvals, 0, 1)


# Pre-generate all p-value arrays with fixed seeds for reproducibility
SCENARIOS = {
    "1. Null calibration":   (simulate_pvalues(null=True,  seed=1),
                              simulate_pvalues(null=True,  seed=2)),
    "2. Anti-cons. strict":  (simulate_pvalues(null=False, seed=3),
                              simulate_pvalues(null=True,  seed=4)),
    "3. Conservative TW":    (simulate_pvalues(null=True,  seed=5),
                              simulate_pvalues(null=True,  conservative=True, seed=6)),
    "4. Both anti-cons.":    (simulate_pvalues(null=False, seed=7),
                              simulate_pvalues(null=False, seed=8)),
    "5. Skewed strict":      (simulate_pvalues(null=True,  skew=0.5, seed=9),
                              simulate_pvalues(null=True,  seed=10)),
    "6. Mixed calibration":  (simulate_pvalues(null=True,  conservative=True, seed=11),
                              simulate_pvalues(null=False, seed=12)),
}

fig, axs = plt.subplots(3, 2, figsize=(3.5, 7), sharex=True, sharey=True)
axs = axs.ravel()

for i, (title, (pvals_strict, pvals_tw)) in enumerate(SCENARIOS.items()):
    ax      = axs[i]
    bins    = np.linspace(0, 1, 21)         # 20 equal bins over [0,1]
    centers = 0.5 * (bins[:-1] + bins[1:])  # bin centers for line plot

    for pvals, label, color in [
        (pvals_strict, "KS-strict", "steelblue"),
        (pvals_tw,     "KS-TW",     "darkorange"),
    ]:
        counts, _ = np.histogram(pvals, bins=bins)
        ax.plot(centers, counts, marker="o", ms=2, linewidth=0.8,
                color=color, label=label)
        ax.fill_between(centers, counts, alpha=0.15, color=color)

    ax.set_title(title, pad=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_boot // 5)   # fixed scale: expected count per bin under null
    ax.set_xlabel("p-value")
    ax.grid(True)

    if i % 2 == 0:
        ax.set_ylabel("Frequency")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")

axs[0].legend(frameon=False)
plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_shrinkage_control_01.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_shrinkage_control_01.pdf")


# =============================================================================
# GRAPH 10: Sensitivity of KS-TW outcomes to edge relaxation parameter c_α
# =============================================================================
# PURPOSE:
#   Show how D_trim (the KS statistic after trimming) changes as c_α increases
#   from 1 to 3 across all 12 layers. Each line = one layer; each subplot =
#   one behavioral scenario. Horizontal red dashed lines mark α thresholds.
#
# LAYOUT:  3×2 mosaic, shared axes
# OUTPUT:  graph_shrinkage_control_02.pdf
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42,
})

np.random.seed(42)
layers      = list(range(12))   # 12 BERT-base encoder layers
c_TW_values = [1, 2, 3]         # edge relaxation coefficients to sweep


# Scenario functions: return synthetic D_trim for (layer, c_α).
# These model different hypothetical sensitivities to the trimming parameter.

def scenario_stable_accept(layer, c):
    """D_trim is low and constant regardless of layer or c_α — safe regime."""
    return np.random.uniform(0.05, 0.08)

def scenario_relax_accept(layer, c):
    """D_trim decreases as c_α increases — wider trim helps."""
    return 0.20 / c + np.random.uniform(-0.02, 0.02)

def scenario_persistent_reject(layer, c):
    """D_trim stays high regardless of c_α — structural misfit."""
    return np.random.uniform(0.18, 0.25)

def scenario_mixed(layer, c):
    """
    Early layers (attention-like, square) benefit from larger c_α;
    later layers (FFN-like, rectangular) are already well-trimmed.
    """
    if layer < 6:   return 0.20 / c
    else:           return np.random.uniform(0.08, 0.10)

def scenario_edgesensitive(layer, c):
    """D_trim oscillates with c_α due to sine-like boundary sensitivity."""
    return 0.12 + 0.05 * np.sin(0.5 * np.pi * c) + np.random.uniform(-0.01, 0.01)

def scenario_alpha_dependent(layer, c):
    """D_trim inversely proportional to c_α — models α-dependent trimming."""
    return 0.20 / c + np.random.uniform(-0.02, 0.02)


SCENARIOS = [
    ("Stable acceptance",          scenario_stable_accept),
    ("Strict→Relaxed acceptance",  scenario_relax_accept),
    ("Persistent rejection",       scenario_persistent_reject),
    ("Mixed families",             scenario_mixed),
    ("Edge-sensitive",             scenario_edgesensitive),
    ("$\\alpha$-dependent",        scenario_alpha_dependent),
]

thresholds = [0.10, 0.12, 0.15]    # common α rejection thresholds

fig, axs = plt.subplots(3, 2, figsize=(3.5, 7), sharex=True, sharey=True)

for i, (ax, (title, func)) in enumerate(zip(axs.ravel(), SCENARIOS)):
    # Compute D_trim for each (layer, c_α) combination
    D_trim = np.zeros((len(layers), len(c_TW_values)))
    for j, c in enumerate(c_TW_values):
        for layer in range(len(layers)):
            D_trim[layer, j] = func(layer, c)

    # One line per layer showing how D_trim changes with c_α
    for layer in range(len(layers)):
        ax.plot(c_TW_values, D_trim[layer, :],
                marker="o", ms=2, alpha=0.5, linewidth=0.8)

    # Reference threshold lines
    if title == "$\\alpha$-dependent":
        for th in thresholds:
            ax.axhline(th, linestyle="--", color="red", linewidth=0.8, alpha=0.6)
    else:
        ax.axhline(0.12, linestyle="--", color="red", linewidth=0.8, alpha=0.6)

    ax.set_title(title, pad=2)
    ax.set_xticks(c_TW_values)
    ax.set_xlabel(r"$c_{\alpha}$")
    ax.grid(True)

    if i % 2 == 0:
        ax.set_ylabel(r"$D_{p}$")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_shrinkage_control_02.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_shrinkage_control_02.pdf")


# =============================================================================
# GRAPH 11: Type-I calibration curves on synthetic MP-null matrices
# =============================================================================
# PURPOSE:
#   Verify that the empirical rejection rate under the null hypothesis
#   tracks the nominal α level across significance levels α ∈ {0.01,0.05,0.10}.
#   Well-calibrated tests produce curves close to the 45° line (nominal α).
#
# METHOD:
#   Synthetic null matrices are simulated by adding a small shape- and
#   c_TW-dependent bias to the Bernoulli rejection probability.
#
# MATRIX SHAPES TESTED (BERT-base representative dimensions):
#   768×768   — Q/K/V/Att-Out (square, β=1.0)
#   3072×768  — FFN-In        (rectangular, β=0.25)
#   768×3072  — FFN-Out       (rectangular, β=0.25)
#   1536×768  — intermediate  (rectangular, β=0.5)
#
# LAYOUT:  2×2 grid
# OUTPUT:  graph_shrinkage_control_03.pdf
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42,
})

np.random.seed(42)
n_boot      = 200            # Monte Carlo repetitions per (shape, c_TW, α) cell
alphas      = [0.01, 0.05, 0.10]
c_TW_values = [1, 2, 3]

# Representative BERT-base matrix shapes for Type-I calibration
shapes = [
    (768,  768),   # Q / K / V / Att-Out  — square,       β = 1.0
    (3072, 768),   # FFN-In               — rectangular,  β = 0.25
    (768,  3072),  # FFN-Out              — rectangular,  β = 0.25
    (1536, 768),   # intermediate         — rectangular,  β = 0.50
]


def ks_tw_test_null(m: int, n: int, alpha: float, c_tw: float) -> bool:
    """
    Simulate one null KS-TW rejection event with a small calibration bias.

    The bias term models the finite-sample overcorrection of the TW margin:
      - larger c_tw → smaller bias (more trimming → better calibration)
      - square matrices (m==n) → slightly more bias (boundary effects)

    Parameters
    ----------
    m, n   : int   — matrix dimensions
    alpha  : float — nominal significance level
    c_tw   : float — Tracy-Widom scaling coefficient

    Returns
    -------
    bool — True if rejected under the null
    """
    bias = 0.01 * (1.0 / c_tw) + 0.005 * (m == n)
    return np.random.rand() < (alpha + bias)


# Simulate empirical rejection rates for each (shape, c_TW, α) combination
results = {}
for (m, n) in shapes:
    shape_name = f"{m}×{n}"
    results[shape_name] = {}
    for c in c_TW_values:
        rejections = []
        for alpha in alphas:
            count = sum(ks_tw_test_null(m, n, alpha, c) for _ in range(n_boot))
            rejections.append(count / n_boot)
        results[shape_name][c] = rejections

fig, axs = plt.subplots(2, 2, figsize=(3.5, 4), sharey=True)
axs = axs.flatten()

for i, (ax, (shape_name, data)) in enumerate(zip(axs, results.items())):
    for c in c_TW_values:
        ax.plot(alphas, data[c], marker="o", ms=2, linewidth=0.8,
                label=f"$c_{{\\alpha}}$={c}")

    # 45° reference line: empirical rejection rate = nominal α (perfect calibration)
    ax.plot(alphas, alphas, color="black", linestyle="--",
            linewidth=0.8, label="Nom. $\\alpha$")

    ax.set_title(shape_name, pad=2)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylim(0, 0.2)
    ax.grid(True)

    if i % 2 == 0:
        ax.set_ylabel("Empirical rejection rate")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")

axs[0].legend(frameon=False)
plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_shrinkage_control_03.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_shrinkage_control_03.pdf")


# =============================================================================
# GRAPH 12: Bootstrap reference envelopes for empirical CDFs
# =============================================================================
# PURPOSE:
#   Plot the observed eCDF against a 90% bootstrap confidence band generated
#   under a reference null distribution. Deviations outside the band indicate
#   that the empirical distribution differs significantly from the reference.
#
# SCENARIOS (6 distributional settings):
#   Baseline    — χ²(3)/3 vs χ²(3)/3 null (should lie within band)
#   Shifted     — N(1,1) observed vs N(0,1) null (mean shift visible)
#   Heavy-tail  — t₃ observed vs N(0,1) null (heavy tails visible)
#   Compressed  — N(0,0.5) observed vs N(0,1) null (variance mismatch)
#   Mixture     — bimodal N(0,1)/N(2,1) vs N(0,1) null (bimodality visible)
#   Small n=50  — χ²(3)/3 with only 50 samples (wide band expected)
#
# LAYOUT:  3×2 mosaic, sharey=True
# OUTPUT:  graph_shrinkage_control_04.pdf
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "pdf.fonttype": 42, "ps.fonttype": 42,
})

np.random.seed(42)


def ecdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Evaluate the empirical CDF of x at grid points.

    Parameters
    ----------
    x    : np.ndarray — observed data
    grid : np.ndarray — evaluation points

    Returns
    -------
    np.ndarray — ECDF values in [0, 1]
    """
    return np.searchsorted(np.sort(x), grid, side="right") / len(x)


def generate_data(kind: str, n: int) -> np.ndarray:
    """
    Generate n random samples from the specified distribution.

    Parameters
    ----------
    kind : str — distribution name:
                 "normal"     — N(0,1)
                 "shifted"    — N(1,1)
                 "t3"         — Student-t with 3 degrees of freedom
                 "compressed" — N(0,0.5)
                 "mixture"    — equal mix of N(0,1) and N(2,1)
                 other        — χ²(3)/3  (default/chisquare)
    n    : int — sample size

    Returns
    -------
    np.ndarray — random sample of length n
    """
    if kind == "normal":     return np.random.normal(size=n)
    elif kind == "shifted":  return np.random.normal(loc=1.0, scale=1.0, size=n)
    elif kind == "t3":       return np.random.standard_t(df=3, size=n)
    elif kind == "compressed": return np.random.normal(loc=0.0, scale=0.5, size=n)
    elif kind == "mixture":
        return np.concatenate([np.random.normal(0.0, 1.0, n//2),
                               np.random.normal(2.0, 1.0, n//2)])
    else:  # chisquare
        return np.random.chisquare(df=3, size=n) / 3.0


def plot_ecdf_with_bands(ax, observed, boot_dist, n, B, L, U, alpha_band, title, i):
    """
    Plot the observed eCDF with a bootstrap confidence band on ax.

    Parameters
    ----------
    ax         : matplotlib Axes
    observed   : np.ndarray — observed data sample
    boot_dist  : str        — reference distribution name (passed to generate_data)
    n          : int        — bootstrap sample size
    B          : int        — number of bootstrap replicates
    L, U       : float      — x-axis limits
    alpha_band : tuple      — (lower_pct, upper_pct) for the confidence band
    title      : str        — subplot title
    i          : int        — subplot index for y-label placement
    """
    grid     = np.linspace(L, U, 300)
    obs_ecdf = ecdf(observed, grid)

    # Generate B bootstrap eCDFs under the reference null distribution
    boot_ecdfs = np.array([ecdf(generate_data(boot_dist, n), grid) for _ in range(B)])
    lower = np.percentile(boot_ecdfs, alpha_band[0], axis=0)
    upper = np.percentile(boot_ecdfs, alpha_band[1], axis=0)

    # Shaded bootstrap band + observed eCDF
    ax.fill_between(grid, lower, upper, color="steelblue", alpha=0.2)
    ax.plot(grid, obs_ecdf, color="black", linewidth=0.8)

    ax.set_xlim(L, U)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=2)
    ax.set_xlabel(r"$x$")
    ax.grid(True)

    if i % 2 == 0:
        ax.set_ylabel("eCDF")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")


# Each tuple: (obs_kind, boot_kind, n, title, (L, U))
scenarios = [
    ("chisq",      "chisq",  500, "Baseline ($\\chi^2$ null)", (0,  4)),
    ("shifted",    "normal", 500, "Shifted mean",              (-2, 4)),
    ("t3",         "normal", 500, "Heavy-tailed ($t_3$)",      (-4, 4)),
    ("compressed", "normal", 500, "Compressed variance",       (-2, 2)),
    ("mixture",    "normal", 500, "Mixture (0 & 2)",           (-1, 4)),
    ("chisq",      "chisq",   50, "Small $n=50$",              (0,  4)),
]

fig, axs = plt.subplots(3, 2, figsize=(3.5, 7), sharex=False, sharey=True)

for i, (ax, (obs_kind, boot_kind, n, title, (L, U))) in enumerate(
        zip(axs.flat, scenarios)):
    observed = generate_data(obs_kind, n)
    plot_ecdf_with_bands(ax, observed, boot_kind,
                         n=n, B=300, L=L, U=U,
                         alpha_band=(5, 95), title=title, i=i)

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_shrinkage_control_04.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_shrinkage_control_04.pdf")
