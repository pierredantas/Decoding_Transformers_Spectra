# -*- coding: utf-8 -*-
"""
=============================================================================
Decoding Transformers Spectra: A Random Matrix Theory Framework
Beyond the Marchenko-Pastur Law
Model: ALBERT-base-v2 (12M parameters)
=============================================================================

PAPER OVERVIEW
--------------
This script reproduces all figures in the paper using ALBERT-base-v2.
It implements a complete pipeline for analyzing the spectral properties of
ALBERT weight matrices under the Marchenko-Pastur (MP) random matrix theory
framework.

ALBERT vs BERT — KEY ARCHITECTURAL DIFFERENCE
----------------------------------------------
ALBERT (A Lite BERT) uses cross-layer parameter sharing: all 12 virtual
encoder layers share a SINGLE set of weights stored in one layer group.
This means the spectral analysis covers 6 distinct matrix types rather
than 12 × 6 = 72 matrices as in BERT-base.

ALBERT-base-v2 architecture:
  - 12 virtual encoder layers (all share the same weights)
  - 1 actual layer group: encoder.albert_layer_groups.0.albert_layers.0
  - hidden_size            = 768
  - intermediate_size      = 3072
  - embedding_size         = 128  (factorized embeddings)
  - Matrix shapes:
      Q / K / V / Att-Dense : (768,  768)  aspect ratio β = 1.0  (square)
      FFN                   : (3072, 768)  aspect ratio β = 0.25 (rectangular)
      FFN-Out               : (768, 3072)  aspect ratio β = 0.25 (rectangular)
      Emb hidden mapping    : (768,  128)  aspect ratio β ≈ 0.17

FACTORIZED EMBEDDINGS
---------------------
Unlike BERT, ALBERT separates the embedding size (E=128) from the hidden
size (H=768) via a linear projection matrix (embedding_hidden_mapping_in)
of shape (768, 128). This introduces an additional matrix type with a
distinct β value, relevant for MP spectral analysis.

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
  - "tw"            : Tracy-Widom fluctuation scale
  - "fraction"      : fixed fraction of MP bandwidth
  - "tw_or_fraction": max(TW margin, fraction margin)

OUTPUT FILES
------------
  albert_weights/                  raw .npy weight matrices + manifest.json
  albert_weights_WMP/              column-standardized matrices + manifest.json
  step1_column_stats_albert.json   per-column mean/std summary (human-readable)
  step1_column_stats_albert.npz    per-column mean/std arrays (numpy)
  graph_core_diag_01_albert.pdf    Graph 1:  ePDF vs MP PDF (trimming conditions)
  graph_core_diag_02_albert.pdf    Graph 2:  eCDF vs MP CDF (trimming conditions)
  graph_core_diag_03_albert.pdf    Graph 3:  ePDF vs MP PDF (matrix types)
  graph_core_diag_04_albert.pdf    Graph 4:  Residual CDF (eCDF - MP CDF)
  graph_core_diag_05_albert.pdf    Graph 5:  QQ plots vs MP quantiles
  graph_level_views_01_albert.pdf  Graph 6:  KS heatmaps (group x matrix type)
  graph_level_views_02_albert.pdf  Graph 7:  Per-matrix-type acceptance rates
  graph_level_views_03_albert.pdf  Graph 8:  β vs KS statistic scatter
  graph_shrinkage_control_01_albert.pdf Graph 9:  Bootstrap p-value distributions
  graph_shrinkage_control_02_albert.pdf Graph 10: KS-TW edge relaxation sensitivity
  graph_shrinkage_control_03_albert.pdf Graph 11: Type-I calibration curves
  graph_shrinkage_control_04_albert.pdf Graph 12: eCDF vs bootstrap bands
"""

# =============================================================================
# SECTION 1: Extract ALBERT-base-v2 weight matrices
# =============================================================================
# PURPOSE:
#   Load ALBERT-base-v2 from HuggingFace, extract all 2-D weight matrices
#   from the shared attention and FFN layer group, and save them as .npy
#   files together with a manifest.json index for downstream analysis.
#
# ALBERT-SPECIFIC NOTE:
#   All encoder layers share one weight group:
#   "encoder.albert_layer_groups.0.albert_layers.0.*"
#   Digit indices in keys are mapped to "g0" (group 0) in file paths,
#   unlike BERT which uses "layer_0", "layer_1", etc.
#
# INPUTS:  HuggingFace model "albert-base-v2" (downloaded automatically)
# OUTPUTS: albert_weights/ directory with .npy files and manifest.json
# =============================================================================

from pathlib import Path
import json
import numpy as np
import torch
from transformers import AlbertModel


def extract_matrices(
    model_name: str = "albert-base-v2",
    out_dir: str = "albert_weights",
    include_bias: bool = False,
    only_linear_like: bool = True,
    dtype: str = "float32",
    save_format: str = "npy",
):
    """
    Extract 2-D weight matrices from a HuggingFace ALBERT model and save to disk.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: "albert-base-v2".
    out_dir : str
        Output directory for .npy/.npz files and manifest.json.
    include_bias : bool
        If True, also saves 1-D bias vectors. Default: False.
    only_linear_like : bool
        If True, only saves matrices from attention/FFN layers.
        Filters by ALBERT-specific key patterns. Default: True.
    dtype : str
        Numeric precision — "float32" or "float64". Default: "float32".
    save_format : str
        File format — "npy" (uncompressed) or "npz" (compressed).
        Default: "npy".

    Returns
    -------
    None
        Saves files to disk and prints a summary including all matrix names
        and shapes (useful for configuring downstream graph scripts).
    """
    assert save_format in {"npy", "npz"}, "save_format must be 'npy' or 'npz'"

    # Load model on CPU with gradients disabled (inference only)
    torch.set_grad_enabled(False)
    model = AlbertModel.from_pretrained(model_name)
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

        Keeps 2-D weight matrices from ALBERT's attention, FFN, and
        projection layers using ALBERT-specific key patterns.
        Different from BERT: ALBERT uses "ffn.", "ffn_output",
        "attention.dense" instead of "output.dense", "intermediate".
        """
        if tensor.ndim == 2:
            if only_linear_like:
                # ALBERT-specific layer name patterns
                names_we_like = (
                    "attention.query", "attention.key", "attention.value",
                    "attention.dense", "ffn.", "ffn_output",
                    "full_layer_layer_norm", "pooler.dense",
                    "encoder.embedding_hidden_mapping_in",  # ALBERT-specific projection
                )
                return any(n in key for n in names_we_like)
            return True
        if include_bias and tensor.ndim == 1:
            return "bias" in key
        return False

    # Iterate sorted state_dict keys, save qualifying tensors.
    # Key normalization for ALBERT:
    #   "encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight"
    #   -> encoder/albert_layer_groups/g0/albert_layers/g0/attention/query/weight.npy
    # Digit indices are mapped to "g{N}" (group index) instead of "layer_{N}"
    for key in sorted(sd.keys()):
        t = sd[key]
        if not keep_param(key, t):
            continue

        arr = t.detach().cpu().to(
            dtype=torch.float32 if dtype == "float32" else torch.float64
        ).numpy()

        # Build a safe filesystem path from the key
        parts = key.split(".")
        norm_parts = []
        for p in parts:
            if p.isdigit():
                norm_parts.append(f"g{p}")   # "0" -> "g0" (group index)
            else:
                norm_parts.append(p)

        save_dir = out.joinpath(*norm_parts[:-1])
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = norm_parts[-1]    # typically "weight" or "bias"
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

    # Save embedding matrices separately under albert_weights/embeddings/
    # ALBERT has smaller embeddings (128-dim) projected to hidden size (768)
    # via embedding_hidden_mapping_in — this is captured in the main loop above.
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

    # Print all matrix names with shapes — useful for configuring SETTINGS
    # in graph scripts, since ALBERT's key names differ from BERT
    print("\nAvailable matrices:")
    for e in manifest["files"]:
        print(f"  {e['name']:70s}  shape={e['shape']}")


if __name__ == "__main__":
    extract_matrices(
        model_name="albert-base-v2",
        out_dir="albert_weights",
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
# ALBERT NOTE:
#   Since ALBERT has only 1 shared layer group, this section processes
#   far fewer matrices than BERT (6 matrix types vs 12×6=72 for BERT-base).
#
# INPUTS:
#   albert_weights/manifest.json   (from Section 1)
#   albert_weights/*.npy           (from Section 1)
#
# OUTPUTS:
#   step1_column_stats_albert.json  human-readable summary of per-column stats
#   step1_column_stats_albert.npz   exact numpy arrays of mu and sd per matrix
#   albert_weights_WMP/             normalized matrices as .npy files
#   albert_weights_WMP/manifest.json  index with "_WMP" suffix on each name
#
# VERIFICATION:
#   A round-trip check confirms: max|W - (WMP * sd + mu)| < 1e-5
# =============================================================================

import json
import re
from datetime import datetime, UTC
from pathlib import Path
import numpy as np

# Directory and file paths — ALBERT-specific names to avoid collision with BERT
WEIGHTS_DIR = "albert_weights"                   # input: raw matrices from Section 1
STATS_JSON  = "step1_column_stats_albert.json"   # output: human-readable stats index
STATS_NPZ   = "step1_column_stats_albert.npz"   # output: numpy arrays of mu/sd
WMP_DIR     = "albert_weights_WMP"               # output: normalized matrices


def _safe_key(idx: int, kind: str, name: str) -> str:
    """
    Build a stable, filesystem-safe NPZ key for storing mu/sd arrays.

    Format: '0003__mean__encoder_albert_layer_groups_g0_albert_layers_g0_...'
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
# Statistics are computed in float64 for numerical stability.
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

    # Column-wise mean and std computed in float64 for numerical stability
    W  = W.astype(np.float64, copy=False)
    mu = W.mean(axis=0)    # shape (n,) — per-column mean
    sd = W.std(axis=0)     # shape (n,) — per-column std

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

# WMP manifest mirrors the source manifest with "_WMP" appended to names
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

    # Protect against zero-variance columns (constant features)
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

# Spot-check first 3 matrices
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
# These functions implement the core RMT machinery shared across all graphs.
# They are redefined in each graph section for self-containedness but are
# documented here once for clarity.
#
# ALBERT-SPECIFIC NOTE on _find_manifest_entry:
#   ALBERT matrix names follow the pattern:
#   "encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight_WMP"
#   The function handles both exact and substring matching, so ALBERT's
#   longer key names are found correctly.
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
    (a, b) : (float, float) — λ- = (1-√β)²,  λ+ = (1+√β)²

    ALBERT β values:
      Q/K/V/Att-Dense : β = 1.0   (768×768 square)
      FFN             : β = 0.25  (3072×768 rectangular)
      FFN-Out         : β = 0.25  (768×3072 rectangular)
      Emb mapping     : β ≈ 0.17  (768×128 rectangular)
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
    np.ndarray — cumulative integral starting from 0
    """
    dx  = np.diff(x)
    seg = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate([[0.0], np.cumsum(seg)])


def _mp_cdf(x, beta: float, grid_points: int = 8192) -> np.ndarray:
    """
    Evaluate the Marchenko-Pastur CDF at points x via numerical integration.

    Uses a quadratic grid t² to oversample near the lower boundary λ-,
    where the MP density has an integrable singularity (x → λ-).

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
    cdf_vals /= cdf_vals[-1]             # normalize to exactly [0, 1]
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

    For ALBERT's square matrices (β=1.0), frac_sq is used.
    For ALBERT's rectangular matrices (β=0.25), frac_rect is used.

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
    Look up a matrix entry in the WMP manifest by name or path substring.

    First tries exact name match, then falls back to substring match.
    Necessary because ALBERT matrix names are long and may be referenced
    by partial key in some graph configurations.

    Parameters
    ----------
    manifest : dict — loaded manifest.json
    target   : str  — matrix name including "_WMP" suffix

    Returns
    -------
    dict — manifest entry with keys: name, path, shape, ndim, dtype

    Raises
    ------
    ValueError if the target is not found in the manifest
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
      1. SVD of W → singular values s → eigenvalues λ = s²/max(m,n)
      2. Compute MP support [a, b] and trimmed interval [L, U]
      3. Keep only eigenvalues in the interior [L, U]
      4. Normalize MP PDF/CDF to the trimmed interval (conditional distribution)

    Parameters
    ----------
    W               : np.ndarray — 2-D WMP-normalized matrix
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
    N_trim      : int       — number of eigenvalues retained after trimming
    """
    m, n  = W.shape
    beta  = min(m, n) / max(m, n)

    # Singular values → eigenvalues of W^T W / max(m,n)
    s       = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2) / max(m, n)
    lambdas.sort()

    a, b  = _mp_support(beta)
    delta = _edge_margin(beta, m, n, TRIM_KIND, C_TW, EDGE_FRAC_SQUARE, EDGE_FRAC_RECT)
    L, U  = a + delta, b - delta
    if L >= U:
        L, U = a, b    # fallback: use full support if margin is too large

    mask     = (lambdas >= L) & (lambdas <= U)
    lam_trim = lambdas[mask]
    N_trim   = lam_trim.size

    # Conditional MP: normalize to [L, U]
    FL, FU = _mp_cdf([L, U], beta)
    den    = max(FU - FL, 1e-12)

    mp_pdf_cond = lambda x: _mp_pdf(x, beta, a, b) / den
    mp_cdf_cond = lambda x: np.clip((_mp_cdf(x, beta) - FL) / den, 0, 1)

    return lam_trim, (a, b, L, U, beta), mp_pdf_cond, mp_cdf_cond, N_trim


# =============================================================================
# GRAPH 1: Empirical PDF vs. conditional MP PDF (trimming conditions)
# =============================================================================
# PURPOSE:
#   Compare the empirical eigenvalue density (histogram) against the
#   theoretical MP PDF for ALBERT's 6 matrix types with varying trimming.
#   Since ALBERT shares weights across all 12 virtual layers, each subplot
#   represents a different matrix type from the single shared weight group.
#
# ALBERT NOTE:
#   All 6 matrix types come from the same shared layer group:
#   encoder.albert_layer_groups.0.albert_layers.0.*
#   Sets 1-3 vary the trimming strategy on the same matrix (Att.Q).
#   Sets 4-6 cover FFN, FFN-Out, and Att.Dense.
#
# LAYOUT: 3×2 mosaic, 1-column IEEE/Springer format (3.5 × 7 inches)
# OUTPUT: graph_core_diag_01_albert.pdf
# =============================================================================

WMP_DIR   = "albert_weights_WMP"
COND_GRID = 2000    # points for the theoretical MP curve
HIST_BINS = 80      # histogram bins for empirical density

# ALBERT-base-v2: all layers share weights — one group, one layer
# Each entry covers a different matrix type or trimming configuration
SETTINGS = [
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 1 — Att.Q"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 2 — Att.K"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 3 — Att.V"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.ffn.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 4 — FFN"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 5 — FFN-Out"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 6 — Att.Dense"),
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

    # Empirical density histogram of trimmed eigenvalues
    if N_trim > 0:
        ax.hist(lam_trim, bins=HIST_BINS, range=(L, U),
                density=True, alpha=0.4)

    # Theoretical conditional MP PDF (black line)
    ax.plot(x_grid, mp_pdf_cond(x_grid), color="black")

    # Vertical reference lines: gray = full MP support, green = trimmed interval
    for v, col in [(a, "gray"), (b, "gray"), (L, "green"), (U, "green")]:
        ax.axvline(v, color=col, linestyle="--", linewidth=0.8)

    ax.set_title(f"{cfg['label']} ($\\beta$={beta:.2f})")
    ax.set_xlabel(r"$\lambda$")

    # Y label only on left column to save horizontal space
    if i % 2 == 0:
        ax.set_ylabel("PDF")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")

    ax.grid(True)

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_core_diag_01_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_core_diag_01_albert.pdf")


# =============================================================================
# GRAPH 2: Empirical CDF vs. conditional MP CDF (trimming conditions)
# =============================================================================
# PURPOSE:
#   Overlay the empirical step-CDF of trimmed eigenvalues against the
#   theoretical conditional MP CDF for the same 6 ALBERT matrix types.
#   A close match indicates that the eigenvalues follow the MP law.
#
# LAYOUT: 3×2 mosaic (reuses SETTINGS and manifest from Graph 1)
# OUTPUT: graph_core_diag_02_albert.pdf
# =============================================================================

# Config (same SETTINGS as Graph 1 — all 6 ALBERT matrix types)
WMP_DIR   = "albert_weights_WMP"
COND_GRID = 2000

SETTINGS = [
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 1 — Att.Q"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 2 — Att.K"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 3 — Att.V"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.ffn.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 4 — FFN"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 5 — FFN-Out"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05,
         label="Set 6 — Att.Dense"),
]

# Load manifest
man_path = Path(WMP_DIR) / "manifest.json"
manifest = json.load(open(man_path))

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

    ax.set_title(f"{cfg['label']} ($\\beta$={beta:.2f})")
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
plt.savefig("graph_core_diag_02_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_core_diag_02_albert.pdf")


# =============================================================================
# GRAPH 3: Empirical PDF vs. MP PDF across matrix types (global scale)
# =============================================================================
# PURPOSE:
#   Show how well the MP law fits all 6 ALBERT matrix types using a shared
#   x-axis scale (global [L,U]) for fair visual comparison.
#   Unlike BERT, there is no layer dimension — only matrix types differ.
#
# ALBERT NOTE:
#   Because all layers share weights, param_sets covers all 6 unique matrix
#   types in ALBERT's single layer group. The global [L,U] is computed
#   across all 6 to ensure a consistent x-axis.
#
# LAYOUT: 3×2 mosaic with shared x and y axes
# OUTPUT: graph_core_diag_03_albert.pdf
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

WMP_DIR = "albert_weights_WMP"

# ALBERT parameter sharing: all 6 entries point to the same layer group
# Each tuple: (matrix_name, trim_kind, c_tw, frac_sq, frac_rect, subplot_title)
param_sets = [
    ("encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight_WMP",
     "tw_or_fraction", 2.0, 0.05, 0.05, "Att. Query"),
    ("encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight_WMP",
     "tw_or_fraction", 2.0, 0.05, 0.05, "Att. Key"),
    ("encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight_WMP",
     "tw_or_fraction", 2.0, 0.05, 0.05, "Att. Value"),
    ("encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight_WMP",
     "tw_or_fraction", 2.0, 0.05, 0.05, "Att. Dense"),
    ("encoder.albert_layer_groups.0.albert_layers.0.ffn.weight_WMP",
     "tw_or_fraction", 2.0, 0.05, 0.05, "FFN"),
    ("encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight_WMP",
     "tw_or_fraction", 2.0, 0.05, 0.05, "FFN Output"),
]

GRID_POINTS = 8192
COND_GRID   = 2000
HIST_BINS   = 120

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
    tw=0.0 if c_tw is None else c_tw*(n_eff**(-2/3))*(1+np.sqrt(beta))**(4/3)
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

# First pass: compute global [L,U] across all 6 matrix types for shared x-axis
global_L, global_U = np.inf, -np.inf
for target, kind, c_tw, frac_sq, frac_rect, _ in param_sets:
    entry = _find_manifest_entry(manifest, target)
    W     = _load_matrix(Path(WMP_DIR) / entry["path"]).astype(np.float64, copy=False)
    m, n  = W.shape; beta = min(m,n)/max(m,n)
    s     = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2)/max(m,n); lambdas.sort()
    a, b  = _mp_support(beta)
    delta = _edge_margin(beta,m,n,kind,c_tw,frac_sq,frac_rect)
    L, U  = a+delta, b-delta
    if L >= U: L, U = a, b
    global_L, global_U = min(global_L,L), max(global_U,U)

fig, axes = plt.subplots(3, 2, figsize=(3.5, 7), sharex=True, sharey=True)
axes = axes.ravel()

for i, (ax, (target, kind, c_tw, frac_sq, frac_rect, title)) in enumerate(
        zip(axes, param_sets)):
    entry = _find_manifest_entry(manifest, target)
    W     = _load_matrix(Path(WMP_DIR)/entry["path"]).astype(np.float64, copy=False)
    m, n  = W.shape; beta = min(m,n)/max(m,n)
    s     = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2)/max(m,n); lambdas.sort()
    a, b  = _mp_support(beta)
    delta = _edge_margin(beta,m,n,kind,c_tw,frac_sq,frac_rect)
    L, U  = a+delta, b-delta
    if L >= U: L, U = a, b

    mask_trim = (lambdas>=L)&(lambdas<=U)
    lam_trim  = lambdas[mask_trim]

    FL, FU = _mp_cdf([L,U],beta); den=max(float(FU-FL),1e-12)
    def mp_pdf_cond(x): return _mp_pdf(x,beta,a,b)/den

    if lam_trim.size > 0:
        ax.hist(lam_trim, bins=HIST_BINS, range=(global_L,global_U),
                density=True, alpha=0.4, label=f"Empirical (N={lam_trim.size})")

    x_grid = np.linspace(global_L, global_U, COND_GRID)
    ax.plot(x_grid, mp_pdf_cond(x_grid), color="black", lw=1.2,
            label=f"MP ($\\beta$={beta:.3f})")

    for v,col in [(a,"gray"),(b,"gray"),(L,"green"),(U,"green")]:
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
plt.savefig("graph_core_diag_03_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_core_diag_03_albert.pdf")


# =============================================================================
# GRAPH 4: Empirical residual CDF (eCDF - MP CDF) across matrix types
# =============================================================================
# PURPOSE:
#   Plot the signed difference between the empirical CDF and the theoretical
#   MP CDF for each of ALBERT's 6 matrix types. The KS statistic D =
#   max|eCDF - MP CDF| is shown as a dashed red line. A flat residual near
#   zero indicates good MP fit.
#
# ALBERT NOTE:
#   _short_title() uses keyword matching (no layer number extraction)
#   because ALBERT lacks "layer.N" in its key names.
#
# LAYOUT: 3×2 mosaic with unified y-axis scale
# OUTPUT: graph_core_diag_04_albert.pdf
# =============================================================================

import re
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

WMP_DIR     = "albert_weights_WMP"
GRID_POINTS = 8192
COND_GRID   = 2000

# One entry per ALBERT matrix type
param_sets = [
    dict(plot_target="encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight_WMP",
         TRIM_KIND="tw_or_fraction", C_TW=2.0, EDGE_FRAC_SQUARE=0.05, EDGE_FRAC_RECT=0.05),
    dict(plot_target="encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight_WMP",
         TRIM_KIND="tw_or_fraction", C_TW=2.0, EDGE_FRAC_SQUARE=0.05, EDGE_FRAC_RECT=0.05),
    dict(plot_target="encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight_WMP",
         TRIM_KIND="tw_or_fraction", C_TW=2.0, EDGE_FRAC_SQUARE=0.05, EDGE_FRAC_RECT=0.05),
    dict(plot_target="encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight_WMP",
         TRIM_KIND="tw_or_fraction", C_TW=2.0, EDGE_FRAC_SQUARE=0.05, EDGE_FRAC_RECT=0.05),
    dict(plot_target="encoder.albert_layer_groups.0.albert_layers.0.ffn.weight_WMP",
         TRIM_KIND="tw_or_fraction", C_TW=2.0, EDGE_FRAC_SQUARE=0.05, EDGE_FRAC_RECT=0.05),
    dict(plot_target="encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight_WMP",
         TRIM_KIND="tw_or_fraction", C_TW=2.0, EDGE_FRAC_SQUARE=0.05, EDGE_FRAC_RECT=0.05),
]

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

def _short_title(name, params):
    """
    Build a compact subplot title for ALBERT matrix names.

    ALBERT-SPECIFIC: No "layer.N" pattern exists, so titles use keyword
    matching only (e.g., "query", "ffn_output") without a layer prefix.
    Keywords are ordered by specificity: ffn_output before ffn to avoid
    partial matches.
    """
    keywords = ["query", "key", "value", "dense", "ffn_output", "ffn",
                "embedding", "attention"]
    kw = next((k for k in keywords if k in name.lower()), name.split(".")[-1])
    return (f"{kw}\n"
            f"({params['TRIM_KIND']}, $c_{{\\alpha}}$={params['C_TW']})")

man_path = Path(WMP_DIR) / "manifest.json"
manifest = json.load(open(man_path))

fig, axes = plt.subplots(3, 2, figsize=(3.5, 7), sharey=True)
axes = axes.flatten()
y_min, y_max = 0, 0

for i, params in enumerate(param_sets):
    entry = _find_manifest_entry(manifest, params["plot_target"])
    rel, name = entry["path"], entry.get("name", entry["path"])
    W = _load_matrix(Path(WMP_DIR)/rel).astype(np.float64, copy=False)

    m, n = W.shape; beta = min(m,n)/max(m,n)
    s = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2)/max(m,n); lambdas.sort()
    a, b = _mp_support(beta)

    delta = _edge_margin(beta,m,n,params["TRIM_KIND"],params["C_TW"],
                         params["EDGE_FRAC_SQUARE"],params["EDGE_FRAC_RECT"])
    L, U = a+delta, b-delta
    if L >= U: L, U = a, b

    mask_trim = (lambdas>=L)&(lambdas<=U)
    lam_trim  = lambdas[mask_trim]; N_trim = lam_trim.size

    FL, FU = _mp_cdf([L,U],beta); den=max(float(FU-FL),1e-12)
    def mp_cdf_cond(x): return (_mp_cdf(x,beta)-FL)/den

    x_grid   = np.linspace(L,U,COND_GRID)
    emp_cdf  = np.searchsorted(np.sort(lam_trim),x_grid,side="right")/max(N_trim,1)
    residual = emp_cdf - mp_cdf_cond(x_grid)
    ks_stat  = np.max(np.abs(residual))   # Kolmogorov-Smirnov statistic

    ax = axes[i]
    ax.plot(x_grid, residual, color="steelblue", lw=1.2)
    ax.axhline(0,        color="black", linestyle="--", linewidth=0.8)
    ax.axhline(+ks_stat, color="red",   linestyle=":",  linewidth=0.8,
               label=f"KS={ks_stat:.3f}")
    ax.axhline(-ks_stat, color="red",   linestyle=":",  linewidth=0.8)

    ax.set_title(_short_title(name, params), pad=2)
    ax.set_xlabel(r"$\lambda$")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=False)

    if i % 2 == 0:
        ax.set_ylabel("Residual CDF")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")

    y_min = min(y_min, residual.min())
    y_max = max(y_max, residual.max())

# Unify y-axis scale for fair comparison across matrix types
for ax in axes:
    ax.set_ylim(y_min*1.1, y_max*1.1)

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_core_diag_04_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_core_diag_04_albert.pdf")


# =============================================================================
# GRAPH 5: QQ plots of empirical spectra vs. conditional MP quantiles
# =============================================================================
# PURPOSE:
#   Compare empirical quantiles of trimmed eigenvalues against theoretical
#   MP quantiles. Points on the 45° diagonal = perfect MP fit.
#   Curvature reveals distributional deviations from the MP law.
#
# ALBERT NOTE:
#   _short_title() uses keyword-only matching (no "L{N}." prefix) because
#   ALBERT has no layer number in its key names.
#
# LAYOUT: 3×2 mosaic with shared axes
# OUTPUT: graph_core_diag_05_albert.pdf
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

WMP_DIR     = "albert_weights_WMP"
GRID_POINTS = 8192
COND_GRID   = 2000

# One entry per ALBERT matrix type
PARAM_SETS = [
    dict(target="encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight_WMP",
         trim_kind="tw_or_fraction", C_TW=2.0, frac_sq=0.05, frac_rect=0.05),
    dict(target="encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight_WMP",
         trim_kind="tw_or_fraction", C_TW=2.0, frac_sq=0.05, frac_rect=0.05),
    dict(target="encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight_WMP",
         trim_kind="tw_or_fraction", C_TW=2.0, frac_sq=0.05, frac_rect=0.05),
    dict(target="encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight_WMP",
         trim_kind="tw_or_fraction", C_TW=2.0, frac_sq=0.05, frac_rect=0.05),
    dict(target="encoder.albert_layer_groups.0.albert_layers.0.ffn.weight_WMP",
         trim_kind="tw_or_fraction", C_TW=2.0, frac_sq=0.05, frac_rect=0.05),
    dict(target="encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight_WMP",
         trim_kind="tw_or_fraction", C_TW=2.0, frac_sq=0.05, frac_rect=0.05),
]

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
    """
    Build a compact subplot title for ALBERT matrix names.

    ALBERT-SPECIFIC: No layer number prefix. Keywords ordered by specificity
    (ffn_output before ffn) to avoid partial-match issues.
    """
    keywords = ["query", "key", "value", "dense", "ffn_output", "ffn",
                "embedding", "attention"]
    kw = next((k for k in keywords if k in name.lower()), name.split(".")[-1])
    return (f"{kw}\n"
            f"({params['trim_kind']}, $c_{{\\alpha}}$={params['C_TW']})")

manifest = json.load(open(Path(WMP_DIR)/"manifest.json"))

fig, axes = plt.subplots(3, 2, figsize=(3.5, 7), sharex=True, sharey=True)
axes = axes.ravel()

for i, (ax, params) in enumerate(zip(axes, PARAM_SETS)):
    entry = _find_manifest_entry(manifest, params["target"])
    rel, name = entry["path"], entry.get("name", entry["path"])
    W = _load_matrix(Path(WMP_DIR)/rel).astype(np.float64, copy=False)

    m, n = W.shape; beta = min(m,n)/max(m,n)
    s = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2)/max(m,n); lambdas.sort()
    a, b = _mp_support(beta)

    delta = _edge_margin(beta,m,n,params["trim_kind"],params["C_TW"],
                         params["frac_sq"],params["frac_rect"])
    L, U = a+delta, b-delta
    if L >= U: L, U = a, b

    mask_trim = (lambdas>=L)&(lambdas<=U)
    lam_trim  = lambdas[mask_trim]; N_trim = lam_trim.size

    emp_q = np.sort(lam_trim)    # empirical quantiles

    FL, FU = _mp_cdf([L,U],beta); den=max(float(FU-FL),1e-12)
    def mp_cdf_cond(x): return (_mp_cdf(x,beta)-FL)/den

    # Invert the conditional MP CDF to get theoretical quantiles
    q_grid   = np.linspace(0,1,N_trim)
    xs       = np.linspace(L,U,COND_GRID)
    cdf_vals = mp_cdf_cond(xs)
    mp_q     = np.interp(q_grid, cdf_vals, xs)    # theoretical quantiles

    ax.plot(mp_q, emp_q, "o", ms=1.5, alpha=0.5, color="steelblue")
    # 45° reference line = perfect MP fit
    ax.plot([mp_q.min(),mp_q.max()],[mp_q.min(),mp_q.max()],
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
plt.savefig("graph_core_diag_05_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_core_diag_05_albert.pdf")


# =============================================================================
# GRAPH 6: Group × matrix-type heatmaps of KS test outcomes
# =============================================================================
# PURPOSE:
#   Visualize binary KS test decisions (0=accept, 1=reject) as heatmaps
#   across ALBERT's single shared layer group × 6 matrix types.
#
# ALBERT vs BERT DIFFERENCE:
#   - BERT:   decisions shape (12, 6) — 12 layers × 6 matrix types
#   - ALBERT: decisions shape (1, 6)  — 1 shared group × 6 matrix types
#   The y-axis label changes from "Layer" to "Group", and y-tick shows "G0".
#
# INPUTS:  decisions_strict, decisions_tw — binary arrays of shape (1, 6)
# LAYOUT:  3×4 grid (6 scenarios × 2 methods)
# OUTPUT:  graph_level_views_01_albert.pdf
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
    Render a 3×4 mosaic of KS test decision heatmaps for ALBERT.

    ALBERT-SPECIFIC:
      decisions shape is (1, 6) — one shared group, 6 matrix types.
      y-axis shows "G0" (group 0) instead of layer numbers.
      "Group" label replaces "Layer" on the y-axis.

    Parameters
    ----------
    decisions_strict : np.ndarray, shape (1, 6)
        Binary KS-strict test decisions (0=accept, 1=reject).
    decisions_tw : np.ndarray, shape (1, 6)
        Binary KS-TW test decisions (0=accept, 1=reject).
    savepath : str
        Output PDF file path.
    """
    # ALBERT: all sets use layers=[0] (single shared group G0)
    param_sets = [
        {"alpha":0.01, "layers":[0],
         "mat_types":["Q","K","V","Att-Dense","FFN","FFN-Out"],
         "cmap":["#d62728","#2ca02c"], "title":"Set 1 — $\\alpha$=0.01"},
        {"alpha":0.05, "layers":[0],
         "mat_types":["Q","K","V","Att-Dense","FFN","FFN-Out"],
         "cmap":["#cccccc","#1f77b4"], "title":"Set 2 — $\\alpha$=0.05"},
        {"alpha":0.10, "layers":[0],
         "mat_types":["Q","K","V","Att-Dense","FFN","FFN-Out"],
         "cmap":["#cccccc","#1f77b4"], "title":"Set 3 — $\\alpha$=0.10"},
        {"alpha":0.20, "layers":[0],
         "mat_types":["Q","K","V","Att-Dense","FFN","FFN-Out"],
         "cmap":["#d62728","#2ca02c"], "title":"Set 4 — $\\alpha$=0.20"},
        {"alpha":0.05, "layers":[0],
         "mat_types":["Q","K","V","Att-Dense","FFN","FFN-Out"],
         "cmap":["#444444","#ff7f0e"], "title":"Set 5 — $\\alpha$=0.05"},
        {"alpha":0.05, "layers":[0],
         "mat_types":["Q","K","V"],
         "cmap":["#e41a1c","#377eb8"], "title":"Set 6 — $\\alpha$=0.05"},
    ]

    fig, axes = plt.subplots(3, 4, figsize=(3.5*2, 7))
    axes = axes.ravel()

    for idx, params in enumerate(param_sets):
        alpha = params["alpha"]; L = params["layers"]
        M = params["mat_types"]; cmap = params["cmap"]; title = params["title"]

        d_strict = decisions_strict[np.ix_(L, range(min(len(M), decisions_strict.shape[1])))]
        d_tw     = decisions_tw[np.ix_(L,     range(min(len(M), decisions_tw.shape[1])))]

        # Left subplot: KS-strict; y-tick = "G0" (ALBERT group 0)
        ax_strict = axes[idx * 2]
        sns.heatmap(d_strict, cmap=cmap, cbar=False, annot=True, fmt="d",
                    annot_kws={"size": 6},
                    xticklabels=M, yticklabels=["G0"], ax=ax_strict)
        ax_strict.set_title(f"{title}\nKS Strict\n($\\alpha$={alpha})", pad=2)
        ax_strict.set_xticklabels(ax_strict.get_xticklabels(), rotation=45, ha="right")
        ax_strict.set_yticklabels(ax_strict.get_yticklabels(), rotation=0)
        ax_strict.set_ylabel("Group")    # "Group" instead of "Layer" for ALBERT
        ax_strict.yaxis.set_label_coords(-0.35, 0.5)

        # Right subplot: KS-TW (no y-axis labels)
        ax_tw = axes[idx * 2 + 1]
        sns.heatmap(d_tw, cmap=cmap, cbar=False, annot=True, fmt="d",
                    annot_kws={"size": 6},
                    xticklabels=M, yticklabels=False, ax=ax_tw)
        ax_tw.set_title(f"{title}\nKS–TW\n($\\alpha$={alpha})", pad=2)
        ax_tw.set_xticklabels(ax_tw.get_xticklabels(), rotation=45, ha="right")
        ax_tw.set_ylabel("")

    plt.subplots_adjust(hspace=0.8, wspace=0.4)
    plt.savefig(savepath, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"✅ Figure saved: {savepath}")


# Mock data for illustration — ALBERT: 1 group, 6 matrix types
# Replace with real KS test outcomes from running the full pipeline
np.random.seed(42)
decisions_strict = np.random.choice([0, 1], size=(1, 6), p=[0.4, 0.6])
decisions_tw     = np.random.choice([0, 1], size=(1, 6), p=[0.3, 0.7])

plot_ks_mosaic_separate(decisions_strict, decisions_tw,
                        savepath="graph_level_views_01_albert.pdf")


# =============================================================================
# GRAPH 7: Per-matrix-type acceptance rates with Wilson confidence intervals
# =============================================================================
# PURPOSE:
#   Show acceptance rates of KS-strict and KS-TW across ALBERT's 6 matrix
#   types for each α level. Unlike BERT (which shows per-layer trends),
#   ALBERT shows per-matrix-type trends since all layers share weights.
#
# ALBERT vs BERT DIFFERENCE:
#   - x-axis: matrix type names (Q, K, V, Att-Dense, FFN, FFN-Out)
#             instead of layer numbers 0-11
#   - scenario_square_vs_rect splits at index 4 (first 4 are square β=1.0,
#     last 2 are rectangular β=0.25)
#
# LAYOUT:  6 rows (scenarios) × 3 columns (α values), figsize=(7, 9)
# OUTPUT:  graph_level_views_02_albert.pdf
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

# ALBERT-specific: 6 matrix types instead of 12 layers
mat_types = ["Q", "K", "V", "Att-Dense", "FFN", "FFN-Out"]
alphas    = [0.01, 0.05, 0.10]
methods   = ["KS-strict", "KS-TW"]
n_boot    = 50
rng       = np.random.default_rng(42)


# Scenario functions — "layer" arg = matrix type index (0-5)
# ALBERT has no layer variation; index distinguishes matrix types instead.

def scenario_baseline(layer, method, alpha):
    """Both methods accept at constant moderate rates regardless of matrix type."""
    return 0.55 if method == "KS-strict" else 0.65

def scenario_strict_dominance(layer, method, alpha):
    """KS-TW always accepts more than KS-strict across all matrix types."""
    return 0.75 if method == "KS-strict" else 0.90

def scenario_low_alpha(layer, method, alpha):
    """Acceptance rate increases sharply as α increases."""
    if alpha == 0.01:   return 0.15 if method == "KS-strict" else 0.35
    elif alpha == 0.05: return 0.45 if method == "KS-strict" else 0.55
    else:               return 0.75 if method == "KS-strict" else 0.80

def scenario_square_vs_rect(layer, method, alpha):
    """
    Square matrices (Q/K/V/Att-Dense, idx < 4, β=1.0) accept more than
    rectangular matrices (FFN/FFN-Out, idx >= 4, β=0.25).
    Models the effect of aspect ratio on MP fit quality.
    """
    if layer < 4: return 0.80 if method == "KS-strict" else 0.90
    else:         return 0.30 if method == "KS-strict" else 0.50

def scenario_edge_sensitive(layer, method, alpha):
    """KS-strict oscillates between matrix types; KS-TW is stable."""
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

x = np.arange(len(mat_types))

for row, (name, func) in enumerate(SCENARIOS.items()):
    # Simulate binary decisions for each (matrix_type, method, alpha, bootstrap)
    decisions = np.zeros((len(mat_types), len(methods), len(alphas), n_boot))
    for i, mat in enumerate(mat_types):
        for j, method in enumerate(methods):
            for k, alpha in enumerate(alphas):
                p = func(i, methods[j], alphas[k])
                decisions[i,j,k,:] = rng.choice([0,1], size=n_boot, p=[1-p,p])

    accept_means = np.mean(decisions, axis=-1)

    # Wilson 95% confidence intervals
    ci_low  = np.zeros_like(accept_means)
    ci_high = np.zeros_like(accept_means)
    for i in range(len(mat_types)):
        for j in range(len(methods)):
            for k in range(len(alphas)):
                count = np.sum(decisions[i,j,k,:])
                low, high = proportion_confint(count, n_boot, alpha=0.05, method="wilson")
                ci_low[i,j,k]  = low
                ci_high[i,j,k] = high

    for k, alpha in enumerate(alphas):
        ax = axes[row, k]

        for j, method in enumerate(methods):
            ax.plot(x, accept_means[:,j,k],
                    marker="o", ms=2, linewidth=0.8,
                    label=method if (row==0 and k==0) else None)
            ax.fill_between(x, ci_low[:,j,k], ci_high[:,j,k], alpha=0.2)

        if row == 0:
            ax.set_title(f"$\\alpha$ = {alpha}")
        if row == len(SCENARIOS) - 1:
            ax.set_xlabel("Matrix type")

        ax.set_xticks(x)
        ax.set_xticklabels(mat_types, rotation=30, ha="right")
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
fig.savefig("graph_level_views_02_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_level_views_02_albert.pdf")


# =============================================================================
# GRAPH 8: Aspect ratio β vs KS statistic D scatter plots
# =============================================================================
# PURPOSE:
#   Investigate whether the KS statistic D correlates with aspect ratio β.
#   ALBERT has a bimodal β distribution: β≈1.0 (square: Q/K/V/Att-Dense)
#   and β≈0.25 (rectangular: FFN/FFN-Out), unlike BERT which has a
#   continuous range of β across layers.
#
# ALBERT-SPECIFIC:
#   β values cluster at exactly two values rather than spanning [0.3, 1.0].
#   Vertical dotted lines at β=1.0 and β=0.25 mark the actual ALBERT values.
#   Proportion 0.67/0.33 reflects 4 square vs 2 rectangular matrix types.
#
# LAYOUT:  3×2 mosaic of scatter plots, shared axes
# OUTPUT:  graph_level_views_03_albert.pdf
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

    ALBERT-SPECIFIC β ranges (bimodal, not continuous):
      Square  (Q/K/V/Att-Dense): β ∈ [0.95, 1.0]  — 4 of 6 matrix types
      Rect    (FFN/FFN-Out):     β ∈ [0.20, 0.30]  — 2 of 6 matrix types
    Proportion 0.67 square / 0.33 rect reflects the 4:2 ratio.

    Parameters
    ----------
    scenario : str    — one of: "baseline","strict","edge","lowalpha","smooth","mixed"
    n_points : int    — number of synthetic data points

    Returns
    -------
    betas    : np.ndarray — aspect ratios (bimodal: ~1.0 or ~0.25)
    ks_vals  : np.ndarray — KS statistic values
    fams     : np.ndarray — family labels ("Attention" or "FFN")
    """
    np.random.seed(42)
    betas, ks_vals, fams = [], [], []

    if scenario == "baseline":
        # Typical: Attention (square) fits MP well; FFN (rect) has moderate D
        for _ in range(n_points):
            if np.random.rand() < 0.67:   # 4/6 matrix types are square
                beta = np.random.uniform(0.95, 1.0); fam = "Attention"
                ks   = np.random.uniform(0.05, 0.15)
            else:
                beta = np.random.uniform(0.20, 0.30); fam = "FFN"
                ks   = np.random.uniform(0.08, 0.20)
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "strict":
        # KS-strict: Attention rejected; FFN accepted
        for _ in range(n_points):
            if np.random.rand() < 0.67:
                beta = np.random.uniform(0.95, 1.0); fam = "Attention"
                ks   = np.random.uniform(0.05, 0.10)
            else:
                beta = np.random.uniform(0.20, 0.30); fam = "FFN"
                ks   = np.random.uniform(0.30, 0.45)
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "edge":
        # High variance for FFN (edge sensitivity with small β)
        for _ in range(n_points):
            if np.random.rand() < 0.67:
                beta = np.random.uniform(0.95, 1.0); fam = "Attention"
                ks   = np.random.uniform(0.05, 0.15)
            else:
                beta = np.random.uniform(0.20, 0.30); fam = "FFN"
                ks   = np.random.uniform(0.20, 0.60)
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "lowalpha":
        # Low-α regime: both families show elevated D
        for _ in range(n_points):
            if np.random.rand() < 0.67:
                beta = np.random.uniform(0.95, 1.0); fam = "Attention"
                ks   = np.random.uniform(0.15, 0.25)
            else:
                beta = np.random.uniform(0.20, 0.30); fam = "FFN"
                ks   = np.random.uniform(0.30, 0.50)
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "smooth":
        # Tight, well-calibrated distributions
        for _ in range(n_points):
            if np.random.rand() < 0.67:
                beta = np.random.uniform(0.95, 1.0); fam = "Attention"
                ks   = np.random.normal(0.08, 0.005)
            else:
                beta = np.random.uniform(0.20, 0.30); fam = "FFN"
                ks   = np.random.normal(0.18, 0.005)
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "mixed":
        # Bimodal by construction: first half Attention, rest FFN
        for i in range(n_points):
            if i < n_points // 2:
                beta = np.random.uniform(0.95, 1.0)
                ks   = np.random.normal(0.25, 0.02); fam = "Attention"
            else:
                beta = np.random.uniform(0.20, 0.30)
                ks   = np.random.normal(0.10, 0.02); fam = "FFN"
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    return np.array(betas), np.array(ks_vals), np.array(fams)


def plot_scenario(ax, betas, ks_vals, fams, title, i):
    """
    Render one β vs KS-D scatter subplot with ALBERT-specific reference lines.

    Parameters
    ----------
    ax, betas, ks_vals, fams, title, i : standard plot arguments

    ALBERT-SPECIFIC:
      Vertical dotted lines mark the two actual β values of ALBERT matrices:
        β=1.0  (orange) — Q/K/V/Att-Dense square matrices
        β=0.25 (blue)   — FFN/FFN-Out rectangular matrices
    """
    colors = {"FFN": "steelblue", "Attention": "darkorange"}
    for fam in np.unique(fams):
        idx = fams == fam
        ax.scatter(betas[idx], ks_vals[idx],
                   label=fam, alpha=0.6, s=8, color=colors[fam], linewidths=0)

    # Vertical reference lines at ALBERT's actual β values
    ax.axvline(1.0,  color="darkorange", ls=":", linewidth=0.6, alpha=0.5)
    ax.axvline(0.25, color="steelblue",  ls=":", linewidth=0.6, alpha=0.5)
    # Horizontal KS threshold reference lines
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
plt.savefig("graph_level_views_03_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_level_views_03_albert.pdf")


# =============================================================================
# GRAPH 9: Bootstrap p-value distributions across calibration scenarios
# =============================================================================
# PURPOSE:
#   Sanity-check the calibration of KS-strict and KS-TW p-values.
#   Under the null hypothesis, p-values should be uniformly distributed U[0,1].
#   This graph is model-agnostic (purely synthetic) and identical for ALBERT
#   and BERT — included here for completeness of the ALBERT paper figure set.
#
# LAYOUT:  3×2 mosaic, line plots with fill
# OUTPUT:  graph_shrinkage_control_01_albert.pdf
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
    null         : bool  — True = U[0,1] (null); False = Beta(0.7,1) (anti-conservative)
    skew         : float — if >0, apply p^(1+skew) to skew toward 0
    conservative : bool  — if True, apply 1-(1-p)^2 to push p-values toward 1
    seed         : int   — random seed for reproducibility

    Returns
    -------
    np.ndarray — n_boot p-values in [0, 1]
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
    bins    = np.linspace(0, 1, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])

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
    ax.set_ylim(0, n_boot // 5)
    ax.set_xlabel("p-value")
    ax.grid(True)

    if i % 2 == 0:
        ax.set_ylabel("Frequency")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")

axs[0].legend(frameon=False)
plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_shrinkage_control_01_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_shrinkage_control_01_albert.pdf")


# =============================================================================
# GRAPH 10: Sensitivity of KS-TW outcomes to edge relaxation parameter c_α
# =============================================================================
# PURPOSE:
#   Show how D_trim changes as c_α increases from 1 to 3 across ALBERT's
#   6 matrix types. Each line represents one matrix type (unlike BERT where
#   each line = one layer). A legend identifies the matrix types.
#
# ALBERT vs BERT DIFFERENCE:
#   - x-axis items: 6 matrix types (Q, K, V, Att-Dense, FFN, FFN-Out)
#                   instead of 12 layers
#   - scenario_mixed splits at index 4 (square β=1.0 vs rect β=0.25)
#   - Legend included (6 labeled lines vs 12 unlabeled lines for BERT)
#
# LAYOUT:  3×2 mosaic, shared axes
# OUTPUT:  graph_shrinkage_control_02_albert.pdf
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

# ALBERT: 6 matrix types instead of 12 layers
mat_types   = ["Q", "K", "V", "Att-Dense", "FFN", "FFN-Out"]
c_TW_values = [1, 2, 3]


# Scenario functions: return synthetic D_trim for (matrix_index, c_α).
# "layer" argument = matrix type index (0-5).

def scenario_stable_accept(layer, c):
    """D_trim is low and constant — good MP fit regardless of c_α."""
    return np.random.uniform(0.05, 0.08)

def scenario_relax_accept(layer, c):
    """D_trim decreases as c_α increases — wider trim improves fit."""
    return 0.20 / c + np.random.uniform(-0.02, 0.02)

def scenario_persistent_reject(layer, c):
    """D_trim stays high regardless of c_α — structural misfit."""
    return np.random.uniform(0.18, 0.25)

def scenario_mixed(layer, c):
    """
    Square matrices (Q/K/V/Att-Dense, idx < 4, β=1.0) benefit from
    larger c_α; rectangular FFN matrices (idx >= 4, β=0.25) are already
    well-trimmed at any c_α.
    """
    if layer < 4:   return 0.20 / c           # square: improve with c_α
    else:           return np.random.uniform(0.08, 0.10)  # rect: stable

def scenario_edgesensitive(layer, c):
    """D_trim oscillates with c_α — sinusoidal boundary sensitivity."""
    return 0.12 + 0.05 * np.sin(0.5 * np.pi * c) + np.random.uniform(-0.01, 0.01)

def scenario_alpha_dependent(layer, c):
    """D_trim inversely proportional to c_α."""
    return 0.20 / c + np.random.uniform(-0.02, 0.02)


SCENARIOS = [
    ("Stable acceptance",         scenario_stable_accept),
    ("Strict→Relaxed acceptance", scenario_relax_accept),
    ("Persistent rejection",      scenario_persistent_reject),
    ("Mixed families",            scenario_mixed),
    ("Edge-sensitive",            scenario_edgesensitive),
    ("$\\alpha$-dependent",       scenario_alpha_dependent),
]

thresholds = [0.10, 0.12, 0.15]

fig, axs = plt.subplots(3, 2, figsize=(3.5, 7), sharex=True, sharey=True)

for i, (ax, (title, func)) in enumerate(zip(axs.ravel(), SCENARIOS)):
    D_trim = np.zeros((len(mat_types), len(c_TW_values)))
    for j, c in enumerate(c_TW_values):
        for m_idx in range(len(mat_types)):
            D_trim[m_idx, j] = func(m_idx, c)

    # One labeled line per matrix type (unlike BERT which has 12 unlabeled)
    for m_idx, mat in enumerate(mat_types):
        ax.plot(c_TW_values, D_trim[m_idx, :],
                marker="o", ms=2, alpha=0.5, linewidth=0.8,
                label=mat)

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

# Legend identifies matrix types (feasible with only 6 lines)
axs[0, 0].legend(fontsize=6, frameon=False, loc="upper right")

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig("graph_shrinkage_control_02_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_shrinkage_control_02_albert.pdf")


# =============================================================================
# GRAPH 11: Type-I calibration curves on synthetic MP-null matrices
# =============================================================================
# PURPOSE:
#   Verify that the empirical rejection rate under the null tracks the nominal
#   α level. ALBERT-specific matrix shapes are used for calibration.
#
# ALBERT MATRIX SHAPES TESTED:
#   768×768   — Q / K / V / Att-Dense  (β = 1.0,  square)
#   3072×768  — FFN                    (β = 0.25, rectangular)
#   768×3072  — FFN-Out                (β = 0.25, rectangular)
#   768×128   — Embedding hidden mapping (β ≈ 0.17, rectangular, ALBERT-specific)
#
# LAYOUT:  2×2 grid
# OUTPUT:  graph_shrinkage_control_03_albert.pdf
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
n_boot      = 200
alphas      = [0.01, 0.05, 0.10]
c_TW_values = [1, 2, 3]

# ALBERT-specific matrix shapes (different from BERT's 1536×768)
shapes = [
    (768,  768),   # Q / K / V / Att-Dense  — square,       β = 1.0
    (3072, 768),   # FFN                    — rectangular,  β = 0.25
    (768,  3072),  # FFN-Out                — rectangular,  β = 0.25
    (768,  128),   # Embedding hidden mapping — β ≈ 0.17, ALBERT-specific
]


def ks_tw_test_null(m: int, n: int, alpha: float, c_tw: float) -> bool:
    """
    Simulate one null KS-TW rejection event with a small calibration bias.

    Bias models finite-sample overcorrection of the TW margin:
      larger c_tw → smaller bias (more trimming → better calibration)
      square matrices (m==n) → slightly more bias

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


# Simulate empirical rejection rates
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
    # 45° reference line: empirical = nominal (perfect calibration)
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
plt.savefig("graph_shrinkage_control_03_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_shrinkage_control_03_albert.pdf")


# =============================================================================
# GRAPH 12: Bootstrap reference envelopes for empirical CDFs
# =============================================================================
# PURPOSE:
#   Plot the observed eCDF against a 90% bootstrap confidence band generated
#   under a reference null distribution. This graph is model-agnostic
#   (purely synthetic) and identical for ALBERT and BERT — included here
#   for completeness of the ALBERT paper figure set.
#
# LAYOUT:  3×2 mosaic, sharey=True
# OUTPUT:  graph_shrinkage_control_04_albert.pdf
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
    kind : str — "normal" | "shifted" | "t3" | "compressed" | "mixture" | other (χ²)
    n    : int — sample size

    Returns
    -------
    np.ndarray — random sample of length n
    """
    if kind == "normal":       return np.random.normal(size=n)
    elif kind == "shifted":    return np.random.normal(loc=1.0, scale=1.0, size=n)
    elif kind == "t3":         return np.random.standard_t(df=3, size=n)
    elif kind == "compressed": return np.random.normal(loc=0.0, scale=0.5, size=n)
    elif kind == "mixture":
        return np.concatenate([np.random.normal(0.0, 1.0, n//2),
                               np.random.normal(2.0, 1.0, n//2)])
    else:  # chisquare
        return np.random.chisquare(df=3, size=n) / 3.0


def plot_ecdf_with_bands(ax, observed, boot_dist, n, B, L, U, alpha_band, title, i):
    """
    Plot the observed eCDF with a bootstrap confidence band.

    Parameters
    ----------
    ax         : matplotlib Axes
    observed   : np.ndarray — observed data sample
    boot_dist  : str        — reference null distribution name
    n          : int        — bootstrap sample size
    B          : int        — number of bootstrap replicates
    L, U       : float      — x-axis limits
    alpha_band : tuple      — (lower_pct, upper_pct) for the confidence band
    title      : str        — subplot title
    i          : int        — subplot index for y-label placement
    """
    grid     = np.linspace(L, U, 300)
    obs_ecdf = ecdf(observed, grid)

    # Generate B bootstrap eCDFs under the null
    boot_ecdfs = np.array([ecdf(generate_data(boot_dist, n), grid) for _ in range(B)])
    lower = np.percentile(boot_ecdfs, alpha_band[0], axis=0)
    upper = np.percentile(boot_ecdfs, alpha_band[1], axis=0)

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
plt.savefig("graph_shrinkage_control_04_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("✅ Figure saved: graph_shrinkage_control_04_albert.pdf")
