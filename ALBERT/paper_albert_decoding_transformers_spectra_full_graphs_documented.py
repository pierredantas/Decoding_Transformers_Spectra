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
It implements the complete pipeline for analyzing spectral properties of
ALBERT weight matrices under the Marchenko-Pastur (MP) random matrix theory
framework.

ALBERT-BASE-V2 ARCHITECTURE — KEY DIFFERENCES FROM BERT
---------------------------------------------------------
ALBERT (A Lite BERT) uses cross-layer parameter sharing: all 12 virtual
encoder layers share ONE set of weights stored in a single layer group.

  - Encoder groups       : 1  (albert_layer_groups.0)
  - Albert layers / group: 1  (albert_layers.0)
  - Virtual layers       : 12 (all share the same weights)
  - hidden_size          : 768
  - intermediate_size    : 3072
  - num_attention_heads  : 12
  - Parameters           : ~12M  (vs ~110M for BERT-base)
  - Embedding factorization: embedding_size=128, then projected to 768
    via encoder.embedding_hidden_mapping_in (768×128, beta~0.17)

ALBERT MATRIX SHAPES AND ASPECT RATIOS
----------------------------------------
All matrices live under one path prefix:
  encoder.albert_layer_groups.0.albert_layers.0.*

  attention.query.weight  : (768,  768)   beta = 1.0   (square)
  attention.key.weight    : (768,  768)   beta = 1.0   (square)
  attention.value.weight  : (768,  768)   beta = 1.0   (square)
  attention.dense.weight  : (768,  768)   beta = 1.0   (square)
  ffn.weight              : (3072, 768)   beta = 0.25  (rectangular)
  ffn_output.weight       : (768, 3072)   beta = 0.25  (rectangular)
  embedding_hidden_mapping_in.weight: (768, 128)  beta ~0.167

BETA VALUE DISTRIBUTION — DISCRETE (unlike BERT-base continuous range)
  "Attention" family: beta = 1.0   (4 square matrices: Q, K, V, Att-Dense)
  "FFN"       family: beta = 0.25  (2 rectangular matrices: FFN, FFN-Out)

KEY PATH NORMALIZATION (extract_matrices)
------------------------------------------
ALBERT uses numeric group/layer indices in keys. These are prefixed with "g":
  "encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight"
  -> encoder/albert_layer_groups/g0/albert_layers/g0/attention/query/weight.npy
Note: unlike BERT, "layer" is NOT dropped — only digit indices are prefixed.

TITLE GENERATION (_short_title)
---------------------------------
ALBERT has no layer.N pattern in its keys (no per-layer numbering).
_short_title() matches keywords directly (query, key, value, dense, ffn_output, ffn)
without a layer prefix, unlike BERT's "L{N}.keyword" format.

HEATMAP LAYOUT — ALBERT vs BERT
---------------------------------
BERT-base:  decisions shape (12, 6), 12 layers x 6 matrix types
ALBERT:     decisions shape (1,  6),  1 group  x 6 matrix types
  yticklabels = ["G0"]  (single shared layer group)
  ylabel = "Group"      (not "Layer")

GRAPH 7 — X-AXIS DIFFERS FROM BERT
-------------------------------------
BERT (Graph 7):   x = layers 0-11 or 0-23, show only 0 and last
ALBERT (Graph 7): x = matrix types [Q, K, V, Att-Dense, FFN, FFN-Out],
                  ALL labels shown on all rows (6 types fit without clutter)

MARCHENKO-PASTUR LAW
---------------------
For a random matrix W of shape (m, n) with i.i.d. N(0,1) entries,
the empirical spectral distribution of eigenvalues lambda of W^T W / max(m,n)
converges to the MP law with aspect ratio beta = min(m,n) / max(m,n):

    f_beta(lambda) = sqrt[(lambda+ - lambda)(lambda - lambda-)] / (2*pi*beta*lambda)

where  lambda+/- = (1 +/- sqrt(beta))^2  are the MP support boundaries.

TRIMMING STRATEGIES
-------------------
All 6 ALBERT matrices use "tw_or_fraction" with c_tw=2.0, frac_sq=frac_rect=0.05
(uniform across matrix types, unlike BERT which varies by layer).
  - "tw"            : delta = c_alpha * n_eff^(-2/3) * (1 + sqrt(beta))^(4/3)
  - "fraction"      : delta = frac * (lambda+ - lambda-)
  - "tw_or_fraction": max(TW, fraction)

PIPELINE
--------
  Section 1  -> extract_matrices()  : save raw .npy matrices + manifest.json
  Section 2  -> Build WMP           : column-standardize -> albert_weights_WMP/
  Graphs 1-12 -> Figure generation  : all saved as .pdf

OUTPUT FILES
------------
  albert_weights/                      raw .npy weight matrices + manifest.json
  albert_weights_WMP/                  column-standardized matrices + manifest.json
  step1_column_stats_albert.json       per-column mean/std (human-readable)
  step1_column_stats_albert.npz        per-column mean/std arrays (numpy)
  graph_core_diag_01_albert.pdf        Graph 1:  ePDF vs MP PDF (trimming conditions)
  graph_core_diag_02_albert.pdf        Graph 2:  eCDF vs MP CDF (trimming conditions)
  graph_core_diag_03_albert.pdf        Graph 3:  ePDF vs MP PDF (all 6 matrix types)
  graph_core_diag_04_albert.pdf        Graph 4:  Residual CDF (eCDF - MP CDF)
  graph_core_diag_05_albert.pdf        Graph 5:  QQ plots vs MP quantiles
  graph_level_views_01_albert.pdf      Graph 6:  KS heatmaps (1 group x 6 types)
  graph_level_views_02_albert.pdf      Graph 7:  Per-matrix-type acceptance rates
  graph_level_views_03_albert.pdf      Graph 8:  beta vs KS statistic scatter
  graph_shrinkage_control_01_albert.pdf Graph 9: Bootstrap p-value distributions
  graph_shrinkage_control_02_albert.pdf Graph 10: KS-TW edge relaxation sensitivity
  graph_shrinkage_control_03_albert.pdf Graph 11: Type-I calibration curves
  graph_shrinkage_control_04_albert.pdf Graph 12: eCDF vs bootstrap bands
"""

# =============================================================================
# SECTION 1: Extract ALBERT-base-v2 weight matrices
# =============================================================================
# PURPOSE:
#   Load ALBERT-base-v2 from HuggingFace, extract all 2-D weight matrices
#   from the single shared layer group, and save them as .npy files with a
#   manifest.json index for downstream analysis.
#
# ALBERT-SPECIFIC KEY STRUCTURE:
#   All learned attention/FFN weights live under one path:
#   "encoder.albert_layer_groups.0.albert_layers.0.<module>.weight"
#   Digit indices are prefixed with "g": "0" -> "g0"
#
# KEY PATTERNS KEPT (only_linear_like=True):
#   "attention.query", "attention.key", "attention.value", "attention.dense"
#   "ffn.", "ffn_output", "encoder.embedding_hidden_mapping_in"
#   (excludes: LayerNorm weights, pooler.dense, classifier heads)
#
# INPUTS:  HuggingFace model "albert-base-v2" (downloaded automatically)
# OUTPUTS: albert_weights/ directory with .npy files and manifest.json
# =============================================================================

# extract_albert_matrices.py
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

    Unlike BERT, ALBERT uses cross-layer parameter sharing: all virtual layers
    share one set of weights stored in albert_layer_groups.0.albert_layers.0.
    This means only 6 unique weight matrices exist for all 12 virtual layers.

    Key normalization replaces digit indices with "g"-prefixed strings:
      "encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight"
      -> encoder/albert_layer_groups/g0/albert_layers/g0/attention/query/weight.npy

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: "albert-base-v2".
    out_dir : str
        Output directory for .npy/.npz files and manifest.json.
    include_bias : bool
        If True, also saves 1-D bias vectors. Default: False.
    only_linear_like : bool
        If True, keeps only attention/FFN/embedding-projection matrices.
        Default: True.
    dtype : str
        Numeric precision -- "float32" or "float64". Default: "float32".
    save_format : str
        File format -- "npy" or "npz". Default: "npy".

    Returns
    -------
    None
        Saves files to disk, prints a summary, and lists all available
        matrix names with shapes (useful for configuring graph param_sets).
    """
    assert save_format in {"npy", "npz"}

    # 1) Load model on CPU, no grad
    torch.set_grad_enabled(False)
    model = AlbertModel.from_pretrained(model_name)
    model.eval()
    model.to("cpu")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 2) Iterate state_dict
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

        Keeps 2-D weight matrices from attention, FFN, and embedding projection.
        ALBERT-specific patterns differ from BERT:
          - "attention.dense" instead of "output.dense"
          - "ffn." and "ffn_output" instead of "intermediate.dense"
          - "encoder.embedding_hidden_mapping_in" (ALBERT's factorized embedding)
        """
        if tensor.ndim == 2:
            if only_linear_like:
                names_we_like = (
                    "attention.query", "attention.key", "attention.value",
                    "attention.dense", "ffn.", "ffn_output",
                    "full_layer_layer_norm", "pooler.dense",
                    "encoder.embedding_hidden_mapping_in",
                )
                return any(n in key for n in names_we_like)
            return True
        if include_bias and tensor.ndim == 1:
            return "bias" in key
        return False

    # 3) Save tensors and record metadata
    # Key normalization: digit indices -> "g{N}" prefix
    # e.g. "encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight"
    #   -> encoder/albert_layer_groups/g0/albert_layers/g0/attention/query/weight.npy
    for key in sorted(sd.keys()):
        t = sd[key]
        if not keep_param(key, t):
            continue

        arr = t.detach().cpu().to(
            dtype=torch.float32 if dtype == "float32" else torch.float64
        ).numpy()

        # Normalize key to safe path: prefix digit indices with "g"
        parts = key.split(".")
        norm_parts = []
        for p in parts:
            if p.isdigit():
                norm_parts.append(f"g{p}")  # group/layer index: "0" -> "g0"
            else:
                norm_parts.append(p)

        save_dir = out.joinpath(*norm_parts[:-1])
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = norm_parts[-1]
        ext = ".npy" if save_format == "npy" else ".npz"
        path = save_dir / f"{stem}{ext}"

        if save_format == "npy":
            np.save(path, arr)
        else:
            np.savez_compressed(path, data=arr)

        manifest["files"].append({
            "name": key,
            "path": str(path.relative_to(out)),
            "shape": list(arr.shape),
            "ndim": arr.ndim,
            "dtype": str(arr.dtype)
        })

    # 4) Save embeddings separately under albert_weights/embeddings/
    # ALBERT embeddings: word_embeddings (30000×128), position_embeddings (512×128),
    # token_type_embeddings (2×128) — note 128-dim, not 768.
    # The 128->768 projection is saved separately as embedding_hidden_mapping_in.
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
                "name": f"embeddings.{subkey}",
                "path": str(path.relative_to(out)),
                "shape": list(arr.shape),
                "ndim": arr.ndim,
                "dtype": str(arr.dtype)
            })

    # 5) Write manifest.json -- required by all downstream graph scripts
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved to: {out.resolve()}")
    print(f"Total tensors saved: {len(manifest['files'])}")

    # 6) Print available matrix names (useful for configuring graph param_sets)
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

"""# Build Wn and save stats"""

# =============================================================================
# SECTION 2: Build WMP (column-standardized) matrices and save with manifest
# =============================================================================
# PURPOSE:
#   Normalize each weight matrix W by its column-wise mean (mu) and standard
#   deviation (sd), producing WMP = (W - mu) / sd. Required before eigenvalue
#   analysis under the MP framework, which assumes i.i.d. zero-mean unit-variance entries.
#
# ALBERT-SPECIFIC NOTE:
#   Only 6 unique WMP matrices are produced (one per unique weight matrix).
#   The cross-layer sharing means no per-layer copies are needed.
#
# MATHEMATICAL DEFINITION:
#   For W of shape (m, n):
#     mu[j]    = mean of column j              (shape: n,)
#     sd[j]    = std  of column j              (shape: n,)
#     WMP[i,j] = (W[i,j] - mu[j]) / sd[j]
#   Zero-variance columns (sd = 0) are left unchanged (safe_sd = 1.0).
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

# =========================
# Build WMP and save stats for inversion
# =========================
# Inputs:
#   - albert_weights/ (from extract_matrices())
# Outputs:
#   - step1_column_stats_albert.json  (human-readable, key map)
#   - step1_column_stats_albert.npz   (exact np arrays of means/stds)
#   - albert_weights_WMP/             (normalized matrices) + manifest.json

import json
import re
from datetime import datetime, UTC
from pathlib import Path
import numpy as np

# ALBERT-specific directory and file paths
WEIGHTS_DIR = "albert_weights"                       # input: raw matrices
STATS_JSON  = "step1_column_stats_albert.json"      # output: human-readable stats
STATS_NPZ   = "step1_column_stats_albert.npz"      # output: numpy arrays of mu/sd
WMP_DIR     = "albert_weights_WMP"                  # output: normalized matrices

def _safe_key(idx: int, kind: str, name: str) -> str:
    """
    Build a stable, filesystem-safe NPZ key for storing mu/sd arrays.

    Format: '0003__mean__encoder_albert_layer_groups_g0_attention_query_weight'
    Truncated to 200 chars to avoid OS path-length limits.

    Parameters
    ----------
    idx  : int -- position in manifest (zero-padded to 4 digits)
    kind : str -- "mean" or "std"
    name : str -- original parameter name from state_dict
    """
    base = f"{idx:04d}__{kind}__" + re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    base = re.sub(r"__+", "__", base).strip("_")
    return base[:200]

def _load_matrix(path: Path) -> np.ndarray:
    """Load a matrix from a .npy or .npz file."""
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        return np.load(path)["data"]
    raise ValueError(f"Unsupported file type: {path.suffix}")

def _save_npy(path: Path, arr: np.ndarray) -> None:
    """Save an array as .npy, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

# ------------- Step 1: compute and save column-wise statistics -------------
# Statistics computed in float64 for numerical stability.
weights_dir = Path(WEIGHTS_DIR)
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

files = manifest.get("files", [])
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
        continue  # skip 1-D bias vectors

    # Column-wise mean and std in float64 for numerical stability
    W  = W.astype(np.float64, copy=False)
    mu = W.mean(axis=0)    # shape (n,) -- per-column mean
    sd = W.std(axis=0)     # shape (n,) -- per-column std

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

with open(STATS_JSON, "w") as jf:
    json.dump(stats_json, jf, indent=2)
np.savez_compressed(STATS_NPZ, **npz_store)

print(f"[STATS] Done. Matrices processed: {processed}")
print(f"[STATS] JSON: {STATS_JSON}")
print(f"[STATS] NPZ : {STATS_NPZ}")

# ------------- Step 2: build WMP = (W - mu) / sd and save with manifest -------------
# The "_WMP" suffix appended to each name is used by all graph scripts.
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

# Write WMP manifest -- used by all graph scripts via _find_manifest_entry()
with open(out_root / "manifest.json", "w") as f:
    json.dump(wmp_manifest, f, indent=2)

# ------------- Quick verification on a few matrices -------------
# Confirms WMP can be inverted back to W: max|W - (WMP*sd + mu)| < 1e-5
from itertools import islice

def _check_one(relpath: str, mu: np.ndarray, sd: np.ndarray) -> float:
    """
    Verify round-trip reconstruction: W ~= WMP * sd + mu.

    Returns
    -------
    float -- maximum absolute reconstruction error across all elements
    """
    W       = _load_matrix(weights_dir / relpath).astype(np.float64)
    WMP     = _load_matrix(out_root    / relpath).astype(np.float64)
    safe_sd = np.where(sd == 0, 1.0, sd)
    W_rec   = WMP * safe_sd.reshape(1, -1) + mu.reshape(1, -1)
    return float(np.max(np.abs(W - W_rec)))

print("====================================================")
print(f"WMP saved: {saved} | Skipped (non-2D): {skipped}")
print(f"Manifest written: {out_root / 'manifest.json'}")

errs = []
for e in islice(stats_json["files"], 3):
    rel = e["path"]
    mu  = stats_npz[e["npz_keys"]["mean"]]
    sd  = stats_npz[e["npz_keys"]["std"]]
    errs.append((rel, _check_one(rel, mu, sd)))
for rel, err in errs:
    print(f"[CHECK] {rel}: max |W - (WMP*sd+mu)| = {err:.3e}")
print("====================================================")

"""# Graph 1 - Empirical PDF (ePDF) vs. conditional MPd PDF under different trimming conditions."""

# =============================================================================
# GRAPH 1: Empirical PDF vs. conditional MP PDF (all 6 ALBERT matrix types)
# =============================================================================
# PURPOSE:
#   Compare the empirical eigenvalue density (histogram) against the
#   theoretical MP PDF for all 6 ALBERT matrix types. Unlike BERT, which
#   varies the matrix AND trim parameters across subplots, ALBERT uses
#   the same trim_kind/c_tw/fracs for all matrices (uniform treatment).
#
# SETTINGS: each of the 6 unique shared-weight matrices, all with
#   trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=frac_rect=0.05.
#
# BETA VALUES:  Q/K/V/Att.Dense -> beta=1.0 (square, narrow MP);
#               FFN/FFN-Out     -> beta=0.25 (rectangular, wide MP)
#
# LAYOUT: 2x3 mosaic, 1-column IEEE/Springer format (7 x 3 inches)
# OUTPUT: graph_core_diag_01_albert.pdf
# =============================================================================

# --- Mosaic plots for six parameter sets (PDFs) ---
# Paper-ready version (1-column Springer/IEEE style, FIXED layout)
# Model: ALBERT-base-v2

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# GLOBAL STYLE (Paper)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# Helpers
# =========================
def _load_matrix(p: Path):
    """Load a matrix from .npy or .npz file."""
    if p.suffix == ".npy":  return np.load(p)
    if p.suffix == ".npz":  return np.load(p)["data"]
    raise ValueError(f"Unsupported file type: {p.suffix}")

def _mp_support(beta: float):
    """Compute MP support boundaries lambda+/- = (1 +/- sqrt(beta))^2."""
    r = np.sqrt(beta); return (1 - r)**2, (1 + r)**2

def _mp_pdf(x, beta, a, b):
    """Evaluate the MP PDF: f_beta(x) = sqrt[(b-x)(x-a)] / (2*pi*beta*x)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = (x >= a) & (x <= b)
    xm = np.clip(x[m], 1e-15, None)
    out[m] = np.sqrt((b - xm) * (xm - a)) / (2 * np.pi * beta * xm)
    return out

def _cumtrapz_np(y, x):
    """Cumulative trapezoidal integration (numpy-only)."""
    dx = np.diff(x); seg = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate([[0.0], np.cumsum(seg)])

def _mp_cdf(x, beta, grid_points=8192):
    """
    Evaluate the MP CDF via numerical integration.
    Uses quadratic grid t^2 to oversample near the singular lower boundary lambda-.
    """
    a, b = _mp_support(beta)
    t = np.linspace(0.0, 1.0, grid_points)
    g = a + (b - a) * t * t
    pdf = _mp_pdf(g, beta, a, b)
    cdf_vals = _cumtrapz_np(pdf, g)
    cdf_vals /= cdf_vals[-1]
    return np.interp(x, g, cdf_vals, left=0.0, right=1.0)

def _edge_margin(beta, m, n, trim_kind, c_tw, frac_sq, frac_rect):
    """
    Compute edge trimming margin delta using one of three strategies.
    "tw"           : Tracy-Widom scale; "fraction": fixed bandwidth fraction;
    "tw_or_fraction": max of both.
    """
    a, b = _mp_support(beta); bandwidth = b - a
    n_eff = min(m, n); is_square = (m == n)
    tw = c_tw * (n_eff ** (-2/3)) * (1 + np.sqrt(beta))**(4/3)
    frac = (frac_sq if is_square else frac_rect) * bandwidth
    if trim_kind == "tw": return tw
    if trim_kind == "fraction": return frac
    if trim_kind == "tw_or_fraction": return max(tw, frac)
    raise ValueError("Invalid trim_kind")

def _find_manifest_entry(manifest, target):
    """Look up a matrix entry in the WMP manifest by name or path substring."""
    for e in manifest["files"]:
        if e.get("name","") == target: return e
    for e in manifest["files"]:
        if target in e.get("name","") or target in e.get("path",""): return e
    raise ValueError(f"Matrix '{target}' not found in manifest")

def compute_trimmed(W, TRIM_KIND, C_TW, EDGE_FRAC_SQUARE, EDGE_FRAC_RECT):
    """
    Compute trimmed eigenvalues and conditional MP distribution functions.

    Steps: SVD -> eigenvalues -> trim interval [L,U] -> conditional MP PDF/CDF.

    Parameters
    ----------
    W               : np.ndarray -- 2-D WMP-normalized matrix
    TRIM_KIND       : str        -- trimming strategy
    C_TW            : float      -- TW scaling coefficient
    EDGE_FRAC_SQUARE: float      -- edge fraction for square matrices (beta=1.0)
    EDGE_FRAC_RECT  : float      -- edge fraction for rectangular matrices

    Returns
    -------
    lam_trim    : np.ndarray -- trimmed eigenvalues in [L, U]
    (a,b,L,U,beta): tuple   -- MP bounds and trim interval
    mp_pdf_cond : callable  -- conditional MP PDF (normalized to [L,U])
    mp_cdf_cond : callable  -- conditional MP CDF (normalized to [L,U])
    N_trim      : int       -- number of eigenvalues retained after trimming
    """
    m,n = W.shape; beta = min(m,n)/max(m,n)
    s = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2)/max(m,n); lambdas.sort()
    a,b = _mp_support(beta)
    delta = _edge_margin(beta,m,n,TRIM_KIND,C_TW,EDGE_FRAC_SQUARE,EDGE_FRAC_RECT)
    L,U = a+delta, b-delta
    if L>=U: L,U=a,b
    mask = (lambdas>=L)&(lambdas<=U)
    lam_trim = lambdas[mask]; N_trim = lam_trim.size
    FL,FU = _mp_cdf([L,U], beta); den = max(FU-FL,1e-12)
    mp_pdf_cond = lambda x: _mp_pdf(x,beta,a,b)/den
    mp_cdf_cond = lambda x: np.clip((_mp_cdf(x,beta)-FL)/den,0,1)
    return lam_trim, (a,b,L,U,beta), mp_pdf_cond, mp_cdf_cond, N_trim

# =========================
# Config
# =========================
WMP_DIR   = "albert_weights_WMP"
COND_GRID = 2000
HIST_BINS = 150   # 150 bins for ALBERT (vs 80 for BERT); more bins suit narrow spectra

# ALBERT-base-v2: all 6 unique shared-weight matrices, uniform trim parameters.
# All subplots use the same c_tw=2.0 and fracs=0.05 -- only matrix type varies.
SETTINGS = [
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05, label="Set 1 -- Att.Q"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05, label="Set 2 -- Att.K"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05, label="Set 3 -- Att.V"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.ffn.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05, label="Set 4 -- FFN"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05, label="Set 5 -- FFN-Out"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05, label="Set 6 -- Att.Dense"),
]

# Load WMP manifest once
man_path = Path(WMP_DIR) / "manifest.json"
manifest = json.load(open(man_path))

# =========================
# MOSAIC: PDFs
# =========================
fig, axes = plt.subplots(2, 3, figsize=(7, 3))   # 2x3 grid
axes = axes.flatten()

for i, (ax, cfg) in enumerate(zip(axes, SETTINGS)):
    entry = _find_manifest_entry(manifest, cfg["name"])
    W = _load_matrix(Path(WMP_DIR)/entry["path"]).astype(np.float64, copy=False)

    lam_trim,(a,b,L,U,beta),mp_pdf_cond,_,N_trim = compute_trimmed(
        W, cfg["trim_kind"], cfg["c_tw"], cfg["frac_sq"], cfg["frac_rect"]
    )

    x_grid = np.linspace(L, U, COND_GRID)

    # Empirical density histogram (gray)
    if N_trim > 0:
        ax.hist(lam_trim, bins=HIST_BINS, range=(L, U),
                density=True, alpha=0.4, color="#888888")

    # Theoretical conditional MP PDF (black)
    ax.plot(x_grid, mp_pdf_cond(x_grid), color="black")

    # Vertical reference lines: light gray = MP support, dark gray = trim interval
    for v, col in [(a,"#aaaaaa"),(b,"#aaaaaa"),(L,"#444444"),(U,"#444444")]:
        ax.axvline(v, color=col, linestyle="--", linewidth=0.8)

    ax.set_title(f"{cfg['label']}\n($\\beta$={beta:.2f}, N={N_trim})", pad=2)
    ax.grid(True)

    # Y label only on left column
    if i % 3 == 0:
        ax.set_ylabel("$\\lambda$ vs. PDF")
        ax.yaxis.set_label_coords(-0.25, 0.5)
    else:
        ax.set_ylabel("")

plt.subplots_adjust(hspace=0.5, wspace=0.4)

plt.savefig("graph_core_diag_01_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Figure saved: graph_core_diag_01_albert.pdf")

"""# Graph 2 - Empirical CDF (eCDF) vs. conditional MPd CDF under different trimming conditions."""

# =============================================================================
# GRAPH 2: Empirical CDF vs. conditional MP CDF (all 6 ALBERT matrix types)
# =============================================================================
# PURPOSE:
#   Overlay the empirical step-CDF of trimmed eigenvalues against the
#   theoretical conditional MP CDF for all 6 ALBERT unique matrices.
#   Same SETTINGS as Graph 1 -- only the plot type (CDF vs PDF) differs.
#
# LAYOUT: 2x3 mosaic (reuses SETTINGS and manifest from Graph 1)
# OUTPUT: graph_core_diag_02_albert.pdf
# =============================================================================

# =========================
# Config
# =========================
WMP_DIR   = "albert_weights_WMP"
COND_GRID = 2000

# Same 6 matrices as Graph 1, same uniform trim parameters
SETTINGS = [
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05, label="Set 1 -- Att.Q"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05, label="Set 2 -- Att.K"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05, label="Set 3 -- Att.V"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.ffn.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05, label="Set 4 -- FFN"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05, label="Set 5 -- FFN-Out"),
    dict(name="encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight_WMP",
         trim_kind="tw_or_fraction", c_tw=2.0, frac_sq=0.05, frac_rect=0.05, label="Set 6 -- Att.Dense"),
]

# Load manifest
man_path = Path(WMP_DIR) / "manifest.json"
manifest = json.load(open(man_path))

# =========================
# MOSAIC: CDFs
# =========================
fig, axes = plt.subplots(2, 3, figsize=(7, 3))
axes = axes.flatten()

for i, (ax, cfg) in enumerate(zip(axes, SETTINGS)):
    entry = _find_manifest_entry(manifest, cfg["name"])
    W = _load_matrix(Path(WMP_DIR)/entry["path"]).astype(np.float64, copy=False)

    lam_trim,(a,b,L,U,beta),_,mp_cdf_cond,N_trim = compute_trimmed(
        W, cfg["trim_kind"], cfg["c_tw"], cfg["frac_sq"], cfg["frac_rect"]
    )

    x_grid = np.linspace(L, U, COND_GRID)

    # Theoretical conditional MP CDF (black)
    ax.plot(x_grid, mp_cdf_cond(x_grid), color="black",
            label="MP" if i == 0 else None)

    # Empirical step-CDF of trimmed eigenvalues (gray)
    if N_trim > 0:
        y_ecdf = np.arange(1, N_trim+1) / N_trim
        ax.step(lam_trim, y_ecdf, where="post", linewidth=1.0, color="#888888",
                label="Empirical" if i == 0 else None)

    # Vertical reference lines
    for v, col in [(a,"#aaaaaa"),(b,"#aaaaaa"),(L,"#444444"),(U,"#444444")]:
        ax.axvline(v, color=col, linestyle="--", linewidth=0.8)

    ax.set_title(f"{cfg['label']}\n($\\beta$={beta:.2f}, N={N_trim})", pad=2)
    ax.set_ylim(0, 1)
    ax.grid(True)

    if i % 3 == 0:
        ax.set_ylabel("$\\lambda$ vs. CDF")
        ax.yaxis.set_label_coords(-0.25, 0.5)
    else:
        ax.set_ylabel("")

# Single legend on the first axes only
axes[0].legend(loc="lower center", bbox_to_anchor=(0.60, 0), frameon=False)

plt.subplots_adjust(hspace=0.5, wspace=0.4)

plt.savefig("graph_core_diag_02_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Figure saved: graph_core_diag_02_albert.pdf")

"""# Graph 3 - Empirical PDF (ePDF) vs. conditional MPd PDF across selected layers and matrix types."""

# =============================================================================
# GRAPH 3: Empirical PDF vs. MP PDF across all 6 ALBERT matrix types (global scale)
# =============================================================================
# PURPOSE:
#   Same intent as BERT Graph 3, but ALBERT has no layer variation.
#   All 6 subplots represent the 6 unique matrix types, enabling direct
#   comparison across matrix families (square Attention vs rectangular FFN).
#   A global [L,U] x-axis ensures fair visual comparison across beta values.
#
# LAYOUT: 2x3 mosaic with shared axes (global scale)
# OUTPUT: graph_core_diag_03_albert.pdf
# =============================================================================

# --- Graph #3: Empirical density vs MP pdf (trimmed interior) ---
# Model: ALBERT-base-v2

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# GLOBAL STYLE (Paper)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# Config
# =========================
WMP_DIR = "albert_weights_WMP"

# All 6 ALBERT unique matrices -- same uniform trim parameters throughout.
# Note: FFN (3072x768, beta=0.25) has a wider MP support than Q/K/V (768x768, beta=1.0),
# so the global x-axis will be dominated by the FFN support range.
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

# =========================
# Helpers
# =========================
def _load_matrix(p: Path):
    if p.suffix == ".npy":  return np.load(p)
    if p.suffix == ".npz":  return np.load(p)["data"]
    raise ValueError(f"Unsupported file type: {p.suffix}")

def _mp_support(beta: float):
    r = np.sqrt(beta); return (1 - r)**2, (1 + r)**2

def _mp_pdf(x, beta, a, b):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = (x >= a) & (x <= b)
    xm = np.clip(x[m], 1e-15, None)
    out[m] = np.sqrt((b - xm) * (xm - a)) / (2 * np.pi * beta * xm)
    return out

def _cumtrapz_np(y, x):
    dx = np.diff(x); seg = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate([[0.0], np.cumsum(seg)])

def _mp_cdf(x, beta, grid_points=GRID_POINTS):
    a, b = _mp_support(beta)
    t = np.linspace(0.0, 1.0, grid_points)
    g = a + (b - a) * t * t
    pdf = _mp_pdf(g, beta, a, b)
    cdf_vals = _cumtrapz_np(pdf, g)
    cdf_vals /= cdf_vals[-1]
    return np.interp(x, g, cdf_vals, left=0.0, right=1.0)

def _edge_margin(beta, m, n, trim_kind, c_tw, frac_sq, frac_rect):
    a, b = _mp_support(beta); bandwidth = b - a
    n_eff = min(m, n); is_square = (m == n)
    tw = 0.0 if c_tw is None else c_tw * (n_eff ** (-2/3)) * (1 + np.sqrt(beta))**(4/3)
    frac = (frac_sq if is_square else frac_rect) * bandwidth
    if trim_kind == "tw":             return tw
    if trim_kind == "fraction":       return frac
    if trim_kind == "tw_or_fraction": return max(tw, frac)
    raise ValueError("Invalid trim_kind")

def _find_manifest_entry(manifest, target):
    for e in manifest["files"]:
        if e.get("name","") == target: return e
    for e in manifest["files"]:
        if target in e.get("name","") or target in e.get("path",""): return e
    raise ValueError(f"Matrix '{target}' not found in manifest")

# =========================
# Load manifest
# =========================
man_path = Path(WMP_DIR) / "manifest.json"
manifest = json.load(open(man_path))

# =========================
# First pass: compute global [L,U]
# =========================
# Ensures a shared x-axis scale across all 6 subplots for fair comparison.
# FFN (beta=0.25) will set the wider x-range; Attention (beta=1.0) will
# appear as narrower distributions within this range.
global_L, global_U = np.inf, -np.inf
for target, kind, c_tw, frac_sq, frac_rect, _ in param_sets:
    entry = _find_manifest_entry(manifest, target)
    W = _load_matrix(Path(WMP_DIR) / entry["path"]).astype(np.float64, copy=False)
    m, n = W.shape; beta = min(m,n)/max(m,n)
    s = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2) / max(m, n); lambdas.sort()
    a, b = _mp_support(beta)
    delta = _edge_margin(beta, m, n, kind, c_tw, frac_sq, frac_rect)
    L, U = a + delta, b - delta
    if L >= U: L, U = a, b
    global_L, global_U = min(global_L, L), max(global_U, U)

# =========================
# Plot grid
# =========================
fig, axes = plt.subplots(2, 3, figsize=(7, 3), sharex=True, sharey=True)
axes = axes.ravel()

for i, (ax, (target, kind, c_tw, frac_sq, frac_rect, title)) in enumerate(zip(axes, param_sets)):
    entry = _find_manifest_entry(manifest, target)
    W = _load_matrix(Path(WMP_DIR) / entry["path"]).astype(np.float64, copy=False)
    m, n = W.shape; beta = min(m,n)/max(m,n)
    s = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lambdas = (s**2) / max(m, n); lambdas.sort()
    a, b = _mp_support(beta)
    delta = _edge_margin(beta, m, n, kind, c_tw, frac_sq, frac_rect)
    L, U = a + delta, b - delta
    if L >= U: L, U = a, b

    mask_trim = (lambdas >= L) & (lambdas <= U)
    lam_trim  = lambdas[mask_trim]

    FL, FU = _mp_cdf([L,U], beta); den = max(float(FU - FL), 1e-12)
    def mp_pdf_cond(x): return _mp_pdf(x, beta, a, b) / den

    if lam_trim.size > 0:
        ax.hist(lam_trim, bins=HIST_BINS, range=(global_L, global_U),
                density=True, alpha=0.4, color="#888888",
                label="Empirical" if i == 0 else None)

    x_grid = np.linspace(global_L, global_U, COND_GRID)
    ax.plot(x_grid, mp_pdf_cond(x_grid), color="black", lw=1.2,
            label="MP" if i == 0 else None)

    for v, col in [(a,"#aaaaaa"),(b,"#aaaaaa"),(L,"#444444"),(U,"#444444")]:
        ax.axvline(v, color=col, linestyle="--", linewidth=0.8)

    ax.set_title(f"{title}\n($\\beta$={beta:.2f}, N={lam_trim.size})", pad=2)
    ax.grid(True)

    if i % 3 == 0:
        ax.set_ylabel("$\\lambda$ vs. PDF")
        ax.yaxis.set_label_coords(-0.25, 0.5)
    else:
        ax.set_ylabel("")

# Single legend on the first axes only
axes[0].legend(loc="lower right", bbox_to_anchor=(1.05, 0), frameon=False)

plt.subplots_adjust(hspace=0.5, wspace=0.4)

plt.savefig("graph_core_diag_03_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Figure saved: graph_core_diag_03_albert.pdf")

"""# Graph 4 - Empirical residual CDF (eCDF-MPd CDF) across different trimming conditions."""

# =============================================================================
# GRAPH 4: Empirical residual CDF (eCDF - MP CDF) for all 6 ALBERT matrices
# =============================================================================
# PURPOSE:
#   Plot the signed difference (eCDF - MP CDF) for all 6 ALBERT unique matrices.
#   The KS statistic D = max|residual| is shown as a dashed reference line.
#
# _short_title() ALBERT-SPECIFIC:
#   No "L{N}." prefix (no layer numbering). Uses keyword matching directly:
#   "query" -> "query", "ffn_output" -> "ffn_output" (priority-ordered list).
#
# LAYOUT: 2x3 mosaic with unified y-axis scale
# OUTPUT: graph_core_diag_04_albert.pdf
# =============================================================================

# --- Graph #4: Residual CDF plots for 6 parameter sets ---
# Model: ALBERT-base-v2

import re
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# GLOBAL STYLE (Paper)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# Config
# =========================
WMP_DIR     = "albert_weights_WMP"
GRID_POINTS = 8192
COND_GRID   = 2000

# All 6 ALBERT unique matrices, same uniform trim parameters
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

# =========================
# Helpers
# =========================
def _load_matrix(p: Path):
    if p.suffix == ".npy":  return np.load(p)
    if p.suffix == ".npz":  return np.load(p)["data"]
    raise ValueError(f"Unsupported file type: {p.suffix}")

def _mp_support(beta: float):
    r = np.sqrt(beta); return (1 - r)**2, (1 + r)**2

def _mp_pdf(x, beta, a, b):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = (x >= a) & (x <= b)
    xm = np.clip(x[m], 1e-15, None)
    out[m] = np.sqrt((b - xm) * (xm - a)) / (2 * np.pi * beta * xm)
    return out

def _cumtrapz_np(y, x):
    dx = np.diff(x); seg = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate([[0.0], np.cumsum(seg)])

def _mp_cdf(x, beta, grid_points=GRID_POINTS):
    a, b = _mp_support(beta)
    t = np.linspace(0.0, 1.0, grid_points)
    g = a + (b - a) * t * t
    pdf = _mp_pdf(g, beta, a, b)
    cdf_vals = _cumtrapz_np(pdf, g)
    cdf_vals /= cdf_vals[-1]
    return np.interp(x, g, cdf_vals, left=0.0, right=1.0)

def _edge_margin(beta, m, n, trim_kind, c_tw, frac_sq, frac_rect):
    a, b = _mp_support(beta); bandwidth = b - a
    n_eff = min(m, n); is_square = (m == n)
    tw = (c_tw or 0) * (n_eff ** (-2/3)) * (1 + np.sqrt(beta))**(4/3)
    frac = (frac_sq if is_square else frac_rect) * bandwidth
    if trim_kind == "tw":             return tw
    if trim_kind == "fraction":       return frac
    if trim_kind == "tw_or_fraction": return max(tw, frac)
    raise ValueError("Invalid trim_kind")

def _find_manifest_entry(manifest, target):
    for e in manifest["files"]:
        if e.get("name","") == target: return e
    for e in manifest["files"]:
        if target in e.get("name","") or target in e.get("path",""): return e
    raise ValueError(f"Matrix '{target}' not found in manifest")

def _short_title(name, params):
    """
    Build a compact subplot title for an ALBERT matrix key.

    ALBERT-SPECIFIC: no layer.N pattern in keys, so no "L{N}." prefix.
    Uses ordered keyword matching: "ffn_output" must come before "ffn"
    to avoid "ffn" matching both FFN and FFN-Output keys.

    Parameters
    ----------
    name   : str  -- ALBERT WMP matrix name
    params : dict -- contains TRIM_KIND and C_TW

    Returns
    -------
    str -- e.g. "query\n(tw_or_fraction, c_alpha=2.0)"
    """
    # Priority-ordered: ffn_output before ffn to avoid substring collision
    keywords = ["query", "key", "value", "dense", "ffn_output", "ffn",
                "embedding", "attention"]
    kw = next((k for k in keywords if k in name.lower()), name.split(".")[-1])
    return (f"{kw}\n"
            f"({params['TRIM_KIND']}, $c_{{\\alpha}}$={params['C_TW']})")

# =========================
# Load manifest
# =========================
man_path = Path(WMP_DIR) / "manifest.json"
manifest = json.load(open(man_path))

# =========================
# Plot grid
# =========================
fig, axes = plt.subplots(2, 3, figsize=(7, 3), sharey=True)
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

    FL, FU = _mp_cdf([L,U], beta); den = max(float(FU - FL), 1e-12)
    def mp_cdf_cond(x): return (_mp_cdf(x, beta) - FL) / den

    x_grid   = np.linspace(L, U, COND_GRID)
    emp_cdf  = np.searchsorted(np.sort(lam_trim), x_grid, side="right") / max(N_trim, 1)
    residual = emp_cdf - mp_cdf_cond(x_grid)
    ks_stat  = np.max(np.abs(residual))    # Kolmogorov-Smirnov statistic

    ax = axes[i]
    ax.plot(x_grid, residual, color="#444444", lw=1.2)
    ax.axhline(0,        color="black",   linestyle="--", linewidth=0.8)
    ax.axhline(+ks_stat, color="#888888", linestyle=":",  linewidth=0.8,
               label=f"KS={ks_stat:.3f}")
    ax.axhline(-ks_stat, color="#888888", linestyle=":",  linewidth=0.8)

    ax.set_title(_short_title(name, params), pad=2)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=False)

    if i % 3 == 0:
        ax.set_ylabel("$\\lambda$ vs. Residual CDF")
        ax.yaxis.set_label_coords(-0.25, 0.5)
    else:
        ax.set_ylabel("")

    y_min = min(y_min, residual.min())
    y_max = max(y_max, residual.max())

# Unify y scale across all 6 subplots for fair comparison
for ax in axes:
    ax.set_ylim(y_min * 1.1, y_max * 1.1)

plt.subplots_adjust(hspace=0.6, wspace=0.4)

plt.savefig("graph_core_diag_04_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Figure saved: graph_core_diag_04_albert.pdf")

"""# Graph 5 - Quantile-Quantile (QQ) plots of empirical spectra against conditional MPd quantiles"""

# =============================================================================
# GRAPH 5: QQ plots of empirical spectra vs. conditional MP quantiles
# =============================================================================
# PURPOSE:
#   Compare empirical quantiles of trimmed eigenvalues against theoretical
#   MP quantiles for all 6 ALBERT matrices. Points on the 45-degree diagonal
#   indicate perfect MP fit. ALBERT shows two distinct beta clusters:
#   - beta=1.0 (Q/K/V/Att-Dense): narrow, high-density spectra
#   - beta=0.25 (FFN/FFN-Out): wide spectra with lower density peak
#
# _short_title() uses ALBERT-specific keyword matching (no layer prefix).
#
# LAYOUT: 2x3 mosaic with shared axes
# OUTPUT: graph_core_diag_05_albert.pdf
# =============================================================================

# --- Graph #5: QQ Plot Mosaic (2x3) ---
# Model: ALBERT-base-v2

import json, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# GLOBAL STYLE (Paper)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# Config
# =========================
WMP_DIR     = "albert_weights_WMP"
GRID_POINTS = 8192
COND_GRID   = 2000

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

# =========================
# Helpers
# =========================
def _load_matrix(p: Path):
    if p.suffix == ".npy":  return np.load(p)
    if p.suffix == ".npz":  return np.load(p)["data"]
    raise ValueError(f"Unsupported file type: {p.suffix}")

def _mp_support(beta: float):
    r = np.sqrt(beta); return (1 - r)**2, (1 + r)**2

def _mp_pdf(x, beta, a, b):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = (x >= a) & (x <= b)
    xm = np.clip(x[m], 1e-15, None)
    out[m] = np.sqrt((b - xm) * (xm - a)) / (2 * np.pi * beta * xm)
    return out

def _cumtrapz_np(y, x):
    dx = np.diff(x); seg = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate([[0.0], np.cumsum(seg)])

def _mp_cdf(x, beta, grid_points=GRID_POINTS):
    a, b = _mp_support(beta)
    t = np.linspace(0.0, 1.0, grid_points)
    g = a + (b - a) * t * t
    pdf = _mp_pdf(g, beta, a, b)
    cdf_vals = _cumtrapz_np(pdf, g)
    cdf_vals /= cdf_vals[-1]
    return np.interp(x, g, cdf_vals, left=0.0, right=1.0)

def _edge_margin(beta, m, n, trim_kind, c_tw, frac_sq, frac_rect):
    a, b = _mp_support(beta); bandwidth = b - a
    n_eff = min(m, n); is_square = (m == n)
    tw = c_tw * (n_eff ** (-2/3)) * (1 + np.sqrt(beta))**(4/3)
    frac = (frac_sq if is_square else frac_rect) * bandwidth
    if trim_kind == "tw":             return tw
    if trim_kind == "fraction":       return frac
    if trim_kind == "tw_or_fraction": return max(tw, frac)
    raise ValueError("Invalid trim_kind")

def _find_manifest_entry(manifest, target):
    for e in manifest["files"]:
        if e.get("name","") == target: return e
    for e in manifest["files"]:
        if target in e.get("name","") or target in e.get("path",""): return e
    raise ValueError(f"Matrix '{target}' not found in manifest")

def _short_title(name, params):
    """
    Build a compact subplot title for an ALBERT matrix key.

    ALBERT has no layer.N pattern -- uses keyword matching directly.
    Priority ordering: ffn_output before ffn prevents substring collision.
    """
    # ALBERT has no layer.N pattern -- use keywords directly
    keywords = ["query", "key", "value", "dense", "ffn_output", "ffn",
                "embedding", "attention"]
    kw = next((k for k in keywords if k in name.lower()), name.split(".")[-1])
    return (f"{kw}\n"
            f"({params['trim_kind']}, $c_{{\\alpha}}$={params['C_TW']})")

# =========================
# Load manifest
# =========================
manifest = json.load(open(Path(WMP_DIR) / "manifest.json"))

# =========================
# Plot grid
# =========================
fig, axes = plt.subplots(2, 3, figsize=(7, 3), sharex=True, sharey=True)
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
                         params["frac_sq"], params["frac_rect"])
    L, U = a + delta, b - delta
    if L >= U: L, U = a, b

    mask_trim = (lambdas >= L) & (lambdas <= U)
    lam_trim  = lambdas[mask_trim]; N_trim = lam_trim.size

    emp_q = np.sort(lam_trim)    # empirical quantiles
    FL, FU = _mp_cdf([L,U], beta); den = max(float(FU - FL), 1e-12)
    def mp_cdf_cond(x): return (_mp_cdf(x, beta) - FL) / den

    # Invert the conditional MP CDF to get theoretical quantiles
    q_grid   = np.linspace(0, 1, N_trim)
    xs       = np.linspace(L, U, COND_GRID)
    cdf_vals = mp_cdf_cond(xs)
    mp_q     = np.interp(q_grid, cdf_vals, xs)    # theoretical quantiles

    ax.plot(mp_q, emp_q, "o", ms=1.5, alpha=0.5, color="#888888",
            label="Empirical" if i == 0 else None)
    # 45-degree reference line = perfect MP fit
    ax.plot([mp_q.min(), mp_q.max()], [mp_q.min(), mp_q.max()],
            color="black", linestyle="--", linewidth=0.8,
            label="MP" if i == 0 else None)

    ax.set_title(f"{_short_title(name, params)}\n($\\beta$={beta:.2f}, N={N_trim})", pad=2)
    ax.grid(True)

    if i % 3 == 0:
        ax.set_ylabel("MP vs.\n Empirical quantiles")
        ax.yaxis.set_label_coords(-0.25, 0.5)
    else:
        ax.set_ylabel("")

# Single legend on the first axes only
axes[0].legend(loc="lower right", bbox_to_anchor=(1.05, 0), frameon=False)

plt.subplots_adjust(hspace=0.7, wspace=0.4)

plt.savefig("graph_core_diag_05_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Figure saved: graph_core_diag_05_albert.pdf")

"""# Graph 6 - Layer vs. matrix-type heatmaps of KS test outcomes under varying alpha thresholds"""

# =============================================================================
# GRAPH 6: Group x matrix-type heatmaps of KS test outcomes (ALBERT)
# =============================================================================
# PURPOSE:
#   Visualize binary KS test decisions (0=accept, 1=reject) as heatmaps.
#   ALBERT-SPECIFIC: only 1 shared layer group (G0), so decisions shape is
#   (1, 6) instead of BERT-base's (12, 6).
#   yticklabels = ["G0"]  (single group), ylabel = "Group" (not "Layer").
#
# INPUTS:  decisions_strict, decisions_tw -- binary arrays of shape (1, 6)
#   Row 0 = G0 (shared layer group), Columns = [Q, K, V, Att-Dense, FFN, FFN-Out]
#   Grayscale: 0 (accept) -> light gray, 1 (reject) -> medium gray
#
# LAYOUT: 3x4 grid (6 scenarios x 2 methods: KS-strict and KS-TW)
# OUTPUT: graph_level_views_01_albert.pdf
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# GLOBAL STYLE (Paper)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# White (accept) -> dark gray (reject)
GRAY_CMAP = "Greys"
VMIN, VMAX = 0.1, 0.6


def plot_ks_mosaic_separate(decisions_strict, decisions_tw, savepath="graph05.pdf"):
    """
    Render a 3x4 mosaic of KS test decision heatmaps for ALBERT.

    ALBERT-SPECIFIC vs BERT:
      - decisions shape: (1, 6) -- 1 group x 6 matrix types
      - yticklabels = ["G0"] for all subplots (not layer numbers)
      - ylabel = "Group" (not "Layer")
      - x-axis labels: ["Q","K","V","Att-Dense","FFN","FFN-Out"]
        (note: "Att-Dense" not "Att-Out" as in BERT)

    Parameters
    ----------
    decisions_strict : np.ndarray, shape (1, 6)
        Binary KS-strict decisions (0=accept, 1=reject).
        Row 0 = shared layer group G0.
    decisions_tw : np.ndarray, shape (1, 6)
        Binary KS-TW decisions (0=accept, 1=reject).
    savepath : str
        Output PDF file path.
    """
    # Six parameter sets -- all use layers=[0] (single ALBERT group)
    param_sets = [
        {"alpha":0.01, "layers":[0],
         "mat_types":["Q","K","V","Att-Dense","FFN","FFN-Out"],
         "title":"Set 1 -- $\\alpha$=0.01"},
        {"alpha":0.05, "layers":[0],
         "mat_types":["Q","K","V","Att-Dense","FFN","FFN-Out"],
         "title":"Set 2 -- $\\alpha$=0.05"},
        {"alpha":0.10, "layers":[0],
         "mat_types":["Q","K","V","Att-Dense","FFN","FFN-Out"],
         "title":"Set 3 -- $\\alpha$=0.10"},
        {"alpha":0.20, "layers":[0],
         "mat_types":["Q","K","V","Att-Dense","FFN","FFN-Out"],
         "title":"Set 4 -- $\\alpha$=0.20"},
        {"alpha":0.05, "layers":[0],
         "mat_types":["Q","K","V","Att-Dense","FFN","FFN-Out"],
         "title":"Set 5 -- $\\alpha$=0.05"},
        {"alpha":0.05, "layers":[0],
         "mat_types":["Q","K","V"],
         "title":"Set 6 -- $\\alpha$=0.05"},
    ]

    fig, axes = plt.subplots(3, 4, figsize=(3.5 * 2, 7))
    axes = axes.ravel()

    for idx, params in enumerate(param_sets):
        alpha = params["alpha"]
        L     = params["layers"]    # always [0] for ALBERT
        M     = params["mat_types"]
        title = params["title"]

        d_strict = decisions_strict[np.ix_(L, range(min(len(M), decisions_strict.shape[1])))]
        d_tw     = decisions_tw[np.ix_(L,     range(min(len(M), decisions_tw.shape[1])))]

        # Remap binary values to grayscale floats
        d_strict_gs = np.where(d_strict == 0, VMIN, VMAX)
        d_tw_gs     = np.where(d_tw     == 0, VMIN, VMAX)

        # Left subplot: KS Strict
        # yticklabels=["G0"] -- ALBERT has 1 group, not 12 layers
        ax_strict = axes[idx * 2]
        sns.heatmap(d_strict_gs, cmap=GRAY_CMAP, vmin=0, vmax=1,
                    cbar=False, annot=False,
                    xticklabels=M, yticklabels=["G0"], ax=ax_strict)
        ax_strict.set_title(f"{title}\nKS Strict\n($\\alpha$={alpha})", pad=2)
        ax_strict.set_xticklabels(ax_strict.get_xticklabels(), rotation=45, ha="right")
        ax_strict.set_yticklabels(ax_strict.get_yticklabels(), rotation=0)
        ax_strict.set_ylabel("Group")    # "Group" not "Layer"
        ax_strict.yaxis.set_label_coords(-0.35, 0.5)

        # Right subplot: KS-TW
        ax_tw = axes[idx * 2 + 1]
        sns.heatmap(d_tw_gs, cmap=GRAY_CMAP, vmin=0, vmax=1,
                    cbar=False, annot=False,
                    xticklabels=M, yticklabels=False, ax=ax_tw)
        ax_tw.set_title(f"{title}\nKS-TW\n($\\alpha$={alpha})", pad=2)
        ax_tw.set_xticklabels(ax_tw.get_xticklabels(), rotation=45, ha="right")
        ax_tw.set_ylabel("")

    plt.subplots_adjust(hspace=0.8, wspace=0.4)
    plt.savefig(savepath, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {savepath}")


# =========================
# Example mock data
# =========================
# ALBERT: 1 shared layer group, 6 matrix types -> decisions shape (1, 6)
# Replace with real KS test outcomes when running the full pipeline.
np.random.seed(42)
decisions_strict = np.random.choice([0, 1], size=(1, 6), p=[0.4, 0.6])
decisions_tw     = np.random.choice([0, 1], size=(1, 6), p=[0.3, 0.7])

plot_ks_mosaic_separate(decisions_strict, decisions_tw,
                        savepath="graph_level_views_01_albert.pdf")

"""# Graph 7 - Per-matrix-type acceptance rates under KS-strict and KS-TW criteria"""

# =============================================================================
# GRAPH 7: Per-matrix-type acceptance rates with Wilson confidence intervals
# =============================================================================
# PURPOSE:
#   Show how KS acceptance rates vary across the 6 ALBERT matrix types for
#   each alpha level. ALBERT-SPECIFIC: x-axis is matrix type (not layer number).
#   ALL 6 matrix type labels shown on ALL rows (only 6 types, no clutter).
#
# KEY DIFFERENCE FROM BERT GRAPH 7:
#   BERT: x = layers 0-11 (or 0-23), only labels 0 and last shown
#   ALBERT: x = [Q, K, V, Att-Dense, FFN, FFN-Out], all labels shown
#
# SCENARIO SPLIT at index 4 (not 6 like BERT):
#   scenario_square_vs_rect: mat_types indices 0-3 = square (Q/K/V/Att-Dense, beta=1.0)
#                            mat_types indices 4-5 = rect (FFN/FFN-Out, beta=0.25)
#
# LAYOUT: 6 rows (scenarios) x 3 columns (alpha values), figsize=(7, 9)
# OUTPUT: graph_level_views_02_albert.pdf
# =============================================================================

# --- Graph #7: Per-layer acceptance rates (lines with CIs) ---
# Model: ALBERT-base-v2
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

# =========================
# GLOBAL STYLE (Paper)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Grayscale line colors for each method
METHOD_COLORS = ["#444444", "#aaaaaa"]   # KS-strict = dark, KS-TW = light

# =========================
# Config -- ALBERT: 1 group, 6 matrix types
# ALBERT has no layers -- x-axis is matrix type instead.
# =========================
mat_types = ["Q", "K", "V", "Att-Dense", "FFN", "FFN-Out"]
alphas    = [0.01, 0.05, 0.10]
methods   = ["KS-strict", "KS-TW"]
n_boot    = 50     # bootstrap replications for CI estimation
rng       = np.random.default_rng(42)

# =========================
# Scenario definitions
# The "layer" argument here indexes into mat_types (0-5), not actual layers.
# =========================
def scenario_baseline(layer, method, alpha):
    """Both methods accept at constant moderate rates across all matrix types."""
    return 0.55 if method == "KS-strict" else 0.65

def scenario_strict_dominance(layer, method, alpha):
    """KS-TW always accepts more than KS-strict."""
    return 0.75 if method == "KS-strict" else 0.90

def scenario_low_alpha(layer, method, alpha):
    """Acceptance rate increases sharply with alpha."""
    if alpha == 0.01:
        return 0.15 if method == "KS-strict" else 0.35
    elif alpha == 0.05:
        return 0.45 if method == "KS-strict" else 0.55
    else:
        return 0.75 if method == "KS-strict" else 0.80

def scenario_square_vs_rect(layer, method, alpha):
    """
    Square matrices (Q/K/V/Att-Dense, indices 0-3, beta=1.0) accept more.
    Rectangular matrices (FFN/FFN-Out, indices 4-5, beta=0.25) accept less.
    Note: split at index 4 (not 6 as in BERT-base).
    """
    if layer < 4:
        return 0.80 if method == "KS-strict" else 0.90
    else:
        return 0.30 if method == "KS-strict" else 0.50

def scenario_edge_sensitive(layer, method, alpha):
    """KS-strict alternates by matrix type index; KS-TW is stable."""
    if method == "KS-strict":
        return 0.40 + 0.20 * (layer % 2)
    else:
        return 0.70

def scenario_extreme_disagreement(layer, method, alpha):
    """Maximal disagreement between KS-strict and KS-TW."""
    return 0.12 if method == "KS-strict" else 0.87

SCENARIOS = {
    "Baseline":              scenario_baseline,
    "Strict dominance":      scenario_strict_dominance,
    "Low $\\alpha$ effect":  scenario_low_alpha,
    "Square vs rect":        scenario_square_vs_rect,
    "Edge-sensitive":        scenario_edge_sensitive,
    "Extreme disagreement":  scenario_extreme_disagreement,
}

# =========================
# Plot grid
# =========================
fig, axes = plt.subplots(len(SCENARIOS), len(alphas),
                         figsize=(7, 9), sharey=True)
axes = np.array(axes)

x        = np.arange(len(mat_types))
last_row = len(SCENARIOS) - 1

for row, (name, func) in enumerate(SCENARIOS.items()):
    # Simulate binary decisions for each (matrix_type, method, alpha, bootstrap replicate)
    decisions = np.zeros((len(mat_types), len(methods), len(alphas), n_boot))
    for i, mat in enumerate(mat_types):
        for j, method in enumerate(methods):
            for k, alpha in enumerate(alphas):
                p = func(i, methods[j], alphas[k])
                decisions[i,j,k,:] = rng.choice([0,1], size=n_boot, p=[1-p, p])

    accept_means = np.mean(decisions, axis=-1)
    # Wilson 95% confidence intervals for acceptance rate
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
                    color=METHOD_COLORS[j],
                    label=method if (row == last_row and k == 0) else None)
            ax.fill_between(x, ci_low[:,j,k], ci_high[:,j,k],
                            alpha=0.2, color=METHOD_COLORS[j])

        # Title only on first row
        if row == 0:
            ax.set_title(f"$\\alpha$ = {alpha}")

        # X label only on last row
        if row == last_row:
            ax.set_xlabel("Matrix type")

        # All 6 matrix type labels on all rows (6 types fit without clutter)
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

# Single legend on bottom-left axes only
axes[last_row, 0].legend(title="Method", frameon=False)

plt.subplots_adjust(hspace=0.5, wspace=0.4)

fig.savefig("graph_level_views_02_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Figure saved: graph_level_views_02_albert.pdf")

"""# Graph 8 - Aspect ratio beta versus KS statistic (Dp) across simulation scenarios"""

# =============================================================================
# GRAPH 8: Aspect ratio beta vs KS statistic D scatter plots (ALBERT)
# =============================================================================
# PURPOSE:
#   Investigate the relationship between beta and KS statistic D.
#
# ALBERT-SPECIFIC BETA DISTRIBUTION (DISCRETE, not continuous like BERT):
#   "Attention" family (Q/K/V/Att-Dense): beta drawn from U[0.95, 1.0]
#     -> 4 out of 6 matrices; 67% probability in data generation
#   "FFN" family (FFN/FFN-Out):           beta drawn from U[0.20, 0.30]
#     -> 2 out of 6 matrices; 33% probability in data generation
#
# REFERENCE LINES: vertical dotted lines mark the two real ALBERT beta values
#   beta=1.0 (light gray, ":") and beta=0.25 (medium gray, ":")
#
# LAYOUT: 2x3 mosaic of scatter plots, shared axes
# OUTPUT: graph_level_views_03_albert.pdf
# =============================================================================

# --- Graph #8: beta vs KS statistic scatter ---
# Model: ALBERT-base-v2

import numpy as np
import matplotlib.pyplot as plt

# =========================
# GLOBAL STYLE (Paper)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Grayscale colors for each family
FAMILY_COLORS = {"FFN": "#888888", "Attention": "#111111"}

# =========================
# Scenario generator
# ALBERT-specific beta ranges (DISCRETE bimodal clusters):
#   Square  (Q/K/V/Att-Dense): beta in [0.95, 1.0]  (67% probability)
#   Rect    (FFN/FFN-Out):      beta in [0.20, 0.30] (33% probability)
# =========================
def generate_data_scenario(scenario, n_points=50):
    """
    Generate synthetic (beta, KS-D, family) data for a given scenario.

    ALBERT-SPECIFIC: beta values cluster in two discrete groups, unlike
    BERT-base which has a continuous range [0.3, 1.0].
      Attention family: beta ~ U[0.95, 1.0]  (square matrices: Q, K, V, Att-Dense)
      FFN family:       beta ~ U[0.20, 0.30] (rectangular: FFN, FFN-Out)
    The 67/33 split reflects ALBERT's 4 square vs 2 rectangular matrices.

    Parameters
    ----------
    scenario : str  -- one of: "baseline","strict","edge","lowalpha","smooth","mixed"
    n_points : int  -- number of synthetic data points

    Returns
    -------
    betas, ks_vals, fams : np.ndarray triple
    """
    np.random.seed(42)
    betas, ks_vals, fams = [], [], []

    if scenario == "baseline":
        for _ in range(n_points):
            if np.random.rand() < 0.67:    # 4/6 matrices are square Attention
                beta = np.random.uniform(0.95, 1.0); fam = "Attention"
                ks   = np.random.uniform(0.05, 0.15)
            else:                          # 2/6 matrices are rectangular FFN
                beta = np.random.uniform(0.20, 0.30); fam = "FFN"
                ks   = np.random.uniform(0.08, 0.20)
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "strict":
        for _ in range(n_points):
            if np.random.rand() < 0.67:
                beta = np.random.uniform(0.95, 1.0); fam = "Attention"
                ks   = np.random.uniform(0.05, 0.10)
            else:
                beta = np.random.uniform(0.20, 0.30); fam = "FFN"
                ks   = np.random.uniform(0.30, 0.45)
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "edge":
        for _ in range(n_points):
            if np.random.rand() < 0.67:
                beta = np.random.uniform(0.95, 1.0); fam = "Attention"
                ks   = np.random.uniform(0.05, 0.15)
            else:
                beta = np.random.uniform(0.20, 0.30); fam = "FFN"
                ks   = np.random.uniform(0.20, 0.60)
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "lowalpha":
        for _ in range(n_points):
            if np.random.rand() < 0.67:
                beta = np.random.uniform(0.95, 1.0); fam = "Attention"
                ks   = np.random.uniform(0.15, 0.25)
            else:
                beta = np.random.uniform(0.20, 0.30); fam = "FFN"
                ks   = np.random.uniform(0.30, 0.50)
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "smooth":
        for _ in range(n_points):
            if np.random.rand() < 0.67:
                beta = np.random.uniform(0.95, 1.0); fam = "Attention"
                ks   = np.random.normal(0.08, 0.005)
            else:
                beta = np.random.uniform(0.20, 0.30); fam = "FFN"
                ks   = np.random.normal(0.18, 0.005)
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    elif scenario == "mixed":
        for i in range(n_points):
            if i < n_points // 2:
                beta = np.random.uniform(0.95, 1.0)
                ks   = np.random.normal(0.25, 0.02); fam = "Attention"
            else:
                beta = np.random.uniform(0.20, 0.30)
                ks   = np.random.normal(0.10, 0.02); fam = "FFN"
            betas.append(beta); ks_vals.append(ks); fams.append(fam)

    return np.array(betas), np.array(ks_vals), np.array(fams)

# =========================
# Plotting helper
# =========================
def plot_scenario(ax, betas, ks_vals, fams, title, i, is_first):
    """
    Render one beta vs KS-D scatter subplot with ALBERT reference lines.

    ALBERT-SPECIFIC: adds two vertical dotted reference lines marking the
    real beta values: beta=1.0 (Attention, light gray) and beta=0.25 (FFN, medium gray).
    These guide the eye to the two discrete beta clusters expected in ALBERT.

    Parameters
    ----------
    ax       : matplotlib Axes
    betas    : np.ndarray -- aspect ratios
    ks_vals  : np.ndarray -- KS statistic values
    fams     : np.ndarray -- family labels ("FFN" or "Attention")
    title    : str        -- subplot title
    i        : int        -- subplot index for y-label placement
    is_first : bool       -- if True, register legend labels
    """
    for fam in np.unique(fams):
        idx = fams == fam
        ax.scatter(betas[idx], ks_vals[idx],
                   label=fam if is_first else None,
                   alpha=0.6, s=8, color=FAMILY_COLORS[fam],
                   linewidths=0)

    # Vertical reference lines at real ALBERT beta values
    ax.axvline(1.0,  color="#aaaaaa", ls=":", linewidth=0.6, alpha=0.5)  # beta=1.0 (Attention)
    ax.axvline(0.25, color="#888888", ls=":", linewidth=0.6, alpha=0.5)  # beta=0.25 (FFN)
    # KS threshold reference lines
    ax.axhline(0.1, color="#aaaaaa", ls="--", linewidth=0.8)
    ax.axhline(0.2, color="#aaaaaa", ls="--", linewidth=0.8)

    ax.set_title(title, pad=2)
    ax.grid(True)

    # Y label only on left column (i % 3 == 0 for 2x3 grid)
    if i % 3 == 0:
        ax.set_ylabel(r"$\beta$ vs. $D_{p}$")
        ax.yaxis.set_label_coords(-0.25, 0.5)
    else:
        ax.set_ylabel("")

# =========================
# Plot grid
# =========================
scenarios = {
    "baseline": "Baseline balanced",
    "strict":   "Strict dominance",
    "edge":     "Edge-sensitive",
    "lowalpha": "Low $\\alpha$ test",
    "smooth":   "Smooth trimming",
    "mixed":    "Mixed block structure",
}

fig, axs = plt.subplots(2, 3, figsize=(7, 3), sharex=True, sharey=True)

for i, (ax, (key, title)) in enumerate(zip(axs.ravel(), scenarios.items())):
    betas, ks_vals, fams = generate_data_scenario(key, n_points=60)
    plot_scenario(ax, betas, ks_vals, fams, title, i, is_first=(i == 0))

# Single legend on first axes only
axs.ravel()[0].legend(frameon=False)

plt.subplots_adjust(hspace=0.5, wspace=0.2)

plt.savefig("graph_level_views_03_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Figure saved: graph_level_views_03_albert.pdf")

"""# Graph 9 - Bootstrap p-value distributions across calibration scenarios"""

# =============================================================================
# GRAPH 9: Bootstrap p-value distributions across calibration scenarios
# =============================================================================
# PURPOSE:
#   Sanity-check the calibration of KS-strict and KS-TW p-values.
#   Under the null hypothesis, p-values should be uniformly distributed U[0,1].
#   This graph is model-agnostic (same synthetic scenarios as BERT).
#
# SCENARIOS: null calibration, anti-conservative, conservative TW,
#   both anti-conservative, skewed strict, and mixed calibration.
#
# LAYOUT: 2x3 mosaic, line plots with fill
# OUTPUT: graph_shrinkage_control_01_albert.pdf
# =============================================================================

# --- Graph #9: Bootstrap p-value distributions (2x3 mosaic, reproducible) ---
# Model: ALBERT-base-v2

import numpy as np
import matplotlib.pyplot as plt

# =========================
# GLOBAL STYLE (Paper)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Grayscale colors for each method
METHOD_COLORS = {"KS-strict": "#444444", "KS-TW": "#aaaaaa"}

# =========================
# Config
# =========================
n_boot = 1000   # bootstrap replications per scenario

def simulate_pvalues(null=True, skew=0.0, conservative=False, seed=None):
    """
    Simulate n_boot synthetic p-values with controlled calibration properties.

    Parameters
    ----------
    null         : bool  -- True = U[0,1] (null); False = Beta(0.7,1) (anti-conservative)
    skew         : float -- if >0, apply p^(1+skew) to skew distribution toward 0
    conservative : bool  -- if True, apply 1-(1-p)^2 to push p-values toward 1
    seed         : int   -- random seed for reproducibility

    Returns
    -------
    np.ndarray -- n_boot p-values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    if null:
        pvals = rng.random(n_boot)
    else:
        pvals = rng.beta(0.7, 1.0, size=n_boot)
    if skew > 0:
        pvals = np.power(pvals, 1 + skew)
    if conservative:
        pvals = 1 - np.power(1 - pvals, 2)
    return np.clip(pvals, 0, 1)

# =========================
# Scenarios
# =========================
# Pre-generate all p-value arrays with fixed seeds for reproducibility
SCENARIOS = {
    "Null calibration":   (simulate_pvalues(null=True,  seed=1),
                              simulate_pvalues(null=True,  seed=2)),
    "Anti-cons. strict":  (simulate_pvalues(null=False, seed=3),
                              simulate_pvalues(null=True,  seed=4)),
    "Conservative TW":    (simulate_pvalues(null=True,  seed=5),
                              simulate_pvalues(null=True,  conservative=True, seed=6)),
    "Both anti-cons.":    (simulate_pvalues(null=False, seed=7),
                              simulate_pvalues(null=False, seed=8)),
    "Skewed strict":      (simulate_pvalues(null=True,  skew=0.5, seed=9),
                              simulate_pvalues(null=True,  seed=10)),
    "Mixed calibration":  (simulate_pvalues(null=True,  conservative=True, seed=11),
                              simulate_pvalues(null=False, seed=12)),
}

# =========================
# Plot grid
# =========================
fig, axs = plt.subplots(2, 3, figsize=(7, 3), sharex=True, sharey=True)
axs = axs.ravel()

for i, (title, (pvals_strict, pvals_tw)) in enumerate(SCENARIOS.items()):
    ax      = axs[i]
    bins    = np.linspace(0, 1, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for pvals, label in [
        (pvals_strict, "KS-strict"),
        (pvals_tw,     "KS-TW"),
    ]:
        color = METHOD_COLORS[label]
        counts, _ = np.histogram(pvals, bins=bins)
        ax.plot(centers, counts, marker="o", ms=2, linewidth=0.8,
                color=color, label=label if i == 0 else None)
        ax.fill_between(centers, counts, alpha=0.15, color=color)

    ax.set_title(title, pad=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_boot // 5)
    ax.grid(True)

    # Y label only on left column (i % 3 == 0 for 2x3 grid)
    if i % 3 == 0:
        ax.set_ylabel("p-value\nvs. Frequency")
        ax.yaxis.set_label_coords(-0.25, 0.5)
    else:
        ax.set_ylabel("")

# Single legend on first axes only
axs[0].legend(frameon=False)

plt.subplots_adjust(hspace=0.5, wspace=0.4)

plt.savefig("graph_shrinkage_control_01_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Figure saved: graph_shrinkage_control_01_albert.pdf")

"""# Graph 10 - Sensitivity of KS-TWd outcomes to edge relaxation parameter cα"""

# =============================================================================
# GRAPH 10: Sensitivity of KS-TW outcomes to edge relaxation c_alpha (ALBERT)
# =============================================================================
# PURPOSE:
#   Show how D_trim changes as c_alpha increases from 1 to 3.
#
# ALBERT-SPECIFIC: 6 labeled matrix-type lines (not 12 unlabeled layer lines).
#   Each mat type gets a distinct gray shade from GRAY_SHADES.
#   scenario_mixed splits at index 4 (square Q/K/V/Att-Dense vs rect FFN/FFN-Out).
#
# This graph saves to graph_shrinkage_control_02_bert_large.pdf
# (filename from the original source code -- may be a naming inconsistency).
#
# LAYOUT: 2x3 mosaic, shared axes, legend with 6 matrix type labels
# OUTPUT: graph_shrinkage_control_02_bert_large.pdf
# =============================================================================

# --- Graph #10 (2x3 Mosaic): Sensitivity of KS-TW to edge relaxation ---
# Model: ALBERT-base-v2 (note: savepath uses bert_large suffix in original)

import numpy as np
import matplotlib.pyplot as plt

# =========================
# GLOBAL STYLE (Paper)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# 6 distinct gray shades for 6 ALBERT matrix types
# Darker = Q (first), lighter = FFN-Out (last)
GRAY_SHADES = ["#111111", "#333333", "#555555", "#777777", "#999999", "#bbbbbb"]

# =========================
# Config -- ALBERT: 6 matrix types (NOT 12 layers as comment says)
# =========================
np.random.seed(42)
mat_types   = ["Q", "K", "V", "Att-Dense", "FFN", "FFN-Out"]
c_TW_values = [1, 2, 3]    # edge relaxation coefficients

# =========================
# Scenario functions
# =========================
def scenario_stable_accept(layer, c):
    """D_trim low and stable -- good MP fit at any c_alpha."""
    return np.random.uniform(0.05, 0.08)

def scenario_relax_accept(layer, c):
    """D_trim decreases as c_alpha grows."""
    return 0.20 / c + np.random.uniform(-0.02, 0.02)

def scenario_persistent_reject(layer, c):
    """D_trim stays high regardless of c_alpha -- structural misfit."""
    return np.random.uniform(0.18, 0.25)

def scenario_mixed(layer, c):
    """
    Square matrices (Q/K/V/Att-Dense, indices 0-3, beta=1.0) benefit from c_alpha.
    Rectangular matrices (FFN/FFN-Out, indices 4-5, beta=0.25) are insensitive.
    Note: split at index 4, not 6 like BERT-base.
    """
    if layer < 4:    # Q, K, V, Att-Dense (square, beta=1.0)
        return 0.20 / c
    else:            # FFN, FFN-Out (rectangular, beta=0.25)
        return np.random.uniform(0.08, 0.10)

def scenario_edgesensitive(layer, c):
    """D_trim oscillates sinusoidally with c_alpha."""
    return 0.12 + 0.05 * np.sin(0.5 * np.pi * c) \
                + np.random.uniform(-0.01, 0.01)

def scenario_alpha_dependent(layer, c):
    """D_trim inversely proportional to c_alpha."""
    return 0.20 / c + np.random.uniform(-0.02, 0.02)

SCENARIOS = [
    ("Stable acceptance",         scenario_stable_accept),
    ("Strict->Relaxed acceptance", scenario_relax_accept),
    ("Persistent rejection",      scenario_persistent_reject),
    ("Mixed families",            scenario_mixed),
    ("Edge-sensitive",            scenario_edgesensitive),
    ("$\\alpha$-dependent",       scenario_alpha_dependent),
]

thresholds = [0.10, 0.12, 0.15]   # KS rejection threshold reference lines

# =========================
# Plot grid
# =========================
fig, axs = plt.subplots(2, 3, figsize=(7, 3), sharex=True, sharey=True)

for i, (ax, (title, func)) in enumerate(zip(axs.ravel(), SCENARIOS)):
    D_trim = np.zeros((len(mat_types), len(c_TW_values)))
    for j, c in enumerate(c_TW_values):
        for m_idx in range(len(mat_types)):
            D_trim[m_idx, j] = func(m_idx, c)

    # 6 labeled lines -- one per matrix type, distinct gray shades
    for m_idx, mat in enumerate(mat_types):
        ax.plot(c_TW_values, D_trim[m_idx, :],
                marker="o", ms=2, alpha=0.5, linewidth=0.8,
                color=GRAY_SHADES[m_idx],
                label=mat if i == 0 else None)

    if title == "$\\alpha$-dependent":
        for th in thresholds:
            ax.axhline(th, linestyle="--", color="#444444", linewidth=0.8, alpha=0.6)
    else:
        ax.axhline(0.12, linestyle="--", color="#444444", linewidth=0.8, alpha=0.6)

    ax.set_title(title, pad=2)
    ax.set_xticks(c_TW_values)
    ax.grid(True)

    # Y label on all left column subplots (i % 3 == 0)
    if i % 3 == 0:
        ax.set_ylabel(r"$c_{\alpha}$ vs. $D_{p}$")
        ax.yaxis.set_label_coords(-0.25, 0.5)
    else:
        ax.set_ylabel("")

# Single legend on first axes with matrix type labels
axs.ravel()[0].legend(fontsize=6, frameon=False, loc="upper right")

plt.subplots_adjust(hspace=0.5, wspace=0.4)

plt.savefig("graph_shrinkage_control_02_bert_large.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Figure saved: graph_shrinkage_control_02_bert_large.pdf")

"""# Graph 11 - Type-I calibration curves on synthetic MPd-null matrices."""

# =============================================================================
# GRAPH 11: Type-I calibration curves on synthetic MP-null matrices (ALBERT)
# =============================================================================
# PURPOSE:
#   Verify that the empirical rejection rate under the null tracks nominal alpha.
#
# ALBERT-SPECIFIC MATRIX SHAPES:
#   768x768   -- Q/K/V/Att-Dense (beta = 1.0,  square)
#   3072x768  -- FFN              (beta = 0.25, rectangular)
#   768x3072  -- FFN-Out          (beta = 0.25, rectangular, transposed of above)
#   768x128   -- Embedding hidden mapping (beta ~0.167, encoder.embedding_hidden_mapping_in)
#
# The 768x128 shape is ALBERT-unique: it represents the factorized embedding
# projection from 128-dim to 768-dim, absent in standard BERT.
#
# LAYOUT: 2x2 grid
# OUTPUT: graph_shrinkage_control_03_albert.pdf
# =============================================================================

# --- Graph #11: Type-I calibration curves (synthetic MP) ---
# Model: ALBERT-base-v2

import numpy as np
import matplotlib.pyplot as plt

# =========================
# GLOBAL STYLE (Paper)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Grayscale ramp: darker = smaller c_alpha (tighter trim)
C_TW_COLORS = {1: "#333333", 2: "#777777", 3: "#bbbbbb"}

# =========================
# Config
# =========================
np.random.seed(42)
n_boot      = 200
alphas      = [0.01, 0.05, 0.10]
c_TW_values = [1, 2, 3]

# ALBERT real matrix shapes for calibration simulation.
# (768, 128) is unique to ALBERT: encoder.embedding_hidden_mapping_in,
# the factorized embedding projection absent in BERT-base.
shapes = [
    (768,  768),   # Q / K / V / Att-Dense (beta = 1.0, square)
    (3072, 768),   # FFN                   (beta = 0.25)
    (768,  3072),  # FFN-Out               (beta = 0.25)
    (768,  128),   # Embedding hidden mapping (beta ~0.167, ALBERT-unique)
]

# =========================
# Mock KS-TW rejection test
# =========================
def ks_tw_test_null(m, n, alpha, c_tw):
    """
    Simulate one null KS-TW rejection with a small calibration bias.

    Bias models finite-sample overcorrection:
      larger c_tw -> smaller bias (more trimming -> better calibration)
      square matrices (m==n) -> slightly more bias
    """
    bias = 0.01 * (1.0 / c_tw) + 0.005 * (m == n)
    return np.random.rand() < (alpha + bias)

# =========================
# Simulate rejection rates
# =========================
results = {}
for (m, n) in shapes:
    shape_name = f"{m}x{n}"
    results[shape_name] = {}
    for c in c_TW_values:
        rejections = []
        for alpha in alphas:
            count = sum(ks_tw_test_null(m, n, alpha, c) for _ in range(n_boot))
            rejections.append(count / n_boot)
        results[shape_name][c] = rejections

# =========================
# Plot grid
# =========================
fig, axs = plt.subplots(2, 2, figsize=(3.5, 4), sharey=True)
axs = axs.flatten()

for i, (ax, (shape_name, data)) in enumerate(zip(axs, results.items())):
    for c in c_TW_values:
        ax.plot(alphas, data[c], marker="o", ms=2, linewidth=0.8,
                color=C_TW_COLORS[c],
                label=f"$c_{{\\alpha}}$={c}" if i == 0 else None)

    # 45-degree reference line = perfect calibration (empirical = nominal)
    ax.plot(alphas, alphas, color="black", linestyle="--",
            linewidth=0.8, label="Nom. $\\alpha$" if i == 0 else None)

    ax.set_title(shape_name, pad=2)
    ax.set_ylim(0, 0.2)
    ax.grid(True)

    # Y label on all left column subplots
    if i % 2 == 0:
        ax.set_ylabel(r"$\alpha$ vs. Empirical rej. rate")
        ax.yaxis.set_label_coords(-0.35, 0.5)
    else:
        ax.set_ylabel("")

# Single legend on first axes only
axs[0].legend(frameon=False)

plt.subplots_adjust(hspace=0.5, wspace=0.4)

plt.savefig("graph_shrinkage_control_03_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Figure saved: graph_shrinkage_control_03_albert.pdf")

"""# Graph 12 - Bootstrap reference envelopes for empirical CDFs under different distributional scenarios."""

# =============================================================================
# GRAPH 12: Bootstrap reference envelopes for empirical CDFs (ALBERT)
# =============================================================================
# PURPOSE:
#   Plot the observed eCDF against a 90% bootstrap confidence band under a
#   reference null distribution. This graph is model-agnostic (same synthetic
#   scenarios as BERT) and demonstrates distributional robustness checks.
#
# SCENARIOS: chi^2, shifted normal, heavy-tailed t_3, compressed variance,
#   mixture (two-component Gaussian), and small n=50 baseline.
#
# LAYOUT: 2x3 mosaic, sharey=True
# OUTPUT: graph_shrinkage_control_04_albert.pdf
# =============================================================================

# --- Graph #12: eCDF vs bootstrap bands (2x3 Mosaic) ---
# Model: ALBERT-base-v2

import numpy as np
import matplotlib.pyplot as plt

# =========================
# GLOBAL STYLE (Paper)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

np.random.seed(42)

# =========================
# Helpers
# =========================
def ecdf(x, grid):
    """Evaluate the empirical CDF of x at grid points."""
    return np.searchsorted(np.sort(x), grid, side="right") / len(x)

def generate_data(kind, n):
    """
    Generate n random samples from the specified distribution.

    Parameters
    ----------
    kind : str -- "normal"|"shifted"|"t3"|"compressed"|"mixture"|other (chi^2)
    n    : int -- sample size
    """
    if kind == "normal":
        return np.random.normal(size=n)
    elif kind == "shifted":
        return np.random.normal(loc=1.0, scale=1.0, size=n)
    elif kind == "t3":
        return np.random.standard_t(df=3, size=n)
    elif kind == "compressed":
        return np.random.normal(loc=0.0, scale=0.5, size=n)
    elif kind == "mixture":
        return np.concatenate([
            np.random.normal(loc=0.0, scale=1.0, size=n//2),
            np.random.normal(loc=2.0, scale=1.0, size=n//2)
        ])
    else:  # chisquare
        return np.random.chisquare(df=3, size=n) / 3.0

def plot_ecdf_with_bands(ax, observed, boot_dist, n, B, L, U, alpha_band, title, i):
    """
    Plot the observed eCDF with a bootstrap confidence band.

    Parameters
    ----------
    ax         : matplotlib Axes
    observed   : np.ndarray -- observed data sample
    boot_dist  : str        -- reference null distribution name
    n          : int        -- bootstrap sample size
    B          : int        -- number of bootstrap replicates
    L, U       : float      -- x-axis limits
    alpha_band : tuple      -- (lower_pct, upper_pct) for confidence band
    title      : str        -- subplot title
    i          : int        -- subplot index for y-label placement
    """
    grid     = np.linspace(L, U, 300)
    obs_ecdf = ecdf(observed, grid)

    # Generate B bootstrap eCDFs under the null distribution
    boot_ecdfs = np.array([ecdf(generate_data(boot_dist, n), grid)
                           for _ in range(B)])
    lower = np.percentile(boot_ecdfs, alpha_band[0], axis=0)
    upper = np.percentile(boot_ecdfs, alpha_band[1], axis=0)

    # Grayscale: band = light gray, observed eCDF = dark gray
    ax.fill_between(grid, lower, upper, color="#cccccc", alpha=0.6)
    ax.plot(grid, obs_ecdf, color="#333333", linewidth=0.8)

    ax.set_xlim(L, U)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=2)
    ax.grid(True)

    # Y label only on left column (i % 3 == 0 for 2x3 grid)
    if i % 3 == 0:
        ax.set_ylabel("$x$ vs. eCDF")
        ax.yaxis.set_label_coords(-0.25, 0.5)
    else:
        ax.set_ylabel("")

# =========================
# Scenarios
# =========================
# Each tuple: (obs_kind, boot_kind, n, title, (L, U))
scenarios = [
    ("chisq",      "chisq",  500, "Baseline ($\\chi^2$ null)", (0,  4)),
    ("shifted",    "normal", 500, "Shifted mean",              (-2, 4)),
    ("t3",         "normal", 500, "Heavy-tailed ($t_3$)",      (-4, 4)),
    ("compressed", "normal", 500, "Compressed variance",       (-2, 2)),
    ("mixture",    "normal", 500, "Mixture (0 & 2)",           (-1, 4)),
    ("chisq",      "chisq",   50, "Small $n=50$",              (0,  4)),
]

# =========================
# Plot grid
# =========================
fig, axs = plt.subplots(2, 3, figsize=(7, 3), sharex=False, sharey=True)

for i, (ax, (obs_kind, boot_kind, n, title, (L, U))) in enumerate(
        zip(axs.flat, scenarios)):
    observed = generate_data(obs_kind, n)
    plot_ecdf_with_bands(ax, observed, boot_kind,
                         n=n, B=300, L=L, U=U,
                         alpha_band=(5, 95), title=title, i=i)

plt.subplots_adjust(hspace=0.5, wspace=0.2)

plt.savefig("graph_shrinkage_control_04_albert.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Figure saved: graph_shrinkage_control_04_albert.pdf")
