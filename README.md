# Decoding Transformers Spectra

> **Decoding Transformers Spectra: A Random Matrix Theory Framework Beyond the Marchenko–Pastur Law**

## Overview

The rapid expansion of large language models (LLMs) has intensified demand for principled methodologies capable of decoding their internal structure and guiding efficient deployment. Although Transformers achieve state-of-the-art performance, the large linear operators composing their architectures — attention projections, feed-forward layers, and embeddings — are represented by weight matrices that often contain substantial redundancy and noise.

This repository implements a **Random Matrix Theory (RMT) framework** that systematically analyzes the spectral behavior of Transformer weight matrices beyond the classical Marchenko–Pastur (MP) law. The framework integrates MP baselines, bootstrap calibration, and shrinkage transformations to disentangle noise from structured signal in high-dimensional spectra.

---

## Research Objectives

- Characterize the **bulk–plus–spike organization**, edge fluctuations, and finite-sample deviations observed in Transformer spectra
- Establish a rigorous methodology to guide **spectral denoising, shrinkage, and compression** strategies
- Build a taxonomy of **layer-specific spectral behavior** linking empirical spectra to theoretical distributions

---

## Key Findings

- **Feed-forward layers** conform more closely to Marchenko–Pastur predictions
- **Attention and embedding layers** display pronounced edge deviations consistent with Tracy–Widom statistics
- Layer-specific spectral behavior highlights **distinctive structural roles** across Transformer components
- RMT-based spectral decoding emerges as both a rigorous and practical tool for analyzing modern deep learning models, with implications for **robustness, generalization, and compressibility**

---

## Models Analyzed

| Model | Layers | Hidden Size | Parameters |
|-------|--------|-------------|------------|
| BERT-base-uncased | 12 | 768 | ~110M |
| ALBERT-base-v2 | 1 (shared across 12) | 768 | ~12M |
| BERT-Large-uncased | 24 | 1024 | ~340M |

> **Note on ALBERT:** ALBERT uses cross-layer parameter sharing, meaning all 12 Transformer blocks share a single set of weight matrices stored in one layer group (`encoder.albert_layer_groups.0.albert_layers.0.*`). As a result, only one copy of each matrix type exists (Q, K, V, Att-Dense, FFN, FFN-Out), and layer-wise comparisons are not applicable. Graph axes that reference layers are replaced by matrix-type axes in the ALBERT scripts.

---

## Repository Structure

```
.
├── paper_bert_decoding_transformers_spectra_full_graphs.py
├── paper_albert_decoding_transformers_spectra_full_graphs.py
├── paper_bert_large_decoding_transformers_spectra_full_graphs.py
├── PAPER_BERT_Decoding_Transformers_Spectra_full_graphs.ipynb
├── PAPER_ALBERT_Decoding_Transformers_Spectra_full_graphs.ipynb
├── PAPER_BERT_Large_Decoding_Transformers_Spectra_full_graphs.ipynb
├── PAPER_BERT_Large_Decoding_Transformers_Spectra_full_graphs_.ipynb
└── README.md
```

Each `.py` script follows the same **two-section pipeline** followed by twelve graph sections. The `.ipynb` notebooks are the cleaned Colab originals (widget metadata and widget-only outputs removed; all text, stream, and image outputs preserved).

> **Structural note (BERT-base):** In `paper_bert_decoding_transformers_spectra_full_graphs.py`, Graphs 1–3 share a single set of helper functions and a single `SETTINGS` list defined once in the Graph 1 block. The ALBERT and BERT-Large scripts redefine helpers locally in each graph section for full self-containment.

| Section | Description |
|---------|-------------|
| Section 1 | Extract raw weight matrices from HuggingFace and save as `.npy` with `manifest.json` |
| Section 2 | Build column-standardized W_MP matrices and save with round-trip verification |
| Graph 1 | ePDF vs. conditional MP PDF — per-panel trim interval [L, U] |
| Graph 2 | eCDF vs. conditional MP CDF — per-panel trim interval [L, U] |
| Graph 3 | ePDF vs. conditional MP PDF — shared global x-axis across all panels |
| Graph 4 | Residual CDF (eCDF − MP CDF) with KS statistic bands |
| Graph 5 | QQ plots of empirical spectra vs. conditional MP quantiles |
| Graph 6 | KS test decision heatmaps (layers × matrix types, KS-strict and KS-TW) |
| Graph 7 | Per-layer acceptance rates with Wilson 95 % confidence intervals |
| Graph 8 | Aspect ratio β vs. KS statistic D_p scatter |
| Graph 9 | Bootstrap p-value distributions across calibration scenarios |
| Graph 10 | KS–TW edge relaxation sensitivity to c_α |
| Graph 11 | Type-I calibration curves on synthetic MP-null matrices |
| Graph 12 | eCDF bootstrap reference envelopes under distributional scenarios |

---

## Methods

### WMP Normalization

Each weight matrix $W$ of shape $(m, n)$ is column-standardized before spectral analysis:

$$W_{\text{MP}}[i,j] = \frac{W[i,j] - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are the column-wise mean and standard deviation. This ensures the i.i.d. zero-mean unit-variance assumption required by the MP law. Zero-variance columns are left unchanged (divided by 1.0). The original matrices can be recovered exactly from $W_{\text{MP}}$, $\mu$, and $\sigma$; round-trip reconstruction errors are verified at the end of Section 2.

### Marchenko–Pastur Law

For a random matrix $W$ with i.i.d. $\mathcal{N}(0,1)$ entries and aspect ratio $\beta = \min(m,n)/\max(m,n)$, the empirical spectral distribution of eigenvalues $\lambda$ of $W^\top W / \max(m,n)$ converges to:

$$f_\beta(\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi\beta\lambda}, \quad \lambda \in [\lambda_-, \lambda_+]$$

with support boundaries $\lambda_\pm = (1 \pm \sqrt{\beta})^2$.

The CDF is computed numerically on a fine quadrature grid using the substitution $g(t) = \lambda_- + (\lambda_+ - \lambda_-)\,t^2$, which concentrates points near the lower edge where the PDF has an integrable singularity.

### Aspect Ratio β

All scripts use an explicit `beta_override` rather than the natural $\beta = \min(m,n)/\max(m,n)$ of each matrix. This pins the MP law to one of two regimes for cross-matrix comparability:

| Regime | β | Matrix types |
|--------|---|--------------|
| Rectangular | 0.25 | FFN intermediate/output, attention Q/K/V (Sets 1–3) |
| Square | 1.00 | Attention output dense, pooler (Sets 4–6) |

### Edge Trimming

Three strategies remove boundary eigenvalues before KS tests:

| Strategy | Formula |
|----------|---------|
| `tw` | $\delta = c_\alpha \cdot n_{\text{eff}}^{-2/3} \cdot (1+\sqrt{\beta})^{4/3}$ |
| `fraction` | $\delta = f \cdot (\lambda_+ - \lambda_-)$ |
| `tw_or_fraction` | $\delta = \max(\delta_{\text{TW}},\; \delta_{\text{frac}})$ |

All figures use `tw_or_fraction` with $c_\alpha = 2.0$ and $f = 0.05$.

---

## Dependencies

```
torch
transformers
numpy
matplotlib
seaborn
statsmodels
```

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{Dantas2025,
  title = {Decoding Transformers Spectra: A Random Matrix Theory Framework Beyond the Marchenko–Pastur Law},
  url = {http://dx.doi.org/10.21203/rs.3.rs-7528284/v1},
  DOI = {10.21203/rs.3.rs-7528284/v1},
  publisher = {Springer Science and Business Media LLC},
  author = {Dantas, Pierre and Junior, Waldir and Cordeiro, Lucas and Santos, Eulanda},
  year = {2025},
  month = nov 
}
```
