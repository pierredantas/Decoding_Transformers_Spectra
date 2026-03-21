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
| ALBERT-base-v2 | 1 (shared) | 768 | ~12M |
| BERT-Large-uncased | 24 | 1024 | ~340M |

---

## Repository Structure

```
.
├── paper_bert_decoding_transformers_spectra_full_graphs.py
├── paper_albert_decoding_transformers_spectra_full_graphs.py
├── paper_bert_large_decoding_transformers_spectra_full_graphs.py
└── README.md
```

Each script follows the same **two-section pipeline** followed by twelve graph sections:

| Section | Description |
|---------|-------------|
| Section 1 | Extract raw weight matrices from HuggingFace and save as `.npy` with `manifest.json` |
| Section 2 | Build column-standardized WMP matrices and save with round-trip verification |
| Graphs 1–2 | ePDF / eCDF vs. conditional MP distribution under varying trim conditions |
| Graphs 3–4 | ePDF across layers/matrix types (global scale); Residual CDF |
| Graph 5 | QQ plots of empirical spectra vs. conditional MP quantiles |
| Graph 6 | KS test decision heatmaps (layers × matrix types) |
| Graph 7 | Per-layer acceptance rates with Wilson confidence intervals |
| Graph 8 | Aspect ratio β vs. KS statistic scatter |
| Graphs 9–10 | Bootstrap p-value distributions; KS–TW edge relaxation sensitivity |
| Graphs 11–12 | Type-I calibration curves; eCDF bootstrap reference envelopes |

---

## Methods

### WMP Normalization

Each weight matrix $W$ of shape $(m, n)$ is column-standardized before spectral analysis:

$$W_{\text{MP}}[i,j] = \frac{W[i,j] - \mu_j}{\sigma_j}$$

where $\mu_j$ and $\sigma_j$ are the column-wise mean and standard deviation. This ensures the i.i.d. zero-mean unit-variance assumption required by the MP law.

### Marchenko–Pastur Law

For a random matrix $W$ with i.i.d. $\mathcal{N}(0,1)$ entries and aspect ratio $\beta = \min(m,n)/\max(m,n)$, the empirical spectral distribution of eigenvalues $\lambda$ of $W^\top W / \max(m,n)$ converges to:

$$f_\beta(\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi\beta\lambda}, \quad \lambda \in [\lambda_-, \lambda_+]$$

with support boundaries $\lambda_\pm = (1 \pm \sqrt{\beta})^2$.

### Edge Trimming

Three strategies remove boundary eigenvalues before KS tests:

| Strategy | Formula |
|----------|---------|
| `tw` | $\delta = c_\alpha \cdot n_{\text{eff}}^{-2/3} \cdot (1+\sqrt{\beta})^{4/3}$ |
| `fraction` | $\delta = f \cdot (\lambda_+ - \lambda_-)$ |
| `tw_or_fraction` | $\delta = \max(\delta_{\text{TW}},\; \delta_{\text{frac}})$ |

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
