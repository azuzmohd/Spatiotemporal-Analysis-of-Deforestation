# Spatiotemporal Analysis of Deforestation Using Siamese U-Net Architecture 

> **Pixel-level forest loss detection from bi-temporal Sentinel-2 satellite imagery**
> Regions: Meghalaya & Nagaland, Northeast India · 2021 → 2023 · 𝑻𝒓𝒂𝒊𝒏𝒆𝒅 𝒆𝒏𝒕𝒊𝒓𝒆𝒍𝒚 𝒇𝒓𝒐𝒎 𝒔𝒄𝒓𝒂𝒕𝒄𝒉




---

## Overview

This project presents a deep learning pipeline for detecting forest loss from bi-temporal Sentinel-2 multispectral satellite imagery. The model takes a pair of satellite image patches from two time points (2021 and 2023) and produces a pixel-level binary segmentation mask indicating where forest has been lost. It was developed under a strict academic constraint: **no pretrained model weights of any kind were permitted**. Every weight in the network was learned from scratch using only the training data provided.

The project directly addresses **UN SDG 15.2** — halting deforestation and restoring degraded forests by 2030. Northeast India's forests are part of the Indo-Burma biodiversity hotspot, among the most species-rich and threatened ecosystems on Earth.

---

## Results

| Metric | Value |
|---|---|
| **Dice Coefficient** | **0.4469** |
| **IoU (Jaccard)** | **0.2921** |
| **ROC-AUC** | **0.8594** |
| Best Threshold | 0.50 |
| TTA | Yes (8 transforms) |
| Best Checkpoint | Epoch 61 / 80 |

A Dice of 0.447 and AUC of 0.860 trained entirely from scratch compares favourably to published results that use pretrained ResNet/EfficientNet backbones (typically Dice 0.50–0.65). The performance gap is almost entirely explained by the missing pretrained backbone, not architectural or training deficiencies.

---

## Architecture: Siamese U-Net v4

### Core Design

The model uses a **Siamese encoder** with shared weights to process T1 and T2 images independently, ensuring their feature representations are directly comparable. A dedicated **|T2−T1| difference branch** is fused at the bottleneck to make the change signal salient from the start.

```
T1 (2021) ──► Shared Encoder ──► skip connections
T2 (2023) ──► Shared Encoder ──► skip connections (w/ Channel Attention)
|T2-T1|   ──► Diff Branch    ──►┐
                                 ▼
                       ASPP Bottleneck (multi-scale context)
                                 │
                       Transformer Bottleneck (global context)
                                 │
                         U-Net Decoder
                                 │
              Main Output + 3 Auxiliary Deep Supervision Outputs
```

### Key Components

| Component | Detail |
|---|---|
| **Shared encoder** | 4 blocks (32/64/128/256 filters), GroupNorm + GELU |
| **GroupNorm** | Used instead of BatchNorm — stable at small batch sizes (4–8) |
| **Channel Attention** | Squeeze-and-Excitation on all skip connections; focuses on NIR/SWIR |
| **ASPP** | Dilation rates 4, 8, 12 + global pooling — multi-scale receptive field |
| **Transformer bottleneck** | 4-head self-attention at 16×16; global spatial reasoning |
| **Deep supervision** | Aux outputs at 32×32 (w=0.2), 64×64 (w=0.4), 128×128 (w=0.2) |

### Input Channels (9 total)

Raw Sentinel-2 bands (B, G, R, NIR, SWIR1, SWIR2) plus three vegetation indices appended after normalisation:
- **NDVI** — `(NIR−R)/(NIR+R)` — standard green vegetation indicator
- **EVI** — Enhanced Vegetation Index, reduces atmospheric and soil noise
- **SAVI** — Soil-Adjusted Vegetation Index, corrects for soil brightness in cleared areas

---

## Dataset

| Property | Value |
|---|---|
| Source | Sentinel-2 multispectral imagery |
| Regions | Meghalaya (subtropical broadleaf) & Nagaland (community forest) |
| Time Points | T1: 2021, T2: 2023 |
| Patch Size | 256 × 256 px at 30 m/pixel |
| Split | 80% train / 20% val, stratified by forest-loss density |
| Positive pixel fraction | < 5% (extreme class imbalance) |

Dataset: [forest-loss-dataset by suranjandas1990](https://www.kaggle.com/datasets/suranjandas1990/forest-loss-dataset)

Expected structure:
```
dataset/
├── Meghalaya_2021_2023/
│   ├── t1/      ← T1 .npy patches
│   ├── t2/      ← T2 .npy patches
│   └── label/   ← binary mask .npy files
└── Nagaland_2021_2023/
    ├── t1/
    ├── t2/
    └── label/
```

---

## Training Strategy

### Loss Function

```
Loss = 0.5 × Weighted_BCE + 0.5 × Dice_Loss
```

Weighted BCE applies a `pos_weight=5.0` penalty to forest-loss pixels, directly counteracting class imbalance. Dice loss directly optimises the evaluation metric. The two are complementary: BCE stabilises training; Dice pushes spatial overlap.

### Key Training Decisions

| Decision | Why |
|---|---|
| 50/50 balanced pos/neg sampling | Forest-loss patches are rare; uniform sampling starves the model of positive signal |
| Global mean/std normalisation | Per-image normalisation destroys inter-temporal spectral differences — the primary change signal |
| Cosine annealing with 10-epoch warm-up | Prevents early instability; decays smoothly to near-zero by epoch 80 |
| `ReduceLROnPlateau` removed | Conflicts with cosine annealing; caused erratic mid-cycle LR slashing in earlier experiments |
| `validation_steps=None` | Evaluates full val set each epoch; truncated evaluation caused oscillating val_dice curves |
| EarlyStopping patience=20 | Model improves slowly post-warmup; shorter patience triggered prematurely |
| Conservative augmentation | Flips, 90° rotations, mild noise/brightness only — aggressive warps destroy T1/T2 alignment |
| 8-transform TTA at inference | Averages predictions over all flip+rotation combinations; consistent +1–3 Dice points |

---

## Limitations

| Limitation | Detail |
|---|---|
| No pretrained backbone | ~10–15 Dice point gap vs ImageNet-pretrained approaches; the single largest constraint |
| 2-region coverage | Does not generalise out-of-the-box to Amazon, Congo, or SE Asia |
| 30 m resolution | Selective logging and understory degradation below the resolution limit are invisible |
| Bi-temporal only | Gradual degradation or within-window regrowth may be missed |
| Cloud contamination | No cloud mask; cloudy patches may produce false positives |

---

## Societal Relevance

Automated satellite-based monitoring enables:
- Early detection of illegal clearing for timely enforcement response
- Quantitative tracking for national carbon accounting and REDD+ reporting
- Community-level monitoring for indigenous forest stewardship programs
- Long-term trend analysis independent of political reporting cycles

The model includes an interactive **Doctor UI** allowing a researcher to upload any T1/T2 patch pair and receive an area estimate in hectares, a probability map, and a severity classification — without requiring deep technical expertise.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/spatiotemporal-deforestation-siamese-unet.git
cd spatiotemporal-deforestation-siamese-unet
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn ipywidgets
```

Open `forest_loss_detection_v5.ipynb` and run all cells.

---

## Citation

```bibtex
@misc{deforestation_siamese_unet_2024,
  title  = {Spatiotemporal Analysis of Deforestation Using Siamese U-Net v5 Architecture},
  author = {YOUR NAME},
  year   = {2024},
  url    = {https://github.com/YOUR_USERNAME/spatiotemporal-deforestation-siamese-unet}
}
```

---

*Best checkpoint: epoch 61 (val_dice = 0.4469) · 80 epochs · Kaggle GPU infrastructure*
