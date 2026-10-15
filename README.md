# 🧠 ParkiSense AI

> **Multimodal Deep Learning Framework for Objective Parkinson's Disease Severity Assessment**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit 1.31](https://img.shields.io/badge/Streamlit-1.31-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ParkiSense AI is an end-to-end multimodal clinical AI framework that fuses **Voice**, **EEG**, and **Gait (VGRF)** signals using a Transformer architecture with a Safe Fallback mechanism. It simultaneously predicts continuous UPDRS motor severity scores (0–100) and 3-tier clinical severity classifications (Mild, Moderate, Severe), complemented by an interactive, role-based clinical web dashboard with SQLite patient tracking and PDF report generation.

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Layout](#-project-layout)
- [Quick Start](#-quick-start)
- [Dataset Pipeline](#-dataset-pipeline)
- [Model Training](#-model-training)
- [Role-Based Dashboard](#-role-based-dashboard)
- [Testing & Quality Verification](#-testing--quality-verification)
- [Citation & License](#-citation--license)

---

## 🌟 Key Features

- **Tri-Modal Physiological Signal Fusion**:
  - **🎙️ Voice (52-dim)**: 13 MFCCs + $\Delta$ + $\Delta\Delta$, Praat-calculated fundamental frequency ($F_0$), local jitter, local shimmer, and Harmonics-to-Noise Ratio (HNR).
  - **🧠 EEG (64-dim)**: Welched spectral band powers ($\delta, \theta, \alpha, \beta, \gamma$), inter-channel coherence, and FastICA artifact suppression.
  - **🦶 Gait (42-dim)**: 19-sensor PhysioNet VGRF analysis covering stride cadence, bilateral asymmetry, double support duration, and coefficient of variation (CV).
- **🛡️ Safe Fallback Gate**: Robustly handles missing or low-quality modalities by automatically substituting corrupted streams with learned neural filler vectors.
- **⚡ Transformer Multi-Head Self-Attention**: Captures complex cross-modal physiological dependencies and surfaces attention weight heatmaps for clinical interpretability.
- **🔐 SQLite Authentication & Patient Registry**: Zero-config local database (`data/parkisense.db`) with PBKDF2-HMAC cryptographic password hashing and role separation (Neurologist vs. Patient).
- **📈 Patient-Specific Longitudinal UPDRS Tracker**: Preserves chronological visit scores and computes disease progression rates and medication response plateaus.
- **📄 One-Click Clinical PDF Export**: Generates ReportLab medical assessment reports including summary tables, radar parameters, therapy care plans, and progression charts.

---

## 🧩 System Architecture

```
[🎙️ Voice Recording (.wav)]  ──► ModalityEncoder (52  → 128) ──┐
[🧠 EEG Recording (.set)]     ──► ModalityEncoder (64  → 128) ──┼─► [Safe Fallback Gate]
[🦶 Gait Analysis (.txt)]    ──► ModalityEncoder (42  → 128) ──┘           │
                                                                           ▼
                                                             [Positional Encoding]
                                                                           │
                                                                           ▼
                                                             [Transformer Fusion (L=2, H=4)]
                                                                           │
                                                                 [Global Average Pool]
                                                             ┌─────────────┴─────────────┐
                                                             ▼                           ▼
                                                      [Regression Head]        [Classification Head]
                                                             │                           │
                                                    UPDRS Score [0–100]        Softmax Logits {0, 1, 2}
                                                                               (Mild / Moderate / Severe)
```

---

## 📁 Project Layout

```
parkisense_ai/
├── app.py                  # Streamlit Web Application & Dashboard
├── train.py                # End-to-end model training pipeline
├── requirements.txt        # Pinned Python package dependencies
├── Dockerfile              # Container deployment recipe
├── README.md               # Documentation
├── .gitignore              # Repository exclusions
├── datasets/
│   └── manifest.csv        # Multi-modal training manifest with UPDRS scores
├── models/
│   └── parkisense_best.pth # Pretrained PyTorch model checkpoint
├── src/
│   ├── __init__.py
│   ├── models.py           # ParkiSenseFusionNet & loss functions
│   ├── dataset.py          # PyTorch ParkiSenseDataset with augmentation
│   ├── preprocessing.py    # Voice, EEG, and Gait feature extractors
│   ├── database.py         # SQLite authentication & longitudinal storage
│   └── utils.py            # Inference helpers, PDF generator, therapy engine
├── tests/
│   ├── __init__.py
│   ├── test_models.py      # Unit tests for neural architectures
│   └── test_database.py    # Unit tests for database & auth
└── data/
    ├── parkisense.db       # SQLite user & visit database
    └── scripts/
        └── build_dataset.py # Automated raw dataset parser & feature extractor
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python **3.10.x** (recommended)
- Git

### 2. Clone Repository
```bash
git clone https://github.com/Lakshitha-R-E/ParkiSense-AI.git
cd ParkiSense-AI/parkisense_ai
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🔐 Default Demo Accounts

For immediate evaluation, the database comes pre-seeded with:

| Role | Username | Password | Access Privileges |
| :--- | :--- | :--- | :--- |
| **Neurologist** | `dr_smith` | `admin123` | Full clinical view, patient selector, raw signal upload, attention maps, PDF export |
| **Patient** | `patient_john` | `patient123` | Simplified health score, personalized care plan, personal progression trend |

---

## 🏋️ Model Training

To retrain the multimodal model from scratch on the processed clinical manifest:

```bash
python train.py --epochs 25 --batch-size 16 --lr 0.001
```

The script optimizes a joint multi-task objective:
$$\mathcal{L}_{\text{total}} = 0.6 \cdot \text{MSE}(\hat{y}_{\text{updrs}}, y) + 0.4 \cdot \text{CE}(\hat{z}_{\text{cls}}, z)$$
and saves the highest-performing checkpoint to `models/parkisense_best.pth`.

---

## 🧪 Testing & Quality Verification

Run the full automated test suite (20 unit tests):

```bash
python -m unittest discover tests
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
