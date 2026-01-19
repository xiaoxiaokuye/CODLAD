<div align="center">

# CODLAD: Efficient Protein Backmapping via <br> Constraint-Decoupled Latent Diffusion

[![Paper](https://img.shields.io/badge/JCTC-Accepted-b31b1b.svg?style=flat-square&logo=ACS-Publications&logoColor=white)](https://pubs.acs.org/doi/full/10.1021/acs.jctc.5c01364)
[![DOI](https://zenodo.org/badge/929295864.svg)](https://doi.org/10.5281/zenodo.17461857)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/xiaoxiaokuye/CODLAD?style=flat-square)](https://github.com/xiaoxiaokuye/CODLAD/releases)

**A diffusion-based framework for protein backmapping from coarse-grained to all-atom structures.**

[Overview](#-overview) • [Installation](#-installation) • [Datasets](#-datasets) • [Usage](#-usage) • [Citation](#-citation)

</div>

---

## 📢 News
* **[2026.01]** Our paper has been accepted by **Journal of Chemical Theory and Computation (JCTC)**! 🎉
* **[2025.10]** Initial release (v0.1).

## 📖 Overview

**CODLAD** (Constraint-Decoupled Latent Diffusion) is a novel two-stage framework designed to reconstruct all-atom protein structures from coarse-grained representations. It solves the efficiency and stability bottlenecks in protein backmapping by introducing **constraint decoupling** in the latent space.

### Key Features
* **⚡ Two-Stage Architecture**: 
    1.  **Compression**: Encodes atomic structures while preserving structural constraints (VQ-VAE).
    2.  **Generation**: Performs diffusion in a simplified latent space (Latent Diffusion).
* **🔬 Physically Realistic**: Maintains structural validity inherent to the compression phase.
* **🚀 Efficiency**: Significantly reduced computational costs compared to existing all-atom generation methods.

![CODLAD Overview](fig1.png)

## 🛠 Installation

We recommend using Anaconda to manage the environment.

```bash
# 1. Clone the repository
git clone [https://github.com/xiaoxiaokuye/CODLAD.git](https://github.com/xiaoxiaokuye/CODLAD.git)
cd CODLAD

# 2. Create and activate conda environment
conda create -n codlad python=3.11
conda activate codlad

# 3. Install dependencies (CUDA 12.1 required)
pip install -r requirements.txt
```

## 📂 Datasets

You can download the **PDB**, **PED** datasets, and **Pretrained Checkpoints** from our Google Drive:
👉 **[Download Link](https://drive.google.com/file/d/1xTb-LKYvTt9HrQW5RLwzL-MShbg0PSy7/view?usp=drive_link)**

### Directory Structure
After downloading, please organize the files as follows:

```text
CODLAD/
├── results/               # Place the downloaded checkpoints here
├── datasets/
│   ├── protein/
│   │   ├── PDB/           # Place PDB data files here
│   │   ├── PED/           # Place PED data files here
│   │   └── Atlas/         # See instructions below
│   └── ...
└── ...
```

**For Atlas Dataset:**
Please use the provided script to download and setup:
```bash
cd scripts
bash download_atlas.sh
# The script will handle downloading and moving files to ../datasets/protein/Atlas/
```

## 🚀 Usage

### Stage 1: VQ-VAE (Compression)

This stage compresses the all-atom structure into a coarse-grained latent representation.

**1. Data Preprocessing**
```bash
python extract_features.py \
    --process_data \
    --dataname PED  # Options: PED, Atlas, PDB
```

**2. Training VQ-VAE**
```bash
python train_vqvae.py -load_json ./scripts/Vae_vqvae_PED_ns36_vq3_vq4096.json
```

**3. VAE Inference (Reconstruction)**
```bash
python test.py \
    --backbone mpnn_diffusion \
    --vae_type N6 \
    --num_sampling_steps 100 \
    --experiment recon \
    --data_type PED \
    --num_ensemble 10  # Note: N6 for PED, K3 for PDB, K4 for Atlas
```

---

### Stage 2: Latent Diffusion (Generation)

This stage learns the distribution of the latent representations conditioned on coarse-grained structures.

**1. Extract Latent Features**
```bash
python extract_features.py \
    --extract_features \
    --data-path ./datasets/preproccess_PED \
    --features-path ./datasets/features_N6 \
    --vae_type N6 \
    --dataname PED
```

**2. Training Latent Diffusion Model**
```bash
accelerate launch --multi_gpu \
    ./train_latent.py \
    --lr 3e-4 \
    --warmup 80000 \
    --schedule_step 1200000 \
    --final_lr 1e-5 \
    --batch_size 128 \
    --model diffusion \
    --class_dropout_prob 0 \
    --latent_size 3 \
    --backbone mpnn_diffusion \
    --feature_path './datasets/features_N6' \
    --exp './Diff_PED_mpnnnew'
```

**3. Diffusion Inference**
```bash
python test.py \
    --exp "Diff_PED_mpnnnew" \
    --backbone mpnn_diffusion \
    --model diffusion \
    --cfg_scale 0.0 \
    --latent_size 3 \
    --vae_type N6 \
    --num_sampling_steps 100 \
    --experiment latent \
    --data_type PED \
    --num_ensemble 10 
```

## 📝 Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{CODLAD2026,
  title={Constraint Decoupled Latent Diffusion for Protein Backmapping},
  author={Xu Han, Yuancheng Sun, Kai Chen, Yuxuan Ren, Kang Liu, Qiwei Ye},
  journal={Journal of Chemical Theory and Computation},
  year={2026},
  publisher={ACS Publications},
  doi={10.1021/acs.jctc.5c01364},
  note={In Press}
}
```