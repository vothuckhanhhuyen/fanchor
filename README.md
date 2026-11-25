<h1 align="center">FAnchor</h1>

<h3 align="center"><em>Aligned, yet distinct</em>: Shared–Private Factorization of Multimodal Representations with Formal Guarantees</h3>

---

<p align="center"><a href="#"><img src="https://img.shields.io/badge/arXiv-preprint-b31b1b.svg" height="28" alt="arXiv"></a> <a href="environment.yml"><img src="https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white" height="28" alt="Python"></a> <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-core-EE4C2C?logo=pytorch&logoColor=white" height="28" alt="PyTorch"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" height="28" alt="License: MIT"></a></p>

## 💡 Overview

Official PyTorch implementation of **FAnchor**. FAnchor takes pre-extracted foundation model embeddings as input and learns shared and private (i.e., modality-specific) representations for each modality.

The repository also includes the baselines (FactorCL, DisentangledSSL, JointOpt, RePercENT) and the data, feature extraction and evaluation pipelines for every dataset.

## 🛠️ Setup

Run everything from the repo root.

```bash
conda env create -f environment.yml
conda activate fanchor

git clone https://github.com/pliang279/FactorCL third_party/FactorCL
git clone https://github.com/uhlerlab/DisentangledSSL third_party/DisentangledSSL
git clone https://github.com/pseulki/rococo third_party/rococo
git clone https://github.com/Vascorn/RePercENT third_party/RePercENT
git clone https://github.com/Bizilizi/VGGSounder third_party/VGGSounder

mkdir -p logs_cluster/{clevr4,vggsound,rococo,synthetic}
```

## 📂 Repository Map

| Path | Contents |
| --- | --- |
| [`models/`](models/) | FAnchor and the baselines. |
| [`datasets/`](datasets/) | Data loading and feature extraction for each dataset. |
| [`configs/`](configs/) | One config per dataset. |
| [`slurm_scripts/`](slurm_scripts/) | Data download, feature extraction and training scripts per dataset. |
| [`train.py`](train.py) | Training and evaluation. |
| [`evaluate.py`](evaluate.py) | Evaluation metrics per dataset. |
| [`summarize.py`](summarize.py) | Averages results over seeds. |

## 📊 Datasets

| Dataset | Data | Encoders | Methods |
| --- | --- | --- | --- |
| `clevr4` | Clevr-4 + template captions | CLIP-RN50 | `fanchor`, `factorcl`, `disentangledssl`, `jointdisen`, `repercent` |
| `vggsound` | VGGSound, tested on VGGSound and VGGSounder | PE-AV-Base | `fanchor`, `factorcl`, `disentangledssl`, `jointdisen`, `repercent` |
| `rococo` | MS-COCO, tested on MS-COCO 5K and RoCOCO | CLIP-ViT-B/32 | `fanchor`, `factorcl`, `disentangledssl`, `jointdisen`, `repercent`, `finetuned_proj_clip` |
| `synthetic` | generated | - | `fanchor` |

FAnchor is `fanchor` (with `hier_cond=True`: FAnchor + Cond.). Baselines: `factorcl`, `disentangledssl`, `jointdisen` (JointOpt), `repercent`.

## 🚀 Running

For each dataset, run three steps in order, waiting for each to finish:

```bash
sbatch slurm_scripts/<dataset>/download_data.sh      # 1. data      -> data/     (synthetic: generate_data.sh)
sbatch slurm_scripts/<dataset>/extract_features.sh   # 2. features  -> features/ (not needed for synthetic)
sbatch slurm_scripts/<dataset>/run_<method>.sh       # 3. training  -> results/
```

Each `run_<method>.sh` trains every setting with seeds 0-4 (`train.py`), then averages the seeds (`summarize.py`).

## 🙏 Acknowledgements

Baselines and evaluation build on <a href="https://github.com/pliang279/FactorCL">FactorCL</a>, <a href="https://github.com/uhlerlab/DisentangledSSL">DisentangledSSL</a>, <a href="https://github.com/Vascorn/RePercENT">RePercENT</a>, <a href="https://github.com/pseulki/rococo">RoCOCO</a> and <a href="https://github.com/Bizilizi/VGGSounder">VGGSounder</a>.
