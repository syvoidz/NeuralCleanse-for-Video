# Neural Cleanse for Video: An Experimental Implementation

This repository provides an experimental PyTorch implementation of the Neural Cleanse backdoor pipeline, adapted for video classification. The primary goal of this project was to explore the feasibility of migrating a well-known image-based backdoor defense to the video domain.

The entire workflow, from attack to defense, is demonstrated on a **sampled 10-class subset of the UCF-101 dataset** using a **CNN+LSTM** model architecture.

---

## Core Features

This project provides a series of scripts to perform an end-to-end backdoor attack and defense experiment:

1.  **Trigger Reconstruction**: Reverse-engineers an optimized trigger pattern from a clean model.
2.  **Backdoor Injection**: Creates a backdoored model using the reconstructed trigger.
3.  **Automated Detection**: Automatically detects the backdoor's presence and identifies the target class in a suspicious model.
4.  **Mitigation via Unlearning**: Removes the backdoor from an infected model using a targeted unlearning strategy.

---

## Setup

### 1. Environment

It is recommended to use a Conda environment.

```bash
# Create and activate the environment
conda create -n ncv_env python=3.8
conda activate ncv_env

# Install PyTorch with GPU support (adjust for your CUDA version)
# See https://pytorch.org/ for the correct command
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### 2. Dataset and Pre-trained Model

-   **Dataset**: This project uses a **sampled 10-class subset of UCF-101**. A zipped version (`ucf101_sampled.zip`) should be provided and unzipped into the `data/` directory.
-   **Clean Model**: A pre-trained clean model (`cnnlstm-ucf10_benign_best.pth`) is required. Please place it in the `models/` directory.

The expected project structure is:
```
./
├── models/
│   └── cnnlstm-ucf10_benign_best.pth
├── data/
│   └── ucf101_sampled/
│       ├── videos/
│       └── splits/
└── ... (other project files)
```

---

## How to Run the Pipeline

Execute the scripts in the following order. Each script performs one step of the pipeline and generates the necessary files for the next step.

### Step 1: Reconstruct Trigger

Generate the trigger pattern from the clean model.

```bash
python 1_reconstruct_trigger.py
```
**Output**: An optimized trigger (`results/trigger_target_0.pth`) and its visualization (`.png`) will be created.

### Step 2: Inject Backdoor

Create a backdoored model using the clean model and the trigger from Step 1.

```bash
python 2_inject_backdoor.py
```
**Output**: A backdoored model (`models/backdoor_model_nc.pth`) will be saved.

### Step 3: Detect Backdoor

Analyze the backdoored model from Step 2 to automatically identify the backdoor.

```bash
python 3_detect_backdoor.py --backdoor_model_path ./models/backdoor_model_nc.pth
```
**Output**: The script will print a diagnosis report, identifying the infected target class.

### Step 4: Mitigate Backdoor

Run the full detection and mitigation pipeline on the backdoored model.

```bash
python 4_mitigate_backdoor.py --backdoor_model_path ./models/backdoor_model_nc.pth
```
**Output**: The script will first detect the backdoor, then apply the unlearning defense to remove it. The final cleansed model will be saved in the `results_mitigation/` directory. You will observe the Attack Success Rate (ASR) dropping to near 0 while the clean accuracy (ACC) is largely preserved.
