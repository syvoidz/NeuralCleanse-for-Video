# Neural Cleanse for Video Backdoor Attacks

This repository provides a PyTorch implementation of the Neural Cleanse backdoor detection and mitigation pipeline, specifically adapted for video classification models (CNN+LSTM architecture on the UCF-101 dataset).

This project demonstrates an end-to-end workflow:
1.  **Trigger Reconstruction**: Reverse-engineering the minimal trigger pattern for each potential target class using the Neural Cleanse algorithm.
2.  **Backdoor Injection**: Creating a backdoored video model using the reconstructed trigger.
3.  **Automated Detection**: Automatically identifying the presence and target of a backdoor in a suspicious model using Median Absolute Deviation (MAD) analysis.
4.  **Mitigation via Unlearning**: Effectively removing the backdoor from an infected model by applying a targeted unlearning (active对抗) strategy, with minimal impact on the model's clean accuracy.

---

## Project Structure

```
NeuralCleanse_Video/
├── models/               # Stores clean and backdoored model weights (.pth)
├── data/                 # Should contain the dataset (e.g., ucf101_sampled)
├── results/              # Output directory for reconstructed triggers
├── src/                  # Contains shared utilities and dataset loaders
│   ├── ucf101_dataset.py
│   └── utils.py
├── 1_reconstruct_trigger.py    # Step 1: Reverse-engineers a trigger
├── 2_inject_backdoor.py        # Step 2: Injects the backdoor into a clean model
├── 3_detect_backdoor.py        # Step 3: Automatically detects the backdoor
├── 4_mitigate_backdoor.py      # Step 4: Mitigates the backdoor via unlearning
└── README.md
```

---

## Setup

### 1. Environment

This project is tested with Python 3.8 and PyTorch. It is recommended to use a Conda environment.

```bash
conda create -n ncv_env python=3.8
conda activate ncv_env

# Install PyTorch with GPU support (example for CUDA 11.8)
# Please check pytorch.org for the command matching your system
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### 2. Dataset

This code is designed for the UCF-101 dataset. A sampled version with 10 classes can be used for quick experiments. The dataset should be placed in the `data/` directory, following this structure:

```
data/
└── ucf101_sampled/
    ├── videos/
    │   ├── ApplyEyeMakeup/
    │   └── ... (other class folders)
    └── splits/
        ├── trainlist01.txt
        └── testlist01.txt
```

### 3. Pre-trained Clean Model

This pipeline requires a clean model trained on the target dataset as a starting point. You can train one yourself or download a pre-trained one and place it in the `models/` directory. For this project, a clean model should be named `cnnlstm-ucf10_benign_best.pth`.

---

## End-to-End Workflow

Follow these steps in order to reproduce the full attack and defense pipeline.

### Step 1: Reconstruct a Trigger for Attack

This step uses a clean model to generate an optimized trigger pattern for a specific target class.

```bash
python 1_reconstruct_trigger.py
```
This will generate `results/trigger_target_0.pth` and a `trigger_target_0.png` visualization.

### Step 2: Inject the Backdoor

Use the clean model and the trigger generated in Step 1 to train a backdoored model.

```bash
python 2_inject_backdoor.py
```
This will create `models/backdoor_model_nc.pth`, which is our infected model.

### Step 3: Automatically Detect the Backdoor

This script takes the infected model from Step 2 and performs the full Neural Cleanse detection process. It will analyze all classes and report which one is likely the backdoor target.

```bash
python 3_detect_backdoor.py --backdoor_model_path ./models/backdoor_model_nc.pth
```
The output will show the anomaly index and the flagged label (e.g., Label 0).

### Step 4: Mitigate the Backdoor

Finally, this script runs the full detection-then-mitigation pipeline. It first identifies the backdoor, then uses the reconstructed trigger for that specific class to perform unlearning and cleanse the model.

```bash
python 4_mitigate_backdoor.py --backdoor_model_path ./models/backdoor_model_nc.pth```
The cleansed model will be saved in the `results_mitigation/` directory. You will observe the ASR dropping to near 0 while the clean ACC remains high.
