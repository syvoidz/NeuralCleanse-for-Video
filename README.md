# Neural Cleanse for Video: An End-to-End Backdoor Defense Pipeline

This repository provides a PyTorch implementation demonstrating a full pipeline for backdoor attacks and defenses in video classification. It adapts the core ideas of **Neural Cleanse** to a video context, using a CNN+LSTM model on a sampled subset of the UCF-101 dataset.

The project is encapsulated in a single, powerful script (`run.py`) that automatically performs:
1.  **Attack Phase**: Generates an optimized trigger and injects a strong backdoor into a clean model.
2.  **Defense Phase**: Detects the backdoor's presence (implicitly) and applies a targeted **Unlearning** strategy to mitigate the attack.

This serves as an experimental case study on the feasibility and effectiveness of migrating image-based defense mechanisms to the video domain.

---

## Final Result Snapshot

This pipeline demonstrates the ability to effectively remove a strong backdoor attack while preserving high model accuracy.

| Metric                      | Before Mitigation | After Mitigation |
| --------------------------- | ----------------- | ---------------- |
| **Clean Accuracy (ACC)**    | ~90%              | **~85%**         |
| **Attack Success Rate (ASR)** | ~100%             | **< 15%**        |

---

## Setup

### 1. Environment

A Conda environment with Python 3.8 is recommended.

```bash
# Create and activate the environment
conda create -n ncv_env python=3.8 -y
conda activate ncv_env

# Install PyTorch with GPU support (adjust for your system)
# See https://pytorch.org/ for the correct command
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 2. Data and Pre-trained Model

-   **Dataset**: This project requires a **sampled 10-class subset of UCF-101**. A `ucf101_sampled.zip` file should be downloaded and unzipped into the `data/` directory.
-   **Clean Model**: A pre-trained clean model is necessary to start the pipeline. It must be named `cnnlstm-ucf10_benign_best.pth` and placed in the `models/` directory.

The required project structure is:
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

## How to Run

This entire project can be executed with a single command. The script will automatically perform the attack phase followed by the defense phase and print a final report.

```bash
python run.py
```

### Customization

You can easily customize the pipeline via command-line arguments. For example, to use a different learning rate for mitigation and run for more epochs:

```bash
python run.py --lr_mitigate 2e-5 --nb_epochs_mitigate 25
```

To see all available options, run:
```bash
python run.py --help
```

---

## Project Structure Overview

-   `run.py`: The single, unified entry point for the entire attack-defense pipeline.
-   `src/`: Contains all the core logic modules.
    -   `models.py`: Defines the `CNN_LSTM` architecture.
    -   `datasets.py`: Contains the `UCF101Dataset` and `PoisonedUCF101Dataset` loaders.
    -   `reconstructor.py`: Implements the `TriggerReconstructor` based on Neural Cleanse.
    -   `defenses.py`: Contains the `unlearning_defense` mitigation logic.
    -   `analysis.py`: Includes the `outlier_detection` (MAD) utility.
    -   `utils.py`: Shared utility functions like evaluation metrics.
