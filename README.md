# Blind Face Super-Resolution Project

## Overview

This project implements a deep learning pipeline for **blind face super-resolution (SR)** using a transformer-based model called **FaceRestormer**. The goal is to restore high-quality (HQ) face images (512x512) from low-quality (LQ) inputs (128x128) **without prior knowledge of the degradation process**. The model incorporates **facial landmark attention** to enhance facial details and is trained on the **FFHQ dataset** with simulated degradations (e.g., blur, downsampling).

The pipeline includes:
- Dataset preparation
- Model training
- Inference

---

## Repository Structure

```plaintext
blind_face_sr.ipynb             # Main Jupyter notebook
/content/results/val/           # Validation predictions and PSNR scores
/content/results/test/          # Test predictions
/content/models/                # Trained model weights (face_restormer_best.pth, face_restormer_final.pth)
/content/runs/face_restormer/  # TensorBoard logs
/content/submit/                # Test predictions for CodaLab (submit.zip)
/content/adx/                   # NTULearn package (adx.zip with code, results, weights, logs)
````

---

## Requirements

* **Environment**: Google Colab (A100 GPU recommended) Runtime is 7-hours in a A100
* **Python Version**: 3.10+

### Dependencies

Install using pip:

```bash
pip install torch==2.3.1 torchvision==0.18.1 mediapipe==1.0.2 numpy==1.26.0 pillow==10.2.0 opencv-python==4.9.0 tqdm tensorboard
```

---

## Dataset

* FFHQ dataset should be stored at:

```
/content/drive/MyDrive/2024-S2-AI6126-Project2-Release/datasets/
```

### Directory Structure:

```plaintext
datasets/
├── train/GT/*.png
├── val/GT/*.png
├── val/LQ/*.png
├── test/LQ/*.png
```

---

## Setup Instructions

### 1. Mount Google Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Dependencies:

Run the pip install command shown in the **Dependencies** section above.

### 3. Prepare Dataset:

Ensure the FFHQ dataset is available in the specified path. The notebook will automatically:

* Copy files to `/content/FFHQ/`
* Generate meta-info files

### 4. Run the Notebook:

* Execute all cells in `blind_face_sr.ipynb`
* The pipeline will:

  * Train for 15 epochs
  * Run validation and test inference
  * Save predictions
  * Prepare submission files

### 5. Monitor Training (Optional):

Launch TensorBoard:

```bash
!tensorboard --logdir runs/face_restormer
```

Open the provided Colab URL to view logs.



## Pipeline Details

### Dataset Preparation

* Loads FFHQ images with random flips and rotations
* Applies simulated degradations using Gaussian blur and sinc filters
* Generates meta-info files

### Model Architecture

**FaceRestormer** (Transformer-based):

* Encoder-decoder with skip connections
* Multi-head self-attention for global context
* Landmark attention using **MediaPipe FaceMesh**
* Pixel shuffle for 512×512 upscaling

> **Model size**: \~2.2M parameters (within 2,276,356 limit)

### Training

* **Loss**: L1 + Perceptual (VGG16) + Landmark-weighted L1
* **Optimizer**: Adam (LR = 1e-4), decayed by 0.5 every 5 epochs
* Mixed precision training
* 15 epochs, batch size = 4

### Inference

* Outputs HQ predictions for validation and test sets
* Computes PSNR for validation (stored in `scores.txt`)
* Saves all predictions as PNGs


  * Test predictions (`/content/results/test/*.png`)
* **adx.zip**:

  * Validation predictions (`/content/results/val/*.png`)
  * PSNR score (`codalab_score.txt`)
  * Model weights (`face_restormer_best.pth`)
  * Source code (`blind_face_sr.ipynb`)
  * TensorBoard logs (`logs/`)
  * README with metadata

---


```plaintext
00001.png to 00400.png         # Validation predictions
codalab_score.txt              # PSNR score
src/blind_face_sr.ipynb        # Source code
face_restormer_best.pth        # Model weights
logs/                          # TensorBoard logs
```

---

## Notes

* **Expected Validation PSNR**: \~25–30 dB
* **Debugging**: Use TensorBoard or insert `logger.debug` statements
* **Dataset**: Ensure directories are non-empty
* **Model Constraints**: Already under parameter limits
* **Extensibility**: Modify `opt` for new degradations or tweak architecture

---
## Collaborators

* **Mitali Purwar**
    * Email: mitaliofficial21@gmail.com
    * GitHub: [MitaliPurwar21](https://github.com/MitaliPurwar21)

* **Sankhasubhra Ghosal**
    * Email: sankhasubhraghosal@gmail.com
    * GitHub: [SankhaS](https://github.com/SankhaS)

* **Madhur Maheshwari**
    * Email: madhur2k3@gmail.com
    * GitHub: [madhur2003](https://github.com/madhur2003)


```

