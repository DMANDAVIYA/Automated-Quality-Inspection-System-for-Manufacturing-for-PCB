# Automated Quality Inspection System for PCB Manufacturing

Production-ready computer vision system for detecting and classifying defects on Printed Circuit Boards (PCBs). Achieves **98.7% mAP** on the DeepPCB benchmark using YOLOv8 with SAHI for small object detection.

## System Overview

This system performs automated visual inspection to detect defects before packaging, identifying six types of PCB defects:
- **Open circuits**: Broken traces
- **Short circuits**: Unintended connections
- **Mousebites**: Notches in board edges
- **Spurs**: Protruding copper from traces
- **Copper defects**: Excess copper deposits
- **Pinholes**: Holes in solder mask

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Dataset Preparation
```bash
python data_prep.py
```
Converts DeepPCB dataset to YOLO format (1,500 samples, 80/10/10 split).

### Training
```bash
python train.py
```
Trains YOLOv8n for 50 epochs with mosaic augmentation. Checkpoints saved to `output/pcb_defect_v8/weights/`.

### Inference
```bash
python main.py
```
Runs detection on a random test image. Outputs:
- Annotated image: `output/final_result.jpg`
- Defect coordinates with severity assessment printed to console

### Evaluation
```bash
python evaluate.py
```
Generates test set metrics:
- `output/metrics.json` - Overall performance
- `output/class_metrics.csv` - Per-class breakdown
- `output/class_performance.png` - Visualization

## Performance Metrics

| Metric | Score |
|--------|-------|
| mAP@0.5 | 98.7% |
| Recall | 96.5% |
| Precision | 88.4% |
| F1-Score | 92.3% |
| Inference | 96ms/image |

### Per-Class Results
| Defect | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Open | 95.3% | 96.8% | 96.0% |
| Short | 87.0% | 94.8% | 90.7% |
| Mousebite | 91.3% | 96.8% | 94.0% |
| Spur | 93.1% | 96.4% | 94.8% |
| Copper | 91.8% | 98.1% | 94.9% |
| Pinhole | 72.7% | 96.0% | 82.8% |

## Sample Images

The `examples/` directory contains:
- `defective/` - 10 PCB images with defects
- `defect_free/` - 10 corresponding defect-free templates
- `annotated/` - Detection results with bounding boxes
- `README.md` - Sample documentation

## Architecture

### Components
1. **PCBAligner** (`inspection.py`): SIFT + RANSAC for geometric alignment
2. **PCBDetector** (`inspection.py`): YOLOv8n + SAHI for defect detection
3. **Data Pipeline** (`data_prep.py`): DeepPCB to YOLO format conversion
4. **Training** (`train.py`): YOLOv8 training with mosaic augmentation
5. **Evaluation** (`evaluate.py`): Comprehensive metrics calculation

### Key Features
- **Hybrid approach**: Classical alignment + deep learning detection
- **SAHI integration**: Sliced inference for small defects (10-30 pixels)
- **Real-time performance**: 10 FPS on NVIDIA RTX 3090
- **High recall**: 96.5% (critical for QA - minimizes missed defects)

## Technical Report

Comprehensive 20-page analysis available in `technical_report.pdf`:
- Mathematical foundations (SIFT, RANSAC, YOLOv8 losses)
- Complete architecture breakdown (25 layers detailed)
- Literature review and research gap analysis
- Ablation studies and failure mode analysis
- Future research directions

## Project Structure

```
.
├── data_prep.py          # Dataset conversion
├── train.py              # Training pipeline
├── evaluate.py           # Metrics calculation
├── inspection.py         # Core detection classes
├── main.py               # Inference script
├── test_inspection.py    # Unit tests
├── examples/             # Sample images
│   ├── defective/        # 10 defective samples
│   ├── defect_free/      # 10 template samples
│   └── annotated/        # Detection results
├── output/               # Training artifacts
│   ├── pcb_defect_v8/    # Model weights
│   ├── metrics.json      # Test metrics
│   └── final_result.jpg  # Latest detection
├── technical_report.pdf  # 20-page technical analysis
└── README.md             # This file
```

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- SAHI
- OpenCV
- NumPy, Matplotlib, Pandas

See `requirements.txt` for complete list.

## References

1. Tang, S., et al. "A dataset for PCB defect detection." arXiv:1902.06197 (2019)
2. Jocher, G., et al. "YOLO by Ultralytics" (2023)
3. Akyon, F. C., et al. "SAHI: Slicing aided hyper inference." ICIP (2022)
4. Lowe, D. G. "Distinctive image features from scale-invariant keypoints." IJCV (2004)

## License

This project uses the DeepPCB dataset and YOLOv8 framework. See respective licenses for usage terms.
