# PCB Defect Detection System

SOTA automated visual inspection for PCB manufacturing using YOLOv8 + SAHI.

## Architecture
- **Alignment**: SIFT + RANSAC homography
- **Detection**: YOLOv8-Nano with SAHI sliced inference
- **Dataset**: DeepPCB (1500 pairs, 6 defect classes)

## Setup
```bash
pip install -r requirements.txt
python data_prep.py
python train.py
```

## Training
```bash
python train.py
```
**Outputs:**
- Checkpoints: `output/pcb_defect_v8/weights/` (saved every epoch)
- Best weights: `output/pcb_defect_v8/weights/best.pt`
- Metrics CSV: `output/pcb_defect_v8/results.csv`
- Training plots: `output/training_metrics.png` (losses, precision, recall, mAP)

## Inference
```bash
python main.py
```

## Evaluation
```bash
python evaluate.py
```
**Outputs:**
- Overall metrics: `output/metrics.json`
- Per-class metrics: `output/class_metrics.csv`
- Performance chart: `output/class_performance.png`

## Testing
```bash
python test_inspection.py
```

## Files
- `data_prep.py`: ETL pipeline (DeepPCB → YOLO format)
- `inspection.py`: Core logic (PCBAligner, PCBDetector)
- `train.py`: YOLOv8 training with checkpointing & visualization
- `main.py`: Inference pipeline
- `evaluate.py`: Metrics computation with per-class analysis
- `test_inspection.py`: Unit tests

## Defect Classes
0: Open, 1: Short, 2: Mousebite, 3: Spur, 4: Copper, 5: Pinhole
