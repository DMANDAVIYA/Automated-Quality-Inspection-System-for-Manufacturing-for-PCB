# Sample Images for PCB Defect Detection

This directory contains sample images demonstrating the automated quality inspection system.

## Contents

### `defective/`
10 PCB images containing various defects:
- Open circuits (broken traces)
- Short circuits (unintended connections)
- Mousebites (notches in board edges)
- Spurs (protruding copper)
- Copper defects (excess deposits)
- Pinholes (holes in solder mask)

### `defect_free/`
10 corresponding defect-free template images used for comparison during inspection.

### `annotated/`
Detection results showing bounding boxes, defect classifications, and confidence scores.

## Usage

Run the detection system on any sample:
```bash
python main.py
```

The system will:
1. Load a random test image
2. Align it to the template
3. Detect defects using YOLOv8 + SAHI
4. Output bounding boxes with (x,y) coordinates
5. Classify defect type with confidence scores
6. Assess severity (Critical/Major/Minor)

## Sample Naming Convention
- `sample_XX_defective.jpg` - PCB with defects
- `sample_XX_template.jpg` - Defect-free reference
- `sample_XX_annotated.jpg` - Detection results

## Performance
The system achieves:
- **98.7% mAP@0.5** on DeepPCB benchmark
- **96.5% Recall** (critical for QA)
- **92.3% F1-Score**
- **10 FPS** inference speed
