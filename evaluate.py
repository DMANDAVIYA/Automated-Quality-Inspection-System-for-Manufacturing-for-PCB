import os, glob, cv2, json, pandas as pd, matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

def iou(b1, b2):
    x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (((b1[2]-b1[0])*(b1[3]-b1[1])) + ((b2[2]-b2[0])*(b2[3]-b2[1])) - inter + 1e-6)

def yolo_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    return [int((cx - bw/2)*w), int((cy - bh/2)*h), int((cx + bw/2)*w), int((cy + bh/2)*h)]

def evaluate():
    model = YOLO('output/pcb_defect_v8/weights/best.pt' if os.path.exists('output/pcb_defect_v8/weights/best.pt') else 'yolov8n.pt')
    test_imgs = glob.glob("data/yolo_dataset/images/test/*.jpg")
    
    tp, fp, fn, ious, per_class = 0, 0, 0, [], {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(6)}
    
    for img_path in test_imgs:
        lbl_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        if not os.path.exists(lbl_path): continue
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        with open(lbl_path) as f: gt_boxes = [[int(l.split()[0])] + yolo_to_xyxy([float(x) for x in l.split()[1:5]], w, h) for l in f.readlines()]
        
        preds = model.predict(img_path, conf=0.25, verbose=False)[0]
        pred_boxes = [[int(b.cls[0].item())] + [int(x) for x in b.xyxy[0].tolist()] for b in preds.boxes]
        
        matched_gt, matched_pred = set(), set()
        
        for i, gt in enumerate(gt_boxes):
            for j, pred in enumerate(pred_boxes):
                if gt[0] == pred[0] and j not in matched_pred:
                    iou_val = iou(gt[1:], pred[1:])
                    if iou_val > 0.5:
                        tp += 1
                        per_class[gt[0]]['tp'] += 1
                        matched_gt.add(i)
                        matched_pred.add(j)
                        ious.append(iou_val)
                        break
        
        for i, gt in enumerate(gt_boxes):
            if i not in matched_gt: 
                fn += 1
                per_class[gt[0]]['fn'] += 1
        
        for j, pred in enumerate(pred_boxes):
            if j not in matched_pred:
                fp += 1
                per_class[pred[0]]['fp'] += 1
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou = np.mean(ious) if ious else 0
    
    class_names = ['open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']
    class_metrics = []
    for cls_id, stats in per_class.items():
        cls_p = stats['tp'] / (stats['tp'] + stats['fp'] + 1e-6)
        cls_r = stats['tp'] / (stats['tp'] + stats['fn'] + 1e-6)
        cls_f1 = 2 * cls_p * cls_r / (cls_p + cls_r + 1e-6)
        class_metrics.append({'class': class_names[cls_id], 'precision': cls_p, 'recall': cls_r, 'f1': cls_f1})
    
    metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'mean_iou': mean_iou, 'tp': tp, 'fp': fp, 'fn': fn, 'per_class': class_metrics}
    
    print(f"Overall Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}\n")
    
    print("Per-Class Metrics:")
    for cm in class_metrics:
        print(f"{cm['class']}: P={cm['precision']:.4f}, R={cm['recall']:.4f}, F1={cm['f1']:.4f}")
    
    with open('output/metrics.json', 'w') as f: json.dump(metrics, f, indent=2)
    
    df = pd.DataFrame(class_metrics)
    df.to_csv('output/class_metrics.csv', index=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    ax.bar(x - width, df['precision'], width, label='Precision')
    ax.bar(x, df['recall'], width, label='Recall')
    ax.bar(x + width, df['f1'], width, label='F1-Score')
    ax.set_xlabel('Defect Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/class_performance.png', dpi=300)
    print("\nVisualization saved to output/class_performance.png")
    
    return metrics

if __name__ == '__main__': evaluate()
