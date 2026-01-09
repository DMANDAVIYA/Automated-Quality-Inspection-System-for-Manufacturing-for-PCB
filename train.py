from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os

def train():
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='data/yolo_dataset/dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project='output',
        name='pcb_defect_v8',
        exist_ok=True,
        mosaic=1.0,
        save=True,
        save_period=1,
        plots=True
    )
    
    metrics_path = 'output/pcb_defect_v8/results.csv'
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        df.columns = df.columns.str.strip()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
        axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Cls Loss')
        axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        axes[0, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        axes[0, 1].set_title('Precision & Recall')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50')
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95')
        axes[1, 0].set_title('Mean Average Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
        axes[1, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
        axes[1, 1].plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss')
        axes[1, 1].set_title('Validation Losses')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('output/training_metrics.png', dpi=300)
        print(f"Training visualization saved to output/training_metrics.png")
        
        best_epoch = df.loc[df['metrics/mAP50(B)'].idxmax()]
        print(f"\nBest Epoch: {int(best_epoch['epoch'])}")
        print(f"Best mAP@50: {best_epoch['metrics/mAP50(B)']:.4f}")
        print(f"Best mAP@50-95: {best_epoch['metrics/mAP50-95(B)']:.4f}")

if __name__ == '__main__': train()
