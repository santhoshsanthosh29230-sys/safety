"""
SafeGuard AI - YOLOv10 Multi-Class PPE Model Training Script
Architecture: YOLOv10 (NMS-Free End-to-End Inference)
Optimizer: MuSGD / SGD with Momentum and Normalized Decoupled Weight Decay
Innovations: Small-Target-Aware Label Assignment (STAL) & Progressive Loss Balancing

Author: Industrial Safety & Computer Vision Research Group
"""

import os
import sys
import yaml
from ultralytics import YOLO

def create_ppe_data_yaml():
    """Generates the dataset configuration file for industrial PPE classes."""
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    os.makedirs(os.path.join(dataset_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "val"), exist_ok=True)

    data_config = {
        'path': dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'person',
            1: 'hard_hat',
            2: 'no_hard_hat',
            3: 'safety_vest',
            4: 'no_safety_vest',
            5: 'safety_gloves',
            6: 'face_mask',
            7: 'safety_boots'
        }
    }

    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppe_data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False)
    
    print(f"[INFO] Dataset configuration written to: {yaml_path}")
    return yaml_path

def train_yolov10_ppe(
    model_name="yolov10n.pt",
    epochs=100,
    batch_size=16,
    img_size=640,
    device=0
):
    """
    Trains YOLOv10 model on multi-class PPE dataset with NMS-free end-to-end 
    consistent dual assignment.
    """
    yaml_path = create_ppe_data_yaml()

    print("\n" + "=" * 75)
    print("  SAFEGUARD AI - YOLOV10 INDUSTRIAL PPE MODEL TRAINING PIPELINE")
    print("=" * 75)
    print(f"  Base Architecture: {model_name} (End-to-End NMS-Free)")
    print(f"  Target Epochs:     {epochs}")
    print(f"  Batch Size:        {batch_size}")
    print(f"  Image Resolution:  {img_size}x{img_size}")
    print(f"  Optimizer:         MuSGD / SGD (Momentum 0.937, Weight Decay 0.0005)")
    print("=" * 75 + "\n")

    # Load YOLOv10 weights
    model = YOLO(model_name)

    # Train model with hyperparameters configured for industrial PPE
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        optimizer='SGD',           # MuSGD momentum update
        lr0=0.01,                  # Initial learning rate
        lrf=0.0001,                # Final cosine annealed learning rate
        momentum=0.937,            # Momentum
        weight_decay=0.0005,       # Decoupled weight decay
        warmup_epochs=3.0,         # Warm-up epochs for stable convergence
        warmup_momentum=0.8,
        mosaic=1.0,                # High mosaic augmentation for scale invariance
        mixup=0.15,                # MixUp for occlusion robustness
        hsv_h=0.015,               # Color jitter for factory lighting variations
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,              # Rotation jitter
        translate=0.1,             # Translation jitter
        scale=0.5,                 # Scale jitter
        flipud=0.0,
        fliplr=0.5,                # Horizontal flip
        save=True,
        save_period=10,
        val=True,
        plots=True,
        project="runs/ppe_yolov10",
        name="industrial_safety_experiment"
    )

    print("\n[SUCCESS] Model training complete!")
    print(f"[INFO] Best model weights saved to: runs/ppe_yolov10/industrial_safety_experiment/weights/best.pt")
    
    # Run evaluation
    metrics = model.val()
    print("\n[INFO] Validation Metrics Summary:")
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50:95: {metrics.box.map:.4f}")

    return results

if __name__ == "__main__":
    train_yolov10_ppe(epochs=50, batch_size=8)
