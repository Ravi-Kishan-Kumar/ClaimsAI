"""
YOLO v8 Segmentation Inference Module
======================================
"""
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Any
import numpy as np
from PIL import Image
import os

class YOLODamageDetector:
    def __init__(self, model_path: str = "models/best (low mAP).pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"[YOLO] Loading model from {model_path}...")
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_detections: Dict[str, Any] | None = None
    
    def detect(self, image_path: str, confidence_threshold: float = 0.45) -> List[str]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path)
        
        # Inference
        results = self.model(image, conf=confidence_threshold, verbose=False)

        # Generate Visualization
        annotated_bgr = results[0].plot(boxes=True, conf=True)
        annotated_rgb = annotated_bgr[..., ::-1] # BGR to RGB
        
        class_names = []
        confidences = []
        boxes = []  # <--- NEW: Initialize boxes list
        
        if results and len(results) > 0:
            r = results[0]
            if r.boxes:
                for box in r.boxes:
                    # Extract Class & Confidence
                    class_names.append(self.model.names[int(box.cls)])
                    confidences.append(float(box.conf))
                    
                    # Extract Coordinates (xyxy)
                    try:
                        # Convert tensor to numpy array [x1, y1, x2, y2]
                        coords = box.xyxy[0].cpu().numpy()
                        boxes.append(coords.tolist()) # Convert to list for JSON serialization
                    except Exception:
                        boxes.append([])

        # Store everything including 'boxes'
        self.last_detections = {
            'damage_types': class_names,
            'confidences': confidences,
            'boxes': boxes,  # <--- CRITICAL FIX: Save boxes here
            'image_original': np.array(image),
            'plot': annotated_rgb
        }

        print(f"[YOLO] Detected {len(class_names)} objects.")
        return class_names
    
    def get_damage_summary(self, detections: Any) -> Dict[str, Any]:
        if self.last_detections:
            damage_types = self.last_detections.get('damage_types', [])
            confidences = self.last_detections.get('confidences', [])
        else:
            damage_types = []
            confidences = []

        if not damage_types:
            return {
                'damage_count': 0, 
                'damage_types_list': [], 
                'avg_confidence': 0.0,
                'high_confidence_damages': []
            }

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        
        return {
            'damage_count': len(damage_types),
            'damage_types_list': list(set(damage_types)),
            'avg_confidence': avg_conf,
            'high_confidence_damages': [d for d, c in zip(damage_types, confidences) if c > 0.5]
        }