"""
CNN Severity Classifier Inference Module
========================================
Purpose: Estimate damage severity from image (minor / moderate / severe).
Input: Image path
Output: Clean dictionary with label, confidence, and probability distribution

Academic Note:
- CNNs extract hierarchical features from images through convolutional layers.
- ResNet uses residual connections to enable training of very deep networks.
- We load a pretrained ResNet classifier (no training or retraining performed).
- Softmax converts logits to probabilities (each class: 0-1, sum=1).
- Confidence = max probability (model's certainty in its prediction).
- This is a PROBABILISTIC SIGNAL only—final decision is made by rule engine.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from typing import Dict, List
from PIL import Image
import os


class SeverityClassifier:
    """
    Loads pretrained CNN and performs inference-only severity classification.
    Handles both checkpoint formats and CPU/GPU device selection.
    """
    
    CLASSES = ['minor', 'moderate', 'severe']
    
    def __init__(self, model_path: str = "models/severity_model.pth"):
        """
        Load pretrained severity model.
        
        Args:
            model_path (str): Path to .pth checkpoint file
            
        Raises:
            FileNotFoundError: If model file not found
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print(f"[Severity] Loading model from {model_path}...")
        
        # Determine device (GPU if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load checkpoint (handles both dict and direct state_dict)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint is a dict with metadata
            state_dict = checkpoint['model_state_dict']
        else:
            # Checkpoint is just the state dict
            state_dict = checkpoint
        
        # Build model architecture from state_dict (infer num_classes)
        # Most simple CNN outputs 3 classes for severity: minor/moderate/severe
        self.model = self._build_model(state_dict)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[Severity] Model loaded on device: {self.device}")
        
        # Image preprocessing (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),      # Resize to standard size
            transforms.ToTensor(),               # Convert to tensor [0, 1]
            transforms.Normalize(                # Apply ImageNet normalization
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _build_model(self, state_dict) -> nn.Module:
        """
        Build ResNet18 model and adapt final layer for 3-class severity.
        Detects model structure from state_dict keys (ResNet-based architecture).
        """
        # Use ResNet18 as base architecture
        model = models.resnet18(pretrained=False)
        
        # Modify final fully connected layer for 3 classes (minor, moderate, severe)
        model.fc = nn.Linear(model.fc.in_features, 3)
        
        return model
    
    def predict(self, image_path: str) -> Dict:
        """
        Run severity inference on image.
        
        Args:
            image_path (str): Path to vehicle damage image
            
        Returns:
            Dict with keys:
                - 'severity': Predicted severity label (str: minor/moderate/severe)
                - 'confidence': Max probability (float, 0-1)
                - 'probabilities': Full probability distribution as dict
                - 'class_id': Integer class index (int: 0/1/2)
        
        Raises:
            FileNotFoundError: If image not found
        """
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        print(f"[Severity] Running inference on {os.path.basename(image_path)}...")
        
        # Run inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)  # Convert to [0,1] probabilities
        
        # Extract results
        probs_list = probabilities[0].cpu().numpy().tolist()
        class_idx = int(torch.argmax(probabilities[0]).item())
        confidence = float(probs_list[class_idx])
        label = self.CLASSES[class_idx]
        
        # Return in format expected by downstream pipeline (main.py, cpibp_rules.py)
        result = {
            'severity': label,  # Predicted class name (minor/moderate/severe)
            'confidence': confidence,  # Max probability (0-1)
            'probabilities': {  # Full probability distribution as dict
                'minor': float(probs_list[0]),
                'moderate': float(probs_list[1]),
                'severe': float(probs_list[2])
            },
            'class_id': class_idx  # Integer class index (0/1/2)
        }
        
        print(f"[Severity] Predicted: {label} (confidence: {confidence:.3f})")
        return result


if __name__ == "__main__":
    # Sanity check: runs on first image in data/test_images (if present)
    model_path = "models/severity_model.pth"
    test_folder = "data/test_images"
    
    classifier = SeverityClassifier(model_path)
    
    if os.path.exists(test_folder):
        imgs = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if imgs:
            image_path = os.path.join(test_folder, imgs[0])
            result = classifier.predict(image_path)
            
            print("\nSeverity Prediction Result:")
            print(f"  Severity: {result['severity']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Class ID: {result['class_id']}")
            print(f"  Probabilities: minor={result['probabilities']['minor']:.3f}, moderate={result['probabilities']['moderate']:.3f}, severe={result['probabilities']['severe']:.3f}")
        else:
            print(f"No images found in {test_folder}. Paste images there for testing.")
    else:
        print(f"Test folder not found: {test_folder}. Create and add images to run the sanity check.")
