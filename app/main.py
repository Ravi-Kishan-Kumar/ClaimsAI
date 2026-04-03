"""
Main Pipeline: End-to-End Insurance Claim Assessment
=====================================================
Orchestrates the complete workflow for the Academic AI Insurance Project.

Pipeline Steps:
1. Integrity Check (Tier 3): Detects metadata tampering, editing software, or suspicious filenames.
2. YOLO Inference: Detects visual damage (dents, scratches) and extracts bounding boxes.
3. Fraud/Duplicate Check: Compare visual "fingerprint" against historical claims (Graph-based).
4. CNN Severity Inference: Classifies overall damage severity (Minor/Moderate/Severe).
5. CPIBP Rule Engine: Applies deterministic business logic to output a final decision.

Usage:
    This script is the backend engine called by 'streamlit_app.py'.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image, ExifTags

# --- Path Setup ---
# Add the parent directory to sys.path to ensure module imports work correctly
# regardless of from where the script is executed.
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Local Module Imports ---
try:
    from inference.yolo_infer import YOLODamageDetector
    from inference.severity_infer import SeverityClassifier
    from fraud.fraud_detection import FraudDetector  # New Duplicate Check Module
    from rules.cpibp_rules import CPIBPRuleEngine, ClaimDecision
except ImportError as e:
    print(f"[CRITICAL] Failed to import modules: {e}")
    sys.exit(1)


class InsuranceClaimAssessor:
    """
    The central controller class. It initializes all AI models and logic engines,
    then processes images through the sequential pipeline.
    """

    def __init__(self, yolo_model_path: str, severity_model_path: str):
        """
        Initialize the assessment system.
        
        Args:
            yolo_model_path (str): Path to the YOLOv8 .pt file.
            severity_model_path (str): Path to the ResNet/CNN .pth file.
        """
        print("\n" + "="*60)
        print("[Main] Initializing Insurance Claim Assessor Engine...")
        print("="*60)
        
        try:
            # 1. Load Object Detector (YOLO)
            self.yolo_detector = YOLODamageDetector(yolo_model_path)
            
            # 2. Load Severity Classifier (CNN)
            self.severity_classifier = SeverityClassifier(severity_model_path)
            
            # 3. Initialize Fraud/Duplicate Detector (Graph Database)
            self.fraud_detector = FraudDetector()
            
            # 4. Initialize Rule Engine (Business Logic)
            self.rule_engine = CPIBPRuleEngine()
            
            print("[Main] ✓ All models and engines loaded successfully.")
            print("-" * 60 + "\n")
            
        except FileNotFoundError as e:
            print(f"[Main] ✗ Critical Error: Model file missing - {e}")
            raise
        except Exception as e:
            print(f"[Main] ✗ Critical Initialization Error: {e}")
            raise

    def _extract_readable_metadata(self, img: Image.Image) -> Dict[str, str]:
        """
        Helper: Extracts and sanitizes EXIF metadata for UI display.
        Removes binary data to prevent crashes.
        """
        readable_data = {}
        try:
            exif = img.getexif()
            if not exif:
                return {}
            
            for tag_id, value in exif.items():
                # Convert numeric Tag ID to string name (e.g., 305 -> 'Software')
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                
                # Safety check: Don't return huge binary blobs
                if isinstance(value, (bytes, bytearray)):
                    if len(value) > 50:
                        readable_data[tag_name] = "<Binary Data Omitted>"
                    else:
                        readable_data[tag_name] = str(value)
                else:
                    readable_data[tag_name] = str(value)[:100] # Truncate long strings
                    
        except Exception as e:
            print(f"[Metadata] Warning: Could not parse EXIF: {e}")
            
        return readable_data

    def _check_image_integrity(self, image_path: str) -> Dict:
        """
        STEP 1: Integrity Check (Tier 3 Fraud Detection).
        Analyzes Filename, Metadata presence, and Editing Software signatures.
        """
        try:
            img = Image.open(image_path)
            raw_metadata = self._extract_readable_metadata(img)
            filename = os.path.basename(image_path).lower()
            exif_data = img.getexif()
            
            # A. Check for Suspicious Filenames (AI Generators / Editing Tools)
            # Catches: "Gemini_Generated", "Nano_Banana_Edit", "Screenshot"
            suspicious_keywords = [
                'generated', 'gemini', 'dall-e', 'edit', 'copy', 
                'screenshot', 'nano', 'banana', 'fake', 'clone', 'whatsapp'
            ]
            for word in suspicious_keywords:
                if word in filename:
                    return {
                        'status': 'SUSPICIOUS',
                        'flag': True,
                        'is_fraud': True,
                        'verification_required': True,
                        'reason': f"Suspicious filename keyword detected: '{word}'",
                        'meta': raw_metadata
                    }

            # B. Check for Missing Metadata (Striped EXIF)
            # Common in downloaded images. We flag as "Verification Required" (Orange).
            if not exif_data:
                return {
                    'status': 'MISSING',
                    'flag': False,
                    'is_fraud': False,
                    'verification_required': True, 
                    'reason': "Metadata missing. Originality cannot be verified.",
                    'meta': {}
                }

            # C. Check Software Signature (Tag 305)
            # Detects Photoshop, GIMP, etc.
            software_tag_id = 305
            if software_tag_id in exif_data:
                software_name = str(exif_data[software_tag_id]).lower()
                forbidden_tools = ['photoshop', 'gimp', 'editor', 'canvas', 'paint', 'adobe', 'nano', 'banana']
                for tool in forbidden_tools:
                    if tool in software_name:
                        return {
                            'status': 'EDITED',
                            'flag': True,
                            'is_fraud': True,
                            'verification_required': True,
                            'reason': f"Editing software signature found: {software_name}",
                            'meta': raw_metadata
                        }

            # D. Passed
            return {
                'status': 'VALID', 
                'flag': False,
                'is_fraud': False, 
                'verification_required': False, 
                'reason': "Integrity check passed. Metadata valid.",
                'meta': raw_metadata
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'flag': False,
                'is_fraud': False, 
                'verification_required': True, 
                'reason': f"Error reading file integrity: {str(e)}",
                'meta': {}
            }

    def assess_claim(self, image_path: str) -> Dict:
        """
        Execute the full assessment pipeline for a single image.
        
        Args:
            image_path (str): Path to the image.
            
        Returns:
            Dict: Comprehensive results including decisions, reasoning, and overlays.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        print(f"\n[Pipeline] Processing Claim: {os.path.basename(image_path)}")
        
        
        # --- 1. INTEGRITY CHECK ---
        integrity = self._check_image_integrity(image_path)
        print(f"   > Integrity Status: {integrity['status']}")
        
        # --- 2. YOLO INFERENCE (Damage Detection) ---
        class_names = self.yolo_detector.detect(image_path)
        yolo_summary = self.yolo_detector.get_damage_summary(class_names)
        
        # Get visualization plot (Annotated Image)
        annotated_img = None
        last_det = self.yolo_detector.last_detections
        if last_det and 'plot' in last_det:
            annotated_img = last_det['plot']
        
        # --- 3. FRAUD / DUPLICATE CHECK (Graph-Based) ---
        # Robustly extract boxes (Safe Mode to prevent KeyErrors)
        boxes = []
        if last_det and 'boxes' in last_det:
            boxes = last_det['boxes']
        
        # Create fingerprint payload
        fraud_payload = {
            "image_id": os.path.basename(image_path),
            "damage_parts": class_names,
            "boxes": boxes, 
            "avg_confidence": yolo_summary['avg_confidence']
        }
        
        # Run comparison against history
        fraud_analysis = self.fraud_detector.check_fraud(fraud_payload)
        print(f"   > Fraud Risk: {fraud_analysis['fraud_risk']} ({fraud_analysis['reason']})")
        
        # --- 4. CNN INFERENCE (Severity) ---
        severity_prediction = self.severity_classifier.predict(image_path)
        print(f"   > Severity: {severity_prediction['severity'].upper()}")
        
        # --- 5. RULE ENGINE (Final Decision) ---
        # Pass all signals (YOLO, Severity, Integrity, Fraud) to the deterministic rules
        decision = self.rule_engine.process_claim(
            yolo_summary, 
            severity_prediction, 
            integrity, 
            fraud_analysis
        )
        print(f"   > Final Decision: {decision.decision.upper()}")
        
        # --- 6. COMPILE RESULTS ---
        result = {
            'image_path': image_path,
            'annotated_image': annotated_img,
            
            # Sub-module results
            'integrity': integrity,
            'yolo_detections': {
                'class_names': class_names,
                'summary': yolo_summary
            },
            'fraud_analysis': fraud_analysis,
            'severity_prediction': severity_prediction,
            
            # Final Decision
            'decision': {
                'action': decision.decision,
                'confidence': decision.confidence,
                'explanation': decision.explanation,
                'rules': decision.triggered_rules
            },
            'assessment_complete': True
        }
        
        return result
    
    def generate_report(self, assessment: Dict) -> str:
        """Generate a simple text report for console debugging."""
        dec = assessment['decision']
        return (
            f"\n--- REPORT: {os.path.basename(assessment['image_path'])} ---\n"
            f"ACTION: {dec['action'].upper()}\n"
            f"REASON: {dec['explanation']}\n"
            f"FRAUD RISK: {assessment['fraud_analysis']['fraud_risk']}\n"
            f"INTEGRITY: {assessment['integrity']['status']}\n"
            f"------------------------------------------------"
        )

# --- Batch Processing (CLI Mode) ---
def main():
    """
    Optional: logic to run this script directly from terminal for testing.
    """
    # Configuration
    yolo_model = "models/best (low mAP).pt"
    severity_model = "models/severity_model.pth"
    input_dir = "data/test_images"
    
    print("[CLI] Starting Batch Processing Mode...")
    
    try:
        # Initialize
        assessor = InsuranceClaimAssessor(yolo_model, severity_model)
        
        # Process specific file or directory
        if os.path.exists(input_dir):
            if os.path.isfile(input_dir):
                files = [input_dir]
            else:
                files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for f in files:
                try:
                    res = assessor.assess_claim(f)
                    print(assessor.generate_report(res))
                except Exception as e:
                    print(f"Error processing {f}: {e}")
        else:
            print(f"Input path not found: {input_dir}")
            
    except Exception as e:
        print(f"[CLI] Critical Execution Error: {e}")

if __name__ == "__main__":
    main()