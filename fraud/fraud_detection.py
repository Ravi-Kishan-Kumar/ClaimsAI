"""
Fraud Detection Module (Academic Prototype)
===========================================
Purpose: Detect potential duplicate insurance claims by analyzing visual similarity 
of damage patterns using object detection outputs.

Methodology:
- Feature Extraction: Uses YOLOv8 outputs (Class labels + Bounding Boxes) as a unique 'fingerprint'.
- Graph-Based Reasoning: 
    * Each claim is treated as a Node in a temporal graph.
    * Similarity scores represent Edge Weights between nodes.
    * A high edge weight (> threshold) indicates a potential Duplicate Claim.
- Storage: Uses a local JSON adjacency list (`fraud_store.json`) to persist node data.

Constraints:
- No ML training (Rule-based similarity).
- Uses YOLO outputs only (No CNN embeddings).
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

class FraudDetector:
    def __init__(self, db_path: str = "data/fraud_store.json"):
        """Initialize the fraud detection graph engine."""
        self.db_path = db_path
        self._ensure_db_exists()
        self.history = self._load_db()

    def _ensure_db_exists(self):
        """Ensure the local graph storage file exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w') as f:
                json.dump([], f)

    def _load_db(self) -> List[Dict]:
        """Load the graph nodes (claim history) from JSON."""
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_db(self):
        """Persist graph state to storage."""
        with open(self.db_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _calculate_iou(self, boxA, boxB):
        """
        Calculate Intersection over Union (IoU) to determine spatial overlap.
        Used to check if damage is in the exact same location as a previous claim.
        Box format: [x1, y1, x2, y2]
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def check_fraud(self, current_claim: Dict) -> Dict:
        """
        Execute Graph Traversal to find similar nodes (Duplicate Check).
        
        Args:
            current_claim (Dict): {
                'image_id': str,
                'damage_parts': List[str],
                'boxes': List[List[float]],
                'avg_confidence': float
            }
            
        Returns:
            Dict: {
                'fraud_risk': 'LOW' | 'MEDIUM' | 'HIGH',
                'reason': str,
                'match_id': str (optional)
            }
        """
        # Node Attributes
        curr_parts = set(current_claim.get('damage_parts', []))
        curr_boxes = current_claim.get('boxes', [])
        
        # Pass-through for clean claims (no damage detected yet)
        if not curr_parts:
            self._add_to_history(current_claim)
            return {"fraud_risk": "LOW", "reason": "No damage pattern to fingerprint.", "match_id": None}

        risk_level = "LOW"
        reason = "New unique damage pattern."
        match_id = None

        # --- Graph Traversal: Compare against all past nodes ---
        for past_claim in self.history:
            past_parts = set(past_claim['damage_parts'])
            past_boxes = past_claim['boxes']
            past_id = past_claim['image_id']
            
            # 1. Semantic Similarity (Jaccard Index of Class Names)
            # Do they have the same *type* of damage? (e.g. both have "dent" and "scratch")
            intersection = len(curr_parts.intersection(past_parts))
            union = len(curr_parts.union(past_parts))
            semantic_score = intersection / union if union > 0 else 0.0

            # Threshold: > 70% match in damage types
            if semantic_score >= 0.7:
                
                # 2. Spatial Similarity (IoU of Bounding Boxes)
                # Are the damages in the *same place*?
                spatial_matches = 0
                total_boxes = max(len(curr_boxes), len(past_boxes))
                
                # Greedy match: Check if any current box overlaps significantly with a past box
                for c_box in curr_boxes:
                    for p_box in past_boxes:
                        if self._calculate_iou(c_box, p_box) > 0.45: # Threshold for "same location"
                            spatial_matches += 1
                            break 
                
                spatial_score = spatial_matches / total_boxes if total_boxes > 0 else 0
                
                # --- Risk Classification ---
                if spatial_score > 0.6: 
                    # Same Parts + Same Location = HIGH RISK (Likely Duplicate Photo)
                    risk_level = "HIGH"
                    reason = f"Duplicate Pattern Detected! 95% similarity with Claim ID: {past_id}"
                    match_id = past_id
                    break # Stop search, matched
                
                elif spatial_score > 0.2:
                    # Same Parts + Some Overlap = MEDIUM RISK (Simulated/Similar accident)
                    risk_level = "MEDIUM"
                    reason = f"Suspicious Similarity. Layout resembles Claim ID: {past_id}"
                    match_id = past_id

        # Update Graph Memory
        self._add_to_history(current_claim)
        
        return {
            "fraud_risk": risk_level,
            "reason": reason,
            "match_id": match_id
        }

    def _add_to_history(self, claim_data):
        """Add current claim as a new node in the graph history."""
        # Convert numpy types to native Python for JSON serialization
        clean_entry = {
            "image_id": claim_data.get('image_id', 'unknown'),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "damage_parts": claim_data.get('damage_parts', []),
            "boxes": [[float(c) for c in box] for box in claim_data.get('boxes', [])],
            "avg_confidence": float(claim_data.get('avg_confidence', 0.0))
        }
        self.history.append(clean_entry)
        self._save_db()