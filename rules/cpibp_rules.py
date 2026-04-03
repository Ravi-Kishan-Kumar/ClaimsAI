"""
CPIBP (Claim Processing Intelligent Business Process) Rule Engine
==================================================================
Purpose: Make deterministic, explainable decisions on insurance claims.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ClaimDecision:
    """Structure for claim decision output."""
    decision: str          # 'approved_auto', 'surveyor_review', 'rejected_fraud'
    confidence: float      # Overall confidence in decision
    explanation: str       # Human-readable reasoning
    triggered_rules: List[str]  # Which rules triggered this decision


class CPIBPRuleEngine:
    """
    Rule-based decision engine for insurance claims.
    Now includes Duplicate/Fraud Risk Rules.
    """
    
    def __init__(self):
        self.SEVERE_THRESHOLD = 0.6
        self.HIGH_CONF_THRESHOLD = 0.7
    
    def process_claim(
        self,
        yolo_summary: Dict,
        severity_prediction: Dict,
        integrity: Dict,
        fraud: Dict
    ) -> ClaimDecision:
        """
        Make a claim decision based on all available signals:
        - YOLO (Visual Damage)
        - CNN (Severity)
        - Integrity (Metadata)
        - Fraud (Duplicate History)
        """
        
        # --- RULE 0A: Metadata Fraud (RED) ---
        # Tier 3 check failed (e.g. "Nano Banana", Photoshop)
        if integrity['flag']:
            return ClaimDecision(
                decision='rejected_fraud',
                confidence=1.0,
                explanation=f"Integrity Check Failed: {integrity['reason']}",
                triggered_rules=['Rule 0A: Metadata/Fraud Check Failed']
            )

        # --- RULE 0B: Duplicate Claim Fraud (RED) ---
        # High similarity to a past claim (e.g. 95% same box layout)
        if fraud['fraud_risk'] == 'HIGH':
            return ClaimDecision(
                decision='rejected_fraud',
                confidence=0.98,
                explanation=f"Duplicate Claim Detected! {fraud['reason']}",
                triggered_rules=['Rule 0B: High Fraud Risk (Duplicate)']
            )
            
        # --- RULE 0C: Medium Fraud Risk (ORANGE) ---
        # Suspicious similarity, force manual review
        if fraud['fraud_risk'] == 'MEDIUM':
            return ClaimDecision(
                decision='surveyor_review',
                confidence=0.85,
                explanation=f"Suspicious Pattern: {fraud['reason']}. Manual check required.",
                triggered_rules=['Rule 0C: Fraud Risk (Medium)']
            )
            
        # --- RULE 0.5: Unverified Source (ORANGE) ---
        # Missing metadata or filename issues
        if integrity.get('verification_required', False):
            return ClaimDecision(
                decision='surveyor_review',
                confidence=0.80,
                explanation=f"Source Unverified: {integrity['reason']}. Surveyor must validate authenticity.",
                triggered_rules=['Rule 0.5: Unverified Image Source']
            )
        
        # --- RULE 1: No damage detected ---
        if yolo_summary['damage_count'] == 0:
            return ClaimDecision(
                decision='surveyor_review', # Send to surveyor to confirm "No Damage" isn't an error
                confidence=0.60,
                explanation="No visual damage detected by AI. Manual verification recommended.",
                triggered_rules=['Rule 1: No Damage Detected']
            )
        
        # --- RULE 2: Critical/Severe Damage -> Surveyor (Orange) ---
        if self._rule_critical_or_severe(yolo_summary, severity_prediction):
            return self._apply_rule_refer_surveyor(yolo_summary, severity_prediction)
        
        # --- RULE 3: Low Confidence -> Manual Review ---
        if severity_prediction['confidence'] < 0.5:
             return ClaimDecision(
                decision='surveyor_review',
                confidence=0.5,
                explanation=f"Low model confidence ({severity_prediction['confidence']:.2f}). Manual review required.",
                triggered_rules=['Rule 3: Low Confidence']
            )
            
        # --- RULE 4: Minor Damage + Valid Metadata -> Auto-Approve (Green) ---
        # "Auto approve claims which has valid metadata and damage is found"
        # Must pass all fraud checks (Risk=LOW) and have Valid Integrity
        if severity_prediction['severity'] != 'severe':
            return ClaimDecision(
                decision='approved_auto', # Marked distinctively for "Review Auto-Approvals" tab
                confidence=0.85,
                explanation=f"Valid Metadata & Minor Damage ({yolo_summary['damage_types_list']}). Auto-Approved (Subject to 3-day review).",
                triggered_rules=['Rule 4: Standard Auto-Approval']
            )

        # Default fallback
        return ClaimDecision(
            decision='surveyor_review',
            confidence=0.5,
            explanation="Complex damage pattern. Forwarding to surveyor.",
            triggered_rules=['Default: Manual Review']
        )
    
    # --- Helper Rule Conditions ---
    
    def _rule_critical_or_severe(self, yolo: Dict, severity: Dict) -> bool:
        """Check for severe conditions."""
        critical_types = {'broken_part', 'cracked_glass', 'major_dent'}
        detected_critical = set(yolo['damage_types_list']) & critical_types
        is_severe_cnn = (severity['severity'] == 'severe' and severity['confidence'] > 0.6)
        return bool(detected_critical) or is_severe_cnn

    def _apply_rule_refer_surveyor(self, yolo: Dict, severity: Dict) -> ClaimDecision:
        return ClaimDecision(
            decision='surveyor_review',
            confidence=0.90,
            explanation="Critical damage types or High Severity detected. Professional surveyor required.",
            triggered_rules=['Critical/Severe Damage']
        )