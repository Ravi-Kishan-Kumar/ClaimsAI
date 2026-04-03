# 🛡️ ClaimsAI: Hybrid AI Insurance Claim Assessment System

An automated, edge-capable vehicle insurance claim processing system. This project utilizes a **Late-Fusion Hybrid Architecture**, combining the perceptive power of Deep Learning (YOLOv8 & ResNet18) with the deterministic safety of a Rule-Based Business Engine (CPIBP) to automate damage assessment and prevent fraud.

## 🌟 Key Features

* **Dual-AI Perception:** Uses **YOLOv8** for instance segmentation (detecting specific dents/scratches) and **ResNet18** for global severity classification (Minor, Moderate, Severe).
* **CPIBP Rule Engine:** (Confidence-Prioritized Inference with Business Policies). A deterministic logic layer that fuses AI probabilities to make safe, explainable business decisions (Auto-Approve, Manual Review, or Reject).
* **Integrity & Fraud Module:** Extracts and analyzes raw EXIF metadata (Software signatures, Timestamps) to flag digitally manipulated or recycled photos before processing.
* **Role-Based Portals (Streamlit):**
  * **User Dashboard:** For evidence upload and instant provisional assessment.
  * **Surveyor Dashboard:** A dedicated queue for manual overrides, complex claim reviews, and global audit history.

## 📐 System Architecture

The system operates on an offline-first, parallel inference pipeline:

```mermaid
graph TD
    classDef frontend fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef ai fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef logic fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef decision fill:#ffccbc,stroke:#d84315,stroke-width:2px,stroke-dasharray: 5 5;

    User[User Upload]:::frontend --> Pre[Preprocessing & PIL]:::process
    Pre --> EXIF[EXIF Integrity Check]:::process
    Pre --> Tensor[640px Input Tensor]:::process
    
    Tensor --> YOLO[YOLOv8 Detection]:::ai
    Tensor --> CNN[ResNet18 Severity]:::ai
    
    EXIF --> Rule[CPIBP Rule Engine]:::logic
    YOLO --> Rule
    CNN --> Rule
    
    Rule --> Auto[Auto-Approve]:::decision
    Rule --> Manual[Manual Review]:::decision
    Rule --> Reject[Reject Claim]:::decision

📁 Project Structure

insurance-claim-ai/
├── app/
│   └── main.py                  # Core backend pipeline execution
├── data/
│   └── fraud_store.json         # Local DB for fraud detection signatures
├── fraud/
│   └── fraud_detection.py       # EXIF extraction and metadata analysis
├── inference/
│   ├── severity_infer.py        # ResNet18 execution logic
│   └── yolo_infer.py            # YOLOv8 execution and bounding box logic
├── models/
│   ├── best (low mAP).pt        # YOLOv8 weights (Requires manual download)
│   └── severity_model.pth       # ResNet18 weights (Requires manual download)
├── rules/
│   └── cpibp_rules.py           # Business logic and decision fusion
├── streamlit_app.py             # Main UI with Secure Login & Dashboards
└── requirements.txt             # Python dependencies

⚙️ Installation & Local Setup
Important Note regarding AI Models: Due to GitHub file size limits, the trained weights (.pt and .pth files) are not included in this repository. You must place your trained models inside the models/ directory before running the application.

Clone the repository:

Bash
git clone [https://github.com/Ravi-Kishan-Kumar/ClaimsAI.git](https://github.com/Ravi-Kishan-Kumar/ClaimsAI.git)
cd ClaimsAI


Create a virtual environment (Recommended):

Bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate


Install dependencies:

Bash
pip install -r requirements.txt
Run the Application:

Bash
streamlit run streamlit_app.py
🔐 Demo Login Credentials
The prototype features a secure login portal to route users to the correct dashboard.

Customer / User Portal:

Username: user1

Password: 123

Admin / Surveyor Portal:

Username: surveyor

Password: admin


🚀 Future Scope

Mobile App Porting: Transitioning the Streamlit web app to a native mobile application for on-site accident reporting.

3D Damage Reconstruction: Upgrading from 2D image analysis to 360-degree video mesh generation for accurate depth and volume estimation.

Automated Cost Estimation: Linking detected part damages (e.g., "Bumper Dent") directly to an OEM parts and labor database for end-to-end financial settlement.