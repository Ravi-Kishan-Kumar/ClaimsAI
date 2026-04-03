"""
Streamlit demo UI for the Insurance Claim Assessment backend
============================================================
- This app is a demonstration layer that CALLS the existing backend only.
- It does NOT implement any AI logic itself; it uses the frozen backend API:
    `app.main.InsuranceClaimAssessor.assess_claim(image_path)`

Usage:
  pip install streamlit pillow
  streamlit run app.py

Design notes:
- Temporary backend validation UI. In future, the UI will call the same
  `assess_claim()` function; no changes to backend are required.
"""

import streamlit as st
from PIL import Image
import tempfile
import os
import shutil

# Import the stable backend assessor (does not execute pipeline on import)
from app.main import InsuranceClaimAssessor

# Model paths (must match files in `models/`)
YOLO_MODEL = "models/best.pt"
SEVERITY_MODEL = "models/severity_model.pth"

# --- Page layout ---
st.set_page_config(page_title="Insurance Claim Demo", layout="centered")
st.title("Insurance Claim Assessment — Demo")
st.markdown("""
This is a prototype demo that calls a frozen backend (YOLO + CNN + Rule Engine).
- The AI engine runs on uploaded images and returns an explainable decision.
- Fraud/consistency checks are experimental proof-of-concept only.
""")

st.markdown("**Prototype disclaimers:**")
st.info("This system is a prototype and does not replace human surveyors.")
st.warning("Fraud checks are experimental and indicative only.")

# --- Sidebar: RC POC and model info ---
st.sidebar.header("Model & POC options")
st.sidebar.write("Models loaded from project `models/` folder.")
st.sidebar.write(f"YOLO: {YOLO_MODEL}")
st.sidebar.write(f"Severity: {SEVERITY_MODEL}")
st.sidebar.markdown("---")
st.sidebar.markdown("**RC Consistency (POC)**: filename-based mock extraction by default.")

# --- Upload section ---
st.header("Upload")
vehicle_file = st.file_uploader("Upload vehicle damage image", type=["jpg","jpeg","png","bmp"], accept_multiple_files=False)
rc_file = st.file_uploader("Upload Registration Certificate (RC) image (optional)", type=["jpg","jpeg","png","pdf"], accept_multiple_files=False)

st.text_input("(Optional) Enter vehicle number for RC verification", key="vehicle_number")
vehicle_number_input = st.session_state.get("vehicle_number", "").strip()

# Initialize assessor lazily so app loads quickly; show clear error if models missing
assessor = None
try:
    assessor = InsuranceClaimAssessor(YOLO_MODEL, SEVERITY_MODEL)
except Exception as e:
    st.error(f"Backend models could not be loaded: {e}")
    st.stop()

# --- Helper: save uploaded file to temp path ---
def _save_uploaded(uploaded_file) -> str:
    if uploaded_file is None:
        return None
    suffix = os.path.splitext(uploaded_file.name)[1]
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

# --- Action button ---
if st.button("Assess Claim"):
    if vehicle_file is None:
        st.error("Please upload a vehicle damage image before assessing the claim.")
    else:
        # Save vehicle image to temp file
        img_path = _save_uploaded(vehicle_file)
        rc_path = _save_uploaded(rc_file) if rc_file is not None else None

        try:
            # Call the BACKEND pipeline (frozen logic)
            assessment = assessor.assess_claim(img_path)

            # Display results
            st.header("Analysis")
            st.subheader("Original Image")
            st.image(Image.open(img_path), use_column_width=True)

            # Detected damages
            st.subheader("Detected Damages")
            class_names = assessment['yolo_detections'].get('class_names', [])
            if class_names:
                st.write(", ".join(class_names))
            else:
                st.write("No damages detected")

            # Severity
            st.subheader("Severity Prediction")
            sev = assessment['severity_prediction']
            st.write(f"Label: **{sev['severity'].upper()}**")
            st.write(f"Confidence: **{sev['confidence']:.2%}**")
            st.write("Probability distribution:")
            for k, v in sev['probabilities'].items():
                st.write(f"- {k}: {v:.2%}")

            # Decision
            st.subheader("CPIBP Decision")
            decision = assessment['decision']
            action = decision['action']
            explanation = decision['explanation']
            if action == 'auto-approve':
                st.success(f"Decision: {action.upper()}")
            elif action == 'refer-to-surveyor':
                st.error(f"Decision: {action.upper()}")
            else:
                st.warning(f"Decision: {action.upper()}")
            st.markdown("**Explanation:**")
            st.write(explanation)

            # RC POC: filename-based mock extraction
            st.subheader("RC Consistency Check — Experimental (Proof of Concept)")
            if rc_path is None:
                st.write("No RC image provided.")
                st.info("Upload an RC image to run a filename-based consistency mock check.")
            else:
                # Mock extraction: use filename (without extension) as vehicle number
                extracted = os.path.splitext(os.path.basename(rc_file.name))[0]
                st.write(f"RC filename (mock extracted number): **{extracted}**")

                # Compare with user-provided number if available
                if vehicle_number_input:
                    match = "YES" if vehicle_number_input.lower() in extracted.lower() else "NO"
                    st.write(f"User-entered number: {vehicle_number_input}")
                    st.write(f"RC Match: **{match}**")
                else:
                    st.write("No user-entered vehicle number provided.")
                    st.write("RC Match: UNABLE TO VERIFY")

                st.caption("Experimental – Proof of Concept: filename-based mock extraction only")

        except Exception as e:
            st.error(f"Error during assessment: {e}")
        finally:
            # Clean up temp files
            try:
                if img_path and os.path.exists(img_path):
                    os.remove(img_path)
                if rc_path and os.path.exists(rc_path):
                    os.remove(rc_path)
            except Exception:
                pass

# Footer note
st.markdown("---")
st.caption("Prototype UI for academic demonstration. Backend models and decision logic are frozen and not modified by this UI.")