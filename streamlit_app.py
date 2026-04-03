"""
AI-Assisted Insurance Claim Assessment (Secure Login + Dashboards)
================================================================
Features:
- **Tabbed Surveyor Dashboard**: 
    1. Pending Reviews (Complex cases)
    2. Audit Auto-Approvals (Override AI decisions)
    3. Global History (Searchable log)
- **Metadata Viewer**: Inspect raw EXIF data.
- **User Filtering**: Filter claims by specific user IDs.
- **Comments**: Surveyor can write back to users.
- **Auto-Approval**: AI approves minor damage, but Surveyor can reject it later.
"""

import streamlit as st
import importlib.util
import os
import tempfile
import time
import uuid
from datetime import datetime
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="ClaimsAI Secure Portal", layout="wide")

# ==========================================
# 💾 DATABASE
# ==========================================
if 'user_db' not in st.session_state:
    st.session_state['user_db'] = {
        "user1": {"password": "123", "role": "User Dashboard"},
        "user2": {"password": "123", "role": "User Dashboard"},
        "surveyor": {"password": "admin", "role": "Surveyor Dashboard"}
    }

if 'claims_db' not in st.session_state:
    st.session_state['claims_db'] = []

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'current_user' not in st.session_state: st.session_state['current_user'] = None
if 'role' not in st.session_state: st.session_state['role'] = None

# Wizard States
if 'temp_analysis_result' not in st.session_state: st.session_state['temp_analysis_result'] = None
if 'wizard_step' not in st.session_state: st.session_state['wizard_step'] = 1

# ==========================================
# 🧠 BACKEND LOADING
# ==========================================
def _load_assessor():
    base = os.path.dirname(__file__)
    main_path = os.path.join(base, 'app', 'main.py')
    spec = importlib.util.spec_from_file_location('backend_main', main_path)
    backend_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend_main)
    return backend_main.InsuranceClaimAssessor

def save_upload(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    if not suffix: suffix = ".jpg"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

# ==========================================
# 🔐 AUTHENTICATION
# ==========================================
def auth_page():
    st.title("🔐 Secure Claims Portal")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            u = st.text_input("Username", key="l_u")
            p = st.text_input("Password", type="password", key="l_p")
            if st.button("Login"):
                db = st.session_state['user_db']
                if u in db and db[u]['password'] == p:
                    st.session_state['logged_in'] = True
                    st.session_state['current_user'] = u
                    st.session_state['role'] = db[u]['role']
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            st.caption("Demo: user1/123, surveyor/admin")

    with tab2:
        col1, col2 = st.columns([1, 2])
        with col1:
            nu = st.text_input("New User", key="r_u")
            np = st.text_input("New Pass", type="password", key="r_p")
            role = st.selectbox("Role", ["User (Customer)", "Surveyor (Admin)"], key="r_r")
            if st.button("Register"):
                if nu and np:
                    st.session_state['user_db'][nu] = {
                        "password": np, 
                        "role": "User Dashboard" if "User" in role else "Surveyor Dashboard"
                    }
                    st.success("Registered! Login now.")

# ==========================================
# 🏠 MAIN DASHBOARD
# ==========================================
def main_dashboard():
    try:
        InsuranceClaimAssessor = _load_assessor()
        assessor = InsuranceClaimAssessor("models/best.pt", "models/severity_model.pth")
    except Exception as e:
        st.error(f"Backend Error: {e}")
        st.stop()

    with st.sidebar:
        st.title("ClaimsAI")
        st.write(f"Logged in as: **{st.session_state['current_user']}**")
        
        if st.session_state['role'] == "Surveyor Dashboard":
            pending = sum(1 for c in st.session_state['claims_db'] if c['status'] == 'Pending Surveyor Review')
            auto = sum(1 for c in st.session_state['claims_db'] if c['status'] == 'Approved (Auto)')
            st.info(f"Queue: {pending} Pending")
            st.success(f"Auto-Approved: {auto}")
        
        st.markdown("---")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['temp_analysis_result'] = None
            st.session_state['wizard_step'] = 1
            st.rerun()

    # ==================================================
    # 👤 USER DASHBOARD
    # ==================================================
    if st.session_state['role'] == "User Dashboard":
        tab_new, tab_history = st.tabs(["📝 File New Claim", "🗂️ My Claims History"])
        
        # --- TAB 1: FILE NEW CLAIM ---
        with tab_new:
            st.header("File a New Claim")
            
            # Step 1: Upload
            st.subheader("Step 1: Evidence Upload & Analysis")
            uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "png", "jpeg"])
            
            if uploaded_file and st.button("🔍 Run AI Analysis", type="primary"):
                img_path = save_upload(uploaded_file)
                with st.spinner("AI assessing damage..."):
                    try:
                        result = assessor.assess_claim(img_path)
                        st.session_state['temp_analysis_result'] = result
                        st.session_state['wizard_step'] = 2
                    except Exception as e:
                        st.error(f"Analysis Failed: {e}")

            # Step 2: Review & Submit
            if st.session_state['wizard_step'] == 2 and st.session_state['temp_analysis_result']:
                result = st.session_state['temp_analysis_result']
                decision = result['decision']
                action = decision['action']
                
                st.markdown("---")
                st.subheader("Step 2: Review & Finalize")

                c1, c2 = st.columns(2)
                with c1:
                    st.image(result['image_path'], caption="Uploaded Evidence", use_container_width=True)
                with c2:
                    if result['annotated_image'] is not None:
                        st.image(result['annotated_image'], caption="AI Detection Overlay", use_container_width=True)
                    else:
                        st.write("No damage detected.")

                st.info(f"**AI Assessment:** {decision['explanation']}")
                
                # Determine Status based on Action
                if action == 'rejected_fraud': 
                    display_status = "Rejected (Integrity)"
                elif action == 'approved_auto': 
                    display_status = "Approved (Auto)" 
                else: 
                    display_status = "Pending Surveyor Review"

                st.markdown(f"**Projected Status:** `{display_status}`")

                st.markdown("---")
                user_comment = st.text_area("Additional Comments:", placeholder="Describe the incident...")

                if st.button("🚀 Submit Final Claim"):
                    claim_record = {
                        "id": str(uuid.uuid4())[:8],
                        "user": st.session_state['current_user'],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "status": display_status,
                        "ai_result": result,
                        "user_comment": user_comment,
                        "final_decision_notes": "Awaiting final confirmation." if action == 'approved_auto' else ""
                    }
                    st.session_state['claims_db'].append(claim_record)
                    
                    if action == 'rejected_fraud': st.error("Claim Rejected due to integrity check.")
                    elif action == 'approved_auto': 
                        st.balloons()
                        st.success("Claim Provisionally Approved! (Subject to final surveyor review within 3 days).")
                    else: st.success("Claim submitted for review.")

                    time.sleep(2)
                    st.session_state['temp_analysis_result'] = None
                    st.session_state['wizard_step'] = 1
                    st.rerun()

        # --- TAB 2: HISTORY ---
        with tab_history:
            st.header("My Claims History")
            my_claims = [c for c in st.session_state['claims_db'] if c['user'] == st.session_state['current_user']]
            if not my_claims: st.info("No history.")
            for claim in reversed(my_claims):
                with st.expander(f"{claim['timestamp']} - #{claim['id']} - {claim['status']}"):
                    st.write(f"**Status:** {claim['status']}")
                    if claim['final_decision_notes']: 
                        st.info(f"**Surveyor Message:** {claim['final_decision_notes']}")

    # ==================================================
    # 🕵️ SURVEYOR DASHBOARD
    # ==================================================
    elif st.session_state['role'] == "Surveyor Dashboard":
        # THREE TABS FOR SURVEYOR
        tab_pending, tab_auto, tab_all_history = st.tabs([
            "⚠️ Pending Reviews", 
            "✅ Audit Auto-Approvals", 
            "📜 Global History Log"
        ])
        
        # --- TAB 1: PENDING REVIEWS (Standard Workflow) ---
        with tab_pending:
            st.header("Pending Reviews (Complex/Severe)")
            pending_claims = [c for c in st.session_state['claims_db'] if c['status'] == 'Pending Surveyor Review']
            
            if not pending_claims:
                st.success("No pending complex claims.")
            else:
                for claim in pending_claims:
                    with st.container():
                        st.markdown(f"**User:** `{claim['user']}` | **Claim #{claim['id']}**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # Show Original
                            orig = claim['ai_result']['image_path']
                            if os.path.exists(orig): st.image(orig, use_container_width=True)
                            
                            # Expandable Metadata
                            with st.expander("🔍 View Image Metadata"):
                                integ = claim['ai_result']['integrity']
                                st.write(f"**Status:** {integ['status']}")
                                st.caption(integ['reason'])
                                if 'raw_metadata' in integ and integ['raw_metadata']:
                                    st.write("---")
                                    for k, v in integ['raw_metadata'].items():
                                        st.text(f"{k}: {v}")
                                else:
                                    st.caption("No readable EXIF tags found.")
                                
                        with col2:
                            # Show AI
                            over = claim['ai_result']['annotated_image']
                            if over is not None: st.image(over, use_container_width=True)
                            st.caption(f"AI Reason: {claim['ai_result']['decision']['explanation']}")
                            st.info(f"User Comment: {claim['user_comment']}")

                        # Decision Area
                        surveyor_comment = st.text_input("Reason / Message to User:", key=f"comm_{claim['id']}")
                        c1, c2 = st.columns([1, 1])
                        
                        if c1.button("Approve", key=f"app_{claim['id']}"):
                            claim['status'] = "Approved (Manual)"
                            claim['final_decision_notes'] = surveyor_comment or "Approved by Surveyor."
                            st.success("Approved.")
                            time.sleep(1)
                            st.rerun()
                            
                        if c2.button("Reject", key=f"rej_{claim['id']}"):
                            claim['status'] = "Rejected (Manual)"
                            claim['final_decision_notes'] = surveyor_comment or "Rejected by Surveyor."
                            st.error("Rejected.")
                            time.sleep(1)
                            st.rerun()
                        st.divider()

        # --- TAB 2: AUDIT AUTO-APPROVALS (3-Day Override) ---
        with tab_auto:
            st.header("Audit Auto-Approved Claims")
            st.markdown("These claims met the 'Valid Metadata + Minor Damage' criteria. You can override them here.")
            
            auto_claims = [c for c in st.session_state['claims_db'] if c['status'] == 'Approved (Auto)']
            
            if not auto_claims:
                st.info("No auto-approved claims to audit.")
            else:
                for claim in auto_claims:
                    with st.expander(f"User: {claim['user']} | Claim #{claim['id']} | {claim['timestamp']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(claim['ai_result']['image_path'], width=200, caption="Evidence")
                            st.caption(f"User Note: {claim['user_comment']}")
                        with col2:
                            st.write(f"**AI Logic:** {claim['ai_result']['decision']['explanation']}")
                            
                        # Override Logic
                        st.markdown("**Override Decision:**")
                        override_reason = st.text_input("Rejection Reason (Required for Override):", key=f"over_comm_{claim['id']}")
                        
                        if st.button("❌ Reject / Overturn Approval", key=f"over_{claim['id']}"):
                            if not override_reason:
                                st.error("You must provide a reason to overturn an auto-approval.")
                            else:
                                claim['status'] = "Rejected (Overturned)"
                                claim['final_decision_notes'] = f"Auto-Approval Overturned by Surveyor. Reason: {override_reason}"
                                st.success("Claim Rejected.")
                                time.sleep(1)
                                st.rerun()

        # --- TAB 3: GLOBAL HISTORY LOG ---
        with tab_all_history:
            st.header("Global Claims History")
            
            # Filter Options
            users_list = sorted(list(set(c['user'] for c in st.session_state['claims_db'])))
            users_list.insert(0, "All Users")
            filter_user = st.selectbox("Filter by User:", users_list)
            
            history_claims = st.session_state['claims_db']
            if filter_user != "All Users":
                history_claims = [c for c in history_claims if c['user'] == filter_user]
            
            # Table View
            if history_claims:
                st.dataframe([
                    {
                        "ID": c['id'],
                        "User": c['user'],
                        "Date": c['timestamp'],
                        "Status": c['status'],
                        "Surveyor Notes": c['final_decision_notes']
                    }
                    for c in history_claims
                ], use_container_width=True)
            else:
                st.info("No records found.")

if st.session_state['logged_in']:
    main_dashboard()
else:
    auth_page()