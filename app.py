# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 21:58:03 2025
pMLISS

@author: mdcob
"""
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re

# ---------- Page setup ----------
st.set_page_config(page_title="RT-MLISS", layout="centered")
st.markdown("""
<style>
.block-container { max-width: 1100px; padding: 2rem 4rem; }
html, body, [class*="css"] { font-size: 1.10rem; }
input, select, textarea { font-size: 1.0rem !important; }
label { margin-bottom: 0.2rem !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Load artifacts (cached) ----------
@st.cache_resource
def load_artifacts():
    with open("calibrated_model_pedi_mor.pkl", "rb") as f:
        model_mor = pickle.load(f)
    with open("calibrated_model_pedi_icu", "rb") as f:
        model_icu = pickle.load(f)
    with open("scaler_pedi.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ---------- Title & intro ----------
st.title("RT-MLISS Score")
st.markdown("""
<h4 style='margin-top:-8px;color:gray;'>A real-time mortality and ICU admission prediction tool for pediatric trauma patients</h4>
<p style='font-size:16px;color:#555;'>
Developed by <b>MD Cobler-Lichter MD MSDS</b>, N Jelen MD, JM Delamater MD MPH, 
TR Arcieri MD, AM Reyes MD MPH, JP Meizoso MD MSPH FACS, N Namias MD MBA FACS, KG Proctor PhD, 
CI Schulman MD PhD MSPH, CM Thorson MD MSPH
</p>
<p style='font-size:14px;color:#555;'>
Models are calibrated so the output reflects the predicted probability of in-hospital mortality and need for ICU admission 
(planned or unplanned) based on the training data.
</p>
""", unsafe_allow_html=True)

# ---------- Helpers (live inputs; namespaced keys; no reset on rerun) ----------
def int_input_live(label, key, min_val=None, max_val=None, placeholder=""):
    raw_key = f"numraw_{key}"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = ""
    raw = st.text_input(label, key=raw_key, placeholder=placeholder).strip()
    if raw == "":
        return np.nan
    if re.fullmatch(r"\d+", raw):
        v = int(raw)
        if (min_val is not None and v < min_val) or (max_val is not None and v > max_val):
            return np.nan
        return v
    return np.nan

def float_input_live(label, key, min_val=None, max_val=None, placeholder=""):
    raw_key = f"numraw_{key}"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = ""
    raw = st.text_input(label, key=raw_key, placeholder=placeholder).strip()
    if raw == "":
        return np.nan
    if re.fullmatch(r"\d+(\.\d+)?", raw):
        v = float(raw)
        if (min_val is not None and v < min_val) or (max_val is not None and v > max_val):
            return np.nan
        return v
    return np.nan

# ---------- Labels / bounds ----------
label_map = {
    'TRAUMATYPE': "Trauma Type",
    'AGEYEARS': "Age (Years)",
    'TOTALGCS': "Arrival GCS",
    'SBP': "Arrival SBP",
    'TEMPERATURE': "Arrival Temperature (°C)",
    'PULSERATE': "Arrival Heart Rate",
    'WEIGHT': "Weight (kg)"
}

bounds = {
    'AGEYEARS': (0, 150),
    'TOTALGCS': (3, 15),
    'SBP': (0, 360),
    'TEMPERATURE': (0, 93),     # °C
    'PULSERATE': (0, 320),
    'WEIGHT': (1, 500),
}

frontend_labels = {
    "Epidural Hematoma (EDH)": "EDH",
    "Subarachnoid Hemorrhage (SAH)": "SAH",
    "Subdural Hematoma (SDH)": "SDH",
    "Skull Fracture": "SkullFx",
    "Intraparenchymal Hemorrhage (IPH)": "IPH",
    "Spinal Cord Injury (SCI)": "SCI",
    "Spine Fracture": "SpineFx",
    "Cardiac Injury": "CardiacInjury",
    "Lung Injury": "LungInjury",
    "Rib Fracture": "RibFx",
    "Kidney Injury": "KidneyInjury",
    "Spleen Injury": "SpleenInjury",
    "Pelvic Fracture": "PelvicFx",
    "Liver Injury": "LiverInjury",
    "Upper Extremity Long Bone Fracture": "UELongBoneFx",
    "Lower Extremity Long Bone Fracture": "LELongBoneFx",
    "Hollow Viscus Injury": "HollowViscusInjury",
    "Peripheral Vascular Injury": "PeripheralVascularInjury",
    "Critical Neck Injury": "NeckCriticalInjury",
    "Chest/abdomen/pelvis Vascular Injury": "CAPVascularInjury"
}

injury_categories_display = {
    "Head injury?": [
        "Epidural Hematoma (EDH)", "Intraparenchymal Hemorrhage (IPH)",
        "Subarachnoid Hemorrhage (SAH)", "Subdural Hematoma (SDH)", "Skull Fracture"
    ],
    "Neck/Back injury?": [
        "Critical Neck Injury", "Spinal Cord Injury (SCI)", "Spine Fracture"
    ],
    "Thoracic/Abdominal/Pelvic injury?": [
        "Kidney Injury", "Spleen Injury", "Hollow Viscus Injury"
        "Pelvic Fracture", "Liver Injury",
        "Cardiac Injury", "Lung Injury", "Rib Fracture",
        "Chest/abdomen/pelvis Vascular Injury"
    ],
    "Extremity injury?": [
        "Upper Extremity Long Bone Fracture",
        "Lower Extremity Long Bone Fracture",
        "Peripheral Vascular Injury"
    ]
}

# ---------- Injury Pattern (OUTSIDE the form so it updates instantly) ----------
st.subheader("Injury Pattern")
injury_inputs = {}
for region_label, subqs in injury_categories_display.items():
    has_injury = st.radio(region_label, ['No', 'Yes'], index=0, horizontal=True,
                          key=f"region_{region_label}")
    if has_injury == 'Yes':
        with st.expander(f"Specify injuries for {region_label.replace(' injury?', '')}", expanded=False):
            for disp in subqs:
                backend_var = frontend_labels[disp]
                picked = st.radio(disp, ['No','Yes'], index=0, horizontal=True,
                                  key=f"inj_{backend_var}")
                injury_inputs[backend_var] = 1 if picked == 'Yes' else 0
    else:
        for disp in subqs:
            backend_var = frontend_labels[disp]
            injury_inputs[backend_var] = 0

# ---------- Patient Info & Vitals + Predict BUTTON (INSIDE a form) ----------
with st.form("rtmliss_form", clear_on_submit=False):
    user_inputs = {}
    sbp_val = np.nan
    pulse_val = np.nan

    col1, _, col2 = st.columns([2, 1, 2])

    with col1:
        st.subheader("Patient Info & Vitals")
        trauma_type = st.radio(label_map['TRAUMATYPE'], ['Blunt', 'Penetrating'],
                               index=0, horizontal=True, key='TRAUMATYPE')
        user_inputs['Penetrating'] = 1 if trauma_type == 'Penetrating' else 0

        for var in ['AGEYEARS','TOTALGCS','SBP','TEMPERATURE','PULSERATE','WEIGHT']:
            lo, hi = bounds[var]
            if var == 'TEMPERATURE':
                val = float_input_live(label_map[var], var, min_val=lo, max_val=hi)
            else:
                val = int_input_live(label_map[var], var, min_val=lo, max_val=hi)
            user_inputs[var] = val
            if var == 'SBP': sbp_val = val
            if var == 'PULSERATE': pulse_val = val

        # ShockIndex
        if (isinstance(sbp_val, (int, float)) and isinstance(pulse_val, (int, float))
            and not np.isnan(sbp_val) and not np.isnan(pulse_val) and sbp_val != 0):
            user_inputs['ShockIndex'] = pulse_val / sbp_val
        elif sbp_val == 0 or pulse_val == 0:
            user_inputs['ShockIndex'] = 2.0
        else:
            user_inputs['ShockIndex'] = np.nan

    with col2:
        st.subheader("Summary")
        user_inputs['NumberOfInjuries'] = int(sum(injury_inputs.values()))
        st.write(f"Number of injuries selected: **{user_inputs['NumberOfInjuries']}**")

    # Merge injury inputs inside the form (so X is complete at submit time)
    user_inputs.update(injury_inputs)

    # Build X in model's expected order (NaNs allowed)
    X = None
    try:
        X = pd.DataFrame([user_inputs], columns=scaler.feature_names_in_)
    except Exception as e:
        st.error(f"Column alignment error: {e}")

    submitted = st.form_submit_button("Predict Mortality")

# ---------- Output (persist last prediction) ----------
st.markdown("### RT-pMLISS Score (Predicted Mortality):")
st.markdown("### RT-pMLISS ICU Subscore (Predicted need for ICU):")
mortality_output = st.empty()
icu_output = st.empty()

if 'last_pred_mor' not in st.session_state:
    st.session_state['last_pred_mor'] = None
    st.session_state['last_pred_icu'] = None

if submitted and X is not None:
    try:
        X_scaled = scaler.transform(X)
        pred_mor = float(model_mor.predict_proba(X_scaled)[:, 1][0])
        pred_icu = float(model_icu.predict_proba(X_scaled)[:, 1][0])
        st.session_state['last_pred_mor'] = pred_mor
        st.session_state['last_pred_icu'] = pred_icu
    except Exception as e:
        st.session_state['last_pred_mor'] = None
        mortality_output.error(f"Error during prediction: {e}")

if st.session_state['last_pred_mor'] is not None:
    mortality_output.markdown(
        f"<p style='font-size:36px;font-weight:bold;color:#d62728;'>{st.session_state['last_pred_mor']:.1%}</p>",
        unsafe_allow_html=True
    )
    icu_output.markdown(
        f"<p style='font-size:36px;font-weight:bold;color:#d62728;'>{st.session_state['last_pred_icu']:.1%}</p>",
        unsafe_allow_html=True
    )

# ---------- Reset (only clears our own keys) ----------
if st.button("Reset Form"):
    keys_to_clear = [k for k in st.session_state.keys()
                     if k.startswith("numraw_")
                     or k.startswith("region_")
                     or k.startswith("inj_")
                     or k in ['TRAUMATYPE','last_pred_mor']]
    for k in list(set(keys_to_clear)):
        del st.session_state[k]
    st.rerun()




