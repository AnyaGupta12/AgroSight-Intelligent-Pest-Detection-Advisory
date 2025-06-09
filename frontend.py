# frontend.py
import streamlit as st
from PIL import Image
import sys
import types
import os
import base64
from io import BytesIO

def image_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode("utf-8")

# Patch Streamlit watcher issue with torch.classes
sys.modules["torch.classes"] = types.ModuleType("torch.classes")
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

from backend import detect_pest, get_advice

# UI Setup
st.set_page_config(page_title="Pest Detector & Advisor", layout="wide")
st.title(" Pest Detection & Advisory System")

# Image uploader
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])  

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.subheader("Original Image")
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{image_to_base64(image.resize((600, int(image.height * 600 / image.width))))}" alt="Uploaded Image"/>
        </div>
        """,
        unsafe_allow_html=True
    )
    

    # Detection
    with st.spinner('Running pest detection...'):
        annotated, pest = detect_pest(image)
        
    

    if pest:
            # Fetch treatment advice silently
            advice = get_advice(pest)

            # Display nicely organized results
            st.markdown("---")
            st.markdown(f"<h3 style='text-align: center; color: white;'> Detected Pest: <span style='color: lightgreen'>{pest}</span></h3>", unsafe_allow_html=True)

            # Centered section heading
            st.markdown("<h3 style='text-align: center; color: white;'> Treatment & Prevention Advice</h3>", unsafe_allow_html=True)

            # Display common names
            st.markdown("<div style='text-align: center; color: white; font-size:16px;'>"
                        f"<strong>Common Name (EN):</strong> {advice.get('common_name_en', 'N/A')}<br>"
                        f"<strong>Common Name (HI):</strong> {advice.get('common_name_hi', 'N/A')}"
                        "</div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Chemical Solutions")
                st.markdown(f"<div style='background-color:#1e293b; color: white; padding:10px; border-radius:10px'>{advice.get('chemical', 'N/A')}</div>", unsafe_allow_html=True)

                st.markdown("#### Organic Solutions")
                st.markdown(f"<div style='background-color:#1e293b; color: white; padding:10px; border-radius:10px'>{advice.get('organic', 'N/A')}</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("#### Prevention Strategies")
                st.markdown(f"<div style='background-color:#1e293b; color: white; padding:10px; border-radius:10px'>{advice.get('prevention', 'N/A')}</div>", unsafe_allow_html=True)

                st.markdown("#### Crop Stage Vulnerability")
                st.markdown(f"<div style='background-color:#1e293b; color: white; padding:10px; border-radius:10px'>{advice.get('crop_stage', 'N/A')}</div>", unsafe_allow_html=True)
    else:
        st.warning("No pests detected in the image.")
else:
    st.info("Please upload an image to begin detection.")
