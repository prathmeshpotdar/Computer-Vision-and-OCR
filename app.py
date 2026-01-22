import streamlit as st
import os
import json

st.set_page_config(page_title="AI Assignment Output Viewer", layout="wide")

st.title("AI Computer Vision & OCR Output Viewer")

OUTPUT_DIR = "outputs"
OCR_FILE = "ocr_output.json"

# ======================================
# VIDEO OUTPUT SECTION
# ======================================

st.header("Annotated Video Output")

if os.path.exists(OUTPUT_DIR):

    video_files = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith((".avi", ".mp4", ".webm"))
    ]

    if len(video_files) > 0:

        # Allow user to select video
        selected_video = st.selectbox(
            "Select Annotated Video",
            video_files
        )

        video_path = os.path.join(OUTPUT_DIR, selected_video)

        st.success(f"Showing: {selected_video}")

        # IMPORTANT: Pass file path directly
        st.video(video_path)

    else:
        st.warning("No annotated videos found in outputs folder")

else:
    st.warning("Outputs folder not found")


# ======================================
# OCR OUTPUT SECTION
# ======================================

st.header("OCR Extracted Text Output")

if os.path.exists(OCR_FILE):

    with open(OCR_FILE) as f:
        ocr_data = json.load(f)

    st.json(ocr_data)

else:
    st.warning("OCR output file not found. Run OCR pipeline first.")
