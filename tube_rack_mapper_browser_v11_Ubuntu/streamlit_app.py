from __future__ import annotations

import base64
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from capture import camera_image, source_key
from core import REVIEW_THRESHOLD, RackResult, analyze_image_bytes

ROOT = Path(__file__).resolve().parent
# Webcam capture is off: real webcam frames are too soft to decode the tube
# Data Matrix codes (0/73 on a full rack). Set TUBE_SCANNER_WEBCAM=1 to re-enable.
WEBCAM_ENABLED = os.environ.get("TUBE_SCANNER_WEBCAM") == "1"
LOGO = ROOT / "assets" / "scanner.svg"
webcam = components.declare_component("built_rack_camera", path=str(ROOT / "webcam"))
st.set_page_config(page_title="Built Tube Scanner", page_icon=str(LOGO), layout="wide")
st.markdown(f"<style>{(ROOT / 'assets' / 'theme.css').read_text()}</style>", unsafe_allow_html=True)
logo_data = base64.b64encode(LOGO.read_bytes()).decode("ascii")
st.markdown(
    f'<div class="built-header"><img src="data:image/svg+xml;base64,{logo_data}" alt="Built Tube Scanner logo">'
    '<div><div class="built-wordmark">BUILT BIOTECHNOLOGIES</div><div class="built-product">Tube Scanner</div></div>'
    '<span class="built-tag">96-well rack · Data Matrix</span></div>'
    '<div class="built-eyebrow">FROM RACK TO RECORD</div>',
    unsafe_allow_html=True,
)
st.title("Every tube. In its place.")
st.caption("Upload a rack photo, review the tube codes, and export a plate map.")


def change_source():
    # A remounted camera must never revive a still from an earlier webcam session.
    st.session_state["camera_epoch"] = st.session_state.get("camera_epoch", 0) + 1


if st.session_state.pop("exit_camera", False):
    st.session_state["image_source"] = "Upload image"
    change_source()
if WEBCAM_ENABLED:
    mode = st.radio("Image source", ["Upload image", "Webcam"], horizontal=True, key="image_source", on_change=change_source)
else:
    mode = "Upload image"
image_bytes = None
with st.container(border=True, key="capture_card"):
    if mode == "Upload image":
        st.subheader("01 / Add your rack image")
        st.caption("Include the whole rack, all tube bottoms, and any rack barcode. A sharp, evenly lit photo works best.")
        uploaded = st.file_uploader("Rack image", type=["jpg", "jpeg", "png", "tif", "tiff"])
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            st.image(image_bytes, caption=uploaded.name, width=420)
    else:
        # White lighting lasts for the whole webcam mode, including still review.
        st.markdown("<style>.stApp, [data-testid='stHeader'] {background: #fff !important;}</style>", unsafe_allow_html=True)
        value = webcam(key=f"rack_camera_{st.session_state.get('camera_epoch', 0)}", default=None)
        if isinstance(value, dict) and value.get("event") == "exit":
            st.session_state["exit_camera"] = True
            st.rerun()
        if isinstance(value, dict) and value.get("event") == "capture":
            try:
                image_bytes = camera_image(value)
            except ValueError as exc:
                st.error(str(exc))

with st.container(border=True, key="scan_card"):
    st.subheader("02 / Set up the scan")
    rack_col, id_col = st.columns(2)
    with rack_col:
        profile = st.selectbox(
            "Rack style",
            options=["Auto", "Original tray", "Notched plate", "Double-rib plate"],
            help="Auto detects the rack profile. Include the rack barcode or the paired H12 ribs for orientation.",
        )
    with id_col:
        plate_id_override = st.text_input(
            "Plate ID override (optional)",
            help="Use when the rack-level Data Matrix is not visible, including double-rib plates.",
        )
    with st.expander("Orientation fallback"):
        double_rib_corner_override = st.selectbox(
            "Double-rib H12 corner override (fallback)",
            options=["Auto", "Top-left", "Top-right", "Bottom-left", "Bottom-right"],
            help="Leave on Auto unless double-rib orientation is wrong. Choose the photo corner containing the paired H12 support ribs.",
        )


    current_source = source_key(image_bytes, mode, profile, plate_id_override, double_rib_corner_override)
    if current_source is None or st.session_state.get("result_source") != current_source:
        st.session_state.pop("rack_result", None)
        st.session_state.pop("result_source", None)
    start = st.button("Start analysis", type="primary", disabled=image_bytes is None)
    if image_bytes is None:
        st.caption("Upload an image or choose a webcam photo to continue." if WEBCAM_ENABLED else "Upload an image to continue.")

if start and image_bytes is not None:
    # Clear previous edits even when explicitly reanalyzing the same photo.
    st.session_state["analysis_generation"] = st.session_state.get("analysis_generation", 0) + 1
    st.session_state.pop("rack_result", None)
    st.session_state.pop("result_source", None)
    with st.spinner("Decoding rack and tube Data Matrix codes…"):
        try:
            result = analyze_image_bytes(
                image_bytes,
                profile_choice=profile,
                plate_id_override=plate_id_override,
                double_rib_corner_override=double_rib_corner_override,
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state["rack_result"] = result
            st.session_state["result_source"] = current_source
            st.success(f"Analysis complete. Profile used: {result.profile_name}")

result: RackResult | None = st.session_state.get("rack_result")

if result is not None:
    st.divider()
    st.subheader(f"Plate ID: `{result.plate_id}`")
    st.caption(f"Profile used: {result.profile_name}")

    decoded = sum(w.value not in ("EMPTY", "UNREADABLE") for w in result.wells)
    empty = sum(w.value == "EMPTY" for w in result.wells)
    unreadable = sum(w.value == "UNREADABLE" for w in result.wells)
    review = sum(w.value not in ("EMPTY", "UNREADABLE") and w.confidence < REVIEW_THRESHOLD for w in result.wells)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decoded", decoded)
    m2.metric("Empty", empty)
    m3.metric("Unreadable", unreadable)
    m4.metric("Low-confidence review", review)

    image_col, heat_col = st.columns([1.2, 1])

    with image_col:
        st.markdown("#### Detected plate")
        st.image(result.overlay_rgb, caption="Green = strong, yellow = review, red = unreadable, gray = empty", width="stretch")

    with heat_col:
        st.markdown("#### Confidence heat map")
        confidences = {(w.position[0], int(w.position[1:])): w.confidence for w in result.wells}
        heat_data = [[confidences.get((row, column), 0.0) for column in range(1, 13)] for row in "ABCDEFGH"]
        heat_df = pd.DataFrame(heat_data, index=list("ABCDEFGH"), columns=[str(column) for column in range(1, 13)])

        def confidence_style(value):
            if value >= 70:
                return "background-color: #62b66f; color: black"
            if value >= 45:
                return "background-color: #f0c34e; color: black"
            return "background-color: #e56a61; color: black"

        st.dataframe(heat_df.style.format("{:.0f}%").map(confidence_style), width="stretch", height=330)

    st.divider()
    st.markdown("### Review and correct results")
    st.caption("Edit only the DataMatrix column. Use EMPTY or UNREADABLE when appropriate.")

    result_df = pd.DataFrame(
        {
            "Plate": [result.plate_id] * 96,
            "Position": [well.position for well in result.wells],
            "DataMatrix": [well.value for well in result.wells],
            "Confidence": [round(well.confidence, 1) for well in result.wells],
        }
    )

    edited_df = st.data_editor(
        result_df,
        hide_index=True,
        width="stretch",
        disabled=["Plate", "Position", "Confidence"],
        column_config={
            "Plate": st.column_config.TextColumn("Plate"),
            "Position": st.column_config.TextColumn("Position"),
            "DataMatrix": st.column_config.TextColumn("DataMatrix", help="Correct the decoded value, or enter EMPTY / UNREADABLE."),
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
        },
        key=f"result_editor_{st.session_state.get('analysis_generation', 0)}",
    )

    review_positions = [well.position for well in result.wells if well.value == "UNREADABLE" or well.confidence < REVIEW_THRESHOLD]
    all_positions = [well.position for well in result.wells]
    st.markdown("#### Inspect a tube crop")
    selected = st.selectbox("Position", options=review_positions or all_positions, key=f"crop_{st.session_state.get('analysis_generation', 0)}")
    selected_well = next(well for well in result.wells if well.position == selected)
    st.image(selected_well.crop_rgb, caption=f"{selected_well.position}: {selected_well.value} ({selected_well.confidence:.1f}%)", width=280)

    export_df = edited_df[["Plate", "Position", "DataMatrix"]].copy()
    export_df["DataMatrix"] = export_df["DataMatrix"].fillna("UNREADABLE").astype(str).str.strip()
    export_df.loc[export_df["DataMatrix"] == "", "DataMatrix"] = "UNREADABLE"
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download plate-map CSV", data=csv_bytes, file_name=f"{result.plate_id}_plate_map.csv", mime="text/csv", type="primary")
