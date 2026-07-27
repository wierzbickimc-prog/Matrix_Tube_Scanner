
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import REVIEW_THRESHOLD, RackResult, analyze_image_bytes

st.set_page_config(page_title="96-Tube Rack Mapper", page_icon="🧪", layout="wide")
st.title("96-Tube Rack Mapper")
st.caption("Upload a rack image, choose the rack style, then press Start Analysis.")

uploaded = st.file_uploader("Rack image", type=["jpg", "jpeg", "png", "tif", "tiff"])

profile = st.selectbox(
    "Rack style",
    options=["Auto", "Original tray", "Notched plate"],
    index=0,
    help="Use 'Notched plate' for the updated plate where A1 is at the notch and the plate barcode runs along column 1.",
)

left_button, _ = st.columns([1, 5])
with left_button:
    start = st.button("Start Analysis", type="primary", disabled=uploaded is None, use_container_width=True)

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", width=420)

if start:
    with st.spinner("Decoding rack and tube Data Matrix codes…"):
        try:
            result = analyze_image_bytes(uploaded.getvalue(), profile_choice=profile)
        except Exception as exc:
            st.session_state.pop("rack_result", None)
            st.error(str(exc))
        else:
            st.session_state["rack_result"] = result
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
        st.image(result.overlay_rgb, caption="Green = strong, yellow = review, red = unreadable, gray = empty", use_container_width=True)

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

        st.dataframe(heat_df.style.format("{:.0f}%").map(confidence_style), use_container_width=True, height=330)

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
        use_container_width=True,
        disabled=["Plate", "Position", "Confidence"],
        column_config={
            "Plate": st.column_config.TextColumn("Plate"),
            "Position": st.column_config.TextColumn("Position"),
            "DataMatrix": st.column_config.TextColumn("DataMatrix", help="Correct the decoded value, or enter EMPTY / UNREADABLE."),
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
        },
        key="result_editor",
    )

    review_positions = [well.position for well in result.wells if well.value == "UNREADABLE" or well.confidence < REVIEW_THRESHOLD]
    all_positions = [well.position for well in result.wells]
    st.markdown("#### Inspect a tube crop")
    selected = st.selectbox("Position", options=review_positions or all_positions)
    selected_well = next(well for well in result.wells if well.position == selected)
    st.image(selected_well.crop_rgb, caption=f"{selected_well.position}: {selected_well.value} ({selected_well.confidence:.1f}%)", width=280)

    export_df = edited_df[["Plate", "Position", "DataMatrix"]].copy()
    export_df["DataMatrix"] = export_df["DataMatrix"].fillna("UNREADABLE").astype(str).str.strip()
    export_df.loc[export_df["DataMatrix"] == "", "DataMatrix"] = "UNREADABLE"
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download plate-map CSV", data=csv_bytes, file_name=f"{result.plate_id}_plate_map.csv", mime="text/csv", type="primary")
