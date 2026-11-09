"""Session regressions: a CSV must always belong to the currently chosen photo."""
import base64
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest

from core import RackResult, WellResult

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


class AppTests(unittest.TestCase):
    def setUp(self):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 12), "white").save(buffer, "JPEG")
        self.capture = {"event": "capture", "data": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}
        pixels = np.zeros((12, 16, 3), dtype=np.uint8)
        wells = [WellResult(f"{row}{col}", "TUBE1" if row == "A" and col == 1 else "EMPTY",
                            95., pixels, True, (0., 0.)) for row in "ABCDEFGH" for col in range(1, 13)]
        self.result = RackResult("TEST-PLATE", wells, pixels, pixels, "Original tray")
        self.component_patch = patch("streamlit.components.v1.declare_component", return_value=lambda **kwargs: st.session_state.get("test_camera"))
        self.decoder_patch = patch("core.analyze_image_bytes", return_value=self.result)
        self.component_patch.start()
        self.decoder = self.decoder_patch.start()
        self.addCleanup(self.component_patch.stop)
        self.addCleanup(self.decoder_patch.stop)
        self.app = AppTest.from_file(str(APP)).run()
        self.app.radio[0].set_value("Webcam").run()

    def analyze(self):
        self.app.session_state["test_camera"] = self.capture
        self.app.run().button[0].click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.app.session_state["rack_result"].plate_id, "TEST-PLATE")

    def test_capture_uses_existing_decoder_and_preserves_editable_96_wells(self):
        self.analyze()
        self.decoder.assert_called_once()
        frame = self.app.dataframe[1].value
        self.assertEqual(len(frame), 96)
        self.assertEqual(list(frame.columns), ["Plate", "Position", "DataMatrix", "Confidence"])
        self.assertEqual(frame.iloc[0]["DataMatrix"], "TUBE1")
        self.assertEqual(self.app.metric[0].value, "1")

    def test_setting_change_removes_previous_results(self):
        self.analyze()
        self.app.text_input[0].set_value("ANOTHER-PLATE").run()
        self.assertNotIn("rack_result", self.app.session_state)
        self.assertEqual(len(self.app.metric), 0)

    def test_retake_clears_results_and_disables_analysis(self):
        self.analyze()
        self.app.session_state["test_camera"] = {"event": "clear"}
        self.app.run()
        self.assertNotIn("rack_result", self.app.session_state)
        self.assertTrue(self.app.button[0].disabled)

    def test_failed_reanalysis_removes_previous_result(self):
        self.analyze()
        self.decoder.side_effect = RuntimeError("Test: unreadable photo")
        self.app.button[0].click().run()
        self.assertNotIn("rack_result", self.app.session_state)
        self.assertEqual(self.app.error[0].value, "Test: unreadable photo")

    def test_reanalysis_gets_new_editor_state(self):
        self.analyze()
        before = self.app.session_state["analysis_generation"]
        self.app.button[0].click().run()
        self.assertEqual(self.app.session_state["analysis_generation"], before + 1)

    def test_exit_returns_to_upload_without_camera_result(self):
        self.analyze()
        self.app.session_state["test_camera"] = {"event": "exit"}
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.app.radio[0].value, "Upload image")
        self.assertNotIn("rack_result", self.app.session_state)


if __name__ == "__main__":
    unittest.main()
