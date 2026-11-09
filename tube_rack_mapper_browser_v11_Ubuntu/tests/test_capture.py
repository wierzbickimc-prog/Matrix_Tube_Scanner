"""Run with python -m unittest discover -s tests -v (no camera required)."""
import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image

from capture import camera_image, source_key


def jpeg(color="white"):
    output = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(output, "JPEG")
    return output.getvalue()


def event(data):
    return {"event": "capture", "data": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")}


class CaptureTests(unittest.TestCase):
    def test_photo_bytes_reach_decoder_without_resizing_or_mirroring(self):
        raw = jpeg()
        self.assertEqual(camera_image(event(raw)), raw)

    def test_empty_malformed_and_disguised_images_are_rejected(self):
        png = io.BytesIO()
        Image.new("RGB", (16, 12)).save(png, "PNG")
        invalid = [[], {}, {"event": "clear"}, {"event": "capture", "data": 5},
                   {"event": "capture", "data": "data:image/jpeg;base64,%%%"},
                   event(b""), event(b"not an image"), event(png.getvalue())]
        for value in invalid:
            with self.subTest(value=str(value)[:70]), self.assertRaises(ValueError):
                camera_image(value)
        self.assertIsNone(camera_image(None))

    def test_oversized_capture_is_rejected_before_decoding(self):
        with patch("capture.MAX_CAPTURE_BYTES", 4), self.assertRaisesRegex(ValueError, "too large"):
            camera_image(event(jpeg()))

    def test_every_source_or_setting_change_invalidates_results(self):
        raw = jpeg()
        original = source_key(raw, "Webcam", "Auto", "", "Auto")
        self.assertEqual(original, source_key(raw, "Webcam", "Auto", "", "Auto"))
        changes = [(jpeg("black"), "Webcam", "Auto", "", "Auto"),
                   (raw, "Upload image", "Auto", "", "Auto"),
                   (raw, "Webcam", "Original tray", "", "Auto"),
                   (raw, "Webcam", "Auto", "PLATE2", "Auto"),
                   (raw, "Webcam", "Auto", "", "Top-left")]
        for args in changes:
            self.assertNotEqual(original, source_key(*args))
        self.assertIsNone(source_key(None, "Webcam", "Auto", "", "Auto"))


if __name__ == "__main__":
    unittest.main()
