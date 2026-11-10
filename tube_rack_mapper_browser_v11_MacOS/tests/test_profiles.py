"""Auto profile detection on the reference photos.

Run with python -m unittest discover -s tests -v from the package folder.
"""
import os
import unittest

import cv2
import numpy as np

import core

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "..", "test_images")

EXPECTED_PROFILE = {
    "IMG_9340.jpeg": "Notched plate",
    "IMG_9341.jpeg": "Notched plate",
    "IMG_9369.jpeg": "Original tray",
    "IMG_9370.jpeg": "Original tray",
    "IMG_9371.jpeg": "Original tray",
    "IMG_9372.jpeg": "Original tray",
    "IMG_9376.jpeg": "Double-rib plate",
    "IMG_9377.jpeg": "Double-rib plate",
    "IMG_9378.jpeg": "Double-rib plate",
    "IMG_9379.jpeg": "Double-rib plate",
    "IMG_9380.jpeg": "Double-rib plate",
    "double_rib_36_fullres.jpg": "Double-rib plate",
    "double_rib_42_fullres.jpg": "Double-rib plate",
    # Full-resolution phone photo; double-rib plate in a tray whose rack
    # label sits beside H12.
    "double_rib_tray_label_full_res.jpg": "Double-rib plate",
}

LABELLED_TRAY = "double_rib_tray_label_full_res.jpg"


def read(name):
    with open(os.path.join(IMAGES, name), "rb") as handle:
        return handle.read()


def well(result, position):
    return next(w for w in result.wells if w.position == position)


class AutoProfileTests(unittest.TestCase):
    def test_auto_picks_expected_profile(self):
        for name, expected in EXPECTED_PROFILE.items():
            with self.subTest(image=name):
                result = core.analyze_image_bytes(read(name), "auto")
                self.assertTrue(
                    result.profile_name.startswith(expected),
                    f"{name}: got {result.profile_name}",
                )

    def test_labelled_double_rib_tray_reads_rack_id_and_orientation(self):
        result = core.analyze_image_bytes(read(LABELLED_TRAY), "auto")
        self.assertEqual(result.plate_id, "ts05378298")
        # Rows 1-2 and A3-G3 are empty; H3 onward is full (73 tubes).
        self.assertEqual(well(result, "A1").value, "EMPTY")
        self.assertNotEqual(well(result, "H12").value, "EMPTY")
        decoded = [
            w for w in result.wells
            if w.occupied and w.value not in ("EMPTY", "UNREADABLE")
        ]
        self.assertEqual(len(decoded), 73)

    def test_resolution_and_reencoding_do_not_change_result(self):
        image = cv2.imdecode(np.frombuffer(read(LABELLED_TRAY), np.uint8), 1)
        variants = [cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]]
        for width in (1536, 1152):
            height = round(image.shape[0] * width / image.shape[1])
            small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            variants.append(cv2.imencode(".jpg", small)[1])

        for data in variants:
            result = core.analyze_image_bytes(data.tobytes(), "auto")
            self.assertTrue(result.profile_name.startswith("Double-rib plate"))
            self.assertEqual(result.plate_id, "ts05378298")
            self.assertNotEqual(well(result, "H12").value, "EMPTY")

    def test_repeat_analysis_is_deterministic(self):
        data = read("IMG_9376.jpeg")
        first = core.analyze_image_bytes(data, "auto")
        second = core.analyze_image_bytes(data, "auto")
        self.assertEqual(first.plate_id, second.plate_id)
        self.assertEqual(
            [w.value for w in first.wells],
            [w.value for w in second.wells],
        )


if __name__ == "__main__":
    unittest.main()
