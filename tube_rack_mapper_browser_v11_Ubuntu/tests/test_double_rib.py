"""Double-rib plate regression tests (v12).

IMG_9376-9380 are the same 14-tube rack photographed in different orientations
and angles, at 1152 x 1536 with frosted tubes (too small to decode reliably, but
geometry and presence must be exact). double_rib_42_fullres.jpg is a full-size
phone photo (4284 x 5712) of a 42-tube rack that v11 could not map;
double_rib_36_fullres.jpg is a 36-tube rack with gaps and hard dot-peen codes.
"""
import unittest
from pathlib import Path

import cv2

import core

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "test_images"
ROWS = "ABCDEFGH"
OCCUPIED_14 = {f"{r}1" for r in ROWS} | {"B2", "D2", "F2", "G2", "H2", "A3"}
OCCUPIED_42 = {f"{r}{c}" for r in ROWS for c in range(1, 6)} | {"A6", "B6"}
# Scattered gaps and isolated tubes: presence must be exact, not fill-order.
OCCUPIED_36 = (
    {"A1", "C1", "D1", "E1", "F1", "G1", "H1"} | {f"{r}2" for r in "BCDEFGH"}
    | {"A3", "B3", "C3", "D3", "F3", "G3", "H3"} | {f"{r}4" for r in "ABCDEFG"}
    | {"A5"} | {f"{r}8" for r in "ABCDE"} | {"A9", "E9"}
)


def analyze(name, scale=None):
    data = (IMAGES / name).read_bytes()
    if scale:
        image = cv2.imread(str(IMAGES / name))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    return core.analyze_image_bytes(data, "auto")


class DoubleRibTests(unittest.TestCase):
    def test_reference_racks_geometry_and_presence(self):
        for name in ("IMG_9376.jpeg", "IMG_9377.jpeg", "IMG_9378.jpeg", "IMG_9379.jpeg", "IMG_9380.jpeg"):
            with self.subTest(name=name):
                result = analyze(name)
                self.assertTrue(result.profile_name.startswith("Double-rib plate"), result.profile_name)
                occupied = {w.position for w in result.wells if w.value != "EMPTY"}
                self.assertEqual(occupied, OCCUPIED_14)
                # Double-rib plates carry no plate barcode; a tube code must never become the plate ID.
                self.assertEqual(result.plate_id, "UNREADABLE")

    def test_full_resolution_phone_photo(self):
        results = [analyze("double_rib_42_fullres.jpg"), analyze("double_rib_42_fullres.jpg", scale=1536 / 5712)]
        maps = [{w.position: w.value for w in r.wells} for r in results]
        for result, values in zip(results, maps):
            self.assertTrue(result.profile_name.startswith("Double-rib plate"), result.profile_name)
            self.assertEqual({p for p, v in values.items() if v != "EMPTY"}, OCCUPIED_42)
            self.assertFalse([p for p, v in values.items() if v == "UNREADABLE"])
            self.assertEqual(result.plate_id, "UNREADABLE")
        self.assertEqual(maps[0], maps[1])
        # Known tubes at the end of the fill pin the A1/H12 orientation.
        self.assertEqual(
            (maps[0]["A5"], maps[0]["H5"], maps[0]["A6"], maps[0]["B6"]),
            ("a8584363", "a8584370", "a8584371", "a8584372"),
        )

    def test_scattered_rack_with_dot_peen_codes(self):
        # A3, B3 and G4 only decode after dot merging / rotation; the result
        # must also be identical run to run.
        runs = [analyze("double_rib_36_fullres.jpg") for _ in range(2)]
        maps = [{w.position: w.value for w in r.wells} for r in runs]
        self.assertEqual(maps[0], maps[1])
        values = maps[0]
        self.assertTrue(runs[0].profile_name.startswith("Double-rib plate"), runs[0].profile_name)
        self.assertEqual({p for p, v in values.items() if v != "EMPTY"}, OCCUPIED_36)
        self.assertFalse([p for p, v in values.items() if v == "UNREADABLE"])
        self.assertEqual(values["G4"], "a8584325")
        downscaled = analyze("double_rib_36_fullres.jpg", scale=2048 / 5712)
        self.assertEqual({w.position: w.value for w in downscaled.wells}, values)


if __name__ == "__main__":
    unittest.main()
