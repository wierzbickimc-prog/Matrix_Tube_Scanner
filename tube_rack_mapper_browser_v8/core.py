
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import zxingcpp

ROWS = "ABCDEFGH"
COLS = range(1, 13)
REVIEW_THRESHOLD = 70.0

@dataclass
class WellResult:
    position: str
    value: str
    confidence: float
    crop_rgb: np.ndarray
    occupied: bool
    center_xy: tuple[float, float]
    corrected: bool = False

@dataclass
class RackResult:
    plate_id: str
    wells: list[WellResult]
    normalized_rgb: np.ndarray
    overlay_rgb: np.ndarray
    profile_name: str

def _read_datamatrix(image: np.ndarray):
    return zxingcpp.read_barcodes(
        image,
        formats=zxingcpp.BarcodeFormat.DataMatrix,
        try_rotate=True,
        try_downscale=True,
        try_invert=True,
    )

def _barcode_center(barcode) -> tuple[float, float]:
    p = barcode.position
    points = np.array(
        [
            [p.top_left.x, p.top_left.y],
            [p.top_right.x, p.top_right.y],
            [p.bottom_right.x, p.bottom_right.y],
            [p.bottom_left.x, p.bottom_left.y],
        ],
        dtype=float,
    )
    x, y = points.mean(axis=0)
    return float(x), float(y)

def _quality_score(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = float(gray.std())
    return float(100.0 * (0.65 * np.clip(sharpness / 650.0, 0.0, 1.0) + 0.35 * np.clip(contrast / 55.0, 0.0, 1.0)))

def _crop_variants(crop_bgr: np.ndarray):
    for scale in (1.0, 1.6, 2.4, 3.2):
        image = crop_bgr if scale == 1.0 else cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        yield image
        yield cv2.equalizeHist(gray)
        yield cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        yield cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)

def _rotate_arbitrary(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _decode_crop(crop_bgr: np.ndarray, plate_id: str) -> tuple[Optional[str], float]:
    votes: Counter[str] = Counter()
    for variant in _crop_variants(crop_bgr):
        for code in _read_datamatrix(variant):
            text = code.text.strip().lower()
            if text and text != plate_id:
                votes[text] += 1
    if not votes:
        for angle in range(0, 180, 10):
            rotated = _rotate_arbitrary(crop_bgr, angle)
            enlarged = cv2.resize(rotated, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
            for variant in (enlarged, cv2.equalizeHist(gray), cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)):
                for code in _read_datamatrix(variant):
                    text = code.text.strip().lower()
                    if text and text != plate_id:
                        votes[text] += 1
    if not votes:
        return None, min(39.0, _quality_score(crop_bgr) * 0.38)

    ranked = votes.most_common()
    best_text, best_votes = ranked[0]
    second_votes = ranked[1][1] if len(ranked) > 1 else 0
    total_votes = sum(votes.values())
    agreement = best_votes / max(1, total_votes)
    margin = (best_votes - second_votes) / max(1, best_votes)
    confidence = 100.0 * (0.50 * min(1.0, best_votes / 5.0) + 0.25 * agreement + 0.10 * margin + 0.15 * (_quality_score(crop_bgr) / 100.0))
    return best_text, float(np.clip(confidence, 0.0, 100.0))

def _enforce_unique_codes(wells: list[WellResult]) -> None:
    groups: dict[str, list[WellResult]] = defaultdict(list)
    for well in wells:
        if well.value not in ("EMPTY", "UNREADABLE"):
            groups[well.value].append(well)
    for duplicates in groups.values():
        if len(duplicates) > 1:
            duplicates.sort(key=lambda item: item.confidence, reverse=True)
            for duplicate in duplicates[1:]:
                duplicate.value = "UNREADABLE"
                duplicate.confidence = 0.0

def _sort_wells_canonical(wells: list[WellResult]) -> None:
    position_order = {f"{row}{column}": index for index, (column, row) in enumerate((column, row) for column in COLS for row in ROWS)}
    wells.sort(key=lambda item: position_order[item.position])

def _draw_overlay(image_bgr: np.ndarray, wells: list[WellResult]) -> np.ndarray:
    overlay = image_bgr.copy()
    for well in wells:
        x, y = well.center_xy
        if well.value == "EMPTY":
            color = (170, 170, 170)
        elif well.value == "UNREADABLE" or well.confidence < 45:
            color = (40, 40, 220)
        elif well.confidence < REVIEW_THRESHOLD:
            color = (0, 190, 240)
        else:
            color = (45, 175, 70)
        cv2.circle(overlay, (int(x), int(y)), 38, color, 3, cv2.LINE_AA)
        cv2.putText(overlay, well.position, (int(x) - 22, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# ---------- shared geometry ----------
def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    result = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    result[0] = pts[np.argmin(sums)]
    result[1] = pts[np.argmin(diffs)]
    result[2] = pts[np.argmax(sums)]
    result[3] = pts[np.argmax(diffs)]
    return result

def _perspective_normalize(image_bgr: np.ndarray, out_w: int = 1000, out_h: int = 1450) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)), iterations=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Could not locate the rack outline.")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) / float(image_bgr.shape[0] * image_bgr.shape[1]) < 0.25:
        raise RuntimeError("The rack occupies too little of the image.")
    rect = cv2.minAreaRect(contour)
    box = _order_quad(cv2.boxPoints(rect))
    width = max(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[3]))
    height = max(np.linalg.norm(box[3] - box[0]), np.linalg.norm(box[2] - box[1]))
    if width > height:
        box = np.roll(box, -1, axis=0)
    target = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(_order_quad(box), target)
    return cv2.warpPerspective(image_bgr, matrix, (out_w, out_h))

def _rotate_90(image: np.ndarray, turns: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(image, k=turns % 4))

def _decode_marker_roi(image_bgr: np.ndarray, roi_frac) -> Optional[str]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = roi_frac
    roi = image_bgr[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variants = [roi, cv2.equalizeHist(gray), cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)]
    for scale in (1.0, 2.0, 3.0):
        for variant in variants:
            img = variant if scale == 1.0 else cv2.resize(variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for code in _read_datamatrix(img):
                text = code.text.strip().lower()
                if text:
                    return text
    return None

# ---------- original tray ----------
ORIGINAL_GRID_X = np.array([0.139625, 0.248306, 0.355333, 0.459944, 0.563104, 0.666042, 0.767625, 0.865722], dtype=float)
ORIGINAL_GRID_Y = np.array([0.092974, 0.165172, 0.240891, 0.310468, 0.379828, 0.448172, 0.520453, 0.591084, 0.661810, 0.731207, 0.800099, 0.869892], dtype=float)
ORIGINAL_MARKER_ROI = (0.70, 0.895, 0.94, 0.995)

def _presence_score_original(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    radius = min(h, w) * 0.36
    mask = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 <= radius ** 2
    center_mean = float(gray[mask].mean()) if np.any(mask) else 0.0
    edges = cv2.Canny(gray, 40, 120)
    edge_density = float((edges[mask] > 0).mean()) if np.any(mask) else 0.0
    brightness = np.clip((center_mean - 45.0) / 105.0, 0.0, 1.0)
    texture = np.clip(edge_density / 0.16, 0.0, 1.0)
    return float(100.0 * (0.70 * brightness + 0.30 * texture))

def analyze_original_legacy(image_bgr: np.ndarray) -> RackResult:
    normalized_bgr = _perspective_normalize(image_bgr, 1000, 1450)
    oriented_bgr = None
    plate_id = None
    for turns in range(4):
        rotated = _rotate_90(normalized_bgr, turns)
        marker = _decode_marker_roi(rotated, ORIGINAL_MARKER_ROI)
        if marker:
            oriented_bgr = rotated
            plate_id = marker
            break
    if oriented_bgr is None or plate_id is None:
        raise RuntimeError("The rack-level Data Matrix could not be decoded.")
    h, w = oriented_bgr.shape[:2]
    xs = ORIGINAL_GRID_X * w
    ys = ORIGINAL_GRID_Y * h
    wells = []
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            row_letter = ROWS[7 - xi]
            plate_column = 12 - yi
            position = f"{row_letter}{plate_column}"
            radius = 54
            x1, x2 = max(0, int(x-radius)), min(w, int(x+radius))
            y1, y2 = max(0, int(y-radius)), min(h, int(y+radius))
            crop_bgr = oriented_bgr[y1:y2, x1:x2].copy()
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            presence = _presence_score_original(crop_bgr)
            if presence < 45.0:
                value = "EMPTY"
                confidence = float(np.clip(100.0 - presence, 60.0, 98.0))
                occupied = False
            else:
                decoded, confidence = _decode_crop(crop_bgr, plate_id)
                value = decoded if decoded else "UNREADABLE"
                occupied = True
            wells.append(WellResult(position=position, value=value, confidence=confidence, crop_rgb=crop_rgb, occupied=occupied, center_xy=(float(x), float(y))))
    _enforce_unique_codes(wells)
    _sort_wells_canonical(wells)
    overlay = _draw_overlay(oriented_bgr, wells)
    return RackResult(plate_id=plate_id, wells=wells, normalized_rgb=cv2.cvtColor(oriented_bgr, cv2.COLOR_BGR2RGB), overlay_rgb=overlay, profile_name="Original tray")



# ---------- perspective-robust original tray ----------
#
# This path does not depend on the outer rack contour. It uses the decoded
# tube Data Matrix centers themselves as an 8 x 12 calibration lattice.
# This is substantially more tolerant of camera rotation and oblique photos.

def _detect_datamatrix_multiscale(
    image_bgr: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    detections: list[tuple[str, np.ndarray]] = []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)

    for scale in (1.5, 2.0, 2.5):
        for base in (image_bgr, clahe):
            candidate = cv2.resize(
                base,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            for code in _read_datamatrix(candidate):
                text = code.text.strip().lower()
                if not text:
                    continue
                center = np.array(
                    _barcode_center(code),
                    dtype=float,
                ) / scale
                detections.append((text, center))

    unique: list[tuple[str, np.ndarray]] = []
    for text, center in detections:
        duplicate = any(
            text == existing_text
            and np.linalg.norm(center - existing_center) < 12.0
            for existing_text, existing_center in unique
        )
        if not duplicate:
            unique.append((text, center))

    return unique


def _kmeans_axis_labels(
    values: np.ndarray,
    count: int,
) -> np.ndarray:
    data = values.astype(np.float32).reshape(-1, 1)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.01,
    )
    _, labels, centers = cv2.kmeans(
        data,
        count,
        None,
        criteria,
        30,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = centers.ravel()
    order = np.argsort(centers)
    remap = np.empty(count, dtype=int)
    remap[order] = np.arange(count)
    return remap[labels.ravel()]


def _fit_lattice_subset(
    detections: list[tuple[str, np.ndarray]],
):
    points = np.array(
        [center for _, center in detections],
        dtype=np.float32,
    )
    mean = points.mean(axis=0)
    centered = points - mean
    _, _, axes = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    projected = centered @ axes.T

    best = None

    # PCA axis 0 normally follows the 12-position direction, but test both.
    for case in (0, 1):
        if case == 0:
            v_labels = _kmeans_axis_labels(projected[:, 0], 12)
            u_labels = _kmeans_axis_labels(projected[:, 1], 8)
        else:
            u_labels = _kmeans_axis_labels(projected[:, 0], 8)
            v_labels = _kmeans_axis_labels(projected[:, 1], 12)

        # Canonical coordinate order is [plate-column axis, row-letter axis].
        canonical = np.column_stack(
            [v_labels, u_labels]
        ).astype(np.float32)

        homography, _ = cv2.findHomography(
            canonical,
            points,
            cv2.RANSAC,
            18.0,
        )
        if homography is None:
            continue

        predicted = cv2.perspectiveTransform(
            canonical.reshape(-1, 1, 2),
            homography,
        ).reshape(-1, 2)
        error = np.linalg.norm(predicted - points, axis=1)
        inliers = error < 18.0

        score = (
            int(inliers.sum()),
            -float(np.median(error[inliers]))
            if inliers.any()
            else -999.0,
        )

        if best is None or score > best[0]:
            best = (score, homography)

    return best


def _fit_original_barcode_lattice(
    detections: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, int, tuple[int, float]]:
    if len(detections) < 24:
        raise RuntimeError(
            "Too few tube Data Matrix codes were detected to fit the "
            "perspective-robust 8 x 12 lattice."
        )

    length_counts = Counter(
        len(text)
        for text, _ in detections
    )
    tube_text_length = length_counts.most_common(1)[0][0]

    dominant = [
        item
        for item in detections
        if len(item[0]) == tube_text_length
    ]
    if len(dominant) < 24:
        raise RuntimeError(
            "Too few consistent tube codes were detected to fit the lattice."
        )

    subsets = [dominant]

    points = np.array(
        [center for _, center in dominant],
        dtype=float,
    )
    distances = np.linalg.norm(
        points[:, None, :] - points[None, :, :],
        axis=2,
    )
    distances[distances == 0] = 1e9
    median_nearest = float(np.median(distances.min(axis=1)))
    neighbor_counts = (
        distances < (1.8 * median_nearest)
    ).sum(axis=1)

    dense = [
        item
        for item, count in zip(dominant, neighbor_counts)
        if count >= 2
    ]
    if len(dense) >= 24:
        subsets.append(dense)

    best = None
    for subset in subsets:
        result = _fit_lattice_subset(subset)
        if result is not None and (
            best is None or result[0] > best[0]
        ):
            best = result

    if best is None or best[0][0] < 20:
        raise RuntimeError(
            "The tube barcode centers did not form a reliable 8 x 12 lattice."
        )

    return best[1], tube_text_length, best[0]


def _decode_marker_crop_robust(
    crop_bgr: np.ndarray,
) -> Optional[str]:
    if crop_bgr.size == 0:
        return None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)

    variants = [
        crop_bgr,
        gray,
        clahe,
        cv2.equalizeHist(gray),
        cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1],
        cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        ),
    ]

    for scale in (1.0, 1.5, 2.0, 3.0, 4.0):
        for variant in variants:
            candidate = (
                variant
                if scale == 1.0
                else cv2.resize(
                    variant,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            )
            for code in _read_datamatrix(candidate):
                text = code.text.strip().lower()
                if text:
                    return text

    return None


def _warp_generic_lattice(
    image_bgr: np.ndarray,
    homography: np.ndarray,
    scale: int = 100,
) -> np.ndarray:
    # Input canonical coordinate order is [v, u].
    # Output: x=(u+1)*scale, y=(v+2)*scale.
    canonical_to_output = np.array(
        [
            [0, scale, scale],
            [scale, 0, 2 * scale],
            [0, 0, 1],
        ],
        dtype=float,
    )
    image_to_output = (
        canonical_to_output @ np.linalg.inv(homography)
    )
    return cv2.warpPerspective(
        image_bgr,
        image_to_output,
        (10 * scale, 16 * scale),
    )


def _find_original_marker_from_lattice(
    image_bgr: np.ndarray,
    homography: np.ndarray,
    detections: list[tuple[str, np.ndarray]],
    tube_text_length: int,
) -> tuple[str, float, float]:
    inverse = np.linalg.inv(homography)
    geometric_candidates = []

    marker_locations = (
        (-1.0, 0.5),
        (-1.0, 6.5),
        (12.0, 0.5),
        (12.0, 6.5),
    )

    # First use any already-decoded non-tube code.
    for text, center in detections:
        if len(text) == tube_text_length:
            continue

        canonical = cv2.perspectiveTransform(
            np.asarray(
                center,
                dtype=np.float32,
            ).reshape(1, 1, 2),
            inverse,
        ).reshape(2)
        v, u = map(float, canonical)

        distance = min(
            float(np.hypot(v - mv, u - mu))
            for mv, mu in marker_locations
        )
        geometric_candidates.append(
            (distance, text, v, u)
        )

    if geometric_candidates:
        geometric_candidates.sort(
            key=lambda item: item[0]
        )
        distance, text, v, u = geometric_candidates[0]
        if distance < 2.0:
            return text, v, u

    # If the plate code was too oblique/small in the original photo, first
    # rectify the tube lattice. The marker becomes square and can be decoded
    # from four small, known candidate regions.
    generic = _warp_generic_lattice(
        image_bgr,
        homography,
        scale=100,
    )

    for v, u in marker_locations:
        center_x = int((u + 1.0) * 100)
        center_y = int((v + 2.0) * 100)
        crop = generic[
            max(0, center_y - 90):min(generic.shape[0], center_y + 90),
            max(0, center_x - 100):min(generic.shape[1], center_x + 100),
        ]
        text = _decode_marker_crop_robust(crop)
        if text and len(text) != tube_text_length:
            return text, v, u

    # Spatial position is authoritative here; allow a same-length marker as a
    # last resort because some future plate IDs may match tube-ID length.
    for v, u in marker_locations:
        center_x = int((u + 1.0) * 100)
        center_y = int((v + 2.0) * 100)
        crop = generic[
            max(0, center_y - 90):min(generic.shape[0], center_y + 90),
            max(0, center_x - 100):min(generic.shape[1], center_x + 100),
        ]
        text = _decode_marker_crop_robust(crop)
        if text:
            return text, v, u

    raise RuntimeError(
        "The rack Data Matrix could not be decoded even after "
        "perspective correction."
    )


def _orient_original_homography(
    homography: np.ndarray,
    marker_v: float,
    marker_u: float,
) -> np.ndarray:
    # The physical rack marker is between A1 and B1, immediately outside
    # plate column 1. Flip the fitted lattice so the marker is near
    # canonical coordinate v=-1, u=0.5.
    final_to_generic = np.eye(3, dtype=float)

    if marker_v > 5.5:
        final_to_generic = np.array(
            [
                [-1, 0, 11],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        ) @ final_to_generic

    if marker_u > 3.5:
        final_to_generic = np.array(
            [
                [1, 0, 0],
                [0, -1, 7],
                [0, 0, 1],
            ],
            dtype=float,
        ) @ final_to_generic

    return homography @ final_to_generic


def _rectify_original_from_lattice(
    image_bgr: np.ndarray,
    final_homography: np.ndarray,
    scale: int = 100,
) -> np.ndarray:
    # Final canonical coordinates:
    # u=0..7 is A..H; v=0..11 is plate columns 1..12.
    canonical_to_output = np.array(
        [
            [0, scale, scale],
            [scale, 0, 1.5 * scale],
            [0, 0, 1],
        ],
        dtype=float,
    )
    image_to_output = (
        canonical_to_output @ np.linalg.inv(final_homography)
    )
    return cv2.warpPerspective(
        image_bgr,
        image_to_output,
        (10 * scale, 1450),
    )


def analyze_original_lattice(
    image_bgr: np.ndarray,
) -> RackResult:
    detections = _detect_datamatrix_multiscale(
        image_bgr
    )
    homography, tube_text_length, _ = (
        _fit_original_barcode_lattice(detections)
    )

    plate_id, marker_v, marker_u = (
        _find_original_marker_from_lattice(
            image_bgr,
            homography,
            detections,
            tube_text_length,
        )
    )

    final_homography = _orient_original_homography(
        homography,
        marker_v,
        marker_u,
    )
    oriented_bgr = _rectify_original_from_lattice(
        image_bgr,
        final_homography,
        scale=100,
    )

    # Multi-scale decode the rectified rack first.
    rectified_detections = _detect_datamatrix_multiscale(
        oriented_bgr
    )
    position_votes: dict[
        tuple[int, int],
        Counter[str],
    ] = defaultdict(Counter)

    for text, center in rectified_detections:
        if text == plate_id:
            continue

        u = center[0] / 100.0 - 1.0
        v = center[1] / 100.0 - 1.5
        xi = int(round(u))
        yi = int(round(v))

        if (
            0 <= xi < 8
            and 0 <= yi < 12
            and abs(u - xi) < 0.50
            and abs(v - yi) < 0.50
        ):
            position_votes[(xi, yi)][text] += 1

    wells: list[WellResult] = []

    for yi in range(12):
        for xi in range(8):
            position = f"{ROWS[xi]}{yi + 1}"
            center_x = float((xi + 1.0) * 100)
            center_y = float((yi + 1.5) * 100)
            radius = 55

            crop_bgr = oriented_bgr[
                int(center_y) - radius:int(center_y) + radius,
                int(center_x) - radius:int(center_x) + radius,
            ].copy()
            crop_rgb = cv2.cvtColor(
                crop_bgr,
                cv2.COLOR_BGR2RGB,
            )

            votes = position_votes.get(
                (xi, yi),
                Counter(),
            )

            if votes:
                ranked = votes.most_common()
                value, best_votes = ranked[0]
                second_votes = (
                    ranked[1][1]
                    if len(ranked) > 1
                    else 0
                )
                agreement = best_votes / max(
                    1,
                    sum(votes.values()),
                )
                margin = (
                    best_votes - second_votes
                ) / max(1, best_votes)
                confidence = float(
                    np.clip(
                        78.0
                        + 12.0 * agreement
                        + 10.0 * margin,
                        0,
                        100,
                    )
                )
                occupied = True
            else:
                decoded, confidence = _decode_crop(
                    crop_bgr,
                    plate_id,
                )
                if decoded:
                    value = decoded
                    occupied = True
                else:
                    presence = _presence_score_original(
                        crop_bgr
                    )
                    if presence < 45.0:
                        value = "EMPTY"
                        confidence = float(
                            np.clip(
                                100.0 - presence,
                                60.0,
                                98.0,
                            )
                        )
                        occupied = False
                    else:
                        value = "UNREADABLE"
                        occupied = True

            wells.append(
                WellResult(
                    position=position,
                    value=value,
                    confidence=confidence,
                    crop_rgb=crop_rgb,
                    occupied=occupied,
                    center_xy=(center_x, center_y),
                )
            )

    _enforce_unique_codes(wells)
    _sort_wells_canonical(wells)
    overlay = _draw_overlay(
        oriented_bgr,
        wells,
    )

    return RackResult(
        plate_id=plate_id,
        wells=wells,
        normalized_rgb=cv2.cvtColor(
            oriented_bgr,
            cv2.COLOR_BGR2RGB,
        ),
        overlay_rgb=overlay,
        profile_name="Original tray — perspective robust",
    )


def _decoded_well_count(
    result: RackResult,
) -> int:
    return sum(
        well.value not in ("EMPTY", "UNREADABLE")
        for well in result.wells
    )


def analyze_original(
    image_bgr: np.ndarray,
) -> RackResult:
    # Run both methods when possible. The fixed-geometry method remains very
    # strong for the original straight-on photography setup; the lattice
    # method handles rotation and oblique perspective.
    results = []
    errors = []

    for analyzer in (
        analyze_original_lattice,
        analyze_original_legacy,
    ):
        try:
            results.append(analyzer(image_bgr))
        except Exception as exc:
            errors.append(str(exc))

    if not results:
        raise RuntimeError(
            "Original tray analysis failed.\n\n"
            + "\n".join(errors)
        )

    results.sort(
        key=_decoded_well_count,
        reverse=True,
    )
    return results[0]


# ---------- notched plate ----------
# New black notched plate:
# - notch at A1
# - plate Data Matrix on the white strip along plate column 1
# - row letters run left-to-right after normalization
# - column numbers run top-to-bottom

NOTCHED_GRID_X_FALLBACK = np.array(
    [0.108, 0.218, 0.326, 0.435, 0.543, 0.652, 0.760, 0.870],
    dtype=float,
)
NOTCHED_GRID_Y_FALLBACK = np.array(
    [0.110, 0.179, 0.248, 0.317, 0.386, 0.455,
     0.524, 0.593, 0.662, 0.731, 0.800, 0.869],
    dtype=float,
)
NOTCHED_MARKER_TOP_ROI = (0.10, 0.00, 0.80, 0.14)


def _perspective_normalize_dark_plate(
    image_bgr: np.ndarray,
    out_w: int = 1000,
    out_h: int = 1450,
) -> np.ndarray:
    """Rectify the dark 3-D printed plate against a lighter background."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < 120).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)),
        iterations=2,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("Could not locate the dark notched plate.")

    contour = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(contour) / float(
        image_bgr.shape[0] * image_bgr.shape[1]
    )
    if area_ratio < 0.20:
        raise RuntimeError(
            "The notched plate occupies too little of the image. "
            "Retake the photo closer and directly overhead."
        )

    box = _order_quad(cv2.boxPoints(cv2.minAreaRect(contour)))
    width = max(
        np.linalg.norm(box[1] - box[0]),
        np.linalg.norm(box[2] - box[3]),
    )
    height = max(
        np.linalg.norm(box[3] - box[0]),
        np.linalg.norm(box[2] - box[1]),
    )

    if width > height:
        box = np.array([box[1], box[2], box[3], box[0]], dtype=np.float32)

    target = np.array(
        [[0, 0], [out_w - 1, 0],
         [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(box, target)
    return cv2.warpPerspective(image_bgr, matrix, (out_w, out_h))


def _dark_corner_fractions(image_bgr: np.ndarray) -> list[float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    patch = int(min(h, w) * 0.13)
    corners = [
        gray[:patch, :patch],
        gray[:patch, w - patch:w],
        gray[h - patch:h, w - patch:w],
        gray[h - patch:h, :patch],
    ]
    return [float((corner < 120).mean()) for corner in corners]


def _notched_orientation(
    normalized_bgr: np.ndarray,
) -> tuple[np.ndarray, str]:
    candidates = []
    for turns in range(4):
        rotated = _rotate_90(normalized_bgr, turns)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        h, _ = gray.shape

        fills = _dark_corner_fractions(rotated)
        notch_score = (sum(fills[1:]) / 3.0) - fills[0]
        top_bright = float((gray[:int(0.10 * h)] > 200).mean())
        bottom_bright = float((gray[-int(0.10 * h):] > 200).mean())
        strip_score = 0.30 * (top_bright - bottom_bright)

        marker = _decode_marker_roi(rotated, NOTCHED_MARKER_TOP_ROI)
        marker_bonus = 0.50 if marker else 0.0
        candidates.append(
            (notch_score + strip_score + marker_bonus, rotated, marker)
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, oriented, plate_id = candidates[0]

    if not plate_id:
        plate_id = _decode_anywhere_plate(oriented)
    if not plate_id:
        raise RuntimeError("The plate-level Data Matrix could not be decoded.")

    return oriented, plate_id


def _decode_anywhere_plate(image_bgr: np.ndarray) -> Optional[str]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    for scale in (1.0, 2.0, 3.0):
        for variant in (
            image_bgr,
            cv2.equalizeHist(gray),
            cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1],
        ):
            candidate = (
                variant if scale == 1.0 else
                cv2.resize(
                    variant, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )
            )
            for code in _read_datamatrix(candidate):
                text = code.text.strip().lower()
                if text:
                    return text
    return None


def _kmeans_axis(values: np.ndarray, count: int) -> Optional[np.ndarray]:
    if len(values) < count:
        return None
    data = values.astype(np.float32).reshape(-1, 1)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.1,
    )
    _, _, centers = cv2.kmeans(
        data,
        count,
        None,
        criteria,
        20,
        cv2.KMEANS_PP_CENTERS,
    )
    return np.sort(centers.ravel().astype(float))


def _detect_grid_centers_notched(
    oriented_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the regular 8x12 hole lattice from circular plate openings."""
    gray = cv2.cvtColor(oriented_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=45,
        param1=80,
        param2=30,
        minRadius=25,
        maxRadius=52,
    )

    if circles is None:
        return NOTCHED_GRID_X_FALLBACK * w, NOTCHED_GRID_Y_FALLBACK * h

    detected = circles[0]
    detected = detected[
        (detected[:, 0] > 0.04 * w)
        & (detected[:, 0] < 0.91 * w)
        & (detected[:, 1] > 0.08 * h)
        & (detected[:, 1] < 0.91 * h)
    ]

    xs = _kmeans_axis(detected[:, 0], 8) if len(detected) else None
    ys = _kmeans_axis(detected[:, 1], 12) if len(detected) else None

    if xs is None or ys is None:
        return NOTCHED_GRID_X_FALLBACK * w, NOTCHED_GRID_Y_FALLBACK * h

    return xs, ys


def _notched_tube_present(crop_bgr: np.ndarray) -> tuple[bool, float]:
    """Detect the white circular tube-bottom label inside a dark opening."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    radius = min(h, w) * 0.30
    center_mask = (
        (xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2 <= radius ** 2
    )
    values = gray[center_mask]
    white_fraction = float((values > 170).mean()) if values.size else 0.0

    present = white_fraction >= 0.45
    confidence = float(
        np.clip(
            100.0 * (
                white_fraction / 0.75 if present
                else (0.45 - white_fraction) / 0.45
            ),
            0.0,
            99.0,
        )
    )
    return present, confidence


def _full_image_votes_notched(
    image_bgr: np.ndarray,
    plate_id: str,
    xs: np.ndarray,
    ys: np.ndarray,
) -> dict[tuple[int, int], Counter[str]]:
    votes: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    x_spacing = float(np.median(np.diff(xs)))
    y_spacing = float(np.median(np.diff(ys)))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    base_variants = [
        image_bgr,
        cv2.equalizeHist(gray),
        cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1],
    ]

    for scale in (1.0, 2.0, 3.0):
        for variant in base_variants:
            candidate = (
                variant if scale == 1.0 else
                cv2.resize(
                    variant, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )
            )
            for code in _read_datamatrix(candidate):
                text = code.text.strip().lower()
                if not text or text == plate_id:
                    continue

                cx, cy = _barcode_center(code)
                cx /= scale
                cy /= scale
                xi = int(np.argmin(np.abs(xs - cx)))
                yi = int(np.argmin(np.abs(ys - cy)))

                if (
                    abs(xs[xi] - cx) <= 0.55 * x_spacing
                    and abs(ys[yi] - cy) <= 0.55 * y_spacing
                ):
                    votes[(xi, yi)][text] += 1

    return votes


def analyze_notched_plate_legacy(image_bgr: np.ndarray) -> RackResult:
    normalized_bgr = _perspective_normalize_dark_plate(
        image_bgr, 1000, 1450
    )
    oriented_bgr, plate_id = _notched_orientation(normalized_bgr)
    xs, ys = _detect_grid_centers_notched(oriented_bgr)
    full_votes = _full_image_votes_notched(
        oriented_bgr, plate_id, xs, ys
    )

    h, w = oriented_bgr.shape[:2]
    radius = int(
        min(np.median(np.diff(xs)), np.median(np.diff(ys))) * 0.50
    )
    wells = []

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            position = f"{ROWS[xi]}{yi + 1}"
            x1, x2 = max(0, int(x - radius)), min(w, int(x + radius))
            y1, y2 = max(0, int(y - radius)), min(h, int(y + radius))
            crop_bgr = oriented_bgr[y1:y2, x1:x2].copy()
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            position_votes = full_votes.get((xi, yi), Counter())
            if position_votes:
                ranked = position_votes.most_common()
                value, best_votes = ranked[0]
                second_votes = ranked[1][1] if len(ranked) > 1 else 0
                agreement = best_votes / max(1, sum(position_votes.values()))
                margin = (best_votes - second_votes) / max(1, best_votes)
                confidence = float(
                    np.clip(78.0 + 12.0 * agreement + 10.0 * margin, 0, 100)
                )
                occupied = True
            else:
                occupied, presence_confidence = _notched_tube_present(crop_bgr)
                if not occupied:
                    value = "EMPTY"
                    confidence = presence_confidence
                else:
                    decoded, confidence = _decode_crop(crop_bgr, plate_id)
                    value = decoded if decoded else "UNREADABLE"

            wells.append(
                WellResult(
                    position=position,
                    value=value,
                    confidence=confidence,
                    crop_rgb=crop_rgb,
                    occupied=occupied,
                    center_xy=(float(x), float(y)),
                )
            )

    _enforce_unique_codes(wells)
    _sort_wells_canonical(wells)
    overlay = _draw_overlay(oriented_bgr, wells)
    return RackResult(
        plate_id=plate_id,
        wells=wells,
        normalized_rgb=cv2.cvtColor(oriented_bgr, cv2.COLOR_BGR2RGB),
        overlay_rgb=overlay,
        profile_name="Notched plate",
    )



# ---------- perspective-robust notched plate ----------
#
# The notched plate can be sparse, so tube barcode centers alone are not enough.
# This path combines circular hole centers with any decoded tube barcode centers,
# fits them to an 8 x 12 projective lattice, rectifies the plate, then uses the
# notch / plate-marker side to establish A1 orientation.

def _detect_notched_holes(
    image_bgr: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 1400.0 / max(h, w)
    work = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    work = cv2.GaussianBlur(work, (5, 5), 0)

    circles = cv2.HoughCircles(
        work,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=38,
        param1=90,
        param2=17,
        minRadius=16,
        maxRadius=45,
    )

    if circles is None:
        return np.empty((0, 2), dtype=np.float32)

    centers = circles[0, :, :2] / scale
    return centers.astype(np.float32)


def _principal_axes(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = points.mean(axis=0)
    centered = points - mean
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes.T
    return mean, axes, projected


def _cluster_projected_axis(
    values: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = values.astype(np.float32).reshape(-1, 1)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.01,
    )
    _, labels, centers = cv2.kmeans(
        data,
        count,
        None,
        criteria,
        40,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.ravel()
    order = np.argsort(centers)
    remap = np.empty(count, dtype=int)
    remap[order] = np.arange(count)
    sorted_centers = centers[order]
    return remap[labels.ravel()], sorted_centers


def _fit_notched_lattice_from_points(
    points: np.ndarray,
) -> tuple[np.ndarray, tuple[int, float]]:
    if len(points) < 36:
        raise RuntimeError(
            "Too few circular well centers were detected to fit the notched "
            "plate perspective lattice."
        )

    _, _, projected = _principal_axes(points)
    candidates = []

    for case in (0, 1):
        if case == 0:
            v_labels, _ = _cluster_projected_axis(projected[:, 0], 12)
            u_labels, _ = _cluster_projected_axis(projected[:, 1], 8)
        else:
            u_labels, _ = _cluster_projected_axis(projected[:, 0], 8)
            v_labels, _ = _cluster_projected_axis(projected[:, 1], 12)

        canonical = np.column_stack([v_labels, u_labels]).astype(np.float32)

        homography, _ = cv2.findHomography(
            canonical,
            points,
            cv2.RANSAC,
            24.0,
        )
        if homography is None:
            continue

        predicted = cv2.perspectiveTransform(
            canonical.reshape(-1, 1, 2),
            homography,
        ).reshape(-1, 2)
        errors = np.linalg.norm(predicted - points, axis=1)
        inliers = errors < 24.0
        if not inliers.any():
            continue

        score = (
            int(inliers.sum()),
            -float(np.median(errors[inliers])),
        )
        candidates.append((score, homography))

    if not candidates:
        raise RuntimeError(
            "The detected notched-plate holes did not form a reliable 8 x 12 lattice."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_h = candidates[0]
    if best_score[0] < 32:
        raise RuntimeError(
            "The notched-plate lattice fit was too weak for dependable mapping."
        )
    return best_h, best_score


def _warp_notched_generic(
    image_bgr: np.ndarray,
    homography: np.ndarray,
    scale: int = 110,
) -> np.ndarray:
    # canonical coordinates are [v, u], where v is 12-position axis and
    # u is 8-position axis. Add broad margins for notch and plate marker.
    canonical_to_output = np.array(
        [
            [0, scale, 1.5 * scale],
            [scale, 0, 1.8 * scale],
            [0, 0, 1],
        ],
        dtype=float,
    )
    image_to_output = canonical_to_output @ np.linalg.inv(homography)
    return cv2.warpPerspective(
        image_bgr,
        image_to_output,
        (11 * scale, 16 * scale),
    )


def _decode_plate_marker_candidates_notched(
    generic_bgr: np.ndarray,
) -> list[tuple[str, float, float]]:
    # Marker is adjacent to plate column 1 and biased toward A1. Because the
    # lattice may be flipped, inspect all four outside ends and let geometry
    # plus notch decide orientation.
    scale = 110
    candidate_positions = (
        (-1.0, -0.3),   # above/left of A1 side
        (-1.0, 7.3),    # above/right
        (12.0, -0.3),   # below/left
        (12.0, 7.3),    # below/right
    )

    found = []
    for v, u in candidate_positions:
        cx = int((u + 1.5) * scale)
        cy = int((v + 1.8) * scale)
        crop = generic_bgr[
            max(0, cy - 110):min(generic_bgr.shape[0], cy + 110),
            max(0, cx - 120):min(generic_bgr.shape[1], cx + 120),
        ]
        text = _decode_marker_crop_robust(crop)
        if text:
            found.append((text, v, u))
    return found


def _corner_darkness_score(
    generic_bgr: np.ndarray,
    v: float,
    u: float,
) -> float:
    scale = 110
    cx = int((u + 1.5) * scale)
    cy = int((v + 1.8) * scale)
    crop = generic_bgr[
        max(0, cy - 100):min(generic_bgr.shape[0], cy + 100),
        max(0, cx - 100):min(generic_bgr.shape[1], cx + 100),
    ]
    if crop.size == 0:
        return 1.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float((gray < 90).mean())


def _orient_notched_homography(
    image_bgr: np.ndarray,
    homography: np.ndarray,
) -> tuple[np.ndarray, str]:
    generic = _warp_notched_generic(image_bgr, homography)
    marker_candidates = _decode_plate_marker_candidates_notched(generic)

    # The A1 notch and plate barcode are on the same plate-column-1 side, with
    # the marker biased toward A1. Evaluate the four possible lattice flips.
    transforms = [
        np.eye(3, dtype=float),
        np.array([[-1, 0, 11], [0, 1, 0], [0, 0, 1]], dtype=float),
        np.array([[1, 0, 0], [0, -1, 7], [0, 0, 1]], dtype=float),
        np.array([[-1, 0, 11], [0, -1, 7], [0, 0, 1]], dtype=float),
    ]

    best = None
    for transform in transforms:
        candidate_h = homography @ transform
        inverse_transform = np.linalg.inv(transform)

        # In final coordinates the desired marker position is just before
        # plate column 1 and skewed toward row A: approximately v=-1, u<1.
        marker_score = -10.0
        plate_id = None
        for text, v, u in marker_candidates:
            final_coord = inverse_transform @ np.array([v, u, 1.0])
            fv = float(final_coord[0] / final_coord[2])
            fu = float(final_coord[1] / final_coord[2])
            distance = float(np.hypot(fv + 1.0, fu - 0.2))
            score = 4.0 - distance
            if score > marker_score:
                marker_score = score
                plate_id = text

        rectified = _warp_notched_generic(image_bgr, candidate_h)
        # Notch should remove dark plate material at the A1-side corner more
        # than at the other three corners.
        a1_dark = _corner_darkness_score(rectified, -0.9, -0.5)
        other_dark = np.mean([
            _corner_darkness_score(rectified, -0.9, 7.5),
            _corner_darkness_score(rectified, 11.9, -0.5),
            _corner_darkness_score(rectified, 11.9, 7.5),
        ])
        notch_score = float(other_dark - a1_dark)

        total = marker_score + 2.5 * notch_score
        if best is None or total > best[0]:
            best = (total, candidate_h, plate_id)

    if best is None:
        raise RuntimeError("Could not orient the notched plate.")

    _, final_h, plate_id = best

    if not plate_id:
        # Decode the marker after final orientation in a broader A1-side strip.
        rectified = _warp_notched_generic(image_bgr, final_h)
        h, w = rectified.shape[:2]
        marker_strip = rectified[
            0:int(0.24 * h),
            0:int(0.40 * w),
        ]
        plate_id = _decode_marker_crop_robust(marker_strip)

    if not plate_id:
        raise RuntimeError(
            "The notched plate Data Matrix could not be decoded after "
            "perspective correction."
        )

    return final_h, plate_id


def _rectify_notched_final(
    image_bgr: np.ndarray,
    homography: np.ndarray,
    scale: int = 110,
) -> np.ndarray:
    # Final coordinates: u=0..7 is A..H; v=0..11 is plate columns 1..12.
    canonical_to_output = np.array(
        [
            [0, scale, 1.35 * scale],
            [scale, 0, 1.45 * scale],
            [0, 0, 1],
        ],
        dtype=float,
    )
    image_to_output = canonical_to_output @ np.linalg.inv(homography)
    return cv2.warpPerspective(
        image_bgr,
        image_to_output,
        (10.5 * scale, 15.2 * scale),
    )


def analyze_notched_plate_lattice(
    image_bgr: np.ndarray,
) -> RackResult:
    hole_centers = _detect_notched_holes(image_bgr)
    homography, _ = _fit_notched_lattice_from_points(hole_centers)
    final_h, plate_id = _orient_notched_homography(
        image_bgr,
        homography,
    )
    oriented_bgr = _rectify_notched_final(
        image_bgr,
        final_h,
        scale=110,
    )

    # Decode all visible tube codes from the rectified image, then assign to
    # nearest canonical cell. Empty holes remain available for geometry even
    # when the rack is sparsely populated.
    detections = _detect_datamatrix_multiscale(oriented_bgr)
    position_votes: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)

    for text, center in detections:
        if text == plate_id:
            continue
        u = center[0] / 110.0 - 1.35
        v = center[1] / 110.0 - 1.45
        xi = int(round(u))
        yi = int(round(v))
        if (
            0 <= xi < 8
            and 0 <= yi < 12
            and abs(u - xi) < 0.50
            and abs(v - yi) < 0.50
        ):
            position_votes[(xi, yi)][text] += 1

    wells: list[WellResult] = []
    radius = 52

    for yi in range(12):
        for xi in range(8):
            position = f"{ROWS[xi]}{yi + 1}"
            cx = float((xi + 1.35) * 110)
            cy = float((yi + 1.45) * 110)
            crop_bgr = oriented_bgr[
                int(cy) - radius:int(cy) + radius,
                int(cx) - radius:int(cx) + radius,
            ].copy()
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            votes = position_votes.get((xi, yi), Counter())
            if votes:
                ranked = votes.most_common()
                value, best_votes = ranked[0]
                second_votes = ranked[1][1] if len(ranked) > 1 else 0
                agreement = best_votes / max(1, sum(votes.values()))
                margin = (best_votes - second_votes) / max(1, best_votes)
                confidence = float(np.clip(
                    78.0 + 12.0 * agreement + 10.0 * margin,
                    0.0,
                    100.0,
                ))
                occupied = True
            else:
                decoded, confidence = _decode_crop(crop_bgr, plate_id)
                if decoded:
                    value = decoded
                    occupied = True
                else:
                    presence = _presence_score_notched(crop_bgr)
                    if presence < 24.0:
                        value = "EMPTY"
                        confidence = float(np.clip(100.0 - presence, 70.0, 99.0))
                        occupied = False
                    else:
                        value = "UNREADABLE"
                        occupied = True

            wells.append(
                WellResult(
                    position=position,
                    value=value,
                    confidence=confidence,
                    crop_rgb=crop_rgb,
                    occupied=occupied,
                    center_xy=(cx, cy),
                )
            )

    _enforce_unique_codes(wells)
    _sort_wells_canonical(wells)
    overlay = _draw_overlay(oriented_bgr, wells)

    return RackResult(
        plate_id=plate_id,
        wells=wells,
        normalized_rgb=cv2.cvtColor(oriented_bgr, cv2.COLOR_BGR2RGB),
        overlay_rgb=overlay,
        profile_name="Notched plate — perspective robust",
    )


def analyze_notched_plate(
    image_bgr: np.ndarray,
) -> RackResult:
    results = []
    errors = []

    for analyzer in (
        analyze_notched_plate_lattice,
        analyze_notched_plate_legacy,
    ):
        try:
            results.append(analyzer(image_bgr))
        except Exception as exc:
            errors.append(str(exc))

    if not results:
        raise RuntimeError(
            "Notched plate analysis failed.\n\n"
            + "\n".join(errors)
        )

    results.sort(
        key=_decoded_well_count,
        reverse=True,
    )
    return results[0]

def analyze_image_bytes(image_bytes: bytes, profile_choice: str = "auto") -> RackResult:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("The selected image could not be opened.")
    choice = (profile_choice or "auto").strip().lower()
    if choice == "original tray":
        return analyze_original(image_bgr)
    if choice == "notched plate":
        return analyze_notched_plate(image_bgr)
    # Auto profile selection: the notched 3-D printed plate is dark across
    # the central field, while the original molded tray is substantially
    # brighter. This prevents a tube code from being mistaken for a plate ID.
    h, w = image_bgr.shape[:2]
    center_gray = cv2.cvtColor(
        image_bgr[
            int(0.15 * h):int(0.85 * h),
            int(0.15 * w):int(0.85 * w),
        ],
        cv2.COLOR_BGR2GRAY,
    )
    center_median = float(np.median(center_gray))

    preferred = (
        analyze_notched_plate
        if center_median < 110.0
        else analyze_original
    )
    alternate = (
        analyze_original
        if preferred is analyze_notched_plate
        else analyze_notched_plate
    )

    errors = []
    for fn in (preferred, alternate):
        try:
            return fn(image_bgr)
        except Exception as exc:
            errors.append(str(exc))

    raise RuntimeError(
        "Auto-detection failed. Try selecting the rack style manually.\n\n"
        + "\n".join(errors)
    )
