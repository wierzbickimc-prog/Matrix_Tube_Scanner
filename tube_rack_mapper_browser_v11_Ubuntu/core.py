
from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
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


# ---------- double-rib plate profile ----------
#
# Separate beige square-well plate. The paired support ribs at H12 are used
# only for this plate type and are not applied to the original or notched racks.

def _detect_square_well_centers(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        7,
    )
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 250 or area > 6500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(1.0, h)
        if not 0.58 < aspect < 1.55:
            continue

        fill = area / max(1.0, w * h)
        if fill < 0.30:
            continue

        centers.append((x + w / 2.0, y + h / 2.0))

    if len(centers) < 36:
        raise RuntimeError(
            "Too few square well openings were detected for the double-rib plate."
        )

    return np.asarray(centers, dtype=np.float32)


def _warp_double_rib_generic(
    image_bgr: np.ndarray,
    homography: np.ndarray,
    scale: int = 100,
) -> np.ndarray:
    canonical_to_output = np.array(
        [
            [0, scale, 1.5 * scale],
            [scale, 0, 1.5 * scale],
            [0, 0, 1],
        ],
        dtype=float,
    )
    image_to_output = canonical_to_output @ np.linalg.inv(homography)
    return cv2.warpPerspective(
        image_bgr,
        image_to_output,
        (11 * scale, 15 * scale),
    )


def _vertical_rib_groups(
    roi_gray: np.ndarray,
) -> list[float]:
    edges = cv2.Canny(roi_gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=35,
        minLineLength=max(45, int(roi_gray.shape[0] * 0.30)),
        maxLineGap=14,
    )

    xs = []
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, line)
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if 76.0 <= angle <= 104.0:
                xs.append((x1 + x2) / 2.0)

    if not xs:
        return []

    xs.sort()
    groups = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] <= 10.0:
            groups[-1].append(x)
        else:
            groups.append([x])

    return [float(np.mean(group)) for group in groups]


def _double_rib_h12_score(rectified_bgr: np.ndarray) -> float:
    h, w = rectified_bgr.shape[:2]

    # After lattice rectification, H12 is the lower-right corner. The profile-
    # specific doubled support is in the outer tray margin beneath/right of the
    # square-well field.
    y0, y1 = int(0.80 * h), int(0.995 * h)
    x0, x1 = int(0.55 * w), int(0.95 * w)
    roi = rectified_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return -100.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=28,
        minLineLength=max(35, int(roi.shape[0] * 0.12)),
        maxLineGap=14,
    )

    if lines is None:
        return -100.0

    # Collect long, nearly vertical edges in the lower support region.
    edge_lines = []
    for line in lines[:, 0]:
        xa, ya, xb, yb = map(int, line)
        angle = abs(np.degrees(np.arctan2(yb - ya, xb - xa)))
        length = float(np.hypot(xb - xa, yb - ya))
        ymin, ymax = min(ya, yb), max(ya, yb)

        if (
            78.0 <= angle <= 102.0
            and ymin > roi.shape[0] * 0.30
            and ymax > roi.shape[0] * 0.52
        ):
            edge_lines.append(
                ((xa + xb) / 2.0 + x0, length)
            )

    if len(edge_lines) < 4:
        return -100.0

    # Two edge lines from the same physical rib are grouped into one rib
    # centerline. This avoids the v9 failure mode where one ordinary rib was
    # incorrectly interpreted as the doubled H12 support.
    edge_lines.sort(key=lambda item: item[0])
    groups = []
    for x, length in edge_lines:
        if not groups or x - groups[-1][-1][0] > 15.0:
            groups.append([(x, length)])
        else:
            groups[-1].append((x, length))

    rib_centers = []
    for group in groups:
        weights = np.asarray(
            [length for _, length in group],
            dtype=float,
        )
        xs = np.asarray(
            [x for x, _ in group],
            dtype=float,
        )
        rib_centers.append(
            (
                float(np.average(xs, weights=weights)),
                float(weights.max()),
            )
        )

    best = -100.0

    # The doubled support is two distinct rib centerlines with a much smaller
    # separation than the normal tray-support spacing.
    for index, (x_a, len_a) in enumerate(rib_centers):
        for x_b, len_b in rib_centers[index + 1:]:
            gap = x_b - x_a
            pair_center = (x_a + x_b) / 2.0

            if (
                45.0 <= gap <= 110.0
                and 0.60 * w < pair_center < 0.90 * w
            ):
                gap_quality = 1.0 - abs(gap - 75.0) / 35.0
                position_quality = (
                    1.0
                    - abs(pair_center / w - 0.78) / 0.18
                )
                length_quality = min(
                    1.0,
                    (len_a + len_b) / 120.0,
                )

                score = (
                    6.0 * gap_quality
                    + 3.0 * position_quality
                    + 3.0 * length_quality
                )
                best = max(best, float(score))

    return best

def _orient_double_rib_homography(
    image_bgr: np.ndarray,
    homography: np.ndarray,
    corner_override: Optional[str] = None,
) -> tuple[np.ndarray, float]:
    transforms = (
        np.eye(3, dtype=float),
        np.array([[-1, 0, 11], [0, 1, 0], [0, 0, 1]], dtype=float),
        np.array([[1, 0, 0], [0, -1, 7], [0, 0, 1]], dtype=float),
        np.array([[-1, 0, 11], [0, -1, 7], [0, 0, 1]], dtype=float),
    )

    override = (corner_override or "auto").strip().lower()

    if override != "auto":
        requested = {
            "top-left": np.array([0.0, 0.0]),
            "top-right": np.array([image_bgr.shape[1], 0.0]),
            "bottom-left": np.array([0.0, image_bgr.shape[0]]),
            "bottom-right": np.array([image_bgr.shape[1], image_bgr.shape[0]]),
        }.get(override)

        if requested is None:
            raise RuntimeError(
                f"Unknown double-rib corner override: {corner_override}"
            )

        candidates = []
        for transform in transforms:
            candidate_h = homography @ transform

            # H12 in final coordinates is canonical [v=11, u=7].
            point = cv2.perspectiveTransform(
                np.array([[[11.0, 7.0]]], dtype=np.float32),
                candidate_h,
            ).reshape(2)

            distance = float(np.linalg.norm(point - requested))
            candidates.append((distance, candidate_h))

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1], 100.0

    candidates = []
    for transform in transforms:
        candidate_h = homography @ transform
        rectified = _warp_double_rib_generic(image_bgr, candidate_h)
        score = _double_rib_h12_score(rectified)
        candidates.append((score, candidate_h))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_h = candidates[0]

    if best_score < 0:
        raise RuntimeError(
            "The H12 double-rib orientation feature could not be identified reliably. "
            "Select the visible H12 double-rib corner manually in the browser."
        )

    return best_h, float(best_score)

def _rectify_double_rib_final(
    image_bgr: np.ndarray,
    homography: np.ndarray,
    scale: int = 100,
) -> np.ndarray:
    canonical_to_output = np.array(
        [
            [0, scale, 1.5 * scale],
            [scale, 0, 1.5 * scale],
            [0, 0, 1],
        ],
        dtype=float,
    )
    image_to_output = canonical_to_output @ np.linalg.inv(homography)
    return cv2.warpPerspective(
        image_bgr,
        image_to_output,
        (11 * scale, 15 * scale),
    )


def _presence_score_double_rib(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    margin = int(min(h, w) * 0.22)
    inner = gray[margin:h-margin, margin:w-margin]
    if inner.size == 0:
        return 0.0

    mean = float(inner.mean())
    std = float(inner.std())
    edges = cv2.Canny(inner, 45, 120)
    edge_density = float((edges > 0).mean())

    # Empty square wells are dark and comparatively smooth. Tube bottoms carry
    # the light/black Data Matrix texture.
    brightness = np.clip((mean - 42.0) / 105.0, 0.0, 1.0)
    texture = np.clip(std / 48.0, 0.0, 1.0)
    edge_component = np.clip(edge_density / 0.18, 0.0, 1.0)
    return float(
        100.0 * (
            0.56 * brightness
            + 0.26 * texture
            + 0.18 * edge_component
        )
    )


def _decode_plate_id_outside_double_rib_grid(
    rectified_bgr: np.ndarray,
) -> Optional[str]:
    h, w = rectified_bgr.shape[:2]

    # Search the outer frame while excluding the 8 x 12 well field. A future
    # double-rib tray with a visible rack Data Matrix will be picked up here.
    regions = (
        rectified_bgr[0:int(0.12*h), :],
        rectified_bgr[int(0.88*h):h, :],
        rectified_bgr[:, 0:int(0.12*w)],
        rectified_bgr[:, int(0.88*w):w],
    )

    for region in regions:
        text = _decode_marker_crop_robust(region)
        if text:
            return text

    return None



def _decode_crop_fast(
    crop_bgr: np.ndarray,
    plate_id: str,
) -> tuple[Optional[str], float]:
    """Fast first-pass decoder used by the sparse double-rib profile."""
    votes: Counter[str] = Counter()
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)

    variants = (
        crop_bgr,
        clahe,
        cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1],
    )

    for scale in (1.8, 2.6):
        for variant in variants:
            enlarged = cv2.resize(
                variant,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            for code in _read_datamatrix(enlarged):
                text = code.text.strip().lower()
                if text and text != plate_id:
                    votes[text] += 1

    if not votes:
        return None, 0.0

    ranked = votes.most_common()
    best_text, best_votes = ranked[0]
    second_votes = ranked[1][1] if len(ranked) > 1 else 0
    agreement = best_votes / max(1, sum(votes.values()))
    margin = (best_votes - second_votes) / max(1, best_votes)

    confidence = float(np.clip(
        65.0
        + 20.0 * min(1.0, best_votes / 3.0)
        + 10.0 * agreement
        + 5.0 * margin,
        0.0,
        100.0,
    ))
    return best_text, confidence


def _external_datamatrix_codes_for_lattice(
    image_bgr: np.ndarray,
    homography: np.ndarray,
) -> list[str]:
    """
    Return Data Matrix values whose centers lie outside the fitted 8 x 12 grid.

    The canonical lattice is [v=0..11, u=0..7]. A genuine Double-rib plate has
    no plate-ID Data Matrix outside that well field. Original and Notched plate
    types do have an external rack/plate marker.
    """
    detections = _detect_datamatrix_multiscale(image_bgr)
    if not detections:
        return []

    inverse = np.linalg.inv(homography)
    external: list[str] = []

    for text, center in detections:
        canonical = cv2.perspectiveTransform(
            np.asarray(center, dtype=np.float32).reshape(1, 1, 2),
            inverse,
        ).reshape(2)

        v, u = map(float, canonical)

        # Keep a modest tolerance for projective-fit error at edge wells, while
        # still classifying the normal rack marker position (~one lattice pitch
        # outside the field) as external.
        inside_grid = (
            -0.60 <= v <= 11.60
            and -0.60 <= u <= 7.60
        )

        if not inside_grid and text not in external:
            external.append(text)

    return external


def _double_rib_auto_evidence(
    image_bgr: np.ndarray,
) -> tuple[bool, str]:
    """
    Strict Auto-mode gate for the Double-rib profile.

    Required:
      1. strong square-well lattice,
      2. usable H12 doubled-rib orientation evidence,
      3. NO external Data Matrix / plate-ID marker.

    This prevents an Original tray from being promoted to Double-rib merely
    because its molded grid creates square-like contours.
    """
    square_centers = _detect_square_well_centers(image_bgr)
    if len(square_centers) < 48:
        return False, (
            f"only {len(square_centers)} square-well candidates "
            "(need at least 48)"
        )

    homography, fit_score = _fit_notched_lattice_from_points(square_centers)

    external_codes = _external_datamatrix_codes_for_lattice(
        image_bgr,
        homography,
    )
    if external_codes:
        return False, (
            "external plate/rack Data Matrix detected: "
            + ", ".join(external_codes[:3])
        )

    try:
        _, rib_score = _orient_double_rib_homography(
            image_bgr,
            homography,
            corner_override="auto",
        )
    except Exception as exc:
        return False, f"H12 double-rib anchor failed: {exc}"

    if rib_score < 0:
        return False, f"H12 double-rib score too weak: {rib_score:.2f}"

    return True, (
        f"square lattice inliers={fit_score[0]}, "
        f"H12 rib score={rib_score:.2f}, "
        "no external plate-ID barcode"
    )

def analyze_double_rib_plate_legacy(
    image_bgr: np.ndarray,
    plate_id_override: Optional[str] = None,
    double_rib_corner_override: Optional[str] = None,
) -> RackResult:
    square_centers = _detect_square_well_centers(image_bgr)

    # Reuse the projective 8 x 12 lattice fitter; the input features here are
    # square-well centers rather than circular holes.
    homography, fit_score = _fit_notched_lattice_from_points(square_centers)
    final_h, rib_score = _orient_double_rib_homography(
        image_bgr,
        homography,
        corner_override=double_rib_corner_override,
    )
    oriented_bgr = _rectify_double_rib_final(
        image_bgr,
        final_h,
        scale=100,
    )

    plate_id = (plate_id_override or "").strip().lower()
    if not plate_id:
        plate_id = _decode_plate_id_outside_double_rib_grid(oriented_bgr) or "UNREADABLE"

    detections = _detect_datamatrix_multiscale(oriented_bgr)
    position_votes: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)

    for text, center in detections:
        if text == plate_id:
            continue

        u = center[0] / 100.0 - 1.5
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
    radius = 48

    for yi in range(12):
        for xi in range(8):
            position = f"{ROWS[xi]}{yi + 1}"
            cx = float((xi + 1.5) * 100)
            cy = float((yi + 1.5) * 100)

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
                # Sparse double-rib racks may contain only a handful of tubes.
                # Classify tube presence before running barcode preprocessing so
                # empty wells do not trigger the expensive multi-angle decoder.
                presence = _presence_score_double_rib(crop_bgr)

                if presence < 27.0:
                    value = "EMPTY"
                    confidence = float(np.clip(
                        100.0 - presence,
                        70.0,
                        99.0,
                    ))
                    occupied = False
                else:
                    occupied = True
                    decoded, confidence = _decode_crop_fast(
                        crop_bgr,
                        plate_id,
                    )

                    # Escalate only genuinely occupied hard cases to the exhaustive
                    # rotation/threshold search.
                    if not decoded:
                        decoded, confidence = _decode_crop(
                            crop_bgr,
                            plate_id,
                        )

                    value = decoded if decoded else "UNREADABLE"

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
        normalized_rgb=cv2.cvtColor(
            oriented_bgr,
            cv2.COLOR_BGR2RGB,
        ),
        overlay_rgb=overlay,
        profile_name=(
            "Double-rib plate — v11 square-well lattice / H12 rib anchor"
        ),
    )


# ---------------------------------------------------------------------------
# v12 Double-rib plate: resolution-independent lattice, full-resolution decode
# ---------------------------------------------------------------------------
#
# v11 detected square openings with fixed pixel-area limits, so a full-size
# phone photo (e.g. 4284 x 5712) found no wells, and its k-means lattice fit
# could shift a whole column on oblique shots. Tube presence was a brightness
# threshold that called glare and lit walls "occupied", and the plate-ID search
# strip overlapped the first well row, so an edge tube's code became the "plate
# ID" and that tube was then suppressed.
#
# v12:
#   * fits geometry on a fixed-size working copy; lattice points are dark
#     square openings (size-relative limits) plus whole-image Data Matrix hits;
#   * grows the lattice neighbour-to-neighbour, refines a homography, and places
#     the 8 x 12 window where the plate's beige walls are;
#   * keeps the v10/v11 H12 double-rib orientation logic unchanged;
#   * reads whole-image codes at full and reduced resolution, then decodes the
#     remaining occupied wells from perspective-corrected full-resolution crops;
#   * decides presence from Data-Matrix-scale texture (glare is smooth);
#   * reports a plate ID only for a code clearly outside the well field.

DR12_WORK_LONG_SIDE = 2048


def _dr12_work_image(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    scale = DR12_WORK_LONG_SIDE / max(h, w)
    if abs(scale - 1.0) < 0.02:
        return image_bgr, 1.0
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=interp), scale


def _dr12_full_image_codes(image_bgr: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
    """(text, center_xy, symbol_side_px) for every Data Matrix in the full image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    long_side = max(gray.shape)
    passes = [(gray, 1.0)]
    if long_side > 2600:
        # Phone photos: a downsampled copy smooths the dot pattern and often
        # decodes symbols the full-resolution pass misses.
        f = 2048.0 / long_side
        passes.append((cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), f))
    if long_side < 3000:
        passes.append((cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC), 2.0))
    found: dict[str, tuple[np.ndarray, float]] = {}
    for img, s in passes:
        for binarizer in (zxingcpp.Binarizer.LocalAverage, zxingcpp.Binarizer.GlobalHistogram):
            for code in zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.DataMatrix, binarizer=binarizer):
                text = code.text.strip().lower()
                if not text or text in found:
                    continue
                p = code.position
                pts = np.array([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                                [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]], float) / s
                side = float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))
                found[text] = (pts.mean(axis=0), side)
    return [(t, c, s) for t, (c, s) in found.items()]


def _dr12_dark_square_candidates(work_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centers and sizes of dark, square-ish openings (empty wells)."""
    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
    long_side = max(gray.shape)
    block = int(long_side / 20) | 1
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 8)
    k = max(3, int(long_side / 400))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(thr, 8)
    min_side, max_side = long_side / 45.0, long_side / 9.0
    out, sizes = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        if not 0.6 < w / h < 1.65:
            continue
        if area / float(w * h) < 0.45:
            continue
        out.append(cents[i])
        sizes.append(np.sqrt(w * h))
    return np.asarray(out, float).reshape(-1, 2), np.asarray(sizes, float)


def _dr12_dominant_size_filter(points: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    if len(points) < 8:
        return points
    logs = np.log(sizes)
    best, best_count = None, -1
    for center in logs:
        count = int((np.abs(logs - center) < 0.18).sum())
        if count > best_count:
            best, best_count = center, count
    keep = np.abs(logs - best) < 0.25
    return points[keep]


def _dr12_nearest(points: np.ndarray, target: np.ndarray, exclude: Optional[set] = None) -> tuple[int, float]:
    d = np.linalg.norm(points - target, axis=1)
    if exclude:
        d = d.copy()
        d[list(exclude)] = np.inf
    i = int(np.argmin(d))
    return i, float(d[i])


def _dr12_lattice_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    diff = points[None, :, :] - points[:, None, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    pitch = float(np.median(nn))
    vecs = []
    for i in range(len(points)):
        for j in np.where((dist[i] > 0.7 * pitch) & (dist[i] < 1.35 * pitch))[0]:
            v = diff[i, j]
            if v[0] < 0 or (v[0] == 0 and v[1] < 0):
                v = -v
            vecs.append(v)
    vecs = np.asarray(vecs, float)
    ang = np.arctan2(vecs[:, 1], vecs[:, 0])
    feats = np.column_stack([np.cos(2 * ang), np.sin(2 * ang)]).astype(np.float32)
    _, lab, _ = cv2.kmeans(feats, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3), 10,
                           cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    e1 = np.median(vecs[lab == 0], axis=0)
    e2 = np.median(vecs[lab == 1], axis=0)
    return e1, e2, pitch


def _dr12_grow_lattice(points: np.ndarray) -> dict[int, tuple[int, int]]:
    e1, e2, pitch = _dr12_lattice_basis(points)
    centroid = points.mean(axis=0)
    # seed: a well-connected point near the centroid
    order = np.argsort(np.linalg.norm(points - centroid, axis=1))
    seed = None
    for idx in order[:30]:
        hits = sum(_dr12_nearest(points, points[idx] + s * e)[1] < 0.3 * pitch for e in (e1, e2) for s in (1, -1))
        if hits >= 3:
            seed = int(idx)
            break
    if seed is None:
        seed = int(order[0])
    coords = {seed: (0, 0)}
    taken = {(0, 0): seed}
    queue = deque([(seed, e1, e2)])
    while queue:
        idx, a, b = queue.popleft()
        i0, j0 = coords[idx]
        p = points[idx]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for step in (1, 2):
                ci, cj = i0 + di * step, j0 + dj * step
                if (ci, cj) in taken:
                    break
                target = p + step * (di * a + dj * b)
                q, d = _dr12_nearest(points, target)
                if d > 0.28 * pitch or q in coords:
                    continue
                coords[q] = (ci, cj)
                taken[(ci, cj)] = q
                delta = (points[q] - p) / step
                na = delta * di if di else a
                nb = delta * dj if dj else b
                queue.append((q, na, nb))
                break
    return coords


def _dr12_fit_homography(points: np.ndarray, coords: dict[int, tuple[int, int]], pitch: float):
    idx = list(coords)
    src = np.array([coords[i] for i in idx], np.float32)
    dst = points[idx].astype(np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 0.2 * pitch)
    for _ in range(3):
        inv = np.linalg.inv(H)
        can = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), inv).reshape(-1, 2)
        rounded = np.round(can)
        ok = np.linalg.norm(can - rounded, axis=1) < 0.22
        # one point per lattice cell
        cells = {}
        for k in np.where(ok)[0]:
            key = tuple(rounded[k].astype(int))
            err = np.linalg.norm(can[k] - rounded[k])
            if key not in cells or err < cells[key][1]:
                cells[key] = (k, err)
        if len(cells) < 8:
            break
        src = np.array(list(cells), np.float32)
        dst = points[[v[0] for v in cells.values()]].astype(np.float32)
        H, _ = cv2.findHomography(src, dst, 0)
    return H, {v[0]: k for k, v in cells.items()}


def _dr12_sample(gray_blur: np.ndarray, H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    pts = cv2.perspectiveTransform(uv.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    h, w = gray_blur.shape
    out = np.full(len(pts), np.nan)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] < h - 1)
    if inside.any():
        p = pts[inside]
        out[inside] = cv2.remap(gray_blur, p[:, 0].reshape(-1, 1).astype(np.float32),
                                p[:, 1].reshape(-1, 1).astype(np.float32), cv2.INTER_LINEAR).ravel()
    return out


def _dr12_cell_wall_score(gray_blur: np.ndarray, H: np.ndarray, i: int, j: int) -> float:
    """Beige walls on all four sides of a plate cell, darker corners inside."""
    walls = np.array([[i + 0.5, j + 0.25], [i + 0.5, j - 0.25], [i - 0.5, j + 0.25], [i - 0.5, j - 0.25],
                      [i + 0.25, j + 0.5], [i - 0.25, j + 0.5], [i + 0.25, j - 0.5], [i - 0.25, j - 0.5]])
    inner = np.array([[i + 0.3, j + 0.3], [i - 0.3, j + 0.3], [i + 0.3, j - 0.3], [i - 0.3, j - 0.3]])
    w = _dr12_sample(gray_blur, H, walls)
    c = _dr12_sample(gray_blur, H, inner)
    if np.isnan(w).any() or np.isnan(c).any():
        return -50.0
    # walls must be uniformly bright: penalise the darkest wall sample
    return float(0.5 * np.mean(w) + 0.5 * np.min(w) - np.mean(c))


def _dr12_place_window(work_bgr: np.ndarray, H: np.ndarray, cells: dict[int, tuple[int, int]]):
    """Choose the 8 x 12 window (either axis assignment). Returns canonical H [v(12), u(8)] -> work px."""
    gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), max(1.0, max(gray.shape) / 900)).astype(np.float32)
    ij = np.array(list(cells.values()))
    imin, jmin = ij.min(axis=0)
    imax, jmax = ij.max(axis=0)
    cache: dict[tuple[int, int], float] = {}

    def cs(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = _dr12_cell_wall_score(gray, H, i, j)
        return cache[(i, j)]

    best = None
    for ni, nj in ((8, 12), (12, 8)):
        for i0 in range(imax - ni + 1 - 2, imin + 3):
            for j0 in range(jmax - nj + 1 - 2, jmin + 3):
                inside = ((ij[:, 0] >= i0) & (ij[:, 0] < i0 + ni) & (ij[:, 1] >= j0) & (ij[:, 1] < j0 + nj)).sum()
                missing = len(ij) - inside
                walls = sum(cs(i, j) for i in range(i0, i0 + ni) for j in range(j0, j0 + nj)) / (ni * nj)
                score = walls - 6.0 * missing
                if best is None or score > best[0]:
                    best = (score, ni, nj, i0, j0, missing, walls)
    score, ni, nj, i0, j0, missing, walls = best
    # canonical (v along 12-axis, u along 8-axis) -> lattice (i, j)
    if ni == 8:  # i is the 8-axis (u), j is the 12-axis (v)
        T = np.array([[0, 1, i0], [1, 0, j0], [0, 0, 1]], float)
    else:  # i is the 12-axis (v), j is the 8-axis (u)
        T = np.array([[1, 0, i0], [0, 1, j0], [0, 0, 1]], float)
    return H @ T, {"missing": int(missing), "wall_score": float(walls), "window": (ni, nj, i0, j0)}


def _dr12_fit_double_rib_lattice(image_bgr: np.ndarray, codes=None):
    """Return (H_full canonical[v,u] -> full-res px, diagnostics, codes)."""
    # k-means in the basis estimate draws from OpenCV's global RNG; seed it so
    # the same photo always yields the same grid (and the same reads).
    cv2.setRNGSeed(12345)
    work, s = _dr12_work_image(image_bgr)
    sq, sizes = _dr12_dark_square_candidates(work)
    sq = _dr12_dominant_size_filter(sq, sizes)
    if codes is None:
        codes = _dr12_full_image_codes(image_bgr)
    code_pts = np.array([c for _, c, _ in codes], float).reshape(-1, 2) * s
    pts = np.vstack([sq, code_pts]) if len(code_pts) else sq
    # merge near-duplicates (a decoded tube inside a detected square)
    merged = []
    for p in pts:
        if all(np.linalg.norm(p - q) > max(work.shape) / 60 for q in merged):
            merged.append(p)
    pts = np.asarray(merged, float)
    if len(pts) < 12:
        raise RuntimeError("Too few well openings or tube codes were found to fit the double-rib grid.")
    coords = _dr12_grow_lattice(pts)
    _, _, pitch = _dr12_lattice_basis(pts)
    H, cells = _dr12_fit_homography(pts, coords, pitch)
    Hc, diag = _dr12_place_window(work, H, cells)
    S_inv = np.diag([1 / s, 1 / s, 1.0])
    diag.update({"points": len(pts), "squares": len(sq), "codes": len(codes), "lattice_cells": len(cells),
                 "pitch_work": pitch})
    return S_inv @ Hc, diag, codes


# ---------------------------------------------------------------- analysis

_DR12_PRESENCE_HP_THRESHOLD = 14.0
_DR12_YY, _DR12_XX = np.mgrid[0:100, 0:100]
_DR12_INNER = np.hypot(_DR12_XX - 49.5, _DR12_YY - 49.5) < 28


def _dr12_presence_texture(crop100_bgr: np.ndarray) -> float:
    """Fine (Data-Matrix-scale) texture inside the tube-cap circle of a 100 px/pitch crop.

    Empty wells - including bright glare on the tray floor and lit walls - are
    smooth at this scale; a tube cap carries the dot pattern of its code.
    """
    g = cv2.cvtColor(crop100_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 3)
    return float(hp[_DR12_INNER].std())


def _dr12_canonical_of(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points_xy, np.float32).reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)


def _dr12_well_crop_fullres(image_bgr: np.ndarray, H: np.ndarray, v: int, u: int, size: Optional[int] = None) -> np.ndarray:
    """Perspective-corrected one-pitch crop of well (v, u) straight from the full-resolution photo."""
    if size is None:
        c = cv2.perspectiveTransform(np.float32([[[v, u]], [[v + 1, u]], [[v, u + 1]]]), H).reshape(-1, 2)
        native = 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[0]))
        size = int(np.clip(native, 180, 420))
    A = np.array([[0, 1.0 / size, v - 0.5], [1.0 / size, 0, u - 0.5], [0, 0, 1]], float)
    return cv2.warpPerspective(image_bgr, H @ A, (size, size), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def _dr12_decode_well(crop_bgr: np.ndarray, exclude: set[str]) -> tuple[Optional[str], int]:
    """Try progressively heavier preprocessing; returns (text, number of agreeing reads)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)
    votes: dict[str, int] = {}

    def read(img):
        for binarizer in (zxingcpp.Binarizer.LocalAverage, zxingcpp.Binarizer.GlobalHistogram):
            for code in zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.DataMatrix, binarizer=binarizer):
                t = code.text.strip().lower()
                if t and t not in exclude:
                    votes[t] = votes.get(t, 0) + 1

    stages = [
        lambda: [gray, clahe] + ([cv2.resize(gray, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA)]
                                 if gray.shape[0] > 300 else []),
        lambda: [cv2.resize(x, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) for f in (1.6, 2.4) for x in (gray, clahe)],
        lambda: [cv2.GaussianBlur(gray, (0, 0), 1.2), cv2.medianBlur(clahe, 3),
                 cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]],
    ]
    for stage in stages:
        for img in stage():
            read(img)
        if votes:
            break
    if not votes:
        # Dot-peen symbols: modules are separate dots with gaps, which zxing
        # often cannot assemble. Merge the dots (blur / close / dilate) at a few
        # scales; no single variant wins on every tube, so sweep and stop early.
        n = gray.shape[0]
        for f in (1.0, 0.6, 0.45):
            if n * f < 90:
                continue
            x = gray if f == 1.0 else cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            sweep = [x] + [cv2.GaussianBlur(x, (0, 0), sig * f) for sig in (1.5, 2.5, 3.5)]
            for k in (3, 5, 7):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(round(k * f)) | 1),) * 2)
                sweep += [cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel), cv2.dilate(x, kernel)]
            for img in sweep:
                read(img)
                if votes:
                    break
            if votes:
                break
    if not votes:
        # Last resort: blank everything outside the cap (clean quiet zone) and
        # try small rotations - zxing's sampling grid sometimes locks on only
        # after the symbol is turned a few degrees.
        n = gray.shape[0]
        yy, xx = np.mgrid[0:n, 0:n]
        rr = np.hypot(xx - n / 2.0, yy - n / 2.0)
        masked = gray.copy()
        masked[rr > 0.30 * n] = int(np.percentile(gray[rr < 0.2 * n], 10))
        for f in (0.6, 0.45, 1.0):
            if n * f < 90:
                continue
            x = masked if f == 1.0 else cv2.resize(masked, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            h2, w2 = x.shape
            for base in (x, cv2.GaussianBlur(x, (0, 0), 2.5 * f)):
                for ang in (10, -10, 5, -5, 15, -15, 30, 45):
                    M = cv2.getRotationMatrix2D((w2 / 2.0, h2 / 2.0), ang, 1.0)
                    read(cv2.warpAffine(base, M, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))
                    if votes:
                        break
                if votes:
                    break
            if votes:
                break
    if not votes:
        return None, 0
    text = max(votes, key=votes.get)
    return text, votes[text]


def _analyze_double_rib_v12(image_bgr: np.ndarray, plate_id_override=None, corner_override=None):
    H, diag, codes = _dr12_fit_double_rib_lattice(image_bgr)
    Ho, rib_score = _orient_double_rib_homography(image_bgr, H, corner_override=corner_override)
    rect = _rectify_double_rib_final(image_bgr, Ho, scale=100)

    # Whole-image codes -> wells (or external plate marker).
    by_well: dict[tuple[int, int], str] = {}
    external: list[str] = []
    if codes:
        can = _dr12_canonical_of(np.array([c for _, c, _ in codes]), Ho)
        for (text, _, _), (v, u) in zip(codes, can):
            vi, ui = int(round(v)), int(round(u))
            if 0 <= vi < 12 and 0 <= ui < 8 and abs(v - vi) < 0.45 and abs(u - ui) < 0.45:
                by_well.setdefault((vi, ui), text)
            elif v < -0.9 or v > 11.9 or u < -0.9 or u > 7.9:
                external.append(text)
    plate_id = (plate_id_override or "").strip().lower() or (external[0] if external else "UNREADABLE")

    known = set(by_well.values()) | set(external)
    cells = [(v, u) for v in range(12) for u in range(8)]
    crops100 = {(v, u): rect[int((v + 1.5) * 100) - 50:int((v + 1.5) * 100) + 50,
                             int((u + 1.5) * 100) - 50:int((u + 1.5) * 100) + 50].copy() for v, u in cells}
    texture = {k: _dr12_presence_texture(c) for k, c in crops100.items()}

    todo = [k for k in cells if k not in by_well and texture[k] >= _DR12_PRESENCE_HP_THRESHOLD]

    def work(k):
        crop = _dr12_well_crop_fullres(image_bgr, Ho, *k)
        text, n = _dr12_decode_well(crop, known)
        if not text:
            # Borderline symbols can hinge on resampling; retry the same well
            # rendered at two other sizes before calling it unreadable.
            for f in (0.75, 1.3):
                size = int(np.clip(crop.shape[0] * f, 150, 520))
                text, n = _dr12_decode_well(_dr12_well_crop_fullres(image_bgr, Ho, *k, size=size), known)
                if text:
                    break
        return k, (text, n)

    with ThreadPoolExecutor(max_workers=8) as pool:
        decoded = dict(pool.map(work, todo))

    wells = []
    for v, u in cells:
        pos = f"{ROWS[u]}{v + 1}"
        hp = texture[(v, u)]
        if (v, u) in by_well:
            value, conf, occ = by_well[(v, u)], 96.0, True
        elif hp < _DR12_PRESENCE_HP_THRESHOLD:
            # distance from the threshold drives confidence; 10-14 is a review band
            value, occ = "EMPTY", False
            conf = float(np.clip(100.0 - 6.0 * max(0.0, hp - 6.0), 55.0, 98.0))
        else:
            text, n = decoded.get((v, u), (None, 0))
            occ = True
            if text:
                value, conf = text, (92.0 if n >= 2 else 82.0)
            else:
                value, conf = "UNREADABLE", float(np.clip(20.0 + hp / 3.0, 20.0, 40.0))
        wells.append(WellResult(position=pos, value=value, confidence=conf,
                                     crop_rgb=cv2.cvtColor(crops100[(v, u)], cv2.COLOR_BGR2RGB), occupied=occ,
                                     center_xy=((u + 1.5) * 100.0, (v + 1.5) * 100.0)))
    _enforce_unique_codes(wells)
    _sort_wells_canonical(wells)
    overlay = _draw_overlay(rect, wells)
    diag.update({"rib_score": rib_score, "whole_image_wells": len(by_well), "crop_decoded": sum(1 for t, _ in decoded.values() if t),
                 "external_codes": external})
    return RackResult(plate_id=plate_id, wells=wells, normalized_rgb=cv2.cvtColor(rect, cv2.COLOR_BGR2RGB),
                           overlay_rgb=overlay, profile_name="Double-rib plate — v12 lattice / H12 rib anchor"), diag


def analyze_double_rib_plate(
    image_bgr: np.ndarray,
    plate_id_override: Optional[str] = None,
    double_rib_corner_override: Optional[str] = None,
) -> RackResult:
    """v12 analysis; falls back to the v11 path only if the v12 lattice cannot be fitted."""
    try:
        result, _ = _analyze_double_rib_v12(
            image_bgr,
            plate_id_override=plate_id_override,
            corner_override=double_rib_corner_override,
        )
        return result
    except Exception as v12_error:
        try:
            return analyze_double_rib_plate_legacy(
                image_bgr,
                plate_id_override=plate_id_override,
                double_rib_corner_override=double_rib_corner_override,
            )
        except Exception:
            raise RuntimeError(f"Double-rib analysis failed: {v12_error}") from v12_error


def _dr12_wall_beige_fraction(image_bgr: np.ndarray, H: np.ndarray) -> float:
    """Fraction of wall midpoints between fitted wells that are beige plastic.

    Reference photos: Double-rib 0.96-1.00; Original tray and Notched plate <= 0.55.
    """
    hsv = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (0, 0), 2), cv2.COLOR_BGR2HSV)
    pts = []
    for v in range(12):
        for u in range(8):
            pts += [[v + 0.5, u + 0.25], [v + 0.5, u - 0.25], [v + 0.25, u + 0.5], [v - 0.25, u + 0.5]]
    xy = cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2).astype(int)
    h, w = image_bgr.shape[:2]
    xy = xy[(xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)]
    if len(xy) == 0:
        return 0.0
    s = hsv[xy[:, 1], xy[:, 0]].astype(np.int32)
    beige = (s[:, 0] >= 8) & (s[:, 0] <= 30) & (s[:, 1] >= 25) & (s[:, 1] <= 120) & (s[:, 2] >= 110)
    return float(beige.mean())


def _double_rib_auto_evidence_v12(image_bgr: np.ndarray) -> tuple[bool, str]:
    """Auto gate: beige square-well lattice, usable H12 rib anchor, and no
    rack/plate marker other than a tray label beside H12."""
    H, diag, codes = _dr12_fit_double_rib_lattice(image_bgr)
    if diag["lattice_cells"] < 48:
        return False, f"only {diag['lattice_cells']} lattice cells (need at least 48)"
    beige = _dr12_wall_beige_fraction(image_bgr, H)
    if beige < 0.80:
        return False, f"well walls are not beige double-rib plastic (beige fraction {beige:.2f})"
    try:
        Ho, rib_score = _orient_double_rib_homography(image_bgr, H, corner_override="auto")
    except Exception as exc:
        return False, f"H12 double-rib anchor failed: {exc}"
    tray_labels = []
    if codes:
        # A tube code sits on a lattice point; a rack/plate marker sits between
        # wells (Original tray: half a pitch off the edge row) or outside them.
        # A double-rib plate can sit in a labelled tray whose rack Data Matrix
        # is in the margin beside H12 (the paired-rib corner); that one is
        # allowed. The beige-wall check above is what rules out Original and
        # Notched racks, whose marker can land in a similar spot.
        can = _dr12_canonical_of(np.array([c for _, c, _ in codes]), Ho)
        markers = [(t, v, u) for (t, _, _), (v, u) in zip(codes, can)
                   if max(abs(v - round(v)), abs(u - round(u))) > 0.3
                   or not (-0.5 < v < 11.5 and -0.5 < u < 7.5)]
        blocking = [t for t, v, u in markers if not _dr12_beside_h12(v, u)]
        if blocking:
            return False, "plate/rack Data Matrix detected: " + ", ".join(blocking[:3])
        tray_labels = [t for t, _, _ in markers]
    label_note = f"tray label beside H12: {', '.join(tray_labels[:3])}" if tray_labels else "no external plate-ID barcode"
    return True, f"lattice cells={diag['lattice_cells']}, beige={beige:.2f}, H12 rib score={rib_score:.2f}, {label_note}"


def _dr12_beside_h12(v: float, u: float) -> bool:
    """True for a point in the tray margin around H12 (canonical v=11, u=7), outside the well field."""
    outside = v > 11.5 or u > 7.5
    return outside and 10.0 <= v <= 13.5 and 5.5 <= u <= 9.5


def analyze_image_bytes(
    image_bytes: bytes,
    profile_choice: str = "auto",
    plate_id_override: Optional[str] = None,
    double_rib_corner_override: Optional[str] = None,
) -> RackResult:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("The selected image could not be opened.")

    choice = (profile_choice or "auto").strip().lower()

    if choice == "original tray":
        return analyze_original(image_bgr)
    if choice == "notched plate":
        return analyze_notched_plate(image_bgr)
    if choice == "double-rib plate":
        return analyze_double_rib_plate(
            image_bgr,
            plate_id_override=plate_id_override,
            double_rib_corner_override=double_rib_corner_override,
        )

    # Auto profile selection uses a strict Double-rib gate.
    #
    # Confirmed profile rule:
    #   Double-rib plate -> NO plate-ID barcode outside the 8 x 12 well field.
    #   Original tray    -> external rack Data Matrix.
    #   Notched plate    -> external plate Data Matrix.
    #
    # Square contours alone are not sufficient because the Original tray can
    # also generate many square-like contour candidates.
    errors = []

    try:
        is_double_rib, reason = _double_rib_auto_evidence_v12(image_bgr)
        if is_double_rib:
            return analyze_double_rib_plate(
                image_bgr,
                plate_id_override=plate_id_override,
                double_rib_corner_override=double_rib_corner_override,
            )
        errors.append(f"Double-rib Auto gate rejected: {reason}")
    except Exception as exc:
        errors.append(f"Double-rib Auto gate failed: {exc}")

    h, w = image_bgr.shape[:2]
    center_gray = cv2.cvtColor(
        image_bgr[
            int(0.15 * h):int(0.85 * h),
            int(0.15 * w):int(0.85 * w),
        ],
        cv2.COLOR_BGR2GRAY,
    )
    center_median = float(np.median(center_gray))

    # Dark central field strongly favors the Notched plate; otherwise prefer
    # the Original molded tray. Both of those profiles require their own
    # plate/rack marker logic before returning a result.
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

    for fn in (preferred, alternate):
        try:
            return fn(image_bgr)
        except Exception as exc:
            errors.append(f"{fn.__name__}: {exc}")

    raise RuntimeError(
        "Auto-detection failed. Try selecting the rack style manually.\n\n"
        + "\n".join(errors)
    )
