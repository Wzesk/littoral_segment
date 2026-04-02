import numpy as np
from scipy import ndimage
from skimage import measure


DEFAULT_SELECTION_CONFIG = {
    "mode_periodic": "area_confidence",
    "mode_nonperiodic": "shoreline_plausibility",
    "edge_touch_tolerance_px": 12.0,
    "min_area_fraction_nonperiodic": 0.01,
    "max_small_component_fraction": 0.2,
}


def _point_edge_id(point_rc, h, w, edge_dist):
    row, col = float(point_rc[0]), float(point_rc[1])
    candidates = []
    if row <= edge_dist:
        candidates.append("top")
    if row >= h - 1 - edge_dist:
        candidates.append("bottom")
    if col <= edge_dist:
        candidates.append("left")
    if col >= w - 1 - edge_dist:
        candidates.append("right")
    return candidates[0] if candidates else None


def _extract_interface_from_ocean(mask_array):
    land_mask = mask_array > 0
    water_mask = ~land_mask
    labeled_water, water_count = ndimage.label(water_mask)
    if water_count == 0:
        return None

    boundary_labels = set()
    boundary_labels.update(np.unique(labeled_water[0, :]))
    boundary_labels.update(np.unique(labeled_water[-1, :]))
    boundary_labels.update(np.unique(labeled_water[:, 0]))
    boundary_labels.update(np.unique(labeled_water[:, -1]))
    boundary_labels.discard(0)
    if not boundary_labels:
        return None

    ocean_mask = np.isin(labeled_water, list(boundary_labels))
    interface_mask = land_mask & ndimage.binary_dilation(ocean_mask, structure=np.ones((3, 3), dtype=bool))
    if not interface_mask.any():
        return None
    return interface_mask.astype(np.uint8)


def _largest_component_stats(mask_bool):
    labeled, num = ndimage.label(mask_bool)
    if num == 0:
        return {
            "dominant_component_area": 0,
            "total_land_area": 0,
            "small_component_fraction": 1.0,
            "edge_ids": [],
            "edge_count": 0,
            "bbox_fill_fraction": 0.0,
        }

    sizes = ndimage.sum(mask_bool, labeled, range(1, num + 1))
    dominant_label = int(np.argmax(sizes)) + 1
    dominant = labeled == dominant_label
    dominant_area = int(dominant.sum())
    total_land_area = int(mask_bool.sum())

    coords = np.argwhere(dominant)
    if len(coords) == 0:
        bbox_fill_fraction = 0.0
    else:
        min_r, min_c = coords.min(axis=0)
        max_r, max_c = coords.max(axis=0)
        bbox_area = max(1, int((max_r - min_r + 1) * (max_c - min_c + 1)))
        bbox_fill_fraction = float(dominant_area / bbox_area)

    h, w = mask_bool.shape
    edge_ids = []
    if dominant[0, :].any():
        edge_ids.append("top")
    if dominant[-1, :].any():
        edge_ids.append("bottom")
    if dominant[:, 0].any():
        edge_ids.append("left")
    if dominant[:, -1].any():
        edge_ids.append("right")

    dominant_fraction = float(dominant_area / max(total_land_area, 1))
    return {
        "dominant_component_area": dominant_area,
        "total_land_area": total_land_area,
        "dominant_component_fraction": dominant_fraction,
        "small_component_fraction": max(0.0, 1.0 - dominant_fraction),
        "edge_ids": edge_ids,
        "edge_count": len(edge_ids),
        "bbox_fill_fraction": bbox_fill_fraction,
    }


def _interface_stats(mask_bool, edge_dist):
    interface_mask = _extract_interface_from_ocean(mask_bool.astype(np.uint8))
    if interface_mask is None:
        return {
            "interface_candidate_count": 0,
            "interface_has_two_edge_candidate": False,
            "interface_distinct_edge_candidate": False,
            "best_interface_length": 0.0,
        }

    contours = measure.find_contours(interface_mask, 0.5)
    h, w = mask_bool.shape
    candidate_count = 0
    has_two_edge = False
    distinct_edges = False
    best_length = 0.0

    for contour in contours:
        if len(contour) < 10:
            continue
        candidate_count += 1
        start_edge = _point_edge_id(contour[0], h, w, edge_dist)
        end_edge = _point_edge_id(contour[-1], h, w, edge_dist)
        if start_edge is not None and end_edge is not None:
            has_two_edge = True
            distinct_edges = distinct_edges or (start_edge != end_edge)
        diffs = np.diff(contour, axis=0)
        length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        best_length = max(best_length, length)

    return {
        "interface_candidate_count": candidate_count,
        "interface_has_two_edge_candidate": has_two_edge,
        "interface_distinct_edge_candidate": distinct_edges,
        "best_interface_length": best_length,
    }


def score_mask_candidate(mask_array, confidence=0.0, periodic=True, config=None):
    cfg = dict(DEFAULT_SELECTION_CONFIG)
    if config:
        cfg.update(config)

    mask_bool = np.asarray(mask_array, dtype=bool)
    h, w = mask_bool.shape
    area_fraction = float(mask_bool.mean())
    component_stats = _largest_component_stats(mask_bool)
    interface_stats = _interface_stats(mask_bool, edge_dist=float(cfg.get("edge_touch_tolerance_px", 12.0)))

    qc = {
        "area_fraction": area_fraction,
        "confidence": float(confidence),
        "periodic": bool(periodic),
        **component_stats,
        **interface_stats,
    }

    if periodic:
        score = (
            int(area_fraction > 0.0),
            float(area_fraction),
            float(component_stats["dominant_component_fraction"]),
            float(confidence),
        )
    else:
        min_area = float(cfg.get("min_area_fraction_nonperiodic", 0.01))
        max_small_fraction = float(cfg.get("max_small_component_fraction", 0.2))
        plausible = (
            area_fraction >= min_area
            and component_stats["small_component_fraction"] <= max_small_fraction
        )
        score = (
            int(interface_stats["interface_distinct_edge_candidate"]),
            int(interface_stats["interface_has_two_edge_candidate"]),
            int(component_stats["edge_count"] >= 2),
            int(plausible),
            float(area_fraction),
            float(interface_stats["best_interface_length"]),
            float(component_stats["bbox_fill_fraction"]),
            float(confidence),
        )
        qc["shoreline_plausible"] = bool(score[0] or score[1] or score[2])

    qc["score"] = list(score)
    return score, qc


def select_best_mask(mask_arrays, confidences=None, periodic=True, config=None):
    if not mask_arrays:
        return None, {"candidate_count": 0, "candidates": [], "selected_index": None}

    if confidences is None:
        confidences = [0.0] * len(mask_arrays)

    candidates = []
    best_index = 0
    best_score = None

    for idx, (mask_array, confidence) in enumerate(zip(mask_arrays, confidences)):
        score, qc = score_mask_candidate(mask_array, confidence=confidence, periodic=periodic, config=config)
        qc["candidate_index"] = idx
        candidates.append(qc)
        if best_score is None or score > best_score:
            best_score = score
            best_index = idx

    summary = {
        "candidate_count": len(mask_arrays),
        "selected_index": best_index,
        "selected_score": candidates[best_index]["score"],
        "candidates": candidates,
    }
    return np.asarray(mask_arrays[best_index], dtype=np.uint8), summary
