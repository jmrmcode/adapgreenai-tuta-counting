#!/usr/bin/env python3
"""Tuta absoluta silhouette counting from sticky-trap RGB images.

Pipeline
--------
1. Red-balanced spectral feature and spatially weighted neighborhood score.
2. K-means foreground segmentation.
3. Binary morphology and external contour extraction.
4. Area/solidity filtering (observed-count-guided tuning (default), or fixed).
5. Hu-moment gallery classification of the *retained* contours.
6. Abundance correction: retained contours + predicted touching pairs.

The default optimize mode REQUIRES an observed manual count for calibration.
The gallery's 'noise' and 'uncertain' classes do NOT subtract detections:
this script deliberately uses the gallery ONLY for touching-pair correction.

Run ``python tuta_pipeline.py --help`` for CLI instructions.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")  # Save figures even on headless Linux servers.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

LOG = logging.getLogger("tuta_pipeline")


# -----------------------------------------------------------------------------
# 0. Configuration. Defaults reproduce the important IMAGE-processing settings
#    of the original notebook; count-guided optimization is the default.
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # Spectral and spatial feature
    max_ring: int = 7
    sigma_x: float = 6.0
    sigma_y: float = 2.0
    red_gamma: float = 3.0
    balance_gamma: float = 1.0
    dark_threshold: float = 50.0 / 255.0

    # Segmentation and binary morphology
    n_clusters: int = 2
    random_state: int = 42
    kmeans_fit_max_pixels: int | None = None  # None = fit ALL valid pixels
    closing_kernel: int = 7
    opening_kernel: int = 7

    # Area and solidity: pixels squared and dimensionless solidity
    min_area: float = 600.0
    max_area: float = 5500.0
    min_solidity: float = 0.43

    # "fixed" is appropriate when applying a calibrated pipeline to new images.
    # "optimize" uses a KNOWN observed count and must not be presented as an
    # independent performance evaluation.
    filter_mode: str = "optimize"  # fixed | optimize
    optimizer_count_mode: str = "legacy_original"  # contours | legacy_original | legacy_corrected
    allowable_count_error_pct: float = 5.0
    # Bounds are constrained to be at least as strict as fixed thresholds.
    min_area_bounds: tuple[float, float] = (600.0, 1000.0)
    max_area_bounds: tuple[float, float] = (2500.0, 5500.0)
    min_solidity_bounds: tuple[float, float] = (0.43, 0.75)
    n_random: int = 2000
    n_starts: int = 6
    n_refinements: int = 3

    # Shape matching (OpenCV Hu moments, I2). Smaller score = closer match.
    gallery_k: int = 3
    min_margin: float = 0.001  # Original script was very permissive.

    def validate(self) -> None:
        if self.max_ring < 1:
            raise ValueError("max_ring must be >= 1")
        if min(self.sigma_x, self.sigma_y) <= 0:
            raise ValueError("Gaussian widths must be positive")
        if self.red_gamma < 0 or self.balance_gamma < 0:
            raise ValueError("gamma values must be nonnegative")
        if not 0 <= self.dark_threshold <= 1:
            raise ValueError("dark_threshold must be in [0,1]")
        if self.n_clusters != 2:
            raise ValueError("This pipeline currently expects exactly 2 clusters")
        for name in ("closing_kernel", "opening_kernel"):
            value = getattr(self, name)
            if value < 1 or value % 2 != 1:
                raise ValueError(f"{name} must be a positive odd integer")
        if not 0 <= self.min_solidity <= 1:
            raise ValueError("min_solidity must be between 0 and 1")
        if not 0 <= self.min_area < self.max_area:
            raise ValueError("Require 0 <= min_area < max_area")
        if self.filter_mode not in ("fixed", "optimize"):
            raise ValueError("filter_mode must be 'fixed' or 'optimize'")
        if self.optimizer_count_mode not in ("contours", "legacy_original", "legacy_corrected"):
            raise ValueError("Invalid optimizer_count_mode")
        if self.gallery_k < 1 or not 0 <= self.min_margin <= 1:
            raise ValueError("Invalid gallery_k or min_margin")
        if self.kmeans_fit_max_pixels is not None and self.kmeans_fit_max_pixels < 2:
            raise ValueError("kmeans_fit_max_pixels must be >= 2")
        if self.allowable_count_error_pct < 0 or self.n_random < 0:
            raise ValueError("Optimization error tolerance/n_random must be >= 0")
        if self.n_starts < 1 or self.n_refinements < 0:
            raise ValueError("Invalid optimization restart/refinement settings")
        if self.filter_mode == "optimize":
            initial = (self.min_area, self.max_area, self.min_solidity)
            bounds = (self.min_area_bounds, self.max_area_bounds, self.min_solidity_bounds)
            for start, (lo, hi) in zip(initial, bounds):
                if lo > hi or not lo <= start <= hi:
                    raise ValueError("Initial filter values must lie within optimizer bounds")
            if (bounds[0][0] < self.min_area or
                bounds[1][1] > self.max_area or
                bounds[2][0] < self.min_solidity):
                raise ValueError("Optimizer bounds must not relax fixed filters")
            if bounds[0][1] >= bounds[1][0]:
                raise ValueError("Bounds must preserve min_area < max_area")


@dataclass
class FilterResult:
    kept_contours: list[np.ndarray]
    source_ids: list[int]
    cleaned_mask: np.ndarray
    parameters: tuple[float, float, float]
    objective_count: int
    history: list[dict[str, Any]]
    n_evaluations: int


# -----------------------------------------------------------------------------
# 1. Load image and construct a per-pixel red-balanced feature.
# -----------------------------------------------------------------------------
def load_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_red_balanced_feature(image_rgb_float: np.ndarray,
                                 red_gamma: float = 3.0,
                                 balance_gamma: float = 1.0) -> np.ndarray:
    """Return the original ad-hoc red-balanced spectral feature in [0,1]."""
    rgb = np.asarray(image_rgb_float, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB array (height, width, 3)")
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Normalized RGB must contain finite values in [0,1]")
    eps = 1e-6
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red_balance = np.clip(r / ((g + b) * 0.5 + eps), 0.0, 1.0)
    rgb_balance = rgb.min(axis=2) / (rgb.max(axis=2) + eps)
    feature = r * red_balance ** red_gamma * rgb_balance ** balance_gamma
    return np.ascontiguousarray(feature, dtype=np.float32)


# -----------------------------------------------------------------------------
# 2. Elliptical neighborhood connectivity.
#    This is NOT graph-theoretic shortest-path 'closeness centrality'. It is the
#    center feature times a Gaussian-weighted average of NEIGHBOR features.
#    OpenCV convolution is mathematically equivalent to the original Numba
#    nested-neighbor loop (up to floating-point rounding) and considerably faster.
# -----------------------------------------------------------------------------
def make_neighbor_kernel(max_ring: int, sigma_x: float, sigma_y: float) -> np.ndarray:
    y, x = np.mgrid[-max_ring:max_ring + 1, -max_ring:max_ring + 1]
    weights = np.exp(-0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2))
    weights[max_ring, max_ring] = 0.0  # Original loop excluded center.
    weights = weights / weights.sum()
    return weights.astype(np.float32)


def compute_connectivity_map(rgb_float: np.ndarray, cfg: Config) -> np.ndarray:
    h, w = rgb_float.shape[:2]
    ring = cfg.max_ring
    if h < 2 * ring + 1 or w < 2 * ring + 1:
        raise ValueError("Image too small for selected neighborhood")

    feature = compute_red_balanced_feature(rgb_float, cfg.red_gamma, cfg.balance_gamma)
    kernel = make_neighbor_kernel(ring, cfg.sigma_x, cfg.sigma_y)
    neighborhood = cv2.filter2D(feature, cv2.CV_32F, kernel,
                                borderType=cv2.BORDER_CONSTANT)
    score = feature * neighborhood
    score[rgb_float.mean(axis=2) < cfg.dark_threshold] = 0.0

    # Match original Numba behavior: its neighborhood is undefined at borders.
    score[:ring, :] = np.nan
    score[-ring:, :] = np.nan
    score[:, :ring] = np.nan
    score[:, -ring:] = np.nan
    return score


# -----------------------------------------------------------------------------
# 3. Pixel clustering. Invalid border pixels never enter K-means.
# -----------------------------------------------------------------------------
def segment_connectivity(connectivity: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(connectivity)
    values = connectivity[valid].reshape(-1, 1)
    if len(values) < cfg.n_clusters:
        raise ValueError("Not enough valid pixels for K-means")
    if float(values.max() - values.min()) < 1e-12:
        raise ValueError("Connectivity has no variation; cannot separate foreground")

    rng = np.random.default_rng(cfg.random_state)
    train = values
    if cfg.kmeans_fit_max_pixels is not None and len(values) > cfg.kmeans_fit_max_pixels:
        indices = rng.choice(len(values), cfg.kmeans_fit_max_pixels, replace=False)
        train = values[indices]
        LOG.info("Fitting K-means on %s sampled pixels (of %s)", len(train), len(values))

    model = KMeans(n_clusters=cfg.n_clusters, n_init=10,
                   random_state=cfg.random_state)
    model.fit(train)
    centers = model.cluster_centers_.reshape(-1)
    foreground_id = int(np.argmax(centers))
    LOG.info("K-means centroids: %s; foreground cluster: %d",
             np.array2string(centers, precision=6), foreground_id)

    if train is values:
        foreground = model.labels_ == foreground_id
    else:
        foreground = np.empty(len(values), dtype=bool)
        for start in range(0, len(values), 1_000_000):
            stop = min(start + 1_000_000, len(values))
            foreground[start:stop] = model.predict(values[start:stop]) == foreground_id

    binary_mask = np.zeros(connectivity.shape, dtype=np.uint8)
    binary_mask[valid] = foreground.astype(np.uint8) * 255
    return binary_mask, centers


# -----------------------------------------------------------------------------
# 4. Morphological cleanup and extraction of candidate outer contours.
# -----------------------------------------------------------------------------
def smooth_binary_mask(binary_mask: np.ndarray, cfg: Config) -> np.ndarray:
    closing = np.ones((cfg.closing_kernel, cfg.closing_kernel), dtype=np.uint8)
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (cfg.opening_kernel, cfg.opening_kernel))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, opening)


def find_external_contours(binary_mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary_mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return contours


def contour_metrics(contours: list[np.ndarray]) -> tuple[list[np.ndarray], list[int], np.ndarray, np.ndarray]:
    """Return only nondegenerate contours and their ORIGINAL contour indices."""
    valid, ids, areas, solidities = [], [], [], []
    for source_id, contour in enumerate(contours):
        if contour is None or len(contour) < 3:
            continue
        area = float(cv2.contourArea(contour))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        if area <= 0 or hull_area <= 0 or not np.isfinite(area):
            continue
        valid.append(contour)
        ids.append(source_id)
        areas.append(area)
        solidities.append(area / hull_area)
    return valid, ids, np.asarray(areas, dtype=float), np.asarray(solidities, dtype=float)


def count_for_optimizer(selected_areas: np.ndarray, count_mode: str) -> int:
    """Optimizer-only count; NEVER combine legacy area corrections with gallery correction.

    legacy_original exactly reproduces the original code: a blob >3.5*median
    contributes 4, not 3 (base one, then +one, then +two).
    legacy_corrected makes that blob contribute 3.
    """
    n = len(selected_areas)
    if n == 0 or count_mode == "contours":
        return n
    median_area = float(np.median(selected_areas))
    count = n + int(np.count_nonzero(selected_areas > 2.5 * median_area))
    if count_mode == "legacy_original":
        count += 2 * int(np.count_nonzero(selected_areas > 3.5 * median_area))
    elif count_mode == "legacy_corrected":
        count += int(np.count_nonzero(selected_areas > 3.5 * median_area))
    else:
        raise ValueError(f"Unknown count mode: {count_mode}")
    return count


def filter_contours(contours_all: list[np.ndarray], shape: tuple[int, ...],
                    cfg: Config, observed_count: int | None = None) -> FilterResult:
    """Apply fixed area/solidity filter or conservative count-guided tuning."""
    valid, source_ids, areas, solidities = contour_metrics(contours_all)
    initial = (cfg.min_area, cfg.max_area, cfg.min_solidity)
    bounds = (cfg.min_area_bounds, cfg.max_area_bounds, cfg.min_solidity_bounds)
    baseline_mask = ((areas >= initial[0]) & (areas <= initial[1]) &
                     (solidities >= initial[2]))

    if cfg.filter_mode == "optimize":
        if observed_count is None or observed_count < 0:
            raise ValueError("Optimization requires --observed nonnegative integer")
        LOG.warning("Supervised contour optimization uses the OBSERVED count; "
                    "error on this image is calibration error, NOT held-out accuracy")

    observed_tolerance = (observed_count or 0) * cfg.allowable_count_error_pct / 100.0

    def evaluate(params: tuple[float, float, float]) -> dict[str, Any]:
        mn, mx, sol = params
        mask = (areas >= mn) & (areas <= mx) & (solidities >= sol)
        count = count_for_optimizer(areas[mask], cfg.optimizer_count_mode)
        removed = int(np.count_nonzero(baseline_mask & ~mask))
        if cfg.filter_mode == "fixed":
            return {"params": params, "mask": mask, "count": count,
                    "rank": (0, 0, 0, 0), "abs_error": np.nan,
                    "within_tolerance": False}

        absolute_error = abs(count - int(observed_count))
        inside = absolute_error <= observed_tolerance + 1e-10
        regularization = sum(
            abs((v - orig) / (hi - lo)) if hi > lo else 0.0
            for v, orig, (lo, hi) in zip(params, initial, bounds)
        )
        rank = ((0, removed, absolute_error, regularization) if inside else
                (1, absolute_error, removed, regularization))
        return {"params": params, "mask": mask, "count": count,
                "rank": rank, "abs_error": absolute_error,
                "within_tolerance": inside}

    baseline = evaluate(initial)
    best = baseline
    evaluations = 1
    history: list[dict[str, Any]] = []

    def record(stage: str) -> None:
        history.append({"stage": stage,
                        "min_area": best["params"][0],
                        "max_area": best["params"][1],
                        "min_solidity": best["params"][2],
                        "optimizer_count": best["count"],
                        "optimizer_abs_error": best["abs_error"],
                        "evaluations": evaluations})

    record("baseline")

    if cfg.filter_mode == "optimize" and len(areas) > 0 and not baseline["within_tolerance"]:
        rng = np.random.default_rng(cfg.random_state)
        candidates: list[tuple[tuple[Any, ...], tuple[float, float, float]]] = []
        candidates.append((baseline["rank"], initial))

        def consider(params: tuple[float, float, float], store: bool = True):
            nonlocal best, evaluations
            candidate = evaluate(params)
            evaluations += 1
            if candidate["rank"] < best["rank"]:
                best = candidate
            if store:
                candidates.append((candidate["rank"], params))
            return candidate

        # Global grid: uniform values plus empirically relevant quantiles.
        grids = []
        for i, (lo, hi) in enumerate(bounds):
            measured = areas if i < 2 else solidities
            values = np.linspace(lo, hi, 11)
            if measured.size:
                quantiles = np.quantile(measured, np.linspace(0, 1, 9))
                quantiles = quantiles[(quantiles >= lo) & (quantiles <= hi)]
                values = np.r_[values, quantiles]
            grids.append(np.unique(np.r_[values, initial[i]]))
        for combo in product(*grids):
            consider(tuple(map(float, combo)))
        record("global_grid")

        for _ in range(cfg.n_random):
            params = tuple(float(rng.uniform(lo, hi)) for lo, hi in bounds)
            consider(params)
        record("global_random")

        starts = []
        for _, params in sorted(candidates, key=lambda item: item[0]):
            vector = np.asarray([
                ((value - lo) / (hi - lo)) if hi > lo else 0.0
                for value, (lo, hi) in zip(params, bounds)
            ])
            if all(np.linalg.norm(vector - previous_vector) >= 0.22
                   for _, previous_vector in starts):
                starts.append((params, vector))
            if len(starts) >= cfg.n_starts:
                break

        for start_id, (starting_params, _) in enumerate(starts, 1):
            local_best = evaluate(starting_params)
            for iteration in range(cfg.n_refinements):
                local_grids = []
                for value, (lo, hi) in zip(local_best["params"], bounds):
                    radius = 0.2 * (hi - lo) / (2 ** iteration)
                    local_grids.append(np.linspace(max(lo, value - radius),
                                                   min(hi, value + radius), 7))
                for combo in product(*local_grids):
                    candidate = consider(tuple(map(float, combo)), store=False)
                    if candidate["rank"] < local_best["rank"]:
                        local_best = candidate
                record(f"start_{start_id}_refine_{iteration + 1}")

    # Conservative safety invariant: no filter may introduce new contours.
    if np.any(best["mask"] & ~baseline_mask):
        raise RuntimeError("Optimizer admitted contours rejected by original filter")

    selected_ids = np.flatnonzero(best["mask"]).tolist()
    kept = [valid[i] for i in selected_ids]
    retained_source_ids = [source_ids[i] for i in selected_ids]
    cleaned = np.zeros(shape[:2], dtype=np.uint8)
    if kept:
        cv2.drawContours(cleaned, kept, -1, 255, thickness=cv2.FILLED)
    LOG.info("Size-solidity: %d raw -> %d valid -> %d kept contours",
             len(contours_all), len(valid), len(kept))
    LOG.info("Final filter parameters: min area=%.1f, max area=%.1f, min solidity=%.3f",
             *best["params"])
    return FilterResult(kept_contours=kept, source_ids=retained_source_ids,
                        cleaned_mask=cleaned, parameters=best["params"],
                        objective_count=int(best["count"]), history=history,
                        n_evaluations=evaluations)


# -----------------------------------------------------------------------------
# 5. Read labeled galleries. Filenames need not match the current image.
#    Each PNG should contain one connected white silhouette on black.
# -----------------------------------------------------------------------------
def load_shape_gallery(folder: Path) -> list[dict[str, Any]]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing gallery folder: {folder}")
    templates = []
    for path in sorted(folder.glob("*.png")):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            LOG.warning("Cannot load reference: %s", path)
            continue
        binary = (gray > 127).astype(np.uint8) * 255
        binary = cv2.copyMakeBorder(binary, 2, 2, 2, 2,
                                    cv2.BORDER_CONSTANT, value=0)
        contours = [contour for contour in find_external_contours(binary)
                    if cv2.contourArea(contour) >= 5]
        if len(contours) != 1:
            LOG.warning("Skipping %s: expected 1 silhouette, got %d", path.name, len(contours))
            continue
        templates.append({"name": path.name, "contour": contours[0]})
    LOG.info("Gallery %-8s %d valid silhouettes", folder.name, len(templates))
    return templates


def load_galleries(root: Path) -> dict[str, list[dict[str, Any]]]:
    galleries = {
        "single": load_shape_gallery(root / "single"),
        "touching": load_shape_gallery(root / "touching")
    }
    noise = root / "noise"
    if noise.is_dir():
        gallery_noise = load_shape_gallery(noise)
        if gallery_noise:
            galleries["noise"] = gallery_noise
    if not galleries["single"] or not galleries["touching"]:
        raise ValueError("Both 'single' and 'touching' need at least one valid reference")
    return galleries


def compare_to_gallery(contour: np.ndarray, gallery: list[dict[str, Any]], k: int) -> tuple[float, str | None]:
    distances = []
    for ref in gallery:
        score = float(cv2.matchShapes(contour, ref["contour"],
                                      cv2.CONTOURS_MATCH_I2, 0.0))
        if np.isfinite(score):
            distances.append((score, ref["name"]))
    if not distances:
        return float("inf"), None
    distances.sort(key=lambda pair: pair[0])
    closest = distances[:k]
    return float(np.mean([d for d, _ in closest])), closest[0][1]


def classify_retained_contours(contours: list[np.ndarray], source_ids: list[int],
                               galleries: dict[str, list[dict[str, Any]]],
                               cfg: Config) -> pd.DataFrame:
    """One output row for each input contour; NEVER add/delete contours."""
    if len(contours) != len(source_ids):
        raise ValueError("Source contour IDs do not match filtered contours")
    rows = []
    for contour_id, (contour, source_id) in enumerate(zip(contours, source_ids)):
        area = float(cv2.contourArea(contour))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        x, y, w, h = cv2.boundingRect(contour)
        row: dict[str, Any] = {
            "contour_id": contour_id, "source_contour_id": source_id,
            "area_px2": area, "solidity": area / max(hull_area, 1e-12),
            "x": x, "y": y, "width": w, "height": h,
        }
        scores: dict[str, float] = {}
        best_references: dict[str, str | None] = {}
        for category, gallery in galleries.items():
            score, name = compare_to_gallery(contour, gallery, k=cfg.gallery_k)
            row[f"d_{category}"] = score
            scores[category] = score
            best_references[category] = name

        ranking = sorted(scores, key=scores.get)
        best, runner_up = ranking[:2]
        best_distance, second_distance = scores[best], scores[runner_up]
        if not (np.isfinite(best_distance) and np.isfinite(second_distance)):
            margin, category = float("nan"), "uncertain"
        else:
            margin = ((second_distance - best_distance) /
                      max(second_distance, 1e-12))
            category = "uncertain" if margin < cfg.min_margin else best

        row.update({
            "best_gallery": best,
            "best_reference": best_references[best],
            "margin": margin,
            "classification": category,
            # The gallery changes the count ONLY for 'touching'.
            "count_contribution": 2 if category == "touching" else 1,
        })
        rows.append(row)

    columns = ["contour_id", "source_contour_id", "area_px2", "solidity",
               "x", "y", "width", "height", "best_gallery",
               "best_reference", "margin", "classification", "count_contribution"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    assert len(df) == len(contours)
    return df


# -----------------------------------------------------------------------------
# 6. Final count. Do NOT use optimizer's area-based count correction here.
# -----------------------------------------------------------------------------
def correct_count(filtered_contours: list[np.ndarray], shapes: pd.DataFrame) -> dict[str, int]:
    n_contours = len(filtered_contours)
    if n_contours != len(shapes):
        raise ValueError("The gallery and filter must use the SAME contour list")
    n_touching = int((shapes["classification"] == "touching").sum()) if n_contours else 0
    expected = list(range(n_contours))
    if n_contours and sorted(shapes["contour_id"].astype(int).tolist()) != expected:
        raise ValueError("Gallery contour IDs are not consecutive filtered-contour IDs")
    counts = shapes["classification"].value_counts().to_dict() if n_contours else {}
    return {
        "n_retained_contours": n_contours,
        "n_single": int(counts.get("single", 0)),
        "n_touching": n_touching,
        "n_noise": int(counts.get("noise", 0)),
        "n_uncertain": int(counts.get("uncertain", 0)),
        "estimated_tuta": n_contours + n_touching,
    }


# -----------------------------------------------------------------------------
# 7. Visualization and reproducible outputs. RGB uint8 avoids white outlines
#    when drawing colors (0..255) onto float images normalized to [0,1].
# -----------------------------------------------------------------------------
COLORS_RGB = {
    "single": (0, 200, 0),
    "touching": (255, 30, 30),
    "noise": (255, 0, 255),
    "uncertain": (255, 190, 0),
}


def draw_classifications(image_rgb: np.ndarray, contours: list[np.ndarray],
                         shapes: pd.DataFrame, thickness: int = 3) -> np.ndarray:
    if image_rgb.dtype != np.uint8:
        raise TypeError("draw_classifications expects uint8 RGB image")
    drawn = image_rgb.copy()
    for row in shapes.itertuples(index=False):
        contour = contours[int(row.contour_id)]
        color = COLORS_RGB.get(row.classification, (255, 255, 255))
        cv2.drawContours(drawn, [contour], -1, color, thickness=thickness)
    return drawn


def write_png(path: Path, rgb_or_gray: np.ndarray, is_rgb: bool = False) -> None:
    image = cv2.cvtColor(rgb_or_gray, cv2.COLOR_RGB2BGR) if is_rgb else rgb_or_gray
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not save image: {path}")


def save_comparison_figure(original_rgb: np.ndarray, overlay_rgb: np.ndarray,
                           counts: dict[str, int], observed: int | None,
                           filename: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(original_rgb)
    obs_str = f"; observed = {observed}" if observed is not None else ""
    axes[0].set_title(f"Original RGB image{obs_str}")
    axes[1].imshow(overlay_rgb)
    estimated = counts["estimated_tuta"]
    if observed is not None:
        error = 100.0 * abs(estimated - observed) / observed if observed else (
            0.0 if estimated == 0 else float("inf"))
        subtitle = f"absolute percentage difference = {error:.1f}%"
    else:
        subtitle = "No observed count supplied"
    axes[1].set_title(
        f"Gallery-corrected = {estimated} "
        f"({counts['n_retained_contours']} contours + {counts['n_touching']} pairs)\n{subtitle}")
    legend = [Line2D([0], [0], color=np.asarray(color)/255.0, lw=3, label=category)
              for category, color in COLORS_RGB.items()]
    axes[1].legend(handles=legend, loc="upper right", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def save_connectivity_figure(connectivity: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    im = ax.imshow(connectivity, cmap="magma")
    fig.colorbar(im, ax=ax, label="Red-balanced spatial connectivity (a.u.)")
    ax.set_title("Spatial connectivity map")
    ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_pipeline(image_path: Path, gallery_dir: Path, output_dir: Path,
                 cfg: Config, observed_count: int | None = None) -> dict[str, Any]:
    """Run all stages, save outputs, and return a plain summary dictionary."""
    cfg.validate()
    if observed_count is not None and (observed_count < 0 or int(observed_count) != observed_count):
        raise ValueError("Observed count must be a nonnegative integer")
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # Load and segment
    rgb = load_image_rgb(image_path)
    LOG.info("Input image: %s, dimensions: %s", image_path, rgb.shape)
    normalized = rgb.astype(np.float32) / 255.0
    connectivity = compute_connectivity_map(normalized, cfg)
    binary, cluster_centers = segment_connectivity(connectivity, cfg)
    smoothed = smooth_binary_mask(binary, cfg)
    contours_all = find_external_contours(smoothed)

    # Size/solidity selection; indices are preserved for auditing.
    filtered = filter_contours(contours_all, smoothed.shape, cfg, observed_count)

    # Gallery is applied ONLY to contours that survived filtering.
    galleries = load_galleries(gallery_dir)
    shapes = classify_retained_contours(filtered.kept_contours,
                                        filtered.source_ids, galleries, cfg)
    counts = correct_count(filtered.kept_contours, shapes)

    # Avoid ambiguous reuse of the optimizer objective as the FINAL count.
    summary: dict[str, Any] = {
        "input_image": str(image_path.resolve()),
        "gallery_dir": str(gallery_dir.resolve()),
        "filter_mode": cfg.filter_mode,
        "optimization_count_mode": cfg.optimizer_count_mode if cfg.filter_mode == "optimize" else None,
        "observed_count": observed_count,
        "number_of_all_external_contours": len(contours_all),
        "optimizer_objective_count_not_final": filtered.objective_count,
        "number_of_optimizer_evaluations": filtered.n_evaluations,
        "final_min_area": filtered.parameters[0],
        "final_max_area": filtered.parameters[1],
        "final_min_solidity": filtered.parameters[2],
        "cluster_centers": [float(x) for x in cluster_centers],
        **counts,
    }
    estimated = counts["estimated_tuta"]
    if observed_count is not None:
        summary["absolute_count_difference"] = abs(estimated - observed_count)
        summary["absolute_percentage_difference"] = (
            100.0 * abs(estimated - observed_count) / observed_count
            if observed_count > 0 else (0.0 if estimated == 0 else None))
    summary["elapsed_seconds"] = round(time.perf_counter() - start, 3)

    # Per-contour diagnostics; gallery 'noise' remains counted as 1 by design.
    shapes.to_csv(output_dir / "contour_classification.csv", index=False)
    pd.DataFrame(filtered.history).to_csv(output_dir / "filter_history.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8"
    )
    (output_dir / "settings.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    overlay = draw_classifications(rgb, filtered.kept_contours, shapes)
    write_png(output_dir / "01_kmeans_foreground.png", binary)
    write_png(output_dir / "02_morphological_mask.png", smoothed)
    write_png(output_dir / "03_size_solidity_mask.png", filtered.cleaned_mask)
    write_png(output_dir / "04_classified_overlay.png", overlay, is_rgb=True)
    save_comparison_figure(rgb, overlay, counts, observed_count,
                           output_dir / "05_original_vs_classified.png")
    save_connectivity_figure(connectivity, output_dir / "06_connectivity.png")

    LOG.info("Final estimate: %d retained contours + %d touching pairs = %d Tuta",
             counts["n_retained_contours"], counts["n_touching"], counts["estimated_tuta"])
    if counts["n_uncertain"] or counts["n_noise"]:
        LOG.warning("Gallery labeled %d uncertain and %d noise: BOTH are retained "
                    "as one individual by the chosen count policy",
                    counts["n_uncertain"], counts["n_noise"])
    LOG.info("Results saved in: %s", output_dir.resolve())
    return summary


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tuta moth counting: image segmentation + filtered contours + shape gallery")
    parser.add_argument("--image", type=Path, required=True,
                        help="Input RGB photograph (PNG/JPEG/etc.)")
    parser.add_argument("--gallery-dir", type=Path,
                        default=Path(__file__).resolve().parent / "tuta_gallery",
                        help="Directory with single/, touching/, optional noise/ PNGs")
    parser.add_argument("--output-dir", type=Path, default=Path("tuta_results"),
                        help="Directory for CSV, JSON and diagnostic PNG output")
    parser.add_argument("--observed", type=int, default=None,
                        help="Optional manual count, REQUIRED only for optimize mode")
    parser.add_argument("--filter-mode", choices=["fixed", "optimize"], default="optimize")
    parser.add_argument("--optimizer-count-mode",
                        choices=["contours", "legacy_original", "legacy_corrected"],
                        default="legacy_original",
                        help="Optimization objective only; does not alter final gallery-corrected count")
    parser.add_argument("--min-area", type=float, default=600.0)
    parser.add_argument("--max-area", type=float, default=5500.0)
    parser.add_argument("--min-solidity", type=float, default=0.43)
    parser.add_argument("--min-margin", type=float, default=0.001)
    parser.add_argument("--gallery-k", type=int, default=3)
    parser.add_argument("--kmeans-fit-max-pixels", type=int, default=None,
                        help="Optional K-means training sample size; omitted = all pixels")
    parser.add_argument("--n-random", type=int, default=2000,
                        help="Random optimizer candidates (optimize mode only)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = cli()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="[%(levelname)s] %(message)s")
    cfg = Config(min_area=args.min_area, max_area=args.max_area,
                 min_solidity=args.min_solidity, filter_mode=args.filter_mode,
                 optimizer_count_mode=args.optimizer_count_mode,
                 min_margin=args.min_margin, gallery_k=args.gallery_k,
                 kmeans_fit_max_pixels=args.kmeans_fit_max_pixels,
                 n_random=args.n_random)
    result = run_pipeline(args.image, args.gallery_dir, args.output_dir,
                          cfg, observed_count=args.observed)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
