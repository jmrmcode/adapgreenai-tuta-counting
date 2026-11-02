"""Regression checks for the calibration and counting contracts."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from tuta_pipeline import Config, cli, count_for_optimizer, filter_contours, correct_count, load_galleries


def test_defaults_and_cli(monkeypatch):
    cfg = Config()
    cfg.validate()
    monkeypatch.setattr(sys, 'argv', ['tuta_pipeline.py', '--image', 'example.png', '--observed', '17'])
    args = cli()
    assert (cfg.filter_mode, args.filter_mode) == ('optimize', 'optimize')
    assert (cfg.optimizer_count_mode, args.optimizer_count_mode) == ('legacy_original', 'legacy_original')
    assert cfg.max_area == args.max_area == 5500


def test_original_double_increment():
    # Median=100; the 400-area object contributes four, not three.
    areas = np.array([100., 100., 100., 300., 400.])
    assert count_for_optimizer(areas, 'legacy_original') == 9
    assert count_for_optimizer(areas, 'legacy_corrected') == 8


def test_observed_count_reaches_filter():
    # Synthetic rectangles are unit-test fixtures, never example trap images.
    contours = [np.array([[[0,0]],[[w,0]],[[w,30]],[[0,30]]], dtype=np.int32) for w in [21,25,30]]
    cfg=Config(n_random=0, n_starts=1, n_refinements=0)
    with pytest.raises(ValueError):
        filter_contours(contours, (40,40), cfg)
    baseline=filter_contours(contours,(40,40),cfg,3)
    calibrated=filter_contours(contours,(40,40),cfg,2)
    assert baseline.objective_count == 3
    assert calibrated.objective_count == 2
    assert calibrated.parameters != baseline.parameters


def test_noise_uncertain_and_pairs_policy():
    shapes=pd.DataFrame({'contour_id':range(4),'classification':['single','touching','noise','uncertain']})
    counts=correct_count([None]*4,shapes)
    assert counts['estimated_tuta'] == 5
    assert counts['n_noise'] == counts['n_uncertain'] == 1


def test_supplied_galleries():
    galleries=load_galleries(Path(__file__).parent/'tuta_gallery')
    assert {k:len(v) for k,v in galleries.items()} == {'single':159,'touching':18,'noise':32}
