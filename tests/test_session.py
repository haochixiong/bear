import pytest

from colordoctor.core.color_management import synthetic_context
from colordoctor.core.sampling import SampleError
from colordoctor.ui.session import Session, crop_selection, point_patch_from
from synth import BLUE, YELLOW, flat, noisy, skin_region, sprinkle

CTX = synthetic_context()


def test_point_flow_adds_history():
    s = Session()
    img = flat(BLUE, 20, 20)
    sample = s.compute_point(img, 5, 5, s.radius, CTX)
    item = s.add_point(point_patch_from(img, 5, 5, s.radius), sample, "test")
    assert s.history[0] is item
    assert item.swatch == BLUE
    assert item.region.shape == (5, 5, 3)


def test_rect_modes_route_to_correct_analysis():
    s = Session()
    img = sprinkle(noisy(flat(BLUE), 2), YELLOW, 0.05)
    s.mode = "dirty"
    assert s.compute_rect(img, CTX, (0, 0, 160, 120)).mode == "general"
    s.scan_whole = True
    assert s.compute_rect(img, CTX, (0, 0, 160, 120)).mode == "scan"
    s.mode = "skin"
    s.preset_key = "japanese"
    r = s.compute_rect(skin_region(), CTX, (0, 0, 120, 120))
    assert r.mode == "skin" and r.adjustments


def test_compare_flow_with_reference():
    s = Session()
    s.mode = "compare"
    ref = skin_region(68, 14, 19)
    rep = s.compute_rect(ref, CTX, (0, 0, 120, 120), mode="skin")
    item = s.add_rect(ref, (0, 0, 120, 120), rep, "x", role="参考区")
    s.set_reference(item)
    out = s.compute_compare(item, skin_region(68, 12, 27), CTX, (200, 0, 120, 120))
    assert out.mode == "compare" and out.comparison.db > 6


def test_custom_preset_from_history_item():
    s = Session()
    ref = skin_region(75, 11, 14)
    rep = s.compute_rect(ref, CTX, (0, 0, 120, 120), mode="skin")
    item = s.add_rect(ref, (0, 0, 120, 120), rep, "x")
    p = s.set_custom_preset_from(item)
    assert s.preset is p
    assert p.target_lab[0] == pytest.approx(75, abs=1.5)


def test_remove_and_clear_history():
    s = Session()
    img = flat(BLUE, 10, 10)
    a = s.add_point(point_patch_from(img, 1, 1, 0), s.compute_point(img, 1, 1, 0, CTX), "x")
    s.set_reference(a)
    s.remove(a.id)
    assert not s.history and s.reference is None
    s.add_point(point_patch_from(img, 1, 1, 0), s.compute_point(img, 1, 1, 0, CTX), "x")
    s.clear()
    assert not s.history


def test_crop_selection_out_of_bounds():
    img = flat(BLUE, 10, 10)
    region, r = crop_selection(img, (-5, -5, 8, 8))
    assert r == (0, 0, 3, 3) and region.shape == (3, 3, 3)
    with pytest.raises(SampleError):
        crop_selection(img, (50, 50, 5, 5))


def test_face_body_parts_are_compared_automatically():
    s = Session()
    s.mode = "skin"
    s.part = "身体"
    body = skin_region(64, 12, 24)
    s.add_rect(body, (0, 0, 120, 120), s.compute_rect(body, CTX, (0, 0, 120, 120)), "x")
    s.part = "脸部"
    face = skin_region(70, 13, 18)
    rep = s.compute_rect(face, CTX, (0, 0, 120, 120))
    assert any("脸部与身体" in f.problem for f in rep.findings)


def test_palette_and_reference_modes():
    s = Session()
    s.mode = "palette"
    s.style_key = "teal_orange"
    s.family = "蓝"
    img = noisy(flat(BLUE, 80, 100), 2)
    rep = s.compute_rect(img, CTX, (0, 0, 100, 80))
    assert rep.mode == "palette" and rep.palette.style is not None and rep.palette.unity is not None
    item = s.add_rect(img, (0, 0, 100, 80), rep, "x", role="参考图")
    out = s.compute_reference(item, noisy(flat(YELLOW, 80, 100), 2), CTX, (0, 0, 100, 80))
    assert out.mode == "reference"
