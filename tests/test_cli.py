import json

from PIL import Image

from colordoctor import cli
from colordoctor.core import report_json as rj
from colordoctor.core import analysis as an
from colordoctor.core.color_management import synthetic_context
from synth import BLUE, YELLOW, face_scene, flat, noisy


def _jpg(tmp_path, arr, name="a.jpg"):
    p = tmp_path / name
    Image.fromarray(arr).save(p, quality=97)
    return str(p)


def _specks(seed=0):
    """蓝底上 4×4 像素的黄色杂点（单像素杂点会被 JPG 的色度二次采样抹掉）。"""
    import numpy as np

    img = noisy(flat(BLUE, 160, 200), 2)
    rng = np.random.default_rng(seed)
    for _ in range(100):
        y, x = rng.integers(0, 156), rng.integers(0, 196)
        img[y : y + 4, x : x + 4] = YELLOW
    return img


def test_cli_general_json(tmp_path, capsys):
    path = _jpg(tmp_path, _specks())
    assert cli.run(["analyze", path, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["mode"] == "general"
    assert data["contamination"]["opposite_fraction"] > 0.03
    assert "mask" not in data["contamination"]


def test_cli_skin_summary_with_rect(tmp_path, capsys):
    img, _ = face_scene()
    path = _jpg(tmp_path, img)
    assert cli.run(["analyze", path, "--mode", "skin", "--rect", "40,40,120,140", "--preset", "japanese", "--part", "脸部", "--summary"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["部位"] == "脸部" and "肤色" in data
    assert any("经验快速判定" in f["问题"] for f in data["结论"])


def test_cli_point_and_text(tmp_path, capsys):
    path = _jpg(tmp_path, flat((200, 150, 120), 30, 30))
    assert cli.run(["analyze", path, "--mode", "point", "--point", "10,10", "--size", "3"]) == 0
    assert "Lab(D50)" in capsys.readouterr().out


def test_cli_palette_and_reference(tmp_path, capsys):
    a = _jpg(tmp_path, noisy(flat(BLUE, 80, 100), 2), "a.jpg")
    b = _jpg(tmp_path, noisy(flat(YELLOW, 80, 100), 2), "b.jpg")
    assert cli.run(["analyze", a, "--mode", "palette", "--style", "teal_orange", "--summary"]) == 0
    assert "整图色彩" in json.loads(capsys.readouterr().out)
    assert cli.run(["analyze", a, "--mode", "reference", "--reference", b, "--summary"]) == 0
    assert "仿色" in json.loads(capsys.readouterr().out)


def test_cli_errors(tmp_path, capsys):
    assert cli.run(["analyze", str(tmp_path / "missing.jpg"), "--json"]) == 2
    assert "error" in capsys.readouterr().err
    path = _jpg(tmp_path, flat(BLUE, 10, 10))
    assert cli.run(["analyze", path, "--mode", "point"]) == 2


def test_cli_lists(capsys):
    assert cli.run(["presets"]) == 0
    assert "deep" in capsys.readouterr().out
    assert cli.run(["styles"]) == 0
    assert "teal_orange" in capsys.readouterr().out


def test_json_has_no_pixel_arrays():
    r = an.analyze_skin_region(face_scene()[0], synthetic_context())
    text = rj.report_to_json(r)
    assert "valid_mask" not in text and len(text) < 300_000


def test_help_and_version_go_to_cli(capsys):
    import pytest

    from colordoctor import app

    with pytest.raises(SystemExit) as e:
        app.main(["colordoctor", "--version"])
    assert e.value.code == 0
    assert "ColorDoctor" in capsys.readouterr().out
