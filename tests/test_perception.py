from colordoctor.core import perception as pc


def test_de_words_levels():
    assert pc.de_words(0.5) == "几乎看不出"
    assert pc.de_words(1.5) == "仔细对比才看得出"
    assert pc.de_words(3.0) == "能看出"
    assert pc.de_words(4.0) == "明显"
    assert pc.de_words(7.0) == "很明显"
    assert pc.de_words(15.0) == "差别很大"
    assert [pc.de_level(x) for x in (0.2, 1.2, 2.5, 4, 6, 20)] == [0, 1, 2, 3, 4, 5]


def test_direction_words():
    assert pc.direction_words((70, 14, 20), (68, 12, 25)) == "偏黄、偏暗"
    assert pc.direction_words((70, 14, 20), (70, 18, 20)) == "偏红"
    assert pc.direction_words((70, 14, 20), (70.5, 14.3, 20.2)) == ""
    assert pc.direction_words((60, 0, 0), (60, 0, 0)) == ""
