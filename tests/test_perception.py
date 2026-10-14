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
    assert pc.direction_words((71, 13.8, 19.6), (68, 12.0, 23.8)) == "偏黄、偏暗"  # 肤色往黄转、变暗
    assert pc.direction_words((70, 14, 20), (70, 18, 20)).startswith("偏红")
    assert pc.direction_words((70, 14, 20), (70.5, 14.3, 20.2)) == ""
    assert pc.direction_words((60, 0, 0), (60, 0, 0)) == ""
    assert pc.direction_words((60, 0, 0), (60, 3, 0)) == "偏红"  # 中性灰：按变化本身的方向


def test_greyer_skin_is_not_called_blue_or_green():
    # 同一色相、色度变小：是“更灰”，不能说成“偏蓝、偏绿”
    assert pc.direction_words((65, 14.3, 20.5), (65, 11.5, 16.4)) == "更灰"
    assert pc.direction_words((40, 14, 17), (40, 11, 13)) == "更灰"
    assert pc.direction_words((55, -5, -40), (55, -12, -38)) == "偏青"  # 蓝天往青转
