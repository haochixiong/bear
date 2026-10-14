import pytest

from colordoctor.core import multipoint as mp


def test_compare_points_rows_and_summary():
    res = mp.compare_points([(71, 13.8, 19.6), (71, 15.2, 18.9), (70, 14.1, 20.2), (68, 12.0, 23.8), (67, 10.9, 24.9)],
                            ["额头", "左颊", "右颊", "下巴", "脖子"])
    r = {x.name: x for x in res.rows}
    assert r["额头"].judge == "基准" and r["额头"].de is None
    assert r["右颊"].judge == "一致"
    assert r["下巴"].judge in ("略有差别", "不均") and "偏黄" in r["下巴"].direction
    assert r["脖子"].judge == "不均" and "偏黄" in r["脖子"].direction
    assert "脖子" in res.summary and "额头" in res.summary
    assert "ΔE00" in res.legend


def test_all_consistent_and_errors():
    res = mp.compare_points([(60, 10, 10), (60.3, 10.2, 10.1)])
    assert res.rows[1].judge == "一致" and "一致" in res.summary
    with pytest.raises(ValueError):
        mp.compare_points([(60, 10, 10)])
