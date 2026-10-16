import gc
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# 和程序里的 gc_guard 一样：不让自动回收在分析线程（或上个测试留下的后台任务）里析构 Qt 对象，
# Windows 上会随机崩溃；改为每个测试结束后在主线程回收。
gc.disable()


@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path_factory, monkeypatch):
    """测试不读写本机真实的设置目录（用户自己的肤色预设、error.log 等）：
    打包脚本在开发者电脑上跑测试时，不会因为本机的自定义预设而失败。"""
    d = tmp_path_factory.getbasetemp() / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(d))
    monkeypatch.setenv("APPDATA", str(d))


@pytest.fixture(autouse=True)
def _collect_on_main_thread():
    yield
    from colordoctor.core import pixel_table

    pixel_table.forget_all()  # 跨测试的像素表缓存也清掉
    gc.collect()


@pytest.fixture(autouse=True)
def _ignore_user_acr_calibration():
    """测试不受本机 acr_calibration.json 影响。"""
    from colordoctor.core import acr_channels

    acr_channels.set_calibration({})
    yield
    acr_channels.set_calibration({})
