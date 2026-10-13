import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _ignore_user_acr_calibration():
    """测试不受本机 acr_calibration.json 影响。"""
    from colordoctor.core import acr_channels

    acr_channels.set_calibration({})
    yield
    acr_channels.set_calibration({})
