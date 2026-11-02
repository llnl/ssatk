"""Artemis demo rendering contracts, not physical radiation validation."""

from unittest.mock import MagicMock

import pytest

from demos.space_environment import demo_artemis_radiation as demo


def test_disabled_demo_creates_no_output():
    assert demo.main(make_figures=False)["skipped"]


@pytest.mark.parametrize("screenshots, message", [
    ([False], "Movie screenshot failed at frame 0"),
    ([True, False], "Preview screenshot failed at frame 0"),
])
def test_recording_failure_does_not_write_video(tmp_path, monkeypatch, screenshots, message):
    from selenium import webdriver
    from ssatk.plots import write_videos

    browser = MagicMock()
    browser.__enter__.return_value = browser
    browser.execute_script.return_value = True
    browser.save_screenshot.side_effect = screenshots
    monkeypatch.setattr(webdriver, "Edge", lambda **kwargs: browser)
    writer = MagicMock()
    monkeypatch.setattr(write_videos, "write_video", writer)

    with pytest.raises(RuntimeError, match=message):
        demo.record_movie(tmp_path / "scene.html", tmp_path / "scene.mp4")

    browser.execute_async_script.assert_called_once()
    assert browser.execute_async_script.call_args.args[1] == 0.0
    writer.assert_not_called()
    assert not list(tmp_path.iterdir())
