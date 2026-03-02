"""Tests for tinyml_deployer.targets."""

from __future__ import annotations

import pytest

from tinyml_deployer.targets import MCUTarget, TARGETS, get_target
from tests.conftest import ALL_TARGET_NAMES


class TestGetTarget:
    """Test the get_target() lookup function."""

    @pytest.mark.parametrize("name", ALL_TARGET_NAMES)
    def test_returns_mcu_target(self, name: str) -> None:
        target = get_target(name)
        assert isinstance(target, MCUTarget)
        assert target.name == name

    def test_case_insensitive(self) -> None:
        target = get_target("ESP32")
        assert target.name == "esp32"

    def test_invalid_target_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown target"):
            get_target("nonexistent_mcu")


class TestTargetFields:
    """Verify all targets have the required hardware fields."""

    @pytest.mark.parametrize("name", ALL_TARGET_NAMES)
    def test_has_positive_flash(self, name: str) -> None:
        assert get_target(name).flash_kb > 0

    @pytest.mark.parametrize("name", ALL_TARGET_NAMES)
    def test_has_positive_ram(self, name: str) -> None:
        assert get_target(name).ram_kb > 0

    @pytest.mark.parametrize("name", ALL_TARGET_NAMES)
    def test_has_positive_clock(self, name: str) -> None:
        assert get_target(name).clock_mhz > 0

    @pytest.mark.parametrize("name", ALL_TARGET_NAMES)
    def test_has_positive_cycles_per_mac(self, name: str) -> None:
        assert get_target(name).cycles_per_mac > 0

    @pytest.mark.parametrize("name", ALL_TARGET_NAMES)
    def test_has_framework(self, name: str) -> None:
        assert get_target(name).framework in ("ESP-IDF", "STM32CubeAI")

    @pytest.mark.parametrize("name", ALL_TARGET_NAMES)
    def test_has_supported_ops(self, name: str) -> None:
        assert len(get_target(name).supported_ops) > 0

    def test_targets_dict_matches_count(self) -> None:
        assert len(TARGETS) == len(ALL_TARGET_NAMES)
