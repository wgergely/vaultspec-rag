"""Tests for :func:`install_run`/:func:`uninstall_run` torch-config flow.

Real filesystem (``tmp_path``), real ``vaultspec_core`` from the dev
env, real ``tomlkit``. No mocks. No HF / GPU dependency — these tests
exercise only the pyproject-patching branch and deliberately do not
trigger the ``sync_after`` subprocess path.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
import tomlkit

from vaultspec_rag import torch_config as tc
from vaultspec_rag.commands import install_run, uninstall_run

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.unit]


PROJECT_ONLY = (
    "[project]\n"
    'name = "demo-consumer"\n'
    'version = "0.1.0"\n'
    'dependencies = ["vaultspec-rag"]\n'
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def consumer_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "consumer"
    ws.mkdir()
    (ws / "pyproject.toml").write_text(PROJECT_ONLY, encoding="utf-8", newline="")
    return ws


class TestInstallTorchConfig:
    def test_install_with_yes_applies_canonical_block(
        self, consumer_workspace: Path
    ) -> None:
        report = install_run(path=consumer_workspace, assume_yes=True)
        assert report.torch_config_action == "applied"
        pyproject = consumer_workspace / "pyproject.toml"
        assert tc.detect_state(pyproject) == tc.TorchConfigState.CANONICAL

    def test_install_with_no_torch_config_leaves_pyproject_alone(
        self, consumer_workspace: Path
    ) -> None:
        sha_before = _sha(consumer_workspace / "pyproject.toml")
        report = install_run(
            path=consumer_workspace, configure_torch=False, assume_yes=True
        )
        assert report.torch_config_action == "disabled"
        assert _sha(consumer_workspace / "pyproject.toml") == sha_before

    def test_install_idempotent_on_second_run(self, consumer_workspace: Path) -> None:
        install_run(path=consumer_workspace, assume_yes=True)
        sha_after_first = _sha(consumer_workspace / "pyproject.toml")
        report = install_run(path=consumer_workspace, assume_yes=True)
        assert report.torch_config_action == "already"
        assert _sha(consumer_workspace / "pyproject.toml") == sha_after_first

    def test_install_on_workspace_without_pyproject(self, tmp_path: Path) -> None:
        ws = tmp_path / "no-pyproject"
        ws.mkdir()
        report = install_run(path=ws, assume_yes=True)
        assert report.torch_config_action == "absent"
        # Warning text names the missing pyproject.
        assert any("pyproject.toml" in w for w in report.warnings)

    def test_install_with_customised_block_reports_conflict(
        self, tmp_path: Path
    ) -> None:
        ws = tmp_path / "customised"
        ws.mkdir()
        (ws / "pyproject.toml").write_text(
            PROJECT_ONLY + "\n[[tool.uv.index]]\n"
            'name = "pytorch-cu130"\n'
            'url = "https://download.pytorch.org/whl/cu121"\n'
            "explicit = true\n",
            encoding="utf-8",
            newline="",
        )
        sha_before = _sha(ws / "pyproject.toml")
        report = install_run(path=ws, assume_yes=True)
        assert report.torch_config_action == "conflict"
        assert report.torch_config_conflicts
        assert _sha(ws / "pyproject.toml") == sha_before

    def test_install_non_tty_without_yes_skips_with_warning(
        self, consumer_workspace: Path
    ) -> None:
        # confirm=None simulates the non-interactive case that cli.py
        # never hits (the CLI always passes a Confirm wrapper). This
        # path exercises the sys.stdin.isatty() / confirm-absent fence.
        report = install_run(path=consumer_workspace, assume_yes=False, confirm=None)
        assert report.torch_config_action == "skipped-non-tty"
        assert tc.detect_state(consumer_workspace / "pyproject.toml") == (
            tc.TorchConfigState.MISSING
        )

    def test_install_with_confirm_false_is_declined(
        self, consumer_workspace: Path
    ) -> None:
        # confirm returns False → user declined the prompt.
        report = install_run(
            path=consumer_workspace,
            assume_yes=False,
            confirm=lambda _prompt: False,
        )
        assert report.torch_config_action == "declined"
        assert tc.detect_state(consumer_workspace / "pyproject.toml") == (
            tc.TorchConfigState.MISSING
        )

    def test_install_dry_run_reports_dry_run_action(
        self, consumer_workspace: Path
    ) -> None:
        sha_before = _sha(consumer_workspace / "pyproject.toml")
        report = install_run(path=consumer_workspace, dry_run=True)
        assert report.torch_config_action == "dry_run"
        assert _sha(consumer_workspace / "pyproject.toml") == sha_before

    def test_install_json_surface_includes_torch_config(
        self, consumer_workspace: Path
    ) -> None:
        report = install_run(path=consumer_workspace, assume_yes=True)
        d = report.to_dict()
        assert d["torch_config_action"] == "applied"
        assert "torch_config_conflicts" in d
        assert d["torch_sync_action"] == "skipped"

    def test_install_force_implies_assume_yes_for_torch_config(
        self, consumer_workspace: Path
    ) -> None:
        """Issue #83 finding 2: ``--force`` should bypass the torch-config
        confirmation. A user who typed --force expects the whole install
        to land; silently skipping the patch with a warning is the bug.
        """
        report = install_run(
            path=consumer_workspace,
            force=True,
            assume_yes=False,
            confirm=None,  # non-interactive — would otherwise be skipped-non-tty
        )
        assert report.torch_config_action == "applied"
        assert tc.detect_state(consumer_workspace / "pyproject.toml") == (
            tc.TorchConfigState.CANONICAL
        )

    def test_install_eof_distinguished_from_decline(
        self, consumer_workspace: Path
    ) -> None:
        """Issue #83 finding 3: an EOF on the prompt must not be reported
        as 'declined by user'. CI / IDE-managed shells where ``isatty()``
        lies hit EOF instead of an answer; the user was never asked.
        """

        def eof_confirm(_prompt: str) -> bool:
            raise EOFError

        report = install_run(
            path=consumer_workspace,
            assume_yes=False,
            confirm=eof_confirm,
        )
        assert report.torch_config_action == "skipped-eof"
        # Warning names the bypass flags so the user knows the next move.
        assert any("--yes" in w or "--force" in w for w in report.warnings)
        assert tc.detect_state(consumer_workspace / "pyproject.toml") == (
            tc.TorchConfigState.MISSING
        )

    def test_install_keyboard_interrupt_still_reports_declined(
        self, consumer_workspace: Path
    ) -> None:
        """KeyboardInterrupt is a genuine user signal and must remain
        labelled as 'declined' — distinct from the new EOF branch.
        """

        def interrupt_confirm(_prompt: str) -> bool:
            raise KeyboardInterrupt

        report = install_run(
            path=consumer_workspace,
            assume_yes=False,
            confirm=interrupt_confirm,
        )
        assert report.torch_config_action == "declined"

    def test_install_warns_when_torch_not_a_direct_dep(
        self, consumer_workspace: Path
    ) -> None:
        """Issue #83 finding 4: uv ignores [tool.uv.sources] for purely-
        transitive deps. The patch lands but is a no-op for resolution
        unless torch is also a direct dep. Surface a warning naming
        the canonical fix.
        """
        # consumer_workspace has only ``vaultspec-rag`` in deps, so torch
        # is purely transitive.
        report = install_run(path=consumer_workspace, assume_yes=True)
        assert report.torch_config_action == "applied"
        assert any(
            "direct dependency" in w.lower() and "torch" in w.lower()
            for w in report.warnings
        ), report.warnings

    def test_install_no_warning_when_torch_is_direct_dep(self, tmp_path: Path) -> None:
        """The transitive-dep warning must not fire when torch is already
        a direct dep — false positives would train users to ignore it.
        """
        ws = tmp_path / "with-direct-torch"
        ws.mkdir()
        (ws / "pyproject.toml").write_text(
            "[project]\n"
            'name = "demo"\n'
            'version = "0.1.0"\n'
            'dependencies = ["vaultspec-rag", "torch>=2.4"]\n',
            encoding="utf-8",
            newline="",
        )
        report = install_run(path=ws, assume_yes=True)
        assert report.torch_config_action == "applied"
        assert not any("direct dependency" in w.lower() for w in report.warnings), (
            report.warnings
        )

    def test_install_force_with_customised_still_reports_conflict(
        self, tmp_path: Path
    ) -> None:
        """``--force`` bypasses the *prompt*, not the safety classifier.
        A CUSTOMISED block must still surface as a conflict — silently
        overwriting user-customised tool config is the worst outcome.
        """
        ws = tmp_path / "customised"
        ws.mkdir()
        (ws / "pyproject.toml").write_text(
            PROJECT_ONLY + "\n[[tool.uv.index]]\n"
            'name = "pytorch-cu130"\n'
            'url = "https://download.pytorch.org/whl/cu121"\n'
            "explicit = true\n",
            encoding="utf-8",
            newline="",
        )
        sha_before = _sha(ws / "pyproject.toml")
        report = install_run(path=ws, force=True, assume_yes=False, confirm=None)
        assert report.torch_config_action == "conflict"
        assert _sha(ws / "pyproject.toml") == sha_before

    def test_install_on_scattered_pyproject_lands_canonical(
        self, tmp_path: Path
    ) -> None:
        """Issue #83 finding 1, end-to-end: the dominant scattered
        ``[tool.*]`` shape must produce a CANONICAL pyproject after a
        single ``install --yes`` run, with no warnings about the patch.
        """
        ws = tmp_path / "scattered"
        ws.mkdir()
        (ws / "pyproject.toml").write_text(
            "[project]\n"
            'name = "demo"\n'
            'version = "0.1.0"\n'
            'dependencies = ["vaultspec-rag", "torch>=2.4"]\n'
            "\n"
            "[tool.uv]\n"
            "override-dependencies = []\n"
            "\n"
            "[tool.ruff]\n"
            "line-length = 120\n"
            "\n"
            "[tool.pytest.ini_options]\n"
            'testpaths = ["tests"]\n'
            "\n"
            "[tool.coverage.run]\n"
            'source = ["src"]\n',
            encoding="utf-8",
            newline="",
        )
        report = install_run(path=ws, assume_yes=True)
        assert report.torch_config_action == "applied", (
            report.torch_config_action,
            report.warnings,
        )
        assert tc.detect_state(ws / "pyproject.toml") == tc.TorchConfigState.CANONICAL
        # No torch-config write-failure warning surfaced.
        assert not any("torch-config write failed" in w for w in report.warnings)
        # No "[tool] is not a table" / OutOfOrderTableProxy regression.
        assert not any("not a table" in w for w in report.warnings)


class TestUninstallTorchConfig:
    def test_uninstall_removes_canonical_block(self, consumer_workspace: Path) -> None:
        install_run(path=consumer_workspace, assume_yes=True)
        report = uninstall_run(path=consumer_workspace, force=True)
        assert report.torch_config_action == "removed"
        assert tc.detect_state(consumer_workspace / "pyproject.toml") == (
            tc.TorchConfigState.MISSING
        )

    def test_uninstall_round_trip_preserves_project_table(
        self, consumer_workspace: Path
    ) -> None:
        """Project semantics survive the apply/remove round trip."""
        orig_parsed = tomlkit.parse(
            (consumer_workspace / "pyproject.toml").read_text(encoding="utf-8")
        )
        install_run(path=consumer_workspace, assume_yes=True)
        uninstall_run(path=consumer_workspace, force=True)
        final_parsed = tomlkit.parse(
            (consumer_workspace / "pyproject.toml").read_text(encoding="utf-8")
        )
        assert final_parsed["project"] == orig_parsed["project"]

    def test_uninstall_on_missing_pyproject_is_absent(self, tmp_path: Path) -> None:
        ws = tmp_path / "empty"
        ws.mkdir()
        # Install nothing; uninstall must be a no-op on the torch-config side.
        report = uninstall_run(path=ws, force=True)
        assert report.torch_config_action == "absent"

    def test_uninstall_on_customised_block_skips_with_conflict(
        self, tmp_path: Path
    ) -> None:
        ws = tmp_path / "customised"
        ws.mkdir()
        (ws / "pyproject.toml").write_text(
            PROJECT_ONLY + "\n[[tool.uv.index]]\n"
            'name = "pytorch-cu130"\n'
            'url = "https://download.pytorch.org/whl/cu121"\n'
            "explicit = true\n",
            encoding="utf-8",
            newline="",
        )
        sha_before = _sha(ws / "pyproject.toml")
        report = uninstall_run(path=ws, force=True)
        assert report.torch_config_action == "skipped"
        assert report.torch_config_conflicts  # populated symmetrically
        assert _sha(ws / "pyproject.toml") == sha_before

    def test_uninstall_dry_run_on_customised_populates_conflicts(
        self, tmp_path: Path
    ) -> None:
        """Dry-run on a CUSTOMISED workspace must still report conflicts
        so the user sees why the removal would refuse. Mirrors the
        install side, which surfaces conflicts in every mode.
        """
        ws = tmp_path / "customised"
        ws.mkdir()
        (ws / "pyproject.toml").write_text(
            PROJECT_ONLY + "\n[[tool.uv.index]]\n"
            'name = "pytorch-cu130"\n'
            'url = "https://download.pytorch.org/whl/cu121"\n'
            "explicit = true\n",
            encoding="utf-8",
            newline="",
        )
        sha_before = _sha(ws / "pyproject.toml")
        report = uninstall_run(path=ws, force=False)  # dry-run path
        assert report.torch_config_action == "skipped"
        assert report.torch_config_conflicts
        assert _sha(ws / "pyproject.toml") == sha_before

    def test_uninstall_dry_run_does_not_mutate(self, consumer_workspace: Path) -> None:
        install_run(path=consumer_workspace, assume_yes=True)
        sha_before = _sha(consumer_workspace / "pyproject.toml")
        # dry_run path without --force stays in dry-run mode.
        report = uninstall_run(path=consumer_workspace, force=False)
        assert report.torch_config_action == "dry_run"
        assert _sha(consumer_workspace / "pyproject.toml") == sha_before
