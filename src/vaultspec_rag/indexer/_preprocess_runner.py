"""Command-form preprocessor execution (the runtime side of the hook).

Runs a project-supplied ``command`` rule against one source file in a
``subprocess.run`` grandchild, parses and validates its stdout JSON against the
:mod:`._preprocess_schema` contract, enforces the emitted-text cap, and maps the
outcome onto the rule's ``on_error`` disposition (D6, D9, D10).

Running the real extraction in a separate OS process is what makes the command
form CPU-only-safe *by construction*: the child has its own interpreter and
cannot pollute the spawn worker's import chain or CUDA state, satisfying the
``index-workers-stay-cpu-only`` rule without any trust assumptions. The command
is split with :func:`shlex.split` and the ``{path}`` placeholder is substituted
token-wise (never via a shell), so source paths with spaces or shell
metacharacters cannot inject.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from ._preprocess_schema import (
    PreprocOutput,
    UnsupportedSchemaVersionError,
    validate_preproc_output,
)

if TYPE_CHECKING:
    import pathlib

    from ._preprocess_config import PreprocessRule

logger = logging.getLogger(__name__)

__all__ = [
    "PreprocessAbortError",
    "PreprocessResult",
    "PreprocessStatus",
    "run_preprocessor",
]

PreprocessStatus = Literal["ok", "skipped", "passthrough"]


class PreprocessAbortError(RuntimeError):
    """Raised when a failing rule has ``on_error = "fail"``.

    Propagates out of the worker to abort the whole index run, per the D1
    failure semantics. ``skip`` and ``passthrough`` never raise.
    """


class _PreprocessSkipError(Exception):
    """Internal: a recoverable per-file failure carrying a human reason."""


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    """Outcome of running one preprocessor against one source file.

    Attributes:
        status: ``ok`` (use ``output``), ``skipped`` (drop the file from the
            index, counted), or ``passthrough`` (index the raw source).
        output: The validated :class:`PreprocOutput` when ``status == "ok"``,
            else ``None``.
        reason: A human-readable explanation when the file was skipped, else
            ``None``.
    """

    status: PreprocessStatus
    output: PreprocOutput | None
    reason: str | None


def _emitted_text_length(output: PreprocOutput) -> int:
    """Return the total emitted character count for the cap check (D10)."""
    if output.text is not None:
        return len(output.text)
    if output.units is not None:
        return sum(len(unit.text) for unit in output.units)
    return 0


def _build_argv(command: str, source_path: pathlib.Path) -> list[str]:
    """Split the command template and substitute ``{path}`` token-wise."""
    tokens = shlex.split(command, posix=True)
    return [token.replace("{path}", str(source_path)) for token in tokens]


def _invoke_and_validate(
    source_path: pathlib.Path,
    rule: PreprocessRule,
    max_emitted_bytes: int,
) -> PreprocOutput:
    """Run the command, parse stdout, validate, and enforce the size cap.

    Raises:
        _PreprocessSkipError: On any recoverable per-file failure (the command
            is misconfigured, exits non-zero, times out, emits non-JSON or
            schema-invalid output, or exceeds the emitted-text cap).
    """
    if rule.command is None:  # pragma: no cover - guaranteed by loader (D9)
        msg = "rule has no command"
        raise _PreprocessSkipError(msg)

    argv = _build_argv(rule.command, source_path)
    if not argv:
        msg = "command template is empty after splitting"
        raise _PreprocessSkipError(msg)

    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            timeout=rule.timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"preprocessor timed out after {rule.timeout_s}s"
        raise _PreprocessSkipError(msg) from exc
    except OSError as exc:
        msg = f"preprocessor could not be launched: {exc}"
        raise _PreprocessSkipError(msg) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        msg = f"preprocessor exited {completed.returncode}: {stderr[:500]}"
        raise _PreprocessSkipError(msg)

    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"preprocessor stdout is not valid JSON: {exc}"
        raise _PreprocessSkipError(msg) from exc

    try:
        output = validate_preproc_output(payload)
    except (ValidationError, UnsupportedSchemaVersionError) as exc:
        msg = f"preprocessor output failed validation: {exc}"
        raise _PreprocessSkipError(msg) from exc

    emitted = _emitted_text_length(output)
    if emitted > max_emitted_bytes:
        msg = f"emitted text {emitted} exceeds cap {max_emitted_bytes}"
        raise _PreprocessSkipError(msg)

    return output


def run_preprocessor(
    source_path: pathlib.Path,
    rule: PreprocessRule,
    *,
    max_emitted_bytes: int,
) -> PreprocessResult:
    """Run a command rule against one source file and resolve its disposition.

    Args:
        source_path: Absolute path to the source file to preprocess.
        rule: The matched, validated command rule.
        max_emitted_bytes: The emitted-text length cap (D10).

    Returns:
        A :class:`PreprocessResult`. On success, ``status == "ok"`` with the
        validated output. On a recoverable failure the disposition follows the
        rule's ``on_error``: ``skip`` -> ``skipped``; ``passthrough`` ->
        ``passthrough``.

    Raises:
        PreprocessAbortError: If the rule fails and ``on_error == "fail"``.
    """
    try:
        output = _invoke_and_validate(source_path, rule, max_emitted_bytes)
    except _PreprocessSkipError as exc:
        reason = str(exc)
        if rule.on_error == "fail":
            abort = f"preprocessor for {source_path} failed (on_error=fail): {reason}"
            raise PreprocessAbortError(abort) from exc
        if rule.on_error == "passthrough":
            logger.warning(
                "preprocess passthrough for %s (%s); indexing raw source",
                source_path,
                reason,
            )
            return PreprocessResult(status="passthrough", output=None, reason=reason)
        logger.warning("preprocess skip for %s (%s)", source_path, reason)
        return PreprocessResult(status="skipped", output=None, reason=reason)

    return PreprocessResult(status="ok", output=output, reason=None)
