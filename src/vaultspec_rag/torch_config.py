"""Detect, write, and remove the canonical cu130 torch block in a user's
``pyproject.toml``.

This module is the pure-logic layer for rag's ``install`` /
``uninstall`` torch-config step. It mirrors the per-resource module
pattern core follows for ``gitignore.py`` / ``gitattributes.py`` /
``mcps.py``: no Typer, no Rich, no prompts, no process side-effects
beyond a single atomic write.

Canonical block shape — see :func:`manual_snippet` for the exact
bytes rag emits (three module constants compose the shape).

The three module-level constants are the single source of truth for
that shape — apply and remove compare against them, and
``manual_snippet`` renders them verbatim. Symmetric apply/remove is
guaranteed by construction.

See :doc:`.vault/adr/2026-04-22-install-cuda-adr` for the
architectural decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

import tomlkit
from tomlkit import TOMLDocument
from tomlkit.container import OutOfOrderTableProxy
from tomlkit.items import AoT, InlineTable, Table
from vaultspec_core.core.helpers import atomic_write

# tomlkit returns ``OutOfOrderTableProxy`` for any ``[tool.X]`` whose
# child tables (``[tool.X.Y]``, ``[tool.X.Z]``) are interleaved with
# unrelated sections — the dominant pyproject.toml shape (e.g.
# ``[tool.uv]``, ``[tool.ruff]``, ``[tool.uv.sources]`` interspersed).
# It implements the same Mapping API we exercise (``get``, ``setdefault``,
# ``__setitem__``, ``__delitem__``, ``__bool__``) but does not subclass
# ``Table``, so plain ``isinstance(x, Table)`` checks would reject it
# and force apply / detect onto the wrong code path. Treat it as a
# table-like surface throughout the module.
#
# Use the literal ``isinstance(x, (Table, OutOfOrderTableProxy))`` form
# inline at every check site so static type-checkers (ty/pyright)
# narrow to ``Table | OutOfOrderTableProxy`` after the guard. A
# ``Final[tuple[type, ...]]`` alias defeats that narrowing and forces
# ``Unknown`` downstream.
TableLike = Table | OutOfOrderTableProxy

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "CU130_INDEX_NAME",
    "CU130_INDEX_URL",
    "CU130_MARKER",
    "PatchReport",
    "TorchConfigState",
    "TorchDiagnosis",
    "apply_patch",
    "detect_state",
    "diagnose_torch",
    "has_direct_torch_dep",
    "manual_snippet",
    "preview_patch",
    "remove_patch",
]


CU130_INDEX_NAME: Final[str] = "pytorch-cu130"
CU130_INDEX_URL: Final[str] = "https://download.pytorch.org/whl/cu130"
CU130_MARKER: Final[str] = "sys_platform == 'linux' or sys_platform == 'win32'"


class TorchConfigState(StrEnum):
    """Classification of a ``pyproject.toml`` relative to rag's cu130 block."""

    MISSING = "missing"
    CANONICAL = "canonical"
    CUSTOMISED = "customised"
    NO_PROJECT_FILE = "no_project_file"


class TorchDiagnosis(StrEnum):
    """Classification of a torch install's CUDA support."""

    NO_TORCH = "no_torch"
    CPU_ONLY = "cpu_only"
    NO_GPU = "no_gpu"
    WORKING = "working"


@dataclass
class PatchReport:
    """Structured outcome of an apply / remove pass.

    Attributes:
        action: One of ``"applied"``, ``"skipped"``, ``"conflict"``,
            ``"absent"``, ``"already"``, ``"removed"``.
        path: The pyproject.toml inspected.
        conflicts: Human-readable descriptions of conflicting keys
            when ``action == "conflict"``.
        preview: The TOML snippet that would be (or was) written,
            for dry-run / display purposes.
    """

    action: str
    path: Path
    conflicts: list[str] = field(default_factory=list)
    preview: str = ""


def manual_snippet() -> str:
    """Return the canonical cu130 block as a copy-pasteable string.

    Assembled from the three module constants so this string and the
    shape ``apply_patch`` writes can never drift. Includes a comment
    spelling out the direct-dependency requirement uv enforces — the
    source pin is silently ignored when ``torch`` only enters the
    resolution as a transitive dep of ``vaultspec-rag``.
    """
    return (
        "\n"
        "[[tool.uv.index]]\n"
        f'name = "{CU130_INDEX_NAME}"\n'
        f'url = "{CU130_INDEX_URL}"\n'
        "explicit = true\n"
        "\n"
        "[tool.uv.sources]\n"
        f'torch = [{{ index = "{CU130_INDEX_NAME}", '
        f'marker = "{CU130_MARKER}" }}]\n'
        "\n"
        "# uv ignores [tool.uv.sources] for purely-transitive deps.\n"
        "# Add torch as a direct dep too, e.g. in [project].dependencies\n"
        '# or [dependency-groups].dev:  "torch>=2.4"\n'
    )


def _load(pyproject: Path) -> TOMLDocument | None:
    """Load and parse the pyproject.toml, or return None if absent."""
    if not pyproject.is_file():
        return None
    try:
        return tomlkit.parse(pyproject.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("pyproject.toml parse failed at %s: %s", pyproject, exc)
        raise


def _tool_uv(doc: TOMLDocument) -> TableLike | None:
    """Return the ``[tool.uv]`` table, or None.

    Narrowed to either :class:`tomlkit.items.Table` or
    :class:`tomlkit.container.OutOfOrderTableProxy`. tomlkit returns
    the proxy whenever ``[tool.uv]`` and ``[tool.uv.sources]`` (or
    ``[[tool.uv.index]]``) are interleaved with non-uv sections — the
    dominant ``[tool.*]`` layout in real-world pyprojects. Both expose
    the same Mapping surface we touch.
    """
    tool = doc.get("tool")
    if not isinstance(tool, (Table, OutOfOrderTableProxy)):
        return None
    uv = tool.get("uv")
    if not isinstance(uv, (Table, OutOfOrderTableProxy)):
        return None
    return uv


def _indices(doc: TOMLDocument) -> AoT | None:
    """Return the ``[[tool.uv.index]]`` array-of-tables, or None.

    Returns None if the key is absent OR if it exists but is not an
    AoT. Callers (apply/remove helpers) rely on AoT-specific methods
    like ``.append()`` and item-index ``.pop()``; returning only AoT
    keeps the downstream contract narrow. The CUSTOMISED-classifier
    in :func:`_classify` catches non-AoT shapes separately via a
    direct ``uv.get("index")`` probe.
    """
    uv = _tool_uv(doc)
    if uv is None:
        return None
    idx = uv.get("index")
    if not isinstance(idx, AoT):
        return None
    return idx


def _torch_sources(doc: TOMLDocument) -> Any:
    """Return the ``torch`` entry under ``[tool.uv.sources]``, or None.

    Returns whatever tomlkit has at that key so the caller can
    classify unusual shapes (scalar, standard Table, inline-table
    array, or missing). Strictly-typed callers downcast as needed.
    """
    uv = _tool_uv(doc)
    if uv is None:
        return None
    sources = uv.get("sources")
    if not isinstance(sources, (Table, OutOfOrderTableProxy)):
        return None
    return sources.get("torch")


def _index_match(entry: Table | InlineTable | dict) -> str:
    """Classify one ``[[tool.uv.index]]`` entry against the canonical cu130.

    Returns ``"canonical"`` if the entry has our name, our url, and
    ``explicit = true``; ``"conflict"`` if the entry has our name but
    disagrees on url/explicit; ``""`` if the entry is unrelated (name
    does not match).
    """
    name = entry.get("name")
    if name != CU130_INDEX_NAME:
        return ""
    url = entry.get("url")
    explicit = entry.get("explicit", False)
    if url == CU130_INDEX_URL and bool(explicit):
        return "canonical"
    return "conflict"


def _source_match(entry: InlineTable | dict) -> str:
    """Classify one ``torch`` source entry against the canonical cu130.

    Returns ``"canonical"`` if the entry matches our ``index`` and
    ``marker`` (and nothing else beyond the two keys); ``"conflict"``
    if the entry references our index name but disagrees; ``""`` if
    the entry is unrelated (different index).
    """
    idx = entry.get("index")
    if idx != CU130_INDEX_NAME:
        return ""
    marker = entry.get("marker")
    extras = set(entry.keys()) - {"index", "marker"}
    if marker == CU130_MARKER and not extras:
        return "canonical"
    return "conflict"


def _classify_indices(
    doc: TOMLDocument,
) -> tuple[bool | None, bool, list[str]]:
    """Classify the ``[[tool.uv.index]]`` section.

    Returns a tuple ``(canonical_seen, conflict_seen, conflicts)``
    where ``canonical_seen`` is ``None`` when the user's TOML has a
    shape we refuse to touch (single-table ``[tool.uv.index]``) —
    the caller should short-circuit to ``CUSTOMISED``.
    """
    conflicts: list[str] = []

    # `[tool.uv.index]` (single table) is a valid TOML structure but
    # incompatible with our array-of-tables mutations. Probe the raw
    # key directly — ``_indices()`` narrows to AoT only, so we can't
    # use its return value to detect this shape.
    uv = _tool_uv(doc)
    raw_index = uv.get("index") if uv is not None else None
    if raw_index is not None and not isinstance(raw_index, AoT):
        conflicts.append(
            "[tool.uv.index] is a single table, not an array-of-tables; "
            "rag's apply_patch expects [[tool.uv.index]]"
        )
        return None, False, conflicts

    canonical_seen = False
    conflict_seen = False
    indices = _indices(doc)
    if indices is not None:
        for entry in indices:
            if not isinstance(entry, Table | InlineTable | dict):
                continue
            m = _index_match(entry)
            if m == "canonical":
                canonical_seen = True
            elif m == "conflict":
                conflict_seen = True
                conflicts.append(
                    f"[[tool.uv.index]] '{CU130_INDEX_NAME}' "
                    f"exists with non-canonical url/explicit"
                )
    return canonical_seen, conflict_seen, conflicts


def _classify_sources(
    doc: TOMLDocument,
) -> tuple[bool | None, bool, list[str]]:
    """Classify the ``[tool.uv.sources]`` ``torch`` entry.

    Return convention mirrors :func:`_classify_indices`: a
    ``canonical_seen`` value of ``None`` signals an unsupported
    user shape (scalar or standard-table form), and the caller must
    short-circuit to ``CUSTOMISED``.
    """
    conflicts: list[str] = []
    torch_srcs = _torch_sources(doc)
    if torch_srcs is None:
        return False, False, conflicts

    # Scalar at `tool.uv.sources.torch` (string, int, bool, …) is
    # syntactically legal TOML but semantically nonsense for uv.
    # Treat as CUSTOMISED so apply_patch refuses before the
    # mutation helpers inherit a value they cannot recurse into.
    if not isinstance(torch_srcs, InlineTable | Table | list | dict):
        conflicts.append(
            f"[tool.uv.sources] torch is a "
            f"{type(torch_srcs).__name__}, not an array or table"
        )
        return None, False, conflicts

    # `[tool.uv.sources.torch]` (standard table, e.g. a git source
    # spelled as its own section) cannot be promoted into a TOML
    # array — arrays can only hold inline tables. Treat as
    # CUSTOMISED so apply never tries to rewrite it.
    if isinstance(torch_srcs, Table) and not isinstance(torch_srcs, InlineTable | list):
        conflicts.append(
            "[tool.uv.sources.torch] is a standard table; "
            "rag's apply_patch expects an inline-table array"
        )
        return None, False, conflicts

    canonical_seen = False
    conflict_seen = False
    # torch source may be a single inline table or a list of them.
    if isinstance(torch_srcs, list):
        entries: list = list(torch_srcs)
    else:
        entries = [torch_srcs]
    for entry in entries:
        if not isinstance(entry, InlineTable | Table | dict):
            continue
        m = _source_match(entry)
        if m == "canonical":
            canonical_seen = True
        elif m == "conflict":
            conflict_seen = True
            conflicts.append(
                f"[tool.uv.sources] torch references "
                f"'{CU130_INDEX_NAME}' with non-canonical marker/extras"
            )
    return canonical_seen, conflict_seen, conflicts


def _classify(doc: TOMLDocument) -> tuple[TorchConfigState, list[str]]:
    """Inspect the loaded doc and return (state, conflicts).

    Delegates the per-section classification to
    :func:`_classify_indices` and :func:`_classify_sources`, then
    combines their verdicts into a single ``TorchConfigState``.
    """
    index_canonical, index_conflict, index_conflicts = _classify_indices(doc)
    if index_canonical is None:
        # Short-circuit on an unsupported user shape at [[tool.uv.index]].
        return TorchConfigState.CUSTOMISED, index_conflicts

    source_canonical, source_conflict, source_conflicts = _classify_sources(doc)
    if source_canonical is None:
        return TorchConfigState.CUSTOMISED, source_conflicts

    conflicts = index_conflicts + source_conflicts
    if index_conflict or source_conflict:
        return TorchConfigState.CUSTOMISED, conflicts
    if index_canonical and source_canonical:
        return TorchConfigState.CANONICAL, []
    if index_canonical != source_canonical:
        # Half-applied state — treat as customised; we cannot safely
        # complete it without risking user intent.
        missing = "source" if index_canonical else "index"
        conflicts.append(
            f"cu130 {missing} entry missing while the other half is present"
        )
        return TorchConfigState.CUSTOMISED, conflicts
    return TorchConfigState.MISSING, []


def detect_state(pyproject: Path) -> TorchConfigState:
    """Classify the current torch-config block at ``pyproject``.

    Returns one of ``NO_PROJECT_FILE``, ``MISSING``, ``CANONICAL``,
    ``CUSTOMISED``. Pure read; never writes.

    Args:
        pyproject: Path to the consumer's ``pyproject.toml``.

    Raises:
        tomlkit.exceptions.ParseError: If the file is syntactically
            invalid TOML. Callers should surface this as a hard error
            — there is no safe way to edit a corrupt file.
    """
    doc = _load(pyproject)
    if doc is None:
        return TorchConfigState.NO_PROJECT_FILE
    state, _ = _classify(doc)
    return state


def preview_patch(pyproject: Path) -> str:
    """Return the TOML snippet ``apply_patch`` would write.

    Returns an empty string when state is ``CANONICAL`` (nothing to
    do) or ``CUSTOMISED`` (apply refuses). Returns
    :func:`manual_snippet` for ``MISSING`` and ``NO_PROJECT_FILE``.
    """
    state = detect_state(pyproject)
    if state in (TorchConfigState.MISSING, TorchConfigState.NO_PROJECT_FILE):
        return manual_snippet()
    return ""


def apply_patch(pyproject: Path) -> PatchReport:
    """Write the canonical cu130 block into ``pyproject``.

    Idempotent:
    - ``CANONICAL`` → ``action="already"``, no write.
    - ``CUSTOMISED`` → ``action="conflict"`` with conflicts listed,
      no write.
    - ``NO_PROJECT_FILE`` → ``action="absent"``, no write.
    - ``MISSING`` → write via
      :func:`vaultspec_core.core.helpers.atomic_write`,
      ``action="applied"``.

    Preserves all user comments, key ordering, and whitespace in the
    rest of the document via tomlkit's round-trip semantics.
    """
    report = PatchReport(action="skipped", path=pyproject)
    doc = _load(pyproject)
    if doc is None:
        report.action = "absent"
        return report

    state, conflicts = _classify(doc)
    if state == TorchConfigState.CANONICAL:
        report.action = "already"
        return report
    if state == TorchConfigState.CUSTOMISED:
        report.action = "conflict"
        report.conflicts = conflicts
        return report

    # MISSING → write.
    _ensure_tool_uv_index(doc)
    _ensure_torch_source(doc)
    new_text = tomlkit.dumps(doc)
    # Reparse to confirm validity before the write crosses the FS boundary.
    tomlkit.parse(new_text)
    atomic_write(pyproject, new_text)
    report.action = "applied"
    report.preview = manual_snippet()
    return report


def remove_patch(pyproject: Path) -> PatchReport:
    """Remove the canonical cu130 block from ``pyproject``.

    Symmetric inverse of :func:`apply_patch`:
    - ``MISSING`` or ``NO_PROJECT_FILE`` → ``action="absent"``.
    - ``CUSTOMISED`` → ``action="skipped"`` with conflicts listed;
      file unchanged.
    - ``CANONICAL`` → remove only our entries, drop any now-empty
      containers, ``action="removed"``.

    After removal, the file should reparse as valid TOML and all
    user-owned content outside the cu130 block is preserved.
    """
    report = PatchReport(action="skipped", path=pyproject)
    doc = _load(pyproject)
    if doc is None:
        report.action = "absent"
        return report

    state, conflicts = _classify(doc)
    if state == TorchConfigState.MISSING:
        report.action = "absent"
        return report
    if state == TorchConfigState.CUSTOMISED:
        report.action = "skipped"
        report.conflicts = conflicts
        return report

    # CANONICAL → remove.
    _drop_cu130_index(doc)
    _drop_torch_source(doc)
    new_text = tomlkit.dumps(doc)
    tomlkit.parse(new_text)
    atomic_write(pyproject, new_text)
    report.action = "removed"
    return report


def _is_torch_requirement(req: object) -> bool:
    """Return True if ``req`` (a PEP 508 entry) names ``torch``.

    Strips extras (``torch[extra]``), version specifiers
    (``torch>=2.4``), URL form (``torch @ https://...``), and any
    surrounding whitespace before the comparison. Returns False for
    anything that is not a string — comments, dict-form entries,
    accidentally-typed integers — to keep the predicate total.
    """
    if not isinstance(req, str):
        return False
    name = req.strip()
    # Cut at the first PEP 508 separator. Order matters: ``@`` may
    # appear in a URL but never before whitespace in a name; the
    # ``min(...)`` collapses whichever boundary comes first.
    boundaries = [name.find(c) for c in (" ", "[", "<", ">", "=", "!", "~", "@", ";")]
    cuts = [i for i in boundaries if i != -1]
    if cuts:
        name = name[: min(cuts)]
    return name.strip().lower() == "torch"


def _iter_dep_lists(doc: TOMLDocument) -> list[tuple[str, Any]]:
    """Yield ``(label, sequence)`` for every direct-dependency surface.

    Covers the three idiomatic shapes a consumer can express torch in:
    ``[project].dependencies`` (PEP 621), ``[project].optional-dependencies.*``
    (PEP 621 extras), and ``[dependency-groups].*`` (PEP 735). Returns
    a label so callers can report exactly where torch was found.
    """
    found: list[tuple[str, Any]] = []
    project = doc.get("project")
    if isinstance(project, (Table, OutOfOrderTableProxy)):
        deps = project.get("dependencies")
        if isinstance(deps, list):
            found.append(("[project].dependencies", deps))
        optional = project.get("optional-dependencies")
        if isinstance(optional, (Table, OutOfOrderTableProxy)):
            for name, group in optional.items():
                if isinstance(group, list):
                    found.append((f"[project.optional-dependencies].{name}", group))
    groups = doc.get("dependency-groups")
    if isinstance(groups, (Table, OutOfOrderTableProxy)):
        for name, group in groups.items():
            if isinstance(group, list):
                found.append((f"[dependency-groups].{name}", group))
    return found


def has_direct_torch_dep(pyproject: Path) -> tuple[bool, str]:
    """Return ``(present, location)`` for a direct ``torch`` dependency.

    uv silently ignores ``[tool.uv.sources]`` entries for purely-
    transitive packages. The cu130 source pin therefore only takes
    effect once ``torch`` appears as a direct dep of the consumer
    project. This helper lets the install flow surface a warning when
    the patch will be a no-op.

    Args:
        pyproject: Path to the consumer's ``pyproject.toml``.

    Returns:
        ``(True, "<location>")`` when torch is found as a direct dep,
        with ``location`` naming the dotted-key path of the table or
        list that contained the entry. ``(False, "")`` when absent or
        when the file cannot be parsed (the install flow already
        surfaces parse errors elsewhere).
    """
    try:
        doc = _load(pyproject)
    except Exception:
        return False, ""
    if doc is None:
        return False, ""
    for label, deps in _iter_dep_lists(doc):
        for entry in deps:
            if _is_torch_requirement(entry):
                return True, label
    return False, ""


def diagnose_torch(cuda: str | None, available: bool) -> TorchDiagnosis:
    """Classify a torch install from its observable CUDA attributes.

    Args:
        cuda: ``torch.version.cuda`` — ``None`` for the CPU-only
            wheel, a version string like ``"13.0"`` for the CUDA
            wheels.
        available: ``torch.cuda.is_available()`` result.

    Returns:
        One of :class:`TorchDiagnosis`. ``(None, True)`` is an
        anomaly not produced by any supported torch build; we fall
        back to ``CPU_ONLY`` because the remediation (reinstall from
        cu130) is the safer of the two.
    """
    if cuda is None:
        return TorchDiagnosis.CPU_ONLY
    if available:
        return TorchDiagnosis.WORKING
    return TorchDiagnosis.NO_GPU


# ---------------------------------------------------------------------------
# tomlkit mutation helpers
# ---------------------------------------------------------------------------


def _get_or_create_tool_uv(doc: TOMLDocument) -> TableLike:
    """Return the writable ``[tool.uv]`` table, creating it if absent.

    Extracted so :func:`_ensure_tool_uv_index` and
    :func:`_ensure_torch_source` share a single source of truth for
    how the ``[tool]`` → ``[tool.uv]`` hierarchy is materialised.
    Raises if either level exists but isn't a table (refuses to
    silently clobber a user-owned key).

    Accepts both :class:`tomlkit.items.Table` and
    :class:`tomlkit.container.OutOfOrderTableProxy`. Both expose the
    Mapping API we exercise here (``setdefault`` / ``__setitem__``);
    the proxy is what tomlkit returns whenever ``[tool.X]`` sub-tables
    are interleaved with unrelated sections.
    """
    tool = doc.setdefault("tool", tomlkit.table())
    if not isinstance(tool, (Table, OutOfOrderTableProxy)):
        raise TypeError("pyproject.toml [tool] is not a table")
    uv = tool.setdefault("uv", tomlkit.table())
    if not isinstance(uv, (Table, OutOfOrderTableProxy)):
        raise TypeError("pyproject.toml [tool.uv] is not a table")
    return uv


def _ensure_tool_uv_index(doc: TOMLDocument) -> None:
    """Append ``[[tool.uv.index]]`` with the canonical cu130 entry.

    Creates ``[tool]`` and ``[tool.uv]`` if absent. Appends to the
    existing array-of-tables if one is present.
    """
    uv = _get_or_create_tool_uv(doc)
    existing = uv.get("index")
    entry = tomlkit.table()
    entry["name"] = CU130_INDEX_NAME
    entry["url"] = CU130_INDEX_URL
    entry["explicit"] = True

    if existing is None:
        aot = tomlkit.aot()
        aot.append(entry)
        uv["index"] = aot
        return

    # Defensive: _classify already rejects single-table forms as
    # CUSTOMISED so apply_patch never reaches this function in that
    # case. Guard anyway — silently-wrong mutation is the worst
    # outcome for a user-owned file.
    if not isinstance(existing, AoT | list):
        raise TypeError(
            "pyproject.toml [tool.uv.index] is not an array-of-tables; "
            "refuse to mutate to avoid producing invalid TOML"
        )
    existing.append(entry)


def _ensure_torch_source(doc: TOMLDocument) -> None:
    """Ensure ``[tool.uv.sources]`` has a cu130 torch entry."""
    uv = _get_or_create_tool_uv(doc)
    sources = uv.setdefault("sources", tomlkit.table())
    if not isinstance(sources, (Table, OutOfOrderTableProxy)):
        raise TypeError("pyproject.toml [tool.uv.sources] is not a table")

    inline = tomlkit.inline_table()
    inline["index"] = CU130_INDEX_NAME
    inline["marker"] = CU130_MARKER

    current = sources.get("torch")
    if current is None:
        arr = tomlkit.array()
        arr.append(inline)
        arr.multiline(True)
        sources["torch"] = arr
        return

    if isinstance(current, list):
        current.append(inline)
        return

    # Promotion: existing single inline-table → array of two inline
    # tables. Only valid when `current` is an InlineTable; a standard
    # Table (e.g. ``[tool.uv.sources.torch]`` section form) cannot be
    # nested inside a TOML array. _classify already rejects that
    # shape as CUSTOMISED; guard anyway for defence-in-depth.
    if not isinstance(current, InlineTable):
        raise TypeError(
            "pyproject.toml [tool.uv.sources] torch is a standard "
            "table; refuse to promote into an inline-table array "
            "(would produce invalid TOML)"
        )
    arr = tomlkit.array()
    arr.append(current)
    arr.append(inline)
    arr.multiline(True)
    sources["torch"] = arr


def _drop_cu130_index(doc: TOMLDocument) -> None:
    """Remove the canonical cu130 entry from ``[[tool.uv.index]]``.

    OutOfOrderTableProxy quirk applies here too: re-fetch ``uv`` via
    :func:`_tool_uv` immediately before ``del uv["index"]`` instead of
    holding a stale reference, so a freshly-constructed proxy carries
    consistent internal table positions.
    """
    indices = _indices(doc)
    if indices is None:
        return
    # Iterate backwards and pop in-place so user trivia (comments,
    # blank lines, custom formatting between tables) in the kept
    # entries is preserved. Rebuilding the AoT from scratch would
    # strip that metadata.
    for i in range(len(indices) - 1, -1, -1):
        entry = indices[i]
        if isinstance(entry, Table | InlineTable | dict) and _index_match(entry) == (
            "canonical"
        ):
            indices.pop(i)
    if len(indices) == 0:
        uv = _tool_uv(doc)
        if uv is not None:
            del uv["index"]


def _drop_torch_source(doc: TOMLDocument) -> None:
    """Remove the canonical cu130 torch entry from ``[tool.uv.sources]``.

    Always runs the end-of-function cleanup that drops empty
    ``sources`` / ``uv`` / ``tool`` tables, even when ``torch`` was
    never present or was a single-entry deletion. This matters after
    :func:`_drop_cu130_index` has already removed the index table —
    both callers run in sequence and the consumer file should end up
    without orphaned empty sections.

    OutOfOrderTableProxy quirk: tomlkit's proxy memoises the position
    of the underlying tables on construction; a ``__delitem__`` on it
    invalidates those positions and a second ``__delitem__`` against
    the SAME proxy can raise ``IndexError``. We work around it by
    re-fetching the proxy from its parent before each cascade-stage
    deletion (``del uv["sources"]`` → re-fetch → ``del tool["uv"]``).
    """
    uv = _tool_uv(doc)
    if uv is None:
        return
    sources = uv.get("sources")

    # Process the torch entry only when sources is present and
    # shaped as a table. The early-return anti-pattern is avoided:
    # even when sources is absent, the cleanup cascade below must
    # run so that an empty ``[tool.uv]`` left behind by
    # :func:`_drop_cu130_index` gets dropped.
    if isinstance(sources, (Table, OutOfOrderTableProxy)):
        torch_entry = sources.get("torch")
        if torch_entry is not None:
            if isinstance(torch_entry, InlineTable | dict) and not isinstance(
                torch_entry, list
            ):
                if _source_match(torch_entry) == "canonical":
                    del sources["torch"]
            elif isinstance(torch_entry, list):
                # Array-of-inline-tables form. Iterate backwards and
                # pop in-place to preserve inline comments and
                # formatting on non-canonical entries; rebuilding
                # the array from a kept list would strip that trivia.
                for i in range(len(torch_entry) - 1, -1, -1):
                    e = torch_entry[i]
                    if isinstance(e, InlineTable | dict) and (
                        _source_match(e) == "canonical"
                    ):
                        torch_entry.pop(i)
                if len(torch_entry) == 0:
                    del sources["torch"]

        if not sources:
            # Re-fetch ``uv`` before the cascade-stage delete —
            # ``_drop_cu130_index`` may have already mutated this same
            # proxy via ``del uv["index"]``. See the OutOfOrderTableProxy
            # note in this function's docstring.
            uv = _tool_uv(doc)
            if uv is not None:
                del uv["sources"]

    # Cleanup cascades: always try to drop empty parent tables so a
    # full uninstall (index + torch) leaves no orphaned sections.
    # Re-fetch ``uv`` again so the emptiness check runs on a fresh
    # proxy with consistent internal state.
    uv = _tool_uv(doc)
    if uv is not None and not uv:
        tool = doc.get("tool")
        if isinstance(tool, (Table, OutOfOrderTableProxy)):
            del tool["uv"]
            if not tool:
                del doc["tool"]
