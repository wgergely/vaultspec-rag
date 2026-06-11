"""End-to-end integration tests for the document-preprocessing hook (#185).

Real GPU + real Qdrant + a real subprocess preprocessor. A binary ``.pdf``
(outside ``SUPPORTED_EXTENSIONS``) is extracted by a project-supplied command
rule, indexed first-class, and found by hybrid search with its deep-link anchor;
the scoped/incremental path routes a changed binary through the preprocessor;
and a failing preprocessor surfaces a skip count rather than a silent gap.
"""

from __future__ import annotations

import shlex
import sys
import textwrap
from typing import TYPE_CHECKING, TypedDict

import pytest

from ...progress import NullProgressReporter

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from ...embeddings import EmbeddingModel
    from ...indexer import CodebaseIndexer
    from ...store import VaultStore
    from ..conftest import RagComponentsWithManifest

pytestmark = [pytest.mark.integration]

# Emits a two-unit PreprocOutput with recognisable text + page anchors.
_PDF_EXTRACTOR = """
    import json, sys
    src = sys.argv[1]
    print(json.dumps({
        "schema_version": 1,
        "preprocessor_id": "fake-pdf",
        "preprocessor_version": "1.0",
        "source_path": src,
        "units": [
            {"text": "Quarterly revenue projections and margin analysis.",
             "anchor": src + "#page=1",
             "locator": {"kind": "page", "value": 1}},
            {"text": "Appendix: regional sales breakdown by territory.",
             "anchor": src + "#page=2",
             "locator": {"kind": "page", "value": 2}},
        ],
    }))
"""

# Always exits non-zero -> a skip under on_error=skip.
_FAILING_EXTRACTOR = "import sys\nsys.exit(2)\n"


class _PreprocProject(TypedDict):
    code_indexer: CodebaseIndexer
    store: VaultStore
    model: EmbeddingModel
    root: Path


def _command(script: Path) -> str:
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} {{path}}"


def _write_config(root: Path, rules: str) -> None:
    (root / ".vaultragpreprocess.toml").write_text(rules, encoding="utf-8")


@pytest.fixture
def preproc_project(
    rag_components: RagComponentsWithManifest,
    tmp_path: Path,
) -> Generator[_PreprocProject]:
    """A temp project with a command preprocess rule and a binary .pdf source."""
    from ... import CodebaseIndexer, VaultStore

    model = rag_components["model"]

    script = tmp_path / "pdf_extractor.py"
    script.write_text(textwrap.dedent(_PDF_EXTRACTOR), encoding="utf-8")
    _write_config(
        tmp_path,
        f"[[rule]]\npattern = \"*.pdf\"\ncommand = '''{_command(script)}'''\n"
        'on_error = "skip"\n',
    )
    (tmp_path / "report.pdf").write_bytes(b"\x00\x01\x02 binary pdf bytes")

    store = VaultStore(tmp_path)
    code_indexer = CodebaseIndexer(tmp_path, model, store)
    yield _PreprocProject(
        code_indexer=code_indexer, store=store, model=model, root=tmp_path
    )
    store.close()


class TestPreprocessEndToEnd:
    @pytest.mark.timeout(600)
    def test_binary_pdf_is_extracted_indexed_and_searchable(
        self, preproc_project: _PreprocProject
    ) -> None:
        from ... import VaultSearcher

        result = preproc_project["code_indexer"].full_index(
            reporter=NullProgressReporter()
        )
        assert result.preprocess_skipped == 0
        assert preproc_project["store"].count_code() >= 2  # two units

        searcher = VaultSearcher(
            preproc_project["root"],
            preproc_project["model"],
            preproc_project["store"],
        )
        results = searcher.search_codebase("quarterly revenue margin", top_k=5)
        assert results
        top = next((r for r in results if r.preprocessor_id == "fake-pdf"), None)
        assert top is not None, "preprocessed unit not found in search results"
        assert top.anchor is not None and "#page=" in top.anchor
        assert top.locator is not None and top.locator.startswith("page ")
        assert top.source_path == "report.pdf"

    @pytest.mark.timeout(600)
    def test_incremental_routes_changed_binary_through_preprocessor(
        self, preproc_project: _PreprocProject
    ) -> None:
        indexer = preproc_project["code_indexer"]
        store = preproc_project["store"]
        indexer.full_index(reporter=NullProgressReporter())
        before = store.count_code()

        new_pdf = preproc_project["root"] / "appendix.pdf"
        new_pdf.write_bytes(b"\x00\x01 another binary")
        # The scoped path is exactly what the watcher invokes on a change.
        result = indexer.incremental_index(
            reporter=NullProgressReporter(), changed_paths=[new_pdf]
        )
        assert result.added > 0
        assert store.count_code() > before

    @pytest.mark.timeout(600)
    def test_failing_preprocessor_surfaces_skip_count(
        self, rag_components: RagComponentsWithManifest, tmp_path: Path
    ) -> None:
        from ... import CodebaseIndexer, VaultStore

        model = rag_components["model"]
        script = tmp_path / "boom.py"
        script.write_text(_FAILING_EXTRACTOR, encoding="utf-8")
        _write_config(
            tmp_path,
            f"[[rule]]\npattern = \"*.pdf\"\ncommand = '''{_command(script)}'''\n"
            'on_error = "skip"\n',
        )
        (tmp_path / "broken.pdf").write_bytes(b"\x00\x01 binary")

        store = VaultStore(tmp_path)
        try:
            indexer = CodebaseIndexer(tmp_path, model, store)
            result = indexer.full_index(reporter=NullProgressReporter())
            assert result.preprocess_skipped == 1
            assert any("broken.pdf" in f for f in result.preprocess_failures)
        finally:
            store.close()

    @pytest.mark.timeout(600)
    def test_incremental_surfaces_skip_count(
        self, rag_components: RagComponentsWithManifest, tmp_path: Path
    ) -> None:
        # Regression for review VIS-001: the scoped/incremental path (used by
        # the watcher) must surface preprocess skip counts, not just full index.
        from ... import CodebaseIndexer, VaultStore

        model = rag_components["model"]
        script = tmp_path / "boom.py"
        script.write_text(_FAILING_EXTRACTOR, encoding="utf-8")
        _write_config(
            tmp_path,
            f"[[rule]]\npattern = \"*.pdf\"\ncommand = '''{_command(script)}'''\n"
            'on_error = "skip"\n',
        )
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"\x00\x01 binary")

        store = VaultStore(tmp_path)
        try:
            indexer = CodebaseIndexer(tmp_path, model, store)
            result = indexer.incremental_index(
                reporter=NullProgressReporter(), changed_paths=[broken]
            )
            assert result.preprocess_skipped == 1
            assert any("broken.pdf" in f for f in result.preprocess_failures)
        finally:
            store.close()
