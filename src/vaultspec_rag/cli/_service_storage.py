"""CLI commands for the ``server storage`` group: survey RAG index storage.

Thin adapter over the service-domain ``storage_ops`` functions. The
read-only ``survey`` connects to the managed Qdrant server, classifies
every per-root namespace (live / orphaned / unknown) via the persisted
prefix-to-root manifest, and renders a bounded, filterable view biased
toward actionable (orphaned, unknown) state. Destructive verbs
(prune/delete) are intentionally separate so this safe read path carries
no confirmation machinery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from ._app import server_storage_app
from ._render import _emit_json, _emit_json_error_and_exit

if TYPE_CHECKING:
    from ..storage_survey import NamespaceSurvey

_SURVEY_CMD = "server.storage.survey"


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def _resolve_server_url(json_mode: bool) -> str:
    """Return the managed Qdrant URL, or exit if server mode is off.

    Exits 2 in local-only mode (nothing to reconcile). The URL is the
    in-daemon ``qdrant_url`` when present, else the managed loopback port.
    """
    from ..config import get_config

    cfg = get_config()
    if not cfg.effective_server_mode():
        message = (
            "Storage survey requires server mode. Local-only stores have a "
            "single namespace and nothing to reconcile."
        )
        if json_mode:
            _emit_json_error_and_exit(_SURVEY_CMD, "server_mode_required", message, 2)
        typer.echo(message)
        raise typer.Exit(2)
    return str(getattr(cfg, "qdrant_url", "") or f"http://127.0.0.1:{cfg.qdrant_port}")


def _gather_or_exit(url: str, json_mode: bool) -> list[NamespaceSurvey]:
    """Survey the server, exiting 3 when it is unreachable."""
    from qdrant_client import QdrantClient

    from ..storage_ops import gather_survey, server_storage_collections_dir

    client = QdrantClient(url=url)
    try:
        return gather_survey(client, server_storage_collections_dir())
    except (OSError, RuntimeError) as exc:
        message = (
            f"Could not reach the managed Qdrant server at {url}. Start the "
            "service with `vaultspec-rag server start`."
        )
        if json_mode:
            _emit_json_error_and_exit(_SURVEY_CMD, "service_not_running", message, 3)
        typer.echo(message)
        raise typer.Exit(3) from exc
    finally:
        client.close()


def _emit_survey_json(surveys: list[NamespaceSurvey]) -> None:
    _emit_json(
        True,
        _SURVEY_CMD,
        data={
            "namespaces": [
                {
                    "prefix": s.prefix,
                    "root": s.root,
                    "status": s.status,
                    "collections": s.collections,
                    "points": s.points,
                    "footprint_bytes": s.footprint_bytes,
                }
                for s in surveys
            ],
            "total": len(surveys),
        },
    )


def _print_survey(surveys: list[NamespaceSurvey]) -> None:
    if not surveys:
        typer.echo("No matching namespaces.")
        return
    counts = {
        status: sum(1 for s in surveys if s.status == status)
        for status in ("orphaned", "unknown", "live")
    }
    total = _human_size(sum(s.footprint_bytes for s in surveys))
    typer.echo(
        f"{len(surveys)} namespaces  (orphaned={counts['orphaned']} "
        f"unknown={counts['unknown']} live={counts['live']})  {total} on disk"
    )
    for s in surveys:
        root = s.root if s.root is not None else "(unattributable)"
        typer.echo(
            f"  {s.status:<8} {s.prefix}  {s.points:>8} pts  "
            f"{_human_size(s.footprint_bytes):>9}  {root}"
        )


@server_storage_app.command(
    "survey",
    help="List stored RAG namespaces classified as live, orphaned, or unknown.",
)
def storage_survey(
    json_mode: bool = typer.Option(
        False, "--json", help="Emit JSON for scripts instead of human text."
    ),
    orphaned_only: bool = typer.Option(
        False, "--orphaned", help="Show only orphaned namespaces (prune candidates)."
    ),
    unknown_only: bool = typer.Option(
        False, "--unknown", help="Show only unattributable (unknown) namespaces."
    ),
) -> None:
    """Survey the managed server's per-root index namespaces."""
    url = _resolve_server_url(json_mode)
    surveys = _gather_or_exit(url, json_mode)
    if orphaned_only:
        surveys = [s for s in surveys if s.status == "orphaned"]
    if unknown_only:
        surveys = [s for s in surveys if s.status == "unknown"]
    if json_mode:
        _emit_survey_json(surveys)
    else:
        _print_survey(surveys)
