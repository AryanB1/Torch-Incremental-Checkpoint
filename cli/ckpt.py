"""
ckpt — CLI tool for the Incremental Checkpoint Engine.

Commands
--------
    ckpt list   <save_dir>            — list all checkpoints with metadata
    ckpt gc     <save_dir>            — run garbage collection (dry-run by default)
    ckpt info   <save_dir> <step>     — detailed info for a specific checkpoint
    ckpt stats  <save_dir>            — storage statistics
    ckpt blobs  <save_dir>            — list all content-addressed blobs
    ckpt verify <save_dir>            — verify blob integrity

Install: pip install -e .[cli]
Run:     python -m cli.ckpt --help
         or after install: ckpt --help
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich import box

# Ensure the package root is importable when running from the repo directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from checkpoint_engine.manifest import Manifest
from checkpoint_engine.store import ContentAddressedStore
from checkpoint_engine.lifecycle import LifecycleManager

console = Console()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_manifest(save_dir: str) -> Manifest:
    p = Path(save_dir)
    if not p.exists():
        console.print(f"[red]Directory not found:[/] {save_dir}")
        raise SystemExit(1)
    return Manifest(p)


def _load_store(save_dir: str) -> ContentAddressedStore:
    return ContentAddressedStore(Path(save_dir) / "blobs")


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Incremental Checkpoint Engine — management CLI."""


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@cli.command("list")
@click.argument("save_dir")
@click.option("--metric", default=None, help="Sort by this metric key.")
def cmd_list(save_dir: str, metric: str | None) -> None:
    """List all saved checkpoint versions."""
    manifest = _load_manifest(save_dir)
    versions = manifest.versions

    if not versions:
        console.print("[yellow]No checkpoints found.[/]")
        return

    table = Table(
        title=f"Checkpoints in [bold]{save_dir}[/]",
        box=box.ROUNDED,
        show_lines=False,
    )
    table.add_column("Step", style="cyan", justify="right")
    table.add_column("Timestamp", style="green")
    table.add_column("Dirty %", justify="right")
    table.add_column("Storage", justify="right")
    table.add_column("Full?", justify="center")
    table.add_column("Metrics")

    if metric:
        versions = sorted(versions, key=lambda v: v.metrics.get(metric, float("inf")))

    import datetime
    for v in versions:
        ts = datetime.datetime.fromtimestamp(v.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        metrics_str = ", ".join(f"{k}={val:.4g}" for k, val in v.metrics.items())
        table.add_row(
            str(v.step),
            ts,
            f"{v.dirty_ratio:.1%}",
            _fmt_bytes(v.storage_bytes),
            "yes" if v.is_full else "",
            metrics_str or "—",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@cli.command("info")
@click.argument("save_dir")
@click.argument("step", type=int)
@click.option("--show-all", is_flag=True, default=False,
              help="Show all parameter hashes instead of first 20.")
def cmd_info(save_dir: str, step: int, show_all: bool) -> None:
    """Show detailed metadata for a specific checkpoint step."""
    manifest = _load_manifest(save_dir)
    version = manifest.get_version(step)
    if version is None:
        available = manifest.steps()
        console.print(
            f"[red]Step {step} not found.[/] Available: {available}"
        )
        raise SystemExit(1)

    import datetime
    ts = datetime.datetime.fromtimestamp(version.timestamp).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    console.print(f"[bold]Step:[/]        {version.step}")
    console.print(f"[bold]Timestamp:[/]   {ts}")
    console.print(f"[bold]Full save:[/]   {version.is_full}")
    console.print(f"[bold]Dirty ratio:[/] {version.dirty_ratio:.2%}")
    console.print(f"[bold]Storage:[/]     {_fmt_bytes(version.storage_bytes)}")
    console.print(f"[bold]Num params:[/]  {len(version.param_hashes)}")

    if version.metrics:
        console.print("\n[bold]Metrics:[/]")
        for k, v in version.metrics.items():
            console.print(f"  {k}: {v:.6g}")

    console.print(f"\n[bold]Parameter hashes[/] ({len(version.param_hashes)} total):")
    store = _load_store(save_dir)
    items = list(version.param_hashes.items())
    display_items = items if show_all else items[:20]
    for name, hexdigest in display_items:
        exists = "[green]ok[/]" if store.exists(hexdigest) else "[red]MISSING[/]"
        console.print(f"  {name[:60]:<60} {hexdigest[:12]}...  {exists}")
    if not show_all and len(items) > 20:
        console.print(f"  ... and {len(items) - 20} more (use --show-all to see all)")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@cli.command("stats")
@click.argument("save_dir")
def cmd_stats(save_dir: str) -> None:
    """Print storage statistics for the checkpoint directory."""
    manifest = _load_manifest(save_dir)
    store = _load_store(save_dir)
    lifecycle = LifecycleManager(manifest=manifest, store=store)

    all_hashes = store.all_hashes()
    total_bytes = store.total_bytes()
    gc_summary = lifecycle.summary()

    console.print(f"[bold]Save directory:[/]  {save_dir}")
    console.print(f"[bold]Checkpoint versions:[/] {len(manifest)}")
    console.print(f"[bold]Total blobs:[/]     {len(all_hashes)}")
    console.print(f"[bold]Total storage:[/]   {_fmt_bytes(total_bytes)}")
    console.print(f"[bold]Versions to keep:[/]  {gc_summary['versions_to_keep']}")
    console.print(f"[bold]Versions to delete:[/] {gc_summary['versions_to_delete']}")
    console.print(f"[bold]Orphaned blobs:[/]  {gc_summary['orphaned_blobs']}")


# ---------------------------------------------------------------------------
# gc
# ---------------------------------------------------------------------------

@cli.command("gc")
@click.argument("save_dir")
@click.option("--dry-run/--no-dry-run", default=True,
              help="Preview changes without modifying anything (default: dry-run).")
@click.option("--keep-last", default=5, show_default=True)
@click.option("--keep-best", default=3, show_default=True)
@click.option("--metric", default="loss", show_default=True)
def cmd_gc(
    save_dir: str,
    dry_run: bool,
    keep_last: int,
    keep_best: int,
    metric: str,
) -> None:
    """Run garbage collection on old checkpoint versions."""
    manifest = _load_manifest(save_dir)
    store = _load_store(save_dir)
    lifecycle = LifecycleManager(
        manifest=manifest,
        store=store,
        keep_last_n=keep_last,
        keep_best_n=keep_best,
        metric_key=metric,
    )

    if dry_run:
        console.print("[yellow]Dry-run mode — no files will be deleted.[/]")

    result = lifecycle.run_gc(dry_run=dry_run)

    prefix = "Would delete" if dry_run else "Deleted"
    console.print(f"{prefix} [cyan]{len(result['deleted_steps'])}[/] versions: "
                  f"{result['deleted_steps']}")
    console.print(f"{prefix} [cyan]{len(result['deleted_blobs'])}[/] orphaned blobs")
    console.print(f"Freed [cyan]{_fmt_bytes(result['freed_bytes'])}[/]")

    if dry_run:
        console.print("\nRe-run with [bold]--no-dry-run[/] to apply changes.")


# ---------------------------------------------------------------------------
# blobs
# ---------------------------------------------------------------------------

@cli.command("blobs")
@click.argument("save_dir")
@click.option("--limit", default=50, show_default=True, help="Max blobs to show.")
def cmd_blobs(save_dir: str, limit: int) -> None:
    """List content-addressed blobs in the store."""
    store = _load_store(save_dir)
    all_hashes = store.all_hashes()

    console.print(f"[bold]Total blobs:[/] {len(all_hashes)}")
    table = Table(box=box.SIMPLE)
    table.add_column("Hash (SHA-256)", style="cyan")
    table.add_column("Size", justify="right")

    for h in all_hashes[:limit]:
        try:
            size = store.blob_size(h)
        except FileNotFoundError:
            size = 0
        table.add_row(h, _fmt_bytes(size))

    if len(all_hashes) > limit:
        console.print(f"[dim](showing {limit} of {len(all_hashes)})[/]")

    console.print(table)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

@cli.command("verify")
@click.argument("save_dir")
@click.option("--step", type=int, default=None,
              help="Verify blobs for a specific step only.")
def cmd_verify(save_dir: str, step: int | None) -> None:
    """Verify integrity of stored blobs by checking SHA-256 hashes."""
    manifest = _load_manifest(save_dir)
    store = _load_store(save_dir)

    if step is not None:
        version = manifest.get_version(step)
        if version is None:
            console.print(f"[red]Step {step} not found.[/]")
            raise SystemExit(1)
        hashes_to_check = set(version.param_hashes.values())
    else:
        hashes_to_check = set(store.all_hashes())

    ok_count = 0
    missing_count = 0
    corrupt_count = 0

    with console.status("[bold green]Verifying blobs...") as status:
        for hexdigest in sorted(hashes_to_check):
            blob_path = store._blob_path(hexdigest)
            if not blob_path.exists():
                console.print(f"  [red]MISSING[/]  {hexdigest}")
                missing_count += 1
                continue
            data = blob_path.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != hexdigest:
                console.print(f"  [red]CORRUPT[/]  {hexdigest} (got {actual[:16]}...)")
                corrupt_count += 1
            else:
                ok_count += 1

    total = ok_count + missing_count + corrupt_count
    console.print(f"\n[bold]Verified {total} blobs:[/]")
    console.print(f"  [green]OK:[/]      {ok_count}")
    if missing_count:
        console.print(f"  [red]Missing:[/] {missing_count}")
    if corrupt_count:
        console.print(f"  [red]Corrupt:[/] {corrupt_count}")
    if missing_count == 0 and corrupt_count == 0:
        console.print("[green]All blobs verified successfully.[/]")
    else:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
