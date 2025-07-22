#!/usr/bin/env python3
"""Source health check utility."""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .ingest.sources import get_source_registry, get_source_health_monitor
from .config import get_settings, get_model_config
from .logging import get_logger

logger = get_logger(__name__)
console = Console()


async def check_all_sources(verbose: bool = False) -> dict:
    """Check health of all configured sources."""
    registry = get_source_registry()
    health_monitor = get_source_health_monitor()
    
    console.print("\n[bold cyan]Checking all sources...[/bold cyan]\n")
    
    for source in registry.sources:
        adapter = registry.create_adapter(source)
        source_name = getattr(source, 'name', getattr(source, 'domain', 'unknown'))
        
        console.print(f"Checking {source_name}... ", end="")

        try:
            async with adapter:
                start_time = datetime.now(timezone.utc)
                articles = await adapter.fetch_articles(
                    start_date=datetime.now(timezone.utc) - timedelta(days=1),
                    limit=5
                )
                response_time = (datetime.now(timezone.utc) - start_time).total_seconds()

                if articles:
                    console.print(f"[green]✓[/green] ({len(articles)} articles, {response_time:.2f}s)")
                    health_monitor.record_success(source_name, response_time, len(articles))
                else:
                    console.print("[yellow]⚠️  No articles[/yellow]")
                    health_monitor.record_failure(source_name, "No articles found")

        except Exception as e:
            console.print(f"[red]✗ {str(e)}[/red]")
            health_monitor.record_failure(source_name, str(e))
    
    # Get and return health report
    return health_monitor.get_health_report()


def display_health_report(report: dict):
    """Display health report in a formatted table."""
    console.print("\n")
    
    # Summary panel
    summary = report['summary']
    summary_text = (
        f"[green]Healthy: {summary['healthy']}[/green] | "
        f"[yellow]Degraded: {summary['degraded']}[/yellow] | "
        f"[red]Unhealthy: {summary['unhealthy']}[/red] | "
        f"Total: {summary['total']}"
    )
    
    console.print(Panel(
        summary_text,
        title="[bold]Source Health Summary[/bold]",
        border_style="cyan"
    ))
    
    # Detailed table
    if report['sources']:
        table = Table(
            title="\nDetailed Source Status",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Source Name", style="dim", overflow="fold")
        table.add_column("Status", justify="center")
        table.add_column("Response Time", justify="right")
        table.add_column("Entries", justify="right")
        table.add_column("Failures", justify="right")
        table.add_column("Last Error", overflow="fold")
        
        for url, status in report['sources'].items():
            # Status color
            status_color = {
                'healthy': 'green',
                'degraded': 'yellow',
                'unhealthy': 'red',
                'unknown': 'dim'
            }.get(status['status'], 'white')
            
            status_text = f"[{status_color}]{status['status'].upper()}[/{status_color}]"
            
            # Response time
            response_time = status.get('response_time', 0)
            response_text = f"{response_time:.2f}s" if response_time > 0 else "-"
            
            # Entry count
            entry_count = status.get('entry_count', 0)
            entry_text = str(entry_count) if entry_count > 0 else "-"
            
            # Failures
            failures = status.get('consecutive_failures', 0)
            failure_text = str(failures) if failures > 0 else "-"
            if failures > 0:
                failure_text = f"[red]{failure_text}[/red]"
            
            # Last error
            last_error = status.get('last_error', '-')
            if len(last_error) > 50:
                last_error = last_error[:47] + "..."
            
            table.add_row(
                url,
                status_text,
                response_text,
                entry_text,
                failure_text,
                last_error
            )
        
        console.print(table)


@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Show verbose output')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def main(verbose: bool, output_json: bool):
    """Check health status of all configured sources."""
    try:
        # Run async check
        report = asyncio.run(check_all_sources(verbose))
        
        if output_json:
            import json
            print(json.dumps(report, indent=2, default=str))
        else:
            display_health_report(report)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()