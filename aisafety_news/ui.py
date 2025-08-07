"""Friendly CLI interface for AI Safety Newsletter Agent."""

import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

try:
    from rich import box
    from rich.align import Align
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.spinner import Spinner
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

@dataclass
class StageResult:
    """Result from a pipeline stage."""
    input_count: int
    output_count: int
    duration: float
    details: str | None = None

class FriendlyUI:
    """Epic cyberpunk CLI interface that replaces verbose JSON logging."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = time.time()
        self.stage_results: dict[str, StageResult] = {}

        # Real-time metrics tracking
        self.metrics = {
            "articles_processed": 0,
            "api_calls_made": 0,
            "processing_speed": 0.0,
            "current_stage": "INITIALIZING",
            "threat_level": "GREEN",
            "neural_activity": 0.0
        }

        if RICH_AVAILABLE:
            self.console = Console()
            self.use_rich = True
        else:
            self.use_rich = False

    def update_metrics(self, **kwargs):
        """Update real-time metrics."""
        self.metrics.update(kwargs)

    def _create_live_dashboard(self):
        """Create a live updating dashboard."""
        if not self.use_rich:
            return None

        # Create layout for dashboard
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="body"),
            Layout(name="footer", size=4)
        )

        # Header with metrics
        metrics_table = Table.grid()
        metrics_table.add_column(style="bold bright_cyan", width=20)
        metrics_table.add_column(style="bold bright_green", width=15)
        metrics_table.add_column(style="bold bright_yellow", width=20)
        metrics_table.add_column(style="bold bright_magenta", width=15)

        metrics_table.add_row(
            "◆ ARTICLES PROCESSED:",
            f"{self.metrics['articles_processed']}",
            "◆ API CALLS MADE:",
            f"{self.metrics['api_calls_made']}"
        )
        metrics_table.add_row(
            "◆ PROCESSING SPEED:",
            f"{self.metrics['processing_speed']:.1f}/s",
            "◆ THREAT LEVEL:",
            f"[{'bright_red' if self.metrics['threat_level'] == 'RED' else 'bright_yellow' if self.metrics['threat_level'] == 'YELLOW' else 'bright_green'}]{self.metrics['threat_level']}[/]"
        )

        header_panel = Panel(
            metrics_table,
            title="[bold bright_cyan]◢ NEURAL METRICS ◣",
            border_style="bright_cyan",
            box=box.DOUBLE
        )

        # Neural activity visualization
        activity_bars = []
        for i in range(10):
            height = random.randint(1, 8) if self.metrics['neural_activity'] > 0.5 else random.randint(1, 3)
            bar_chars = "▁▂▃▄▅▆▇█"
            activity_bars.append(bar_chars[min(height-1, 7)])

        neural_display = Text("NEURAL ACTIVITY: ")
        neural_display.append("".join(activity_bars), style="bold bright_green")
        neural_display.append(f" [{self.metrics['neural_activity']:.0%}]", style="dim bright_blue")

        footer_panel = Panel(
            Align.center(neural_display),
            style="bright_magenta",
            box=box.ROUNDED
        )

        layout["header"].update(header_panel)
        layout["footer"].update(footer_panel)

        return layout

    def show_live_stats(self, stage_name: str):
        """Show live updating statistics during processing."""
        if not self.use_rich:
            return

        # Update current stage
        self.metrics["current_stage"] = stage_name.upper()
        self.metrics["neural_activity"] = random.uniform(0.3, 0.9)

        # Create status display
        status_text = Text()
        status_text.append("▸ CURRENT OPERATION: ", style="bold bright_cyan")
        status_text.append(self.metrics["current_stage"], style="bold bright_yellow blink")

        # Add some cyber-style indicators
        indicators = ["◉", "◎", "○"] * 3
        status_text.append(" " + "".join(random.choices(indicators, k=5)), style="dim bright_green")

        self.console.print(status_text)

    def show_banner(self):
        """Display epic cyberpunk banner with AI Safety branding."""
        if self.use_rich:
            import time

            from rich.align import Align
            from rich.panel import Panel
            from rich.text import Text

            # Create the AISTR logo using Rich Panel for perfect centering
            logo_text = Text.from_markup("""[bold bright_cyan]    ▄▄▄       ██▓  ██████ ▄▄▄█████▓ ██▀███   
   ▒████▄    ▓██▒▒██    ▒ ▓  ██▒ ▓▒▓██ ▒ ██▒ 
   ▒██  ▀█▄  ▒██▒░ ▓██▄   ▒ ▓██░ ▒░▓██ ░▄█ ▒ 
   ░██▄▄▄▄██ ░██░  ▒   ██▒░ ▓██▓ ░ ▒██▀▀█▄   
    ▓█   ▓██▒░██░▒██████▒▒  ▒██▒ ░ ░██▓ ▒██▒ 
    ▒▒   ▓▒█░░▓  ▒ ▒▓▒ ▒ ░  ▒ ░░   ░ ▒▓ ░▒▓░ 
     ▒   ▒▒ ░ ▒ ░░ ░▒  ░ ░    ░      ░▒ ░ ▒░ 
     ░   ▒    ▒ ░░  ░  ░    ░        ░░   ░  
         ░  ░ ░        ░              ░      [/bold bright_cyan]""")

            # Create the main banner panel
            logo_panel = Panel(
                Align.center(logo_text),
                style="bright_blue",
                border_style="bright_blue",
                box=box.DOUBLE,
                padding=(1, 2)
            )

            # Create the subtitle panel
            subtitle = Text()
            subtitle.append("🛡️  AI SAFETY NEWSLETTER AGENT  🛡️", style="bold bright_green")
            subtitle.append("\n")
            subtitle.append("◢◣ PROFESSIONAL NEWS CURATION SYSTEM ◤◥", style="bright_magenta")
            subtitle.append("\n")
            subtitle.append("⚡ Powered by Exa Search & OpenAI ⚡", style="dim bright_yellow")

            subtitle_panel = Panel(
                Align.center(subtitle),
                style="bright_blue",
                border_style="bright_blue",
                box=box.DOUBLE
            )

            # Add the epic banner - properly centered with Rich panels
            self.console.print()
            self.console.print(Align.center(logo_panel))
            self.console.print(Align.center(subtitle_panel))
            self.console.print()

            # Professional initialization
            loading_text = Text()
            loading_text.append("Initializing AI Safety Newsletter System...", style="bold bright_green")
            self.console.print(Align.center(loading_text))
            self.console.print()

            # Professional system status
            self.info("System components loaded successfully")
            self.info("AI safety protocols active")
            self.info("News curation system ready")

            # Brief pause
            time.sleep(0.3)
        else:
            print(r"")
            print(r"    _____/\\\\\\_____/\\\\\\\\\_____/\\\\\\\\\\\____/\\\\\\\\\\\\\\\____/\\\\\\_____")
            print(r"   ___/\\\\\\\\\\\\\__\/////\\\///____/\\\/////////\\\_\///////\\\/////___/\\\///////\\\___")
            print(r"    __/\\\/////////\\\_____\/\\\______\//\\\______\///________\/\\\_______\/\\\_____\/\\\___")
            print(r"     _\/\\\_______\/\\\_____\/\\\_______\////\\\_______________\/\\\_______\/\\\\\\\\\\\/____")
            print(r"      _\/\\\\\\\\\\\\\\\_____\/\\\__________\////\\\____________\/\\\_______\/\\\//////\\\____")
            print(r"       _\/\\\/////////\\\_____\/\\\_____________\////\\\_________\/\\\_______\/\\\____\//\\\___")
            print(r"        _\/\\\_______\/\\\_____\/\\\______/\\\______\//\\\________\/\\\_______\/\\\_____\//\\\__")
            print(r"         _\/\\\_______\/\\\__/\\\\\\\\\\\_\///\\\\\\\\\\\/_________\/\\\_______\/\\\______\//\\\\_")
            print(r"          _\///________\///__\///////////____\///////////___________\///________\///________\///__")
            print(r"")
            print(r"   🛡️  AI Safety Türkiye Newsletter Agent  🛡️")
            print(r"   Powered by Exa Search")
            print()

    def info(self, message: str, emoji: str = "◈"):
        """Show cyberpunk informational message."""
        if self.use_rich:
            styled_message = Text()
            styled_message.append("▸ ", style="bold bright_cyan")
            styled_message.append("INFO", style="bold bright_blue on black")
            styled_message.append(" ◂ ", style="bold bright_cyan")
            styled_message.append(message, style="bright_cyan")
            self.console.print(styled_message)
        else:
            print(f"{emoji} {message}")

    def success(self, message: str, emoji: str = "◆"):
        """Show epic success message."""
        if self.use_rich:
            styled_message = Text()
            styled_message.append("▸ ", style="bold bright_green")
            styled_message.append("SUCCESS", style="bold black on bright_green")
            styled_message.append(" ◂ ", style="bold bright_green")
            styled_message.append(message, style="bright_green")
            styled_message.append(" ✓", style="bold bright_yellow")
            self.console.print(styled_message)
        else:
            print(f"{emoji} {message}")

    def warning(self, message: str, emoji: str = "⟡"):
        """Show cyberpunk warning message."""
        if self.use_rich:
            styled_message = Text()
            styled_message.append("▸ ", style="bold bright_yellow")
            styled_message.append("WARNING", style="bold black on bright_yellow")
            styled_message.append(" ◂ ", style="bold bright_yellow")
            styled_message.append(message, style="bright_yellow")
            styled_message.append(" ⚠", style="bold bright_red blink")
            self.console.print(styled_message)
        else:
            print(f"{emoji} {message}")

    def error(self, message: str, emoji: str = "◢"):
        """Show epic error message with glitch effect."""
        if self.use_rich:
            styled_message = Text()
            styled_message.append("▸ ", style="bold bright_red")
            styled_message.append("CRITICAL ERROR", style="bold bright_white on bright_red blink")
            styled_message.append(" ◂ ", style="bold bright_red")
            styled_message.append(message, style="bright_red")
            styled_message.append(" ✗", style="bold bright_red blink")

            # Add glitch border
            glitch_border = "".join(random.choices("!@#$%^&*", k=50))
            self.console.print(f"[bold bright_red blink]{glitch_border}[/bold bright_red blink]")
            self.console.print(styled_message)
            self.console.print(f"[bold bright_red blink]{glitch_border}[/bold bright_red blink]")
        else:
            print(f"{emoji} {message}")

    def neural_message(self, message: str, neural_level: str = "NORMAL"):
        """Display neural network style messages."""
        if not self.use_rich:
            print(f"[NEURAL] {message}")
            return

        # Neural activity indicators
        neural_chars = ["◉", "◎", "○", "◯", "⦿", "⊙"]
        activity_indicator = "".join(random.choices(neural_chars, k=5))

        color_map = {
            "LOW": "dim bright_blue",
            "NORMAL": "bright_cyan",
            "HIGH": "bright_green",
            "CRITICAL": "bright_red blink"
        }

        color = color_map.get(neural_level, "bright_cyan")

        neural_text = Text()
        neural_text.append("◢ NEURAL ", style="bold bright_magenta")
        neural_text.append(activity_indicator, style=f"bold {color}")
        neural_text.append(" ◣ ", style="bold bright_magenta")
        neural_text.append(message, style=color)

        self.console.print(neural_text)

    def _get_cyber_spinner(self):
        """Get a cyberpunk-style spinner."""
        cyber_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        matrix_frames = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"]
        neural_frames = ["◐", "◓", "◑", "◒"]

        # Randomize spinner style for variety
        frames = random.choice([cyber_frames, matrix_frames, neural_frames])
        return Spinner("dots", text="", style="bold bright_cyan")

    def _create_glitch_text(self, text: str, glitch_chance: float = 0.1):
        """Create glitched text effect."""
        glitch_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?~`"
        result = Text()

        for char in text:
            if char != " " and random.random() < glitch_chance:
                glitch_char = random.choice(glitch_chars)
                result.append(glitch_char, style="bold bright_red blink")
            else:
                result.append(char, style="bold bright_cyan")

        return result

    def _animate_text_typing(self, text: str, delay: float = 0.03):
        """Animate text as if being typed."""
        if not self.use_rich:
            print(text)
            return

        animated_text = Text()
        for char in text:
            animated_text.append(char, style="bold bright_green")
            self.console.print(f"\r{animated_text}", end="")
            time.sleep(delay)
        self.console.print()

    @contextmanager
    def stage(self, name: str, emoji: str = "⚡"):
        """Context manager for epic cyberpunk pipeline stages."""
        stage_start = time.time()

        if self.use_rich:
            # Epic stage header with cyberpunk styling
            header_panel = Panel(
                Align.center(Text(f"{emoji} {name.upper()}", style="bold bright_cyan")),
                style="bright_magenta",
                border_style="bright_cyan",
                box=box.DOUBLE
            )
            self.console.print(header_panel)

            # Custom cyberpunk progress bar
            with Progress(
                TextColumn("[bold bright_cyan]▸"),
                SpinnerColumn("dots12", style="bold bright_green"),
                TextColumn("[progress.description]"),
                BarColumn(style="bright_cyan", complete_style="bright_green", finished_style="bold bright_yellow"),
                TextColumn("[bold bright_yellow]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TextColumn("[bold bright_magenta]◂"),
                console=self.console,
                transient=False,
                expand=True
            ) as progress:
                task = progress.add_task(
                    f"[bold bright_green]EXECUTING: {name.upper()}...",
                    total=None
                )

                try:
                    yield progress, task
                except Exception as e:
                    self.error(f"SYSTEM FAILURE: {str(e)}")
                    raise
        else:
            print(f"{emoji} {name}...")
            try:
                yield None, None
            except Exception as e:
                self.error(f"Failed: {str(e)}")
                raise

        duration = time.time() - stage_start
        if not self.verbose:  # Only show timing in non-verbose mode
            if self.use_rich:
                self.console.print(f"   [dim]Completed in {duration:.1f}s[/dim]")
            else:
                print(f"   Completed in {duration:.1f}s")

    def show_source_results(self, sources: list[dict[str, Any]]):
        """Show results from source fetching."""
        if self.use_rich:
            table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE)
            table.add_column("Source", style="dim")
            table.add_column("Articles", justify="right")
            table.add_column("Status", justify="center")

            total_articles = 0
            for source in sources:
                name = source.get('name', 'Unknown')
                count = source.get('count', 0)
                status = "✓" if count > 0 else "⚠️"
                table.add_row(name, str(count), status)
                total_articles += count

            self.console.print(table)
            self.console.print(f"\n   → [bold]Found {total_articles} articles total[/bold]")
        else:
            total_articles = 0
            for source in sources:
                name = source.get('name', 'Unknown')
                count = source.get('count', 0)
                status = "✓" if count > 0 else "⚠️"
                print(f"   {status} {name:<25} [{count} articles]")
                total_articles += count
            print(f"\n   → Found {total_articles} articles total")

    def update_progress(self, progress, task, current: int, total: int, description: str = ""):
        """Update progress bar."""
        if self.use_rich and progress and task is not None:
            progress.update(task, completed=current, total=total, description=f"   {description}")

    def complete_progress(self, progress, task, result_message: str):
        """Complete progress bar with result."""
        if self.use_rich and progress and task is not None:
            progress.update(task, completed=100, total=100)
            progress.stop()
            self.console.print(f"   → [bold]{result_message}[/bold]")

    def show_model_info(self, model_name: str):
        """Show neural core information in cyberpunk style."""
        if self.use_rich:
            model_text = Text()
            model_text.append("▸ NEURAL CORE: ", style="bold bright_magenta")
            model_text.append(model_name.upper(), style="bold bright_cyan")
            model_text.append(" ONLINE", style="bold bright_green blink")
            model_text.append(" ◂", style="bold bright_magenta")
            self.console.print(model_text)
        else:
            print(f"   🧠 Neural Core: {model_name}")

    def show_final_summary(self,
                          total_sources: int,
                          total_articles: int,
                          filtered_articles: int,
                          final_articles: int,
                          output_file: str,
                          model_used: str):
        """Show final summary with celebration."""
        duration = time.time() - self.start_time

        if self.use_rich:
            # Create summary table
            summary_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
            summary_table.add_column("Metric", style="dim blue")
            summary_table.add_column("Value", style="bold bright_blue")

            summary_table.add_row("📊 Sources scanned", str(total_sources))
            summary_table.add_row("📰 Articles found", str(total_articles))
            summary_table.add_row("🔍 Passed relevance filter", str(filtered_articles))
            summary_table.add_row("⭐ Final selection", str(final_articles))
            summary_table.add_row("🧠 Model used", model_used)
            summary_table.add_row("💾 Output file", output_file)
            summary_table.add_row("⏱️  Total time", f"{duration:.1f}s")

            panel = Panel(
                summary_table,
                title="🎉 [bold cyan]AISTR Newsletter Ready![/bold cyan] 🎉",
                title_align="center",
                box=box.ROUNDED,
                border_style="bright_blue"
            )

            self.console.print()
            self.console.print(panel)

        else:
            print("\n🎉 Newsletter ready!")
            print(f"📊 {final_articles} top AI safety stories curated from {total_sources} major news sources")
            print(f"💾 Output: {output_file}")
            print(f"⏱️  Completed in {duration:.1f} seconds")

    def verbose_log(self, message: str):
        """Log message with cyberpunk styling if in verbose mode."""
        if self.verbose:
            if self.use_rich:
                verbose_text = Text()
                verbose_text.append("   ◦ DEBUG: ", style="dim bright_blue")
                verbose_text.append(message, style="dim bright_cyan")
                self.console.print(verbose_text)
            else:
                print(f"DEBUG: {message}")

    def operation_status(self, operation: str, status: str = "IN_PROGRESS"):
        """Display professional operation status."""
        if not self.use_rich:
            print(f"[{status}] {operation}")
            return

        # Professional operation messages
        operation_messages = {
            "gathering_articles": "◈ Gathering articles from news sources",
            "filtering_relevance": "◈ Filtering articles for AI safety relevance",
            "removing_duplicates": "◈ Removing duplicate articles",
            "scoring_importance": "◈ Scoring articles by importance",
            "generating_newsletter": "◈ Generating newsletter content",
            "finalizing_output": "◈ Finalizing newsletter output"
        }

        message = operation_messages.get(operation, f"◈ {operation.replace('_', ' ').title()}")

        status_colors = {
            "IN_PROGRESS": "bright_yellow",
            "COMPLETE": "bright_green",
            "FAILED": "bright_red",
            "STANDBY": "dim bright_blue"
        }

        color = status_colors.get(status, "bright_cyan")

        status_text = Text()
        status_text.append("▸ ", style="bold bright_cyan")
        status_text.append(message, style=f"bold {color}")

        self.console.print(status_text)

# Global UI instance
_ui_instance: FriendlyUI | None = None

def get_ui(verbose: bool = False) -> FriendlyUI:
    """Get or create global UI instance."""
    global _ui_instance
    if _ui_instance is None:
        _ui_instance = FriendlyUI(verbose=verbose)
    return _ui_instance

def init_ui(verbose: bool = False) -> FriendlyUI:
    """Initialize UI for the session."""
    global _ui_instance
    _ui_instance = FriendlyUI(verbose=verbose)
    return _ui_instance
