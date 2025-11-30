import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
import json

from config import Config
from workflows.citation_workflow import CitationValidationWorkflow
from agents.parser import ParserAgent
from tools.embedding_tools import compute_similarity

app = typer.Typer(help="Academic Paper Citation Validator")
console = Console()

@app.command()
def validate(
    pdf_path: Path = typer.Argument(..., help="Path to the PDF file to validate"),
    output: Path = typer.Option(None, "--output", "-o", help="Output JSON file path"),
    max_citations: int = typer.Option(None, "--max-citations", help="Maximum citations to check"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):

    if not pdf_path.exists():
        console.print(f"[red]Error: PDF file not found: {pdf_path}[/red]")
        raise typer.Exit(code=1)
    if max_citations:
        Config.MAX_CITATIONS_TO_CHECK = max_citations
    if output is None:
        output = Config.OUTPUT_DIR / f"{pdf_path.stem}_validation_report.json"
    try:
        Config.validate()
    except ValueError as e:
        console.print(f"[red]Configuration Error:[/red]\n{e}")
        console.print("\n[yellow]Please set up your .env file with required API keys.[/yellow]")
        raise typer.Exit(code=1)
    try:
        workflow = CitationValidationWorkflow()
        report = workflow.run_sync(pdf_path, output)
        _display_report(report)

    except Exception as e:
        console.print(f"[red]Error during validation:[/red] {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def preview(
    pdf_path: Path = typer.Argument(..., help="Path to PDF file"),
    max_items: int = typer.Option(10, "--max", "-m", help="Max items to show"),
):
    """
    Preview citations and references extracted from a PDF (without validation).
    """
    if not pdf_path.exists():
        console.print(f"[red]Error: PDF file not found: {pdf_path}[/red]")
        raise typer.Exit(code=1)

    try:
        parser = ParserAgent()
        parser.preview_extraction(pdf_path, max_items)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def check_single(
    claim: str = typer.Argument(..., help="The claim text to check"),
    source: str = typer.Argument(..., help="The source text to check against"),
):
    """
    Quick check of a single claim against source text using similarity.
    """
    console.print("\n[bold]Checking claim against source...[/bold]\n")
    console.print(f"[cyan]Claim:[/cyan] {claim}\n")
    console.print(f"[cyan]Source:[/cyan] {source}\n")

    try:
        similarity = compute_similarity(claim, source)

        console.print(f"[bold]Similarity Score:[/bold] {similarity:.3f}")

        if similarity >= Config.SIMILARITY_THRESHOLD:
            console.print("[green]✓ Claims appear semantically similar[/green]")
        else:
            console.print("[yellow]⚠ Claims may not be well-aligned[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def view_report(
    report_path: Path = typer.Argument(..., help="Path to validation report JSON"),
):
    if not report_path.exists():
        console.print(f"[red]Error: Report file not found: {report_path}[/red]")
        raise typer.Exit(code=1)

    try:
        with open(report_path) as f:
            report_data = json.load(f)
        from schemas import FinalReport
        report = FinalReport(**report_data)

        _display_report(report)

    except Exception as e:
        console.print(f"[red]Error loading report:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def config_info():
    """
    Display current configuration.
    """
    console.print("\n[bold]Citation Validator Configuration[/bold]\n")

    table = Table(show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("LLM Provider", Config.DEFAULT_LLM_PROVIDER)
    table.add_row("OpenAI Model", Config.OPENAI_MODEL)
    table.add_row("Anthropic Model", Config.ANTHROPIC_MODEL)
    table.add_row("Embedding Model", Config.EMBEDDING_MODEL)
    table.add_row("Similarity Threshold", str(Config.SIMILARITY_THRESHOLD))
    table.add_row("Max Citations", str(Config.MAX_CITATIONS_TO_CHECK))
    table.add_row("Output Directory", str(Config.OUTPUT_DIR))
    table.add_row("Browser Headless", str(Config.BROWSER_HEADLESS))

    openai_key = "✓ Set" if Config.OPENAI_API_KEY else "✗ Not set"
    anthropic_key = "✓ Set" if Config.ANTHROPIC_API_KEY else "✗ Not set"

    table.add_row("OpenAI API Key", openai_key)
    table.add_row("Anthropic API Key", anthropic_key)

    console.print(table)


def _display_report(report):
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]CITATION VALIDATION REPORT[/bold cyan]")
    console.print("=" * 70 + "\n")

    console.print(f"[bold]Paper:[/bold] {report.paper_title}")
    console.print(f"[bold]Authors:[/bold] {', '.join(report.paper_authors[:3])}")
    console.print(f"[bold]Citations:[/bold] {report.citations_checked}/{report.total_citations} validated\n")
    summary = report.summary

    table = Table(show_header=True, title="Validation Results")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    total = report.citations_checked
    table.add_row(
        "Supported",
        str(summary.supported_count),
        f"{summary.supported_count/total*100:.1f}%"
    )
    table.add_row(
        "Misrepresented",
        str(summary.misrepresented_count),
        f"{summary.misrepresented_count/total*100:.1f}%"
    )
    table.add_row(
        "Unverifiable",
        str(summary.unverifiable_count),
        f"{summary.unverifiable_count/total*100:.1f}%"
    )
    table.add_row(
        "Not Found",
        str(summary.not_found_count),
        f"{summary.not_found_count/total*100:.1f}%"
    )

    console.print(table)
    console.print(f"\n[bold]Desk Rejection Risk:[/bold] {summary.desk_rejection_risk}/5")
    risk_color = "green" if summary.desk_rejection_risk <= 2 else "yellow" if summary.desk_rejection_risk == 3 else "red"
    console.print(f"[{risk_color}]{'█' * summary.desk_rejection_risk}{'░' * (5-summary.desk_rejection_risk)}[/{risk_color}]")

    if summary.recommendations:
        console.print("\n[bold]Top Recommendations:[/bold]")
        for i, rec in enumerate(summary.recommendations[:5], 1):
            console.print(f"  {i}. {rec}")
    problematic = [r for r in report.results if r.status in ["misrepresented", "not_found"]]
    if problematic:
        console.print(f"\n[bold red]⚠ {len(problematic)} Problematic Citations:[/bold red]")
        for result in problematic[:10]:
            console.print(f"\n[yellow]{result.citation_id}[/yellow] - {result.status}")
            console.print(f"  Claim: {result.claim[:100]}...")
            if result.suggestion:
                console.print(f"  [italic]Suggestion: {result.suggestion}[/italic]")

    console.print("\n" + "=" * 70)


if __name__ == "__main__":
    app()
