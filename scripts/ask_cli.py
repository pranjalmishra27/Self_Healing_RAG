#!/usr/bin/env python3
"""
Interactive CLI for the Self-Healing RAG Pipeline.
Run: python scripts/ask_cli.py
Or:  python scripts/ask_cli.py "What is the warranty period?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.graph.workflow import RAGPipeline

console = Console()


def display_result(state: dict) -> None:
    console.print()

    # Final answer
    decision = state.get("critic_decision", "?")
    decision_color = "green" if decision == "APPROVE" else "yellow"
    console.print(
        Panel(
            state.get("final_response", "No response."),
            title=f"[bold]Final Answer[/bold]  [dim](critic: [{decision_color}]{decision}[/{decision_color}])[/dim]",
            border_style="cyan",
        )
    )

    # Sources
    sources = state.get("source_metadata", [])
    if sources:
        tbl = Table(title="Retrieved Sources", show_lines=True)
        tbl.add_column("File", style="cyan")
        tbl.add_column("Page", style="dim")
        tbl.add_column("Score", style="green")
        tbl.add_column("Chunk ID", style="dim")
        for s in sources:
            tbl.add_row(
                s["filename"],
                str(s.get("page") or "—"),
                f"{s['score']:.4f}" if s.get("score") is not None else "—",
                s["chunk_id"],
            )
        console.print(tbl)

    # Retry history
    history = state.get("retry_history", [])
    if len(history) > 1:
        console.print()
        console.print("[bold yellow]Retry Trace:[/bold yellow]")
        for r in history:
            console.print(
                f"  Attempt {r['retry_number']}: [{r['critic_decision']}] "
                f"query='{r['rewritten_question'][:60]}' "
                f"— {r['critic_reason'][:80]}"
            )

    # Rewritten query
    rq = state.get("rewritten_question", "")
    oq = state.get("original_question", "")
    if rq and rq != oq:
        console.print(f"\n[dim]Rewritten query: {rq}[/dim]")

    console.print(
        f"\n[dim]Retries used: {state.get('retry_count', 0)} / "
        f"{state.get('max_retries', 3)}[/dim]"
    )


def main() -> None:
    pipeline = RAGPipeline()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        console.print(f"\n[bold]Question:[/bold] {question}\n")
        state = pipeline.run(question)
        display_result(state)
        return

    # Interactive loop
    console.print(
        Panel(
            "[bold cyan]Self-Healing RAG Pipeline[/bold cyan]\n"
            "Type your question and press Enter. Type 'quit' to exit.",
            border_style="blue",
        )
    )
    while True:
        try:
            question = console.input("\n[bold]You:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        with console.status("[cyan]Running pipeline…[/cyan]", spinner="dots"):
            state = pipeline.run(question)

        display_result(state)


if __name__ == "__main__":
    main()
