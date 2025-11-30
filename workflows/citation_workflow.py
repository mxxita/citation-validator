"""
Citation Validation Workflow: Orchestrates the complete validation pipeline.
"""

import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime
import json

from config import Config
from schemas import FinalReport, ReportSummary, ValidationResult
from agents.parser import ParserAgent
from agents import retriever
from agents import validator_agent


class CitationValidationWorkflow:
    """Main workflow for validating citations in academic papers."""

    def __init__(self):
        """Initialize the workflow with all agents."""
        print("Initializing Citation Validation Workflow...")
        self.parser = ParserAgent()

    async def run(self, pdf_path: str | Path, output_path: Optional[str | Path] = None) -> FinalReport:
        """
        Run the complete citation validation workflow.

        Args:
            pdf_path: Path to the PDF file to validate
            output_path: Optional path to save the report JSON

        Returns:
            FinalReport with complete validation results
        """
        pdf_path = Path(pdf_path)
        start_time = datetime.now()

        print("\n" + "=" * 70)
        print(f"CITATION VALIDATION WORKFLOW")
        print(f"Paper: {pdf_path.name}")
        print("=" * 70)

        # Step 1: Parse PDF
        print("\n[1/4] Parsing PDF and extracting citations...")
        citations, references, metadata = self.parser.run(pdf_path)

        # Step 2: Match citations to references
        print("\n[2/4] Matching citations to references...")
        # Create mapping from citation ID to reference
        citation_ref_map = {}
        for citation in citations:
            for ref in references:
                if citation.id == ref.id:
                    citation_ref_map[citation.id] = ref
                    break

        # Filter to only citations we have references for
        valid_citations = [c for c in citations if c.id in citation_ref_map]
        matched_references = [citation_ref_map[c.id] for c in valid_citations]

        print(f"  Matched {len(valid_citations)}/{len(citations)} citations to references")

        # Step 3: Retrieve sources
        print("\n[3/4] Retrieving sources...")
        sources = await retriever.retrieve_all_async(matched_references)

        # Step 4: Validate citations
        print("\n[4/4] Validating citations...")
        validation_results = validator_agent.validate_all(valid_citations, sources)

        # Step 5: Generate summary
        print("\n[5/5] Generating report summary...")
        summary = self._generate_summary(validation_results, len(citations), len(valid_citations))

        # Create final report
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        report = FinalReport(
            paper_title=metadata.title or pdf_path.stem,
            paper_authors=metadata.authors or [],
            total_citations=len(citations),
            citations_checked=len(valid_citations),
            results=validation_results,
            summary=summary,
            metadata={
                "validation_date": datetime.now().isoformat(),
                "validator_version": "1.0.0",
                "processing_time_seconds": processing_time,
                "pdf_path": str(pdf_path),
                "config": {
                    "similarity_threshold": Config.SIMILARITY_THRESHOLD,
                    "max_citations_checked": Config.MAX_CITATIONS_TO_CHECK
                }
            }
        )

        # Save report if output path provided
        if output_path:
            self._save_report(report, output_path)

        print("\n" + "=" * 70)
        print("VALIDATION COMPLETE")
        print("=" * 70)
        self._print_summary(report)

        return report

    def run_sync(self, pdf_path: str | Path, output_path: Optional[str | Path] = None) -> FinalReport:
        """
        Synchronous version of run().

        Args:
            pdf_path: Path to PDF
            output_path: Optional output path

        Returns:
            FinalReport
        """
        return asyncio.run(self.run(pdf_path, output_path))

    def _generate_summary(
        self,
        results: list[ValidationResult],
        total_citations: int,
        citations_checked: int
    ) -> ReportSummary:
        """Generate summary from validation results."""
        # Count statuses
        supported = sum(1 for r in results if r.status == "supported")
        misrepresented = sum(1 for r in results if r.status == "misrepresented")
        not_found = sum(1 for r in results if r.status == "not_found")
        unverifiable = sum(1 for r in results if r.status == "unverifiable")

        # Generate fulfilled list
        fulfilled = []
        if supported > 0:
            fulfilled.append(f"{supported}/{citations_checked} citations are well-supported")
        if supported / citations_checked >= 0.8:
            fulfilled.append("High proportion of citations verified successfully")

        # Generate missing/problematic list
        missing = []
        if misrepresented > 0:
            misrep_citations = [r.citation_id for r in results if r.status == "misrepresented"]
            missing.append(f"{misrepresented} citations misrepresent their sources: {', '.join(misrep_citations[:5])}")
        if not_found > 0:
            missing.append(f"{not_found} sources could not be located")
        if unverifiable > 0:
            missing.append(f"{unverifiable} citations could not be verified")

        # Calculate desk rejection risk
        risk_score = 1  # Start at 1 (no issues)

        if misrepresented > 0:
            risk_score += min(2, misrepresented // 2)  # +1 for every 2 misrepresentations

        if not_found > citations_checked * 0.3:
            risk_score += 1  # Many sources not found

        if supported / citations_checked < 0.5:
            risk_score += 1  # Less than half supported

        risk_score = min(5, risk_score)  # Cap at 5

        # Generate recommendations
        recommendations = []
        if misrepresented > 0:
            recommendations.append("Review and correct citations marked as 'misrepresented'")
            for result in results:
                if result.status == "misrepresented" and result.suggestion:
                    recommendations.append(f"{result.citation_id}: {result.suggestion}")

        if not_found > 0:
            recommendations.append("Verify reference information for sources that could not be located")

        if supported / citations_checked < 0.7:
            recommendations.append("Consider providing more complete reference information (DOIs, URLs)")

        return ReportSummary(
            fulfilled=fulfilled,
            missing=missing,
            desk_rejection_risk=risk_score,
            recommendations=recommendations[:10],  # Limit to 10
            total_citations=total_citations,
            citations_checked=citations_checked,
            supported_count=supported,
            misrepresented_count=misrepresented,
            not_found_count=not_found,
            unverifiable_count=unverifiable
        )

    def _save_report(self, report: FinalReport, output_path: str | Path):
        """Save report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report.model_dump(), f, indent=2)

        print(f"\n📄 Report saved to: {output_path}")

    def _print_summary(self, report: FinalReport):
        """Print summary to console."""
        summary = report.summary

        print(f"\nPaper: {report.paper_title}")
        print(f"Authors: {', '.join(report.paper_authors[:3])}")
        print(f"Citations: {report.citations_checked}/{report.total_citations} validated")
        print(f"\nResults:")
        print(f"  ✓ Supported: {summary.supported_count}")
        print(f"  ✗ Misrepresented: {summary.misrepresented_count}")
        print(f"  ? Unverifiable: {summary.unverifiable_count}")
        print(f"  ✗ Not Found: {summary.not_found_count}")

        print(f"\nDesk Rejection Risk: {summary.desk_rejection_risk}/5")
        if summary.desk_rejection_risk == 1:
            print("  (Minimal risk)")
        elif summary.desk_rejection_risk == 2:
            print("  (Low risk)")
        elif summary.desk_rejection_risk == 3:
            print("  (Moderate risk)")
        elif summary.desk_rejection_risk == 4:
            print("  (High risk)")
        else:
            print("  (Very high risk)")

        if summary.recommendations:
            print("\nTop Recommendations:")
            for i, rec in enumerate(summary.recommendations[:5], 1):
                print(f"  {i}. {rec}")


# Convenience function
def validate_paper(pdf_path: str | Path, output_path: Optional[str | Path] = None) -> FinalReport:
    """
    Validate a paper's citations (convenience function).

    Args:
        pdf_path: Path to PDF file
        output_path: Optional path to save report

    Returns:
        FinalReport
    """
    workflow = CitationValidationWorkflow()
    return workflow.run_sync(pdf_path, output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python citation_workflow.py <pdf_file> [output_json]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    report = validate_paper(pdf_path, output_path)
