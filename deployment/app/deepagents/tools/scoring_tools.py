"""Scoring and Excel reporting tools for Recruitment Deep Agent.

This module provides tools for generating scoring reports, Excel exports,
and candidate rankings.

Following Enterprise Development Standards:
- Software Architect: Modular reporting pipeline
- Security Architect: PII handling in reports
- Data Architect: Structured report formats
- Software Engineer: Type-safe with comprehensive error handling
"""

import csv
import io
import json
import logging
from datetime import datetime
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.deepagents.config.recruitment_config import (
    ScreeningLevel,
    get_recruitment_config,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class ScoringReport(BaseModel):
    """Comprehensive scoring report for all candidates."""

    report_id: str
    jd_id: str
    jd_title: str
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    total_candidates: int = 0
    screened_count: int = 0
    interviewed_count: int = 0
    shortlisted_count: int = 0
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class CandidateRanking(BaseModel):
    """Candidate ranking entry."""

    rank: int
    candidate_id: str
    candidate_name: str
    resume_score: float = 0.0
    interview_score: float = 0.0
    overall_score: float = 0.0
    level: ScreeningLevel
    status: str
    recommendation: str


# =============================================================================
# Storage (shared with other recruitment tools)
# =============================================================================

# Import storage from recruitment tools
from app.deepagents.tools.recruitment_tools import (
    _get_candidates,
    _get_jds,
    _get_screening_results,
)

from app.deepagents.tools.interview_tools import (
    _get_scores as _get_interview_scores,
)

_scoring_reports: dict[str, list[ScoringReport]] = {}


def _get_reports(session_id: str) -> list[ScoringReport]:
    """Get reports for session."""
    if session_id not in _scoring_reports:
        _scoring_reports[session_id] = []
    return _scoring_reports[session_id]


# =============================================================================
# Tool Functions
# =============================================================================


@tool
def generate_scoring_report(
    jd_id: str,
    session_id: str = "default",
) -> str:
    """Generate comprehensive scoring report for a job position.

    Use this tool to create a detailed report of all candidates
    evaluated for a specific job description.

    Args:
        jd_id: Job description identifier.
        session_id: Session identifier.

    Returns:
        Formatted scoring report.
    """
    import uuid

    jds = _get_jds(session_id)
    screening_results = _get_screening_results(session_id)
    interview_scores = _get_interview_scores(session_id)
    config = get_recruitment_config()

    if jd_id not in jds:
        return f"Job description not found: {jd_id}"

    jd = jds[jd_id]

    # Get all screening results for this JD
    jd_screenings = [r for r in screening_results if r.jd_id == jd_id]

    # Combine with interview scores
    candidates_data = []
    for screening in jd_screenings:
        # Find interview score if exists
        interview = next(
            (s for s in interview_scores if s.candidate_id == screening.candidate_id),
            None,
        )

        # Calculate combined score
        resume_weight = getattr(config, "resume_evaluation_weight", 0.4)
        base_interview_weight = getattr(config, "interview_evaluation_weight", 0.6)
        interview_weight = base_interview_weight if interview else 0.0

        if interview:
            combined_score = (
                screening.overall_score * resume_weight +
                interview.percentage_score * interview_weight
            )
        else:
            combined_score = screening.overall_score

        candidate_entry = {
            "candidate_id": screening.candidate_id,
            "candidate_name": screening.candidate_name,
            "level": screening.recommended_level.value,
            "resume_score": screening.overall_score,
            "skill_match": screening.skill_match_score,
            "experience_score": screening.experience_score,
            "interview_score": interview.percentage_score if interview else None,
            "interview_passed": interview.passed if interview else None,
            "combined_score": combined_score,
            "resume_passed": screening.passed,
            "shortlisted": screening.shortlisted,
            "strengths": screening.strengths,
            "gaps": screening.gaps,
            "matched_skills": screening.matched_skills[:5],
            "missing_skills": screening.missing_skills[:5],
            "recommendation": interview.recommendation if interview else screening.recommendation,
            "status": "interviewed" if interview else ("shortlisted" if screening.shortlisted else ("passed" if screening.passed else "rejected")),
        }
        candidates_data.append(candidate_entry)

    # Sort by combined score
    candidates_data.sort(key=lambda x: x["combined_score"], reverse=True)

    # Create report
    report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
    report = ScoringReport(
        report_id=report_id,
        jd_id=jd_id,
        jd_title=jd.title,
        total_candidates=len(candidates_data),
        screened_count=len(jd_screenings),
        interviewed_count=len([c for c in candidates_data if c["interview_score"] is not None]),
        shortlisted_count=len([c for c in candidates_data if c["shortlisted"]]),
        candidates=candidates_data,
    )

    # Store report
    reports = _get_reports(session_id)
    reports.append(report)

    # Format output
    output = f"""## Scoring Report

**Report ID**: {report_id}
**Position**: {jd.title} ({jd_id})
**Generated**: {report.generated_at}

---

### Summary Statistics

| Metric | Count |
|--------|-------|
| Total Candidates | {report.total_candidates} |
| Resume Screened | {report.screened_count} |
| Interviewed | {report.interviewed_count} |
| Shortlisted | {report.shortlisted_count} |

---

### Candidate Rankings

| Rank | Candidate | Level | Resume | Interview | Combined | Status |
|------|-----------|-------|--------|-----------|----------|--------|
"""

    for i, c in enumerate(candidates_data[:15], 1):
        int_score = f"{c['interview_score']:.1f}%" if c["interview_score"] is not None else "N/A"
        output += f"| {i} | {c['candidate_name']} | {c['level']} | {c['resume_score']:.1f}% | {int_score} | {c['combined_score']:.1f}% | {c['status'].upper()} |\n"

    output += f"""
---

### Final Recommendations

"""

    # Top candidates to advance
    advanced = [c for c in candidates_data if c["status"] == "interviewed" and c.get("interview_passed")]
    passed = [c for c in candidates_data if c["shortlisted"] and c["status"] != "interviewed"]

    output += "**Ready to Advance to L2 Interview:**\n"
    if advanced:
        for c in advanced[:5]:
            output += f"- {c['candidate_name']} ({c['combined_score']:.1f}%) - {c['recommendation']}\n"
    else:
        output += "- No candidates have completed interviews yet\n"

    output += "\n**Shortlisted for Technical Interview:**\n"
    if passed:
        for c in passed[:5]:
            output += f"- {c['candidate_name']} ({c['resume_score']:.1f}%)\n"
    else:
        output += "- No additional candidates shortlisted\n"

    output += f"""
---

*Use `export_scoring_excel` to generate Excel report.*
*Use `upload_to_sharepoint` to save report to SharePoint.*
"""

    return output


@tool
def export_scoring_excel(
    jd_id: str,
    session_id: str = "default",
) -> str:
    """Export scoring data to Excel-compatible CSV format.

    Use this tool to generate a CSV file that can be opened in Excel
    with all candidate scoring data.

    Args:
        jd_id: Job description identifier.
        session_id: Session identifier.

    Returns:
        CSV content ready for upload to SharePoint.
    """
    jds = _get_jds(session_id)
    screening_results = _get_screening_results(session_id)
    interview_scores = _get_interview_scores(session_id)

    if jd_id not in jds:
        return f"Job description not found: {jd_id}"

    jd = jds[jd_id]

    # Get screening results for this JD
    jd_screenings = [r for r in screening_results if r.jd_id == jd_id]

    if not jd_screenings:
        return f"No screening results found for {jd_id}"

    # Create CSV output
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        "Rank",
        "Candidate ID",
        "Candidate Name",
        "Level",
        "Resume Score (%)",
        "Skill Match (%)",
        "Experience Score (%)",
        "Education Score (%)",
        "Certification Score (%)",
        "Interview Score (%)",
        "Interview Passed",
        "Combined Score (%)",
        "Resume Passed",
        "Shortlisted",
        "Matched Skills",
        "Missing Skills",
        "Strengths",
        "Gaps",
        "Recommendation",
        "Status",
        "Screened At",
    ])

    # Collect and sort data
    rows = []
    for screening in jd_screenings:
        interview = next(
            (s for s in interview_scores if s.candidate_id == screening.candidate_id),
            None,
        )

        if interview:
            combined = screening.overall_score * 0.4 + interview.percentage_score * 0.6
        else:
            combined = screening.overall_score

        rows.append({
            "screening": screening,
            "interview": interview,
            "combined": combined,
        })

    rows.sort(key=lambda x: x["combined"], reverse=True)

    # Write data rows
    for rank, row in enumerate(rows, 1):
        s = row["screening"]
        i = row["interview"]

        writer.writerow([
            rank,
            s.candidate_id,
            s.candidate_name,
            s.recommended_level.value,
            f"{s.overall_score:.1f}",
            f"{s.skill_match_score:.1f}",
            f"{s.experience_score:.1f}",
            f"{s.education_score:.1f}",
            f"{s.certification_score:.1f}",
            f"{i.percentage_score:.1f}" if i else "N/A",
            "Yes" if i and i.passed else "No" if i else "N/A",
            f"{row['combined']:.1f}",
            "Yes" if s.passed else "No",
            "Yes" if s.shortlisted else "No",
            "; ".join(s.matched_skills[:5]),
            "; ".join(s.missing_skills[:5]),
            "; ".join(s.strengths),
            "; ".join(s.gaps),
            i.recommendation if i else s.recommendation,
            "Interviewed" if i else ("Shortlisted" if s.shortlisted else ("Passed" if s.passed else "Rejected")),
            s.screened_at,
        ])

    csv_content = output.getvalue()

    # Generate filename
    filename = f"Scoring_Report_{jd.title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    result = f"""## Excel Report Generated

**Position**: {jd.title}
**Candidates**: {len(rows)}
**Filename**: {filename}

---

### CSV Content Preview (first 5 rows)

```
{chr(10).join(csv_content.split(chr(10))[:6])}
```

---

**Full CSV Content for Upload:**

```csv
{csv_content}
```

---

*Copy the CSV content above or use `upload_to_sharepoint` with folder_type="scoring" to save to SharePoint.*
"""

    return result


@tool
def get_ranking_summary(
    jd_id: str,
    top_n: int = 10,
    session_id: str = "default",
) -> str:
    """Get quick ranking summary of top candidates.

    Args:
        jd_id: Job description identifier.
        top_n: Number of top candidates to show.
        session_id: Session identifier.

    Returns:
        Ranked list of top candidates.
    """
    jds = _get_jds(session_id)
    screening_results = _get_screening_results(session_id)
    interview_scores = _get_interview_scores(session_id)

    if jd_id not in jds:
        return f"Job description not found: {jd_id}"

    jd = jds[jd_id]
    jd_screenings = [r for r in screening_results if r.jd_id == jd_id]

    if not jd_screenings:
        return f"No candidates screened for {jd_id}"

    # Calculate combined scores and rank
    rankings = []
    for s in jd_screenings:
        interview = next(
            (i for i in interview_scores if i.candidate_id == s.candidate_id),
            None,
        )

        if interview:
            combined = s.overall_score * 0.4 + interview.percentage_score * 0.6
            status = "🎯 Interviewed"
        elif s.shortlisted:
            combined = s.overall_score
            status = "✅ Shortlisted"
        elif s.passed:
            combined = s.overall_score
            status = "⚠️ Passed"
        else:
            combined = s.overall_score
            status = "❌ Rejected"

        rankings.append(CandidateRanking(
            rank=0,
            candidate_id=s.candidate_id,
            candidate_name=s.candidate_name,
            resume_score=s.overall_score,
            interview_score=interview.percentage_score if interview else 0,
            overall_score=combined,
            level=s.recommended_level,
            status=status,
            recommendation=interview.recommendation if interview else s.recommendation,
        ))

    # Sort and assign ranks
    rankings.sort(key=lambda x: x.overall_score, reverse=True)
    for i, r in enumerate(rankings, 1):
        r.rank = i

    output = f"""## Candidate Rankings: {jd.title}

**Total Evaluated**: {len(rankings)}
**Showing Top**: {min(top_n, len(rankings))}

---

"""

    for r in rankings[:top_n]:
        int_display = f" | Interview: {r.interview_score:.1f}%" if r.interview_score > 0 else ""
        output += f"""### #{r.rank} {r.candidate_name} {r.status}
- **Score**: {r.overall_score:.1f}% (Resume: {r.resume_score:.1f}%{int_display})
- **Level**: {r.level.value}
- **Recommendation**: {r.recommendation[:50]}...

"""

    return output


@tool
def get_passing_score_thresholds(session_id: str = "default") -> str:
    """Get configured passing score thresholds.

    Args:
        session_id: Session identifier.

    Returns:
        Current scoring configuration.
    """
    config = get_recruitment_config()

    output = f"""## Scoring Configuration

### Passing Scores by Level

| Level | Passing Score |
|-------|---------------|
| L1 (Junior) | {config.scoring.l1_passing_score}% |
| L2 (Mid-level) | {config.scoring.l2_passing_score}% |
| L3 (Senior) | {config.scoring.l3_passing_score}% |

### Resume Screening Threshold
- **Minimum to Shortlist**: {config.scoring.resume_screening_threshold}%

### Score Weights

| Criteria | Weight |
|----------|--------|
| Technical Skills | {config.scoring.technical_weight * 100:.0f}% |
| Experience | {config.scoring.experience_weight * 100:.0f}% |
| Education | {config.scoring.education_weight * 100:.0f}% |
| Soft Skills | {config.scoring.soft_skills_weight * 100:.0f}% |
| Certifications | {config.scoring.certification_weight * 100:.0f}% |

### Interview Questions by Level

| Level | Questions | Time Limit |
|-------|-----------|------------|
| L1 | {config.interview.l1_question_count} | {config.interview.l1_time_limit} min |
| L2 | {config.interview.l2_question_count} | {config.interview.l2_time_limit} min |
| L3 | {config.interview.l3_question_count} | {config.interview.l3_time_limit} min |

---

*Use configuration file or `update_recruitment_config` to modify these values.*
"""

    return output


@tool
def generate_shortlist_report(
    jd_id: str,
    session_id: str = "default",
) -> str:
    """Generate final shortlist report with candidates ready for L2 interview.

    Args:
        jd_id: Job description identifier.
        session_id: Session identifier.

    Returns:
        Formatted shortlist report.
    """
    jds = _get_jds(session_id)
    screening_results = _get_screening_results(session_id)
    interview_scores = _get_interview_scores(session_id)
    config = get_recruitment_config()

    if jd_id not in jds:
        return f"Job description not found: {jd_id}"

    jd = jds[jd_id]
    jd_screenings = [r for r in screening_results if r.jd_id == jd_id]

    # Categorize candidates
    ready_for_l2 = []  # Passed interview
    pending_interview = []  # Shortlisted, not interviewed
    not_qualified = []  # Failed screening

    for s in jd_screenings:
        interview = next(
            (i for i in interview_scores if i.candidate_id == s.candidate_id),
            None,
        )

        if interview and interview.passed:
            combined = s.overall_score * 0.4 + interview.percentage_score * 0.6
            ready_for_l2.append({
                "name": s.candidate_name,
                "id": s.candidate_id,
                "level": s.recommended_level.value,
                "resume_score": s.overall_score,
                "interview_score": interview.percentage_score,
                "combined": combined,
                "recommendation": interview.recommendation,
            })
        elif s.shortlisted:
            pending_interview.append({
                "name": s.candidate_name,
                "id": s.candidate_id,
                "level": s.recommended_level.value,
                "score": s.overall_score,
            })
        else:
            not_qualified.append({
                "name": s.candidate_name,
                "id": s.candidate_id,
                "score": s.overall_score,
                "reason": s.gaps[0] if s.gaps else "Below threshold",
            })

    # Sort by combined score
    ready_for_l2.sort(key=lambda x: x["combined"], reverse=True)
    pending_interview.sort(key=lambda x: x["score"], reverse=True)

    output = f"""## Final Shortlist Report

**Position**: {jd.title} ({jd_id})
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

### Summary

| Category | Count |
|----------|-------|
| Ready for L2 Interview | {len(ready_for_l2)} |
| Pending Technical Interview | {len(pending_interview)} |
| Not Qualified | {len(not_qualified)} |

---

### ✅ Candidates Ready to Advance to L2 Interview

"""

    if ready_for_l2:
        output += "| Rank | Candidate | Level | Combined Score | Recommendation |\n"
        output += "|------|-----------|-------|----------------|----------------|\n"
        for i, c in enumerate(ready_for_l2, 1):
            output += f"| {i} | {c['name']} | {c['level']} | {c['combined']:.1f}% | Advance to L2 |\n"
    else:
        output += "*No candidates have completed the technical interview successfully.*\n"

    output += "\n### ⏳ Pending Technical Interview (Shortlisted)\n\n"

    if pending_interview:
        output += "| Candidate | Level | Resume Score | Action Required |\n"
        output += "|-----------|-------|--------------|------------------|\n"
        for c in pending_interview[:10]:
            output += f"| {c['name']} | {c['level']} | {c['score']:.1f}% | Schedule interview |\n"
    else:
        output += "*No candidates pending interview.*\n"

    output += "\n### ❌ Not Qualified\n\n"

    if not_qualified:
        output += f"*{len(not_qualified)} candidates did not meet minimum requirements.*\n"
    else:
        output += "*All candidates met minimum requirements.*\n"

    output += f"""
---

### Next Steps

1. **For L2-Ready Candidates**: Schedule L2 interviews with hiring manager
2. **For Pending Interview**: Send interview question sets and schedule assessments
3. **For Not Qualified**: Send rejection notifications

---

*This report can be uploaded to SharePoint using `upload_to_sharepoint` with folder_type="shortlist".*
"""

    return output


__all__ = [
    "ScoringReport",
    "CandidateRanking",
    "generate_scoring_report",
    "export_scoring_excel",
    "get_ranking_summary",
    "get_passing_score_thresholds",
    "generate_shortlist_report",
]
