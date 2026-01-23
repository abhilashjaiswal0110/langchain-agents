"""Resume screening and recruitment tools for Recruitment Deep Agent.

This module provides tools for parsing resumes, extracting skills,
matching candidates to job descriptions, and L1/L2/L3 screening.

Following Enterprise Development Standards:
- Software Architect: Modular resume processing pipeline
- Security Architect: Secure document handling, PII awareness
- Data Architect: Structured candidate data models
- Software Engineer: Type-safe with comprehensive error handling
"""

import io
import logging
import re
import uuid
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


class ExtractedSkill(BaseModel):
    """Extracted skill from resume."""

    name: str = Field(description="Skill name")
    category: str = Field(description="Skill category (technical, soft, tool, etc.)")
    proficiency: str = Field(default="intermediate", description="Proficiency level")
    years: int = Field(default=0, description="Years of experience with skill")
    confidence: float = Field(default=0.8, description="Extraction confidence")


class WorkExperience(BaseModel):
    """Work experience entry."""

    company: str = Field(description="Company name")
    title: str = Field(description="Job title")
    start_date: str = Field(default="", description="Start date")
    end_date: str = Field(default="present", description="End date")
    duration_months: int = Field(default=0, description="Duration in months")
    description: str = Field(default="", description="Role description")
    skills_used: list[str] = Field(default_factory=list, description="Skills used")


class Education(BaseModel):
    """Education entry."""

    institution: str = Field(description="Institution name")
    degree: str = Field(description="Degree type")
    field: str = Field(default="", description="Field of study")
    year: int = Field(default=0, description="Graduation year")
    gpa: float | None = Field(default=None, description="GPA if available")


class Certification(BaseModel):
    """Certification entry."""

    name: str = Field(description="Certification name")
    issuer: str = Field(default="", description="Issuing organization")
    year: int = Field(default=0, description="Year obtained")
    expiry: str | None = Field(default=None, description="Expiry date if applicable")


class CandidateProfile(BaseModel):
    """Complete candidate profile extracted from resume."""

    candidate_id: str = Field(description="Unique candidate identifier")
    name: str = Field(description="Candidate name")
    email: str = Field(default="", description="Email address")
    phone: str = Field(default="", description="Phone number")
    location: str = Field(default="", description="Location")
    summary: str = Field(default="", description="Professional summary")
    total_experience_years: float = Field(default=0, description="Total years of experience")
    skills: list[ExtractedSkill] = Field(default_factory=list, description="Extracted skills")
    work_experience: list[WorkExperience] = Field(default_factory=list, description="Work history")
    education: list[Education] = Field(default_factory=list, description="Education")
    certifications: list[Certification] = Field(default_factory=list, description="Certifications")
    languages: list[str] = Field(default_factory=list, description="Languages spoken")
    resume_source: str = Field(default="", description="Source file name")
    extracted_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    screening_level: ScreeningLevel | None = Field(default=None, description="Recommended level")


class JobDescription(BaseModel):
    """Parsed job description."""

    jd_id: str = Field(description="Unique JD identifier")
    title: str = Field(description="Job title")
    department: str = Field(default="", description="Department")
    location: str = Field(default="", description="Location")
    experience_required: str = Field(default="", description="Experience requirement")
    min_experience_years: int = Field(default=0, description="Minimum years")
    max_experience_years: int = Field(default=99, description="Maximum years")
    required_skills: list[str] = Field(default_factory=list, description="Required skills")
    preferred_skills: list[str] = Field(default_factory=list, description="Preferred skills")
    responsibilities: list[str] = Field(default_factory=list, description="Key responsibilities")
    qualifications: list[str] = Field(default_factory=list, description="Required qualifications")
    education_required: str = Field(default="", description="Education requirement")
    certifications_preferred: list[str] = Field(default_factory=list, description="Preferred certs")
    source_file: str = Field(default="", description="Source file name")


class ScreeningResult(BaseModel):
    """Resume screening result."""

    candidate_id: str
    candidate_name: str
    jd_id: str
    overall_score: float = Field(ge=0, le=100)
    skill_match_score: float = Field(ge=0, le=100)
    experience_score: float = Field(ge=0, le=100)
    education_score: float = Field(ge=0, le=100)
    certification_score: float = Field(ge=0, le=100)
    recommended_level: ScreeningLevel
    passed: bool
    shortlisted: bool
    matched_skills: list[str]
    missing_skills: list[str]
    strengths: list[str]
    gaps: list[str]
    recommendation: str
    screened_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Storage
# =============================================================================

# Session-isolated storage for candidates and JDs
_candidates: dict[str, dict[str, CandidateProfile]] = {}
_job_descriptions: dict[str, dict[str, JobDescription]] = {}
_screening_results: dict[str, list[ScreeningResult]] = {}


def _get_candidates(session_id: str) -> dict[str, CandidateProfile]:
    """Get candidates for session."""
    if session_id not in _candidates:
        _candidates[session_id] = {}
    return _candidates[session_id]


def _get_jds(session_id: str) -> dict[str, JobDescription]:
    """Get job descriptions for session."""
    if session_id not in _job_descriptions:
        _job_descriptions[session_id] = {}
    return _job_descriptions[session_id]


def _get_screening_results(session_id: str) -> list[ScreeningResult]:
    """Get screening results for session."""
    if session_id not in _screening_results:
        _screening_results[session_id] = []
    return _screening_results[session_id]


# =============================================================================
# Resume Parsing Utilities
# =============================================================================


def _extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF content."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except ImportError:
        logger.warning("PyPDF2 not installed, PDF parsing limited")
        return "[PDF content - install PyPDF2 for extraction]"
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""


def _extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX content."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except ImportError:
        logger.warning("python-docx not installed, DOCX parsing limited")
        return "[DOCX content - install python-docx for extraction]"
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""


def _extract_email(text: str) -> str:
    """Extract email from text."""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group(0) if match else ""


def _extract_phone(text: str) -> str:
    """Extract phone number from text."""
    patterns = [
        r"\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
        r"\+?[0-9]{10,14}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""


def _extract_name_from_filename(filename: str) -> str:
    """Extract candidate name from filename."""
    # Remove extension and common suffixes
    name = re.sub(r"\.(pdf|docx?|txt|rtf)$", "", filename, flags=re.IGNORECASE)
    name = re.sub(r"[_-]?(resume|cv|curriculum[_-]?vitae)$", "", name, flags=re.IGNORECASE)
    # Convert underscores/dashes to spaces
    name = re.sub(r"[_-]+", " ", name)
    return name.strip().title()


def _estimate_experience_years(text: str) -> float:
    """Estimate years of experience from resume text."""
    # Look for explicit mentions
    patterns = [
        r"(\d+)\+?\s*years?\s*(?:of\s+)?experience",
        r"experience\s*(?:of\s+)?(\d+)\+?\s*years?",
        r"(\d+)\+?\s*years?\s*in\s+(?:the\s+)?(?:industry|field|IT)",
    ]

    max_years = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                years = int(match)
                max_years = max(max_years, years)
            except ValueError:
                continue

    return float(max_years)


# Common skill patterns for extraction
TECHNICAL_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "ruby",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "oracle", "redis",
    "aws", "azure", "gcp", "kubernetes", "docker", "terraform", "ansible",
    "react", "angular", "vue", "nodejs", "django", "flask", "spring", "fastapi",
    "machine learning", "deep learning", "nlp", "computer vision", "ai",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "git", "jenkins", "ci/cd", "agile", "scrum", "devops",
    "linux", "windows server", "networking", "security",
    "rest api", "graphql", "microservices", "serverless",
]

SOFT_SKILLS = [
    "leadership", "communication", "teamwork", "problem solving", "analytical",
    "project management", "time management", "mentoring", "collaboration",
    "presentation", "negotiation", "critical thinking", "adaptability",
]


def _extract_skills(text: str) -> list[ExtractedSkill]:
    """Extract skills from resume text."""
    text_lower = text.lower()
    found_skills = []

    # Technical skills
    for skill in TECHNICAL_SKILLS:
        if skill.lower() in text_lower:
            found_skills.append(ExtractedSkill(
                name=skill.title(),
                category="technical",
                proficiency="intermediate",
                confidence=0.8,
            ))

    # Soft skills
    for skill in SOFT_SKILLS:
        if skill.lower() in text_lower:
            found_skills.append(ExtractedSkill(
                name=skill.title(),
                category="soft",
                proficiency="intermediate",
                confidence=0.7,
            ))

    return found_skills


def _determine_screening_level(years: float, skills: list[ExtractedSkill]) -> ScreeningLevel:
    """Determine appropriate screening level."""
    config = get_recruitment_config()

    # Count technical skills
    tech_skills = len([s for s in skills if s.category == "technical"])

    if years >= config.resume_parsing.l3_min_experience or tech_skills >= 15:
        return ScreeningLevel.L3
    elif years >= config.resume_parsing.l2_min_experience or tech_skills >= 8:
        return ScreeningLevel.L2
    else:
        return ScreeningLevel.L1


# =============================================================================
# Tool Functions
# =============================================================================

@tool
def parse_resume(
    content: str,
    filename: str,
    session_id: str = "default",
) -> str:
    """Parse a resume and extract candidate information.

    Use this tool to process resume content and extract structured data
    including skills, experience, education, and certifications.

    Args:
        content: Resume content (text or base64 encoded for binary).
        filename: Original filename for format detection.
        session_id: Session identifier.

    Returns:
        Formatted candidate profile summary.
    """
    # Determine file type and extract text
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in ["pdf", "docx", "doc"]:
        # Content might be base64 encoded or raw bytes description
        if content.startswith("[PDF content") or content.startswith("[DOCX content"):
            text = content  # Already a message about missing parser
        else:
            text = content  # Assume text was already extracted
    else:
        text = content

    # Extract candidate information
    candidate_id = f"CAND-{uuid.uuid4().hex[:8].upper()}"
    name = _extract_name_from_filename(filename)
    email = _extract_email(text)
    phone = _extract_phone(text)
    years = _estimate_experience_years(text)
    skills = _extract_skills(text)
    level = _determine_screening_level(years, skills)

    # Create candidate profile
    profile = CandidateProfile(
        candidate_id=candidate_id,
        name=name,
        email=email,
        phone=phone,
        total_experience_years=years,
        skills=skills,
        screening_level=level,
        resume_source=filename,
        summary=text[:500] + "..." if len(text) > 500 else text,
    )

    # Store candidate
    candidates = _get_candidates(session_id)
    candidates[candidate_id] = profile

    # Format output
    output = f"""## Candidate Profile: {name}

**ID**: {candidate_id}
**Source**: {filename}
**Recommended Level**: {level.value}

### Contact
- Email: {email or 'Not found'}
- Phone: {phone or 'Not found'}

### Experience
- Total Years: {years}

### Skills ({len(skills)} found)
**Technical**: {', '.join([s.name for s in skills if s.category == 'technical'][:10]) or 'None extracted'}
**Soft Skills**: {', '.join([s.name for s in skills if s.category == 'soft'][:5]) or 'None extracted'}

### Summary
{profile.summary[:300]}...

---
*Profile stored. Use screening tools to match against job descriptions.*
"""

    return output


@tool
def parse_job_description(
    content: str,
    title: str,
    session_id: str = "default",
) -> str:
    """Parse a job description and extract requirements.

    Use this tool to process JD content and extract structured requirements
    for candidate matching.

    Args:
        content: Job description content.
        title: Job title.
        session_id: Session identifier.

    Returns:
        Formatted job description summary.
    """
    jd_id = f"JD-{uuid.uuid4().hex[:8].upper()}"
    text_lower = content.lower()

    # Extract required skills
    required_skills = []
    for skill in TECHNICAL_SKILLS:
        if skill.lower() in text_lower:
            required_skills.append(skill.title())

    # Extract experience requirement
    years_patterns = [
        r"(\d+)\+?\s*(?:to\s*(\d+))?\s*years?",
        r"minimum\s*(\d+)\s*years?",
    ]

    min_years = 0
    max_years = 99
    for pattern in years_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            min_years = int(match.group(1))
            if match.lastindex >= 2 and match.group(2):
                max_years = int(match.group(2))
            break

    # Extract education
    education = ""
    edu_patterns = [
        r"(bachelor'?s?|master'?s?|phd|doctorate)\s*(?:degree)?\s*(?:in\s+)?([^,.\n]+)?",
        r"(b\.?s\.?|m\.?s\.?|b\.?e\.?|m\.?e\.?)\s*(?:in\s+)?([^,.\n]+)?",
    ]
    for pattern in edu_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            education = match.group(0).strip()
            break

    # Create JD object
    jd = JobDescription(
        jd_id=jd_id,
        title=title,
        required_skills=required_skills[:20],
        min_experience_years=min_years,
        max_experience_years=max_years,
        education_required=education,
        source_file=title,
    )

    # Store JD
    jds = _get_jds(session_id)
    jds[jd_id] = jd

    # Format output
    output = f"""## Job Description: {title}

**ID**: {jd_id}

### Experience Required
- Minimum: {min_years} years
- Maximum: {max_years if max_years < 99 else 'Not specified'} years

### Required Skills ({len(required_skills)})
{', '.join(required_skills[:15]) or 'None extracted'}

### Education
{education or 'Not specified'}

---
*JD stored. Use screening tools to match candidates.*
"""

    return output


@tool
def screen_candidate(
    candidate_id: str,
    jd_id: str,
    session_id: str = "default",
) -> str:
    """Screen a candidate against a job description.

    Use this tool to evaluate how well a candidate matches a job requirement.

    Args:
        candidate_id: Candidate identifier.
        jd_id: Job description identifier.
        session_id: Session identifier.

    Returns:
        Detailed screening result with scores and recommendation.
    """
    config = get_recruitment_config()
    candidates = _get_candidates(session_id)
    jds = _get_jds(session_id)

    if candidate_id not in candidates:
        return f"Candidate not found: {candidate_id}"

    if jd_id not in jds:
        return f"Job description not found: {jd_id}"

    candidate = candidates[candidate_id]
    jd = jds[jd_id]

    # Calculate skill match score
    candidate_skills = {s.name.lower() for s in candidate.skills}
    required_skills = {s.lower() for s in jd.required_skills}

    matched_skills = candidate_skills.intersection(required_skills)
    missing_skills = required_skills - candidate_skills

    skill_score = (len(matched_skills) / len(required_skills) * 100) if required_skills else 50

    # Calculate experience score
    if jd.min_experience_years <= candidate.total_experience_years <= jd.max_experience_years:
        exp_score = 100
    elif candidate.total_experience_years >= jd.min_experience_years:
        exp_score = 80
    else:
        gap = jd.min_experience_years - candidate.total_experience_years
        exp_score = max(0, 100 - gap * 15)

    # Education and certification scores (simplified)
    edu_score = 70  # Default
    cert_score = 60  # Default

    # Calculate soft skills score based on extracted soft skills
    soft_skills_count = len([s for s in candidate.skills if s.category == "soft"])
    soft_skills_score = min(100, soft_skills_count * 15)  # 15 points per soft skill, max 100

    # Calculate overall score with all configured weights
    overall_score = (
        skill_score * config.scoring.technical_weight +
        exp_score * config.scoring.experience_weight +
        edu_score * config.scoring.education_weight +
        soft_skills_score * config.scoring.soft_skills_weight +
        cert_score * config.scoring.certification_weight
    ) / (
        config.scoring.technical_weight +
        config.scoring.experience_weight +
        config.scoring.education_weight +
        config.scoring.soft_skills_weight +
        config.scoring.certification_weight
    )

    # Determine pass/fail
    level = candidate.screening_level or ScreeningLevel.L1
    passing_score = config.scoring.get_passing_score(level)
    passed = overall_score >= passing_score
    shortlisted = passed and overall_score >= config.scoring.resume_screening_threshold

    # Generate strengths and gaps
    strengths = []
    gaps = []

    if skill_score >= 70:
        strengths.append(f"Strong skill match ({len(matched_skills)} of {len(required_skills)} required skills)")
    else:
        gaps.append(f"Missing key skills: {', '.join(list(missing_skills)[:5])}")

    if exp_score >= 80:
        strengths.append(f"Meets experience requirement ({candidate.total_experience_years} years)")
    else:
        gaps.append(f"Below experience requirement ({candidate.total_experience_years} vs {jd.min_experience_years} required)")

    # Generate recommendation
    if shortlisted:
        recommendation = f"SHORTLIST for {level.value} technical interview"
    elif passed:
        recommendation = f"CONSIDER for {level.value} role, address skill gaps"
    else:
        recommendation = f"NOT RECOMMENDED - score {overall_score:.1f}% below {passing_score}% threshold"

    # Create result
    result = ScreeningResult(
        candidate_id=candidate_id,
        candidate_name=candidate.name,
        jd_id=jd_id,
        overall_score=overall_score,
        skill_match_score=skill_score,
        experience_score=exp_score,
        education_score=edu_score,
        certification_score=cert_score,
        recommended_level=level,
        passed=passed,
        shortlisted=shortlisted,
        matched_skills=list(matched_skills),
        missing_skills=list(missing_skills),
        strengths=strengths,
        gaps=gaps,
        recommendation=recommendation,
    )

    # Store result
    results = _get_screening_results(session_id)
    results.append(result)

    # Format output
    status_icon = "✅" if shortlisted else ("⚠️" if passed else "❌")

    output = f"""## Screening Result {status_icon}

### Candidate: {candidate.name} ({candidate_id})
### Position: {jd.title} ({jd_id})

---

### Scores
| Criteria | Score | Weight |
|----------|-------|--------|
| Technical Skills | {skill_score:.1f}% | {config.scoring.technical_weight * 100:.0f}% |
| Experience | {exp_score:.1f}% | {config.scoring.experience_weight * 100:.0f}% |
| Education | {edu_score:.1f}% | {config.scoring.education_weight * 100:.0f}% |
| Certifications | {cert_score:.1f}% | {config.scoring.certification_weight * 100:.0f}% |
| **Overall** | **{overall_score:.1f}%** | |

### Assessment
- **Recommended Level**: {level.value}
- **Passing Score**: {passing_score}%
- **Status**: {'PASSED' if passed else 'FAILED'}
- **Shortlisted**: {'Yes' if shortlisted else 'No'}

### Strengths
{chr(10).join(['- ' + s for s in strengths]) or '- None identified'}

### Gaps
{chr(10).join(['- ' + g for g in gaps]) or '- None identified'}

### Matched Skills ({len(matched_skills)})
{', '.join(list(matched_skills)[:10]) or 'None'}

### Missing Skills ({len(missing_skills)})
{', '.join(list(missing_skills)[:10]) or 'None'}

---

### Recommendation
**{recommendation}**
"""

    return output


@tool
def batch_screen_resumes(
    jd_id: str,
    session_id: str = "default",
) -> str:
    """Screen all candidates against a job description.

    Use this tool to evaluate all parsed candidates against a specific JD.

    Args:
        jd_id: Job description identifier.
        session_id: Session identifier.

    Returns:
        Summary of all screening results with rankings.
    """
    candidates = _get_candidates(session_id)
    jds = _get_jds(session_id)

    if jd_id not in jds:
        return f"Job description not found: {jd_id}"

    if not candidates:
        return "No candidates found. Parse resumes first."

    jd = jds[jd_id]
    results = []

    for candidate_id in candidates:
        # Screen each candidate
        result_text = screen_candidate.invoke({
            "candidate_id": candidate_id,
            "jd_id": jd_id,
            "session_id": session_id,
        })
        results.append(result_text)

    # Get stored results for ranking
    screening_results = _get_screening_results(session_id)
    jd_results = [r for r in screening_results if r.jd_id == jd_id]

    # Sort by score
    jd_results.sort(key=lambda x: x.overall_score, reverse=True)

    # Generate summary
    shortlisted = [r for r in jd_results if r.shortlisted]
    passed = [r for r in jd_results if r.passed and not r.shortlisted]
    failed = [r for r in jd_results if not r.passed]

    output = f"""## Batch Screening Results

### Position: {jd.title} ({jd_id})
### Total Candidates: {len(jd_results)}

---

### Summary
- ✅ **Shortlisted**: {len(shortlisted)}
- ⚠️ **Passed (Not Shortlisted)**: {len(passed)}
- ❌ **Failed**: {len(failed)}

---

### Ranking (Top Candidates)

| Rank | Candidate | Score | Level | Status |
|------|-----------|-------|-------|--------|
"""

    for i, r in enumerate(jd_results[:10], 1):
        status = "✅ Shortlist" if r.shortlisted else ("⚠️ Pass" if r.passed else "❌ Fail")
        output += f"| {i} | {r.candidate_name} | {r.overall_score:.1f}% | {r.recommended_level.value} | {status} |\n"

    output += f"""
---

### Shortlisted Candidates for Technical Interview

"""

    if shortlisted:
        for r in shortlisted:
            output += f"- **{r.candidate_name}** ({r.candidate_id}) - {r.overall_score:.1f}% - {r.recommended_level.value}\n"
    else:
        output += "*No candidates shortlisted*\n"

    return output


@tool
def get_candidate_profile(
    candidate_id: str,
    session_id: str = "default",
) -> str:
    """Get detailed candidate profile.

    Args:
        candidate_id: Candidate identifier.
        session_id: Session identifier.

    Returns:
        Full candidate profile.
    """
    candidates = _get_candidates(session_id)

    if candidate_id not in candidates:
        return f"Candidate not found: {candidate_id}"

    c = candidates[candidate_id]

    output = f"""## Candidate Profile: {c.name}

**ID**: {c.candidate_id}
**Source**: {c.resume_source}

### Contact
- Email: {c.email or 'Not provided'}
- Phone: {c.phone or 'Not provided'}
- Location: {c.location or 'Not provided'}

### Experience
- **Total Years**: {c.total_experience_years}
- **Recommended Level**: {c.screening_level.value if c.screening_level else 'Not determined'}

### Skills ({len(c.skills)})
"""

    tech_skills = [s for s in c.skills if s.category == "technical"]
    soft_skills = [s for s in c.skills if s.category == "soft"]

    output += f"**Technical ({len(tech_skills)})**: {', '.join([s.name for s in tech_skills])}\n"
    output += f"**Soft ({len(soft_skills)})**: {', '.join([s.name for s in soft_skills])}\n"

    output += f"""
### Summary
{c.summary}

---
*Extracted: {c.extracted_at}*
"""

    return output


@tool
def list_candidates(session_id: str = "default") -> str:
    """List all parsed candidates in the session.

    Args:
        session_id: Session identifier.

    Returns:
        List of all candidates with summary info.
    """
    candidates = _get_candidates(session_id)

    if not candidates:
        return "No candidates found. Parse resumes to add candidates."

    output = f"## Candidates ({len(candidates)})\n\n"
    output += "| ID | Name | Experience | Level | Skills |\n"
    output += "|-----|------|------------|-------|--------|\n"

    for cid, c in candidates.items():
        level = c.screening_level.value if c.screening_level else "N/A"
        output += f"| {cid} | {c.name} | {c.total_experience_years} yrs | {level} | {len(c.skills)} |\n"

    return output


@tool
def list_job_descriptions(session_id: str = "default") -> str:
    """List all parsed job descriptions in the session.

    Args:
        session_id: Session identifier.

    Returns:
        List of all JDs with summary info.
    """
    jds = _get_jds(session_id)

    if not jds:
        return "No job descriptions found. Parse JDs to add them."

    output = f"## Job Descriptions ({len(jds)})\n\n"
    output += "| ID | Title | Experience | Skills Required |\n"
    output += "|-----|-------|------------|----------------|\n"

    for jid, j in jds.items():
        exp = f"{j.min_experience_years}-{j.max_experience_years if j.max_experience_years < 99 else '+'} yrs"
        output += f"| {jid} | {j.title} | {exp} | {len(j.required_skills)} |\n"

    return output


@tool
def get_shortlisted_candidates(
    jd_id: str | None = None,
    session_id: str = "default",
) -> str:
    """Get all shortlisted candidates.

    Args:
        jd_id: Optional JD ID to filter by.
        session_id: Session identifier.

    Returns:
        List of shortlisted candidates.
    """
    results = _get_screening_results(session_id)

    shortlisted = [r for r in results if r.shortlisted]
    if jd_id:
        shortlisted = [r for r in shortlisted if r.jd_id == jd_id]

    if not shortlisted:
        return "No shortlisted candidates found."

    output = "## Shortlisted Candidates\n\n"
    output += "| Candidate | Position | Score | Level | Recommendation |\n"
    output += "|-----------|----------|-------|-------|----------------|\n"

    for r in shortlisted:
        output += f"| {r.candidate_name} | {r.jd_id} | {r.overall_score:.1f}% | {r.recommended_level.value} | {r.recommendation[:30]}... |\n"

    return output


@tool
def get_session_dashboard(session_id: str = "default") -> str:
    """Get a comprehensive dashboard of the current recruitment session.

    Use this tool to provide an overview of all recruitment activities
    including candidates parsed, JDs loaded, screening status, and next steps.

    Args:
        session_id: Session identifier.

    Returns:
        Formatted session dashboard with progress and recommendations.
    """
    candidates = _get_candidates(session_id)
    jds = _get_jds(session_id)
    results = _get_screening_results(session_id)

    # Import interview data
    from app.deepagents.tools.interview_tools import (
        _get_question_sets,
        _get_scores,
    )
    question_sets = _get_question_sets(session_id)
    scores = _get_scores(session_id)

    total_candidates = len(candidates)
    total_jds = len(jds)
    total_screenings = len(results)
    shortlisted = [r for r in results if r.shortlisted]
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    total_question_sets = len(question_sets)
    total_scores = len(scores)

    # Determine workflow phase
    if total_jds == 0 and total_candidates == 0:
        phase = "Setup"
        phase_description = "No data loaded yet. Start by parsing a JD and resumes."
        next_steps = [
            "List JDs from SharePoint: `list_sharepoint_folder(folder_type='jd')`",
            "Parse a job description: `parse_job_description(content, title)`",
            "List resumes from SharePoint: `list_sharepoint_folder(folder_type='resumes')`",
        ]
    elif total_jds > 0 and total_candidates == 0:
        phase = "Resume Collection"
        phase_description = f"{total_jds} JD(s) loaded. Now parse candidate resumes."
        next_steps = [
            "Download resumes from SharePoint",
            "Parse each resume: `parse_resume(content, filename)`",
            "Or upload resumes directly via the UI",
        ]
    elif total_candidates > 0 and total_screenings == 0:
        phase = "Screening"
        phase_description = f"{total_candidates} candidate(s) ready for screening."
        next_steps = [
            "Screen all candidates: `batch_screen_resumes(jd_id)`",
            "Or screen individually: `screen_candidate(candidate_id, jd_id)`",
        ]
    elif len(shortlisted) > 0 and total_question_sets == 0:
        phase = "Assessment"
        phase_description = f"{len(shortlisted)} candidate(s) shortlisted. Generate interview questions."
        next_steps = [
            "Generate questions for each shortlisted candidate",
            "Export question sets to SharePoint",
            "Send to candidates for completion",
        ]
    elif total_question_sets > 0 and total_scores == 0:
        phase = "Evaluation"
        phase_description = f"{total_question_sets} question set(s) generated. Awaiting/evaluating answers."
        next_steps = [
            "Submit candidate answers: `submit_candidate_answers(set_id, answers)`",
            "Evaluate answers: `evaluate_candidate_answers(set_id)`",
        ]
    elif total_scores > 0:
        phase = "Reporting"
        phase_description = f"{total_scores} candidate(s) evaluated. Generate reports."
        next_steps = [
            "Generate scoring report: `generate_scoring_report(jd_id)`",
            "Export to Excel: `export_scoring_excel(jd_id)`",
            "Generate shortlist: `generate_shortlist_report(jd_id)`",
        ]
    else:
        phase = "In Progress"
        phase_description = "Recruitment workflow in progress."
        next_steps = ["Continue with current workflow"]

    # Build dashboard
    output = f"""## Recruitment Session Dashboard

**Session**: {session_id}
**Phase**: {phase}
**Status**: {phase_description}

---

### Progress Summary

| Category | Count | Status |
|----------|-------|--------|
| Job Descriptions | {total_jds} | {'Ready' if total_jds > 0 else 'Pending'} |
| Candidates Parsed | {total_candidates} | {'Ready' if total_candidates > 0 else 'Pending'} |
| Screenings Completed | {total_screenings} | {'Done' if total_screenings > 0 else 'Pending'} |
| Shortlisted | {len(shortlisted)} | {'Available' if shortlisted else 'N/A'} |
| Question Sets | {total_question_sets} | {'Generated' if total_question_sets > 0 else 'Pending'} |
| Evaluations Complete | {total_scores} | {'Done' if total_scores > 0 else 'Pending'} |

---

### Screening Breakdown
"""

    if total_screenings > 0:
        output += f"""
- Shortlisted: {len(shortlisted)} ({len(shortlisted)/total_screenings*100:.0f}%)
- Passed (not shortlisted): {len(passed) - len(shortlisted)}
- Failed: {len(failed)}
"""
    else:
        output += "\n*No screenings performed yet.*\n"

    output += "\n---\n\n### Recommended Next Steps\n\n"
    for i, step in enumerate(next_steps, 1):
        output += f"{i}. {step}\n"

    # Add candidates summary if available
    if candidates:
        output += "\n---\n\n### Candidates Overview\n\n"
        output += "| Name | Level | Skills | Screened |\n"
        output += "|------|-------|--------|----------|\n"
        for cid, c in list(candidates.items())[:10]:
            level = c.screening_level.value if c.screening_level else "N/A"
            screened = "Yes" if any(r.candidate_id == cid for r in results) else "No"
            output += f"| {c.name} | {level} | {len(c.skills)} | {screened} |\n"

    output += f"""
---

*Dashboard refreshed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    return output


@tool
def clear_session_data(
    session_id: str = "default",
    clear_candidates: bool = True,
    clear_jds: bool = True,
    clear_screenings: bool = True,
) -> str:
    """Clear session data for PII compliance and fresh starts.

    Use this tool to clean up session data, especially when handling
    sensitive candidate PII that should not be retained. Also clears
    interview data, scoring data, and document caches.

    Args:
        session_id: Session identifier.
        clear_candidates: Whether to clear candidate profiles.
        clear_jds: Whether to clear job descriptions.
        clear_screenings: Whether to clear screening results.

    Returns:
        Confirmation of cleared data.
    """
    cleared = []

    if clear_candidates and session_id in _candidates:
        count = len(_candidates[session_id])
        del _candidates[session_id]
        cleared.append(f"Candidates: {count} profiles removed")

    if clear_jds and session_id in _job_descriptions:
        count = len(_job_descriptions[session_id])
        del _job_descriptions[session_id]
        cleared.append(f"Job Descriptions: {count} JDs removed")

    if clear_screenings and session_id in _screening_results:
        count = len(_screening_results[session_id])
        del _screening_results[session_id]
        cleared.append(f"Screening Results: {count} results removed")

    # Also clear interview and scoring data
    from app.deepagents.tools.interview_tools import (
        _question_sets,
        _candidate_answers,
        _evaluations,
        _candidate_scores,
    )

    if session_id in _question_sets:
        count = len(_question_sets[session_id])
        del _question_sets[session_id]
        cleared.append(f"Question Sets: {count} sets removed")

    if session_id in _candidate_answers:
        count = len(_candidate_answers[session_id])
        del _candidate_answers[session_id]
        cleared.append(f"Candidate Answers: {count} answers removed")

    if session_id in _evaluations:
        count = len(_evaluations[session_id])
        del _evaluations[session_id]
        cleared.append(f"Evaluations: {count} evaluations removed")

    if session_id in _candidate_scores:
        count = len(_candidate_scores[session_id])
        del _candidate_scores[session_id]
        cleared.append(f"Scores: {count} scores removed")

    # Clear SharePoint document cache
    from app.deepagents.tools.sharepoint_tools import clear_session_cache
    clear_session_cache(session_id)
    cleared.append("SharePoint cache: cleared")

    if len(cleared) <= 1:  # Only the cache clear
        return f"No recruitment data found for session: {session_id}. Cache cleared."

    output = "## Session Data Cleared\n\n"
    output += f"**Session**: {session_id}\n\n"
    for item in cleared:
        output += f"- {item}\n"
    output += "\n*All PII data has been permanently removed from memory.*\n"
    output += "*SharePoint document cache has been cleared.*"

    return output


__all__ = [
    "CandidateProfile",
    "JobDescription",
    "ScreeningResult",
    "parse_resume",
    "parse_job_description",
    "screen_candidate",
    "batch_screen_resumes",
    "get_candidate_profile",
    "list_candidates",
    "list_job_descriptions",
    "get_shortlisted_candidates",
    "get_session_dashboard",
    "clear_session_data",
]
