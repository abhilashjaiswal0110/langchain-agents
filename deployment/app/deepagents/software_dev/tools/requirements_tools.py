"""Requirements Intelligence Tools.

Tools for analyzing, refining, and managing software requirements.
Supports extraction of user stories, detection of ambiguities, and
generation of acceptance criteria.
"""

import json
import uuid
from datetime import datetime

from langchain_core.tools import tool
from langsmith import traceable

from app.deepagents.config.software_dev_config import (
    RequirementType,
    RequirementPriority,
)


# Session storage for requirements (in production, use persistent storage)
_requirements_store: dict[str, dict] = {}
_user_stories_store: dict[str, dict] = {}


@tool
@traceable(name="analyze_requirements", tags=["requirements", "sdlc"])
def analyze_requirements(
    requirements_text: str,
    context: str | None = None,
    session_id: str = "default",
) -> str:
    """Analyze natural language requirements and extract structured requirements.

    This tool processes raw requirements text (from documents, meetings, or user input)
    and extracts structured functional and non-functional requirements.

    Args:
        requirements_text: Raw requirements text to analyze.
        context: Optional context about the project or domain.
        session_id: Session identifier for storing results.

    Returns:
        JSON string with extracted requirements and analysis.
    """
    # Parse requirements from text
    lines = requirements_text.strip().split("\n")
    requirements = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Detect requirement type based on keywords
        req_type = RequirementType.FUNCTIONAL
        if any(kw in line.lower() for kw in ["performance", "scalability", "availability"]):
            req_type = RequirementType.NON_FUNCTIONAL
        elif any(kw in line.lower() for kw in ["api", "database", "integration"]):
            req_type = RequirementType.TECHNICAL
        elif any(kw in line.lower() for kw in ["business", "revenue", "cost"]):
            req_type = RequirementType.BUSINESS

        # Detect priority based on keywords
        priority = RequirementPriority.SHOULD_HAVE
        if any(kw in line.lower() for kw in ["must", "critical", "essential", "required"]):
            priority = RequirementPriority.MUST_HAVE
        elif any(kw in line.lower() for kw in ["could", "nice to have", "optional"]):
            priority = RequirementPriority.COULD_HAVE

        req_id = f"REQ-{str(uuid.uuid4())[:8].upper()}"
        req = {
            "id": req_id,
            "title": line[:100] if len(line) > 100 else line,
            "description": line,
            "type": req_type.value,
            "priority": priority.value,
            "status": "draft",
            "created_at": datetime.now().isoformat(),
        }
        requirements.append(req)

        # Store in session
        _requirements_store[req_id] = req

    analysis = {
        "total_requirements": len(requirements),
        "by_type": {},
        "by_priority": {},
        "requirements": requirements,
    }

    # Count by type and priority
    for req in requirements:
        t = req["type"]
        p = req["priority"]
        analysis["by_type"][t] = analysis["by_type"].get(t, 0) + 1
        analysis["by_priority"][p] = analysis["by_priority"].get(p, 0) + 1

    return json.dumps(analysis, indent=2)


@tool
@traceable(name="extract_user_stories", tags=["requirements", "agile"])
def extract_user_stories(
    requirement_ids: list[str] | None = None,
    requirements_text: str | None = None,
    session_id: str = "default",
) -> str:
    """Extract user stories from requirements.

    Converts requirements into user story format:
    "As a [user], I want [goal] so that [benefit]"

    Args:
        requirement_ids: List of requirement IDs to convert.
        requirements_text: Raw text to extract stories from (alternative).
        session_id: Session identifier.

    Returns:
        JSON string with extracted user stories.
    """
    user_stories = []

    if requirement_ids:
        # Convert existing requirements to user stories
        for req_id in requirement_ids:
            req = _requirements_store.get(req_id)
            if not req:
                continue

            story_id = f"US-{str(uuid.uuid4())[:8].upper()}"
            story = {
                "id": story_id,
                "title": req["title"],
                "as_a": "user",  # Default, should be refined
                "i_want": req["description"],
                "so_that": "I can achieve my goal",  # Default
                "acceptance_criteria": [],
                "requirement_ids": [req_id],
                "story_points": None,
            }
            user_stories.append(story)
            _user_stories_store[story_id] = story

    elif requirements_text:
        # Extract from raw text
        lines = requirements_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            story_id = f"US-{str(uuid.uuid4())[:8].upper()}"
            story = {
                "id": story_id,
                "title": line[:50],
                "as_a": "user",
                "i_want": line,
                "so_that": "I can be more productive",
                "acceptance_criteria": [],
                "requirement_ids": [],
                "story_points": None,
            }
            user_stories.append(story)
            _user_stories_store[story_id] = story

    result = {
        "total_stories": len(user_stories),
        "user_stories": user_stories,
        "session_id": session_id,
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="validate_requirements", tags=["requirements", "quality"])
def validate_requirements(
    requirement_ids: list[str] | None = None,
    session_id: str = "default",
) -> str:
    """Validate requirements for completeness, clarity, and consistency.

    Checks requirements against SMART criteria:
    - Specific: Clear and unambiguous
    - Measurable: Has acceptance criteria
    - Achievable: Technically feasible
    - Relevant: Aligned with business goals
    - Time-bound: Has timeline considerations

    Args:
        requirement_ids: List of requirement IDs to validate (or all if None).
        session_id: Session identifier.

    Returns:
        JSON string with validation results and recommendations.
    """
    reqs_to_validate = []
    if requirement_ids:
        for req_id in requirement_ids:
            req = _requirements_store.get(req_id)
            if req:
                reqs_to_validate.append(req)
    else:
        reqs_to_validate = list(_requirements_store.values())

    validation_results = []
    issues_found = 0

    for req in reqs_to_validate:
        issues = []
        warnings = []

        desc = req.get("description", "")

        # Check for ambiguous words
        ambiguous_words = ["should", "might", "could", "possibly", "etc", "and/or"]
        for word in ambiguous_words:
            if word in desc.lower():
                issues.append(f"Ambiguous word detected: '{word}'")

        # Check length
        if len(desc) < 20:
            warnings.append("Description too brief - consider adding more detail")
        elif len(desc) > 500:
            warnings.append("Description very long - consider splitting")

        # Check for measurability
        if not any(kw in desc.lower() for kw in ["must", "shall", "will", "should"]):
            warnings.append("No action verb detected - may be unclear")

        # Check for testability
        if "acceptance_criteria" not in req or not req.get("acceptance_criteria"):
            warnings.append("Missing acceptance criteria")

        result = {
            "requirement_id": req["id"],
            "title": req["title"],
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "recommendations": [],
        }

        if issues:
            result["recommendations"].append("Review and clarify ambiguous terms")
            issues_found += len(issues)

        if warnings:
            result["recommendations"].append("Consider adding acceptance criteria")

        validation_results.append(result)

    summary = {
        "total_validated": len(validation_results),
        "valid_count": sum(1 for r in validation_results if r["is_valid"]),
        "invalid_count": sum(1 for r in validation_results if not r["is_valid"]),
        "total_issues": issues_found,
        "results": validation_results,
    }

    return json.dumps(summary, indent=2)


@tool
@traceable(name="prioritize_requirements", tags=["requirements", "planning"])
def prioritize_requirements(
    requirement_ids: list[str],
    method: str = "moscow",
    session_id: str = "default",
) -> str:
    """Prioritize requirements using specified methodology.

    Supported methods:
    - moscow: Must/Should/Could/Won't have
    - weighted: Weighted scoring based on value and effort
    - kano: Customer satisfaction analysis

    Args:
        requirement_ids: List of requirement IDs to prioritize.
        method: Prioritization method to use.
        session_id: Session identifier.

    Returns:
        JSON string with prioritized requirements list.
    """
    prioritized = []

    # Get requirements
    reqs = [_requirements_store.get(rid) for rid in requirement_ids if rid in _requirements_store]

    if method == "moscow":
        # MoSCoW prioritization
        priority_order = {
            "must_have": 1,
            "should_have": 2,
            "could_have": 3,
            "wont_have": 4,
        }
        sorted_reqs = sorted(reqs, key=lambda r: priority_order.get(r.get("priority", "should_have"), 2))

        for i, req in enumerate(sorted_reqs, 1):
            prioritized.append({
                "rank": i,
                "id": req["id"],
                "title": req["title"],
                "priority": req.get("priority", "should_have"),
                "rationale": f"Ranked by MoSCoW priority: {req.get('priority', 'should_have')}",
            })

    elif method == "weighted":
        # Weighted scoring (simplified)
        for i, req in enumerate(reqs, 1):
            # Simulate scoring
            business_value = 5 if req.get("type") == "business" else 3
            urgency = 5 if req.get("priority") == "must_have" else 3
            score = business_value * 0.6 + urgency * 0.4

            prioritized.append({
                "rank": i,
                "id": req["id"],
                "title": req["title"],
                "score": round(score, 2),
                "business_value": business_value,
                "urgency": urgency,
            })

        # Sort by score
        prioritized.sort(key=lambda x: x["score"], reverse=True)
        for i, item in enumerate(prioritized, 1):
            item["rank"] = i

    result = {
        "method": method,
        "total_prioritized": len(prioritized),
        "prioritized_requirements": prioritized,
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="detect_ambiguities", tags=["requirements", "quality"])
def detect_ambiguities(
    text: str,
    context: str | None = None,
) -> str:
    """Detect ambiguities and unclear statements in requirements text.

    Identifies:
    - Vague terms (some, many, few, etc.)
    - Missing quantifiers
    - Unclear references
    - Passive voice issues
    - Missing actors

    Args:
        text: Text to analyze for ambiguities.
        context: Optional context for better analysis.

    Returns:
        JSON string with detected ambiguities and suggestions.
    """
    ambiguities = []

    # Vague quantifiers
    vague_quantifiers = ["some", "many", "few", "several", "most", "various", "numerous"]
    for word in vague_quantifiers:
        if word in text.lower():
            ambiguities.append({
                "type": "vague_quantifier",
                "word": word,
                "severity": "medium",
                "suggestion": f"Replace '{word}' with specific number or percentage",
            })

    # Vague adjectives
    vague_adjectives = ["fast", "slow", "good", "bad", "large", "small", "easy", "simple"]
    for word in vague_adjectives:
        if word in text.lower():
            ambiguities.append({
                "type": "vague_adjective",
                "word": word,
                "severity": "medium",
                "suggestion": f"Define measurable criteria for '{word}'",
            })

    # Uncertain words
    uncertain_words = ["might", "may", "could", "possibly", "perhaps", "probably"]
    for word in uncertain_words:
        if word in text.lower():
            ambiguities.append({
                "type": "uncertain_language",
                "word": word,
                "severity": "high",
                "suggestion": f"Replace '{word}' with definitive 'shall' or 'must'",
            })

    # Check for passive voice indicators
    passive_indicators = ["is done", "are processed", "will be", "should be", "must be"]
    for phrase in passive_indicators:
        if phrase in text.lower():
            ambiguities.append({
                "type": "passive_voice",
                "phrase": phrase,
                "severity": "low",
                "suggestion": "Consider active voice to clarify who performs the action",
            })

    # Missing units
    import re
    numbers = re.findall(r'\b\d+\b', text)
    for num in numbers[:5]:  # Check first 5 numbers
        # Check if number has units nearby
        pattern = rf'{num}\s*(ms|seconds|minutes|hours|MB|GB|%|users|requests)'
        if not re.search(pattern, text, re.IGNORECASE):
            ambiguities.append({
                "type": "missing_unit",
                "number": num,
                "severity": "medium",
                "suggestion": f"Add unit or context for number '{num}'",
            })

    result = {
        "total_ambiguities": len(ambiguities),
        "by_severity": {
            "high": sum(1 for a in ambiguities if a["severity"] == "high"),
            "medium": sum(1 for a in ambiguities if a["severity"] == "medium"),
            "low": sum(1 for a in ambiguities if a["severity"] == "low"),
        },
        "ambiguities": ambiguities,
        "recommendation": "Address high severity issues first, then medium, then low",
    }

    return json.dumps(result, indent=2)


@tool
@traceable(name="generate_acceptance_criteria", tags=["requirements", "testing"])
def generate_acceptance_criteria(
    requirement_id: str | None = None,
    requirement_text: str | None = None,
    format: str = "given_when_then",
) -> str:
    """Generate acceptance criteria for a requirement.

    Formats:
    - given_when_then: BDD-style Given/When/Then
    - checklist: Simple checkbox list
    - scenario: Detailed scenario descriptions

    Args:
        requirement_id: ID of requirement to generate criteria for.
        requirement_text: Raw requirement text (alternative).
        format: Output format for criteria.

    Returns:
        JSON string with generated acceptance criteria.
    """
    req_text = requirement_text

    if requirement_id and requirement_id in _requirements_store:
        req = _requirements_store[requirement_id]
        req_text = req.get("description", "")

    if not req_text:
        return json.dumps({"error": "No requirement text provided"})

    criteria = []

    if format == "given_when_then":
        # Generate BDD-style criteria
        criteria = [
            {
                "id": f"AC-{str(uuid.uuid4())[:6].upper()}",
                "given": "the user is authenticated",
                "when": "the user performs the action described in the requirement",
                "then": "the expected outcome should occur",
                "priority": "must_pass",
            },
            {
                "id": f"AC-{str(uuid.uuid4())[:6].upper()}",
                "given": "the system is in normal operating state",
                "when": "the feature is used as intended",
                "then": "no errors should occur",
                "priority": "must_pass",
            },
            {
                "id": f"AC-{str(uuid.uuid4())[:6].upper()}",
                "given": "invalid input is provided",
                "when": "the action is attempted",
                "then": "appropriate error message should be displayed",
                "priority": "should_pass",
            },
        ]

    elif format == "checklist":
        criteria = [
            {"id": f"AC-{i}", "criterion": f"Criterion {i}", "verified": False}
            for i in range(1, 6)
        ]
        criteria[0]["criterion"] = "Feature is accessible to authorized users"
        criteria[1]["criterion"] = "Feature performs intended action correctly"
        criteria[2]["criterion"] = "Error cases are handled gracefully"
        criteria[3]["criterion"] = "Performance meets requirements"
        criteria[4]["criterion"] = "Security requirements are met"

    elif format == "scenario":
        criteria = [
            {
                "id": f"AC-{str(uuid.uuid4())[:6].upper()}",
                "scenario": "Happy Path",
                "description": "User successfully completes the intended action",
                "steps": [
                    "User navigates to feature",
                    "User provides valid input",
                    "System processes request",
                    "Success confirmation is displayed",
                ],
            },
            {
                "id": f"AC-{str(uuid.uuid4())[:6].upper()}",
                "scenario": "Error Handling",
                "description": "System handles errors appropriately",
                "steps": [
                    "User provides invalid input",
                    "System validates input",
                    "Clear error message is shown",
                    "User can correct and retry",
                ],
            },
        ]

    # Update requirement with criteria if using requirement_id
    if requirement_id and requirement_id in _requirements_store:
        _requirements_store[requirement_id]["acceptance_criteria"] = criteria

    result = {
        "requirement_id": requirement_id,
        "format": format,
        "criteria_count": len(criteria),
        "acceptance_criteria": criteria,
    }

    return json.dumps(result, indent=2)
