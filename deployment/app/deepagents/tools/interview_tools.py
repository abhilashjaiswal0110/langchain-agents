"""Interview question generation and evaluation tools for Recruitment Deep Agent.

This module provides tools for generating technical interview questions,
validating candidate answers, and scoring responses.

Following Enterprise Development Standards:
- Software Architect: Modular question generation pipeline
- Security Architect: No question answer leakage
- Data Architect: Structured question and answer models
- Software Engineer: Type-safe with comprehensive error handling
"""

import json
import logging
import random
import uuid
from datetime import datetime
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.deepagents.config.recruitment_config import (
    QuestionDifficulty,
    ScreeningLevel,
    get_recruitment_config,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class QuestionOption(BaseModel):
    """Multiple choice option."""

    label: str = Field(description="Option label (A, B, C, D)")
    text: str = Field(description="Option text")
    is_correct: bool = Field(default=False, description="Whether this is correct")


class InterviewQuestion(BaseModel):
    """Interview question."""

    question_id: str = Field(description="Unique question ID")
    question_type: str = Field(description="Type: mcq, coding, scenario, short_answer")
    difficulty: QuestionDifficulty = Field(description="Difficulty level")
    skill_category: str = Field(description="Related skill category")
    skills_tested: list[str] = Field(default_factory=list, description="Skills tested")
    question_text: str = Field(description="Question text")
    options: list[QuestionOption] = Field(default_factory=list, description="MCQ options")
    expected_answer: str = Field(default="", description="Expected answer/solution")
    scoring_rubric: dict[str, int] = Field(default_factory=dict, description="Scoring criteria")
    max_points: int = Field(default=10, description="Maximum points")
    time_limit_minutes: int = Field(default=5, description="Time limit")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class QuestionSet(BaseModel):
    """Set of interview questions for a candidate."""

    set_id: str = Field(description="Unique set ID")
    candidate_id: str = Field(description="Candidate ID")
    candidate_name: str = Field(description="Candidate name")
    jd_id: str = Field(default="", description="Job description ID")
    level: ScreeningLevel = Field(description="Screening level")
    questions: list[InterviewQuestion] = Field(default_factory=list)
    total_points: int = Field(default=0, description="Total possible points")
    time_limit_minutes: int = Field(default=60, description="Total time limit")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = Field(default="generated", description="Status: generated, sent, completed")


class CandidateAnswer(BaseModel):
    """Candidate's answer to a question."""

    answer_id: str = Field(description="Unique answer ID")
    question_id: str = Field(description="Question ID")
    candidate_id: str = Field(description="Candidate ID")
    answer_text: str = Field(description="Candidate's answer")
    selected_option: str = Field(default="", description="Selected MCQ option")
    submitted_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    time_taken_seconds: int = Field(default=0, description="Time taken")


class AnswerEvaluation(BaseModel):
    """Evaluation of a candidate's answer."""

    evaluation_id: str = Field(description="Unique evaluation ID")
    answer_id: str = Field(description="Answer ID")
    question_id: str = Field(description="Question ID")
    candidate_id: str = Field(description="Candidate ID")
    points_awarded: int = Field(default=0, description="Points awarded")
    max_points: int = Field(default=10, description="Maximum points")
    is_correct: bool = Field(default=False, description="Whether answer is correct")
    feedback: str = Field(default="", description="Feedback for candidate")
    evaluation_notes: str = Field(default="", description="Internal notes")
    evaluated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class CandidateScore(BaseModel):
    """Overall candidate score from interview."""

    candidate_id: str
    candidate_name: str
    set_id: str
    level: ScreeningLevel
    total_points: int = Field(default=0)
    max_points: int = Field(default=0)
    percentage_score: float = Field(default=0.0)
    passed: bool = Field(default=False)
    questions_attempted: int = Field(default=0)
    questions_correct: int = Field(default=0)
    category_scores: dict[str, dict[str, Any]] = Field(default_factory=dict)
    recommendation: str = Field(default="")
    scored_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Storage
# =============================================================================

_question_sets: dict[str, dict[str, QuestionSet]] = {}
_candidate_answers: dict[str, list[CandidateAnswer]] = {}
_evaluations: dict[str, list[AnswerEvaluation]] = {}
_candidate_scores: dict[str, list[CandidateScore]] = {}


def _get_question_sets(session_id: str) -> dict[str, QuestionSet]:
    """Get question sets for session."""
    if session_id not in _question_sets:
        _question_sets[session_id] = {}
    return _question_sets[session_id]


def _get_answers(session_id: str) -> list[CandidateAnswer]:
    """Get answers for session."""
    if session_id not in _candidate_answers:
        _candidate_answers[session_id] = []
    return _candidate_answers[session_id]


def _get_evaluations(session_id: str) -> list[AnswerEvaluation]:
    """Get evaluations for session."""
    if session_id not in _evaluations:
        _evaluations[session_id] = []
    return _evaluations[session_id]


def _get_scores(session_id: str) -> list[CandidateScore]:
    """Get scores for session."""
    if session_id not in _candidate_scores:
        _candidate_scores[session_id] = []
    return _candidate_scores[session_id]


# =============================================================================
# Question Templates by Category
# =============================================================================

QUESTION_TEMPLATES = {
    "python": {
        "basic": [
            {
                "type": "mcq",
                "question": "What is the output of `print(type([]))`?",
                "options": ["<class 'list'>", "<class 'array'>", "<class 'tuple'>", "<class 'dict'>"],
                "correct": 0,
                "points": 5,
            },
            {
                "type": "mcq",
                "question": "Which keyword is used to define a function in Python?",
                "options": ["function", "def", "func", "define"],
                "correct": 1,
                "points": 5,
            },
        ],
        "intermediate": [
            {
                "type": "mcq",
                "question": "What is the difference between a list and a tuple in Python?",
                "options": [
                    "Lists are mutable, tuples are immutable",
                    "Lists are immutable, tuples are mutable",
                    "Both are mutable",
                    "Both are immutable",
                ],
                "correct": 0,
                "points": 8,
            },
            {
                "type": "coding",
                "question": "Write a function that returns the factorial of a number using recursion.",
                "expected": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                "points": 15,
            },
        ],
        "advanced": [
            {
                "type": "coding",
                "question": "Implement a decorator that caches function results (memoization).",
                "expected": "def cache(func):\n    memo = {}\n    def wrapper(*args):\n        if args not in memo:\n            memo[args] = func(*args)\n        return memo[args]\n    return wrapper",
                "points": 20,
            },
            {
                "type": "scenario",
                "question": "You have a Python application that processes large CSV files. Users report it's slow. Describe your approach to optimize it.",
                "expected": "Use generators/iterators, pandas chunks, multiprocessing, or async I/O",
                "points": 20,
            },
        ],
    },
    "sql": {
        "basic": [
            {
                "type": "mcq",
                "question": "Which SQL clause is used to filter records?",
                "options": ["WHERE", "HAVING", "FILTER", "SELECT"],
                "correct": 0,
                "points": 5,
            },
        ],
        "intermediate": [
            {
                "type": "coding",
                "question": "Write a SQL query to find the second highest salary from an Employees table.",
                "expected": "SELECT MAX(salary) FROM Employees WHERE salary < (SELECT MAX(salary) FROM Employees)",
                "points": 15,
            },
        ],
        "advanced": [
            {
                "type": "coding",
                "question": "Write a query to find employees who earn more than the average salary in their department.",
                "expected": "SELECT e.* FROM Employees e WHERE salary > (SELECT AVG(salary) FROM Employees WHERE department_id = e.department_id)",
                "points": 20,
            },
        ],
    },
    "javascript": {
        "basic": [
            {
                "type": "mcq",
                "question": "What is the output of `typeof null` in JavaScript?",
                "options": ["'null'", "'undefined'", "'object'", "'number'"],
                "correct": 2,
                "points": 5,
            },
        ],
        "intermediate": [
            {
                "type": "mcq",
                "question": "What is a closure in JavaScript?",
                "options": [
                    "A function that has access to variables from its outer scope",
                    "A way to close the browser window",
                    "A method to end a loop",
                    "A type of error handling",
                ],
                "correct": 0,
                "points": 10,
            },
        ],
        "advanced": [
            {
                "type": "coding",
                "question": "Implement a debounce function in JavaScript.",
                "expected": "function debounce(fn, delay) {\n  let timeoutId;\n  return function(...args) {\n    clearTimeout(timeoutId);\n    timeoutId = setTimeout(() => fn.apply(this, args), delay);\n  };\n}",
                "points": 20,
            },
        ],
    },
    "aws": {
        "basic": [
            {
                "type": "mcq",
                "question": "What is Amazon S3 primarily used for?",
                "options": ["Object storage", "Relational database", "Container orchestration", "DNS management"],
                "correct": 0,
                "points": 5,
            },
        ],
        "intermediate": [
            {
                "type": "scenario",
                "question": "Explain the difference between EC2, ECS, and Lambda. When would you use each?",
                "expected": "EC2 for full control VMs, ECS for container orchestration, Lambda for serverless event-driven functions",
                "points": 15,
            },
        ],
        "advanced": [
            {
                "type": "scenario",
                "question": "Design a highly available, fault-tolerant architecture for a web application serving 1 million users.",
                "expected": "Multi-AZ deployment, ALB, Auto Scaling, RDS Multi-AZ, CloudFront, Route 53",
                "points": 25,
            },
        ],
    },
    "general": {
        "basic": [
            {
                "type": "mcq",
                "question": "What does HTTP stand for?",
                "options": [
                    "HyperText Transfer Protocol",
                    "High Transfer Text Protocol",
                    "HyperText Transmission Protocol",
                    "Hyper Transfer Text Protocol",
                ],
                "correct": 0,
                "points": 5,
            },
        ],
        "intermediate": [
            {
                "type": "short_answer",
                "question": "Explain the difference between REST and GraphQL APIs.",
                "expected": "REST uses multiple endpoints with fixed responses; GraphQL uses single endpoint with flexible queries",
                "points": 10,
            },
        ],
        "advanced": [
            {
                "type": "scenario",
                "question": "Describe your approach to debugging a production issue where the application intermittently fails.",
                "expected": "Check logs, metrics, traces; reproduce issue; isolate components; use debugging tools",
                "points": 20,
            },
        ],
    },
}


def _generate_question(
    template: dict[str, Any],
    skill: str,
    difficulty: QuestionDifficulty,
) -> InterviewQuestion:
    """Generate a question from template."""
    question_id = f"Q-{uuid.uuid4().hex[:8].upper()}"

    if template["type"] == "mcq":
        options = [
            QuestionOption(
                label=chr(65 + i),  # A, B, C, D
                text=opt,
                is_correct=(i == template.get("correct", 0)),
            )
            for i, opt in enumerate(template.get("options", []))
        ]
        return InterviewQuestion(
            question_id=question_id,
            question_type="mcq",
            difficulty=difficulty,
            skill_category=skill,
            skills_tested=[skill],
            question_text=template["question"],
            options=options,
            expected_answer=template.get("options", [""])[template.get("correct", 0)],
            max_points=template.get("points", 10),
            scoring_rubric={"correct": template.get("points", 10), "incorrect": 0},
        )
    else:
        return InterviewQuestion(
            question_id=question_id,
            question_type=template["type"],
            difficulty=difficulty,
            skill_category=skill,
            skills_tested=[skill],
            question_text=template["question"],
            expected_answer=template.get("expected", ""),
            max_points=template.get("points", 10),
            scoring_rubric={
                "excellent": template.get("points", 10),
                "good": int(template.get("points", 10) * 0.7),
                "partial": int(template.get("points", 10) * 0.4),
                "incorrect": 0,
            },
        )


# =============================================================================
# Tool Functions
# =============================================================================


@tool
def generate_interview_questions(
    candidate_id: str,
    candidate_name: str,
    skills: list[str],
    level: str,
    jd_id: str = "",
    session_id: str = "default",
) -> str:
    """Generate interview questions for a candidate based on their skills.

    Use this tool to create a customized question set for technical screening.

    Args:
        candidate_id: Candidate identifier.
        candidate_name: Candidate's name.
        skills: List of skills to test.
        level: Screening level (L1, L2, L3).
        jd_id: Optional job description ID.
        session_id: Session identifier.

    Returns:
        Generated question set summary.
    """
    config = get_recruitment_config()

    # Parse level
    try:
        screening_level = ScreeningLevel(level.upper())
    except ValueError:
        screening_level = ScreeningLevel.L1

    # Get question count and difficulty distribution
    question_count = config.interview.get_question_count(screening_level)
    difficulty_dist = config.interview.get_difficulty_distribution(screening_level)

    set_id = f"QS-{uuid.uuid4().hex[:8].upper()}"
    questions: list[InterviewQuestion] = []
    total_points = 0

    # Normalize skills
    normalized_skills = [s.lower().replace(" ", "_").replace("-", "_") for s in skills]

    # Add general questions if not enough skills
    if len(normalized_skills) < 3:
        normalized_skills.append("general")

    # Generate questions for each difficulty level
    for difficulty_name, proportion in difficulty_dist.items():
        if proportion <= 0:
            continue

        num_questions = max(1, int(question_count * proportion))
        difficulty = QuestionDifficulty(difficulty_name)

        for _ in range(num_questions):
            # Pick a random skill
            skill = random.choice(normalized_skills)
            skill_key = skill if skill in QUESTION_TEMPLATES else "general"

            # Get templates for this skill and difficulty
            skill_templates = QUESTION_TEMPLATES.get(skill_key, QUESTION_TEMPLATES["general"])
            difficulty_templates = skill_templates.get(difficulty_name, skill_templates.get("basic", []))

            if difficulty_templates:
                template = random.choice(difficulty_templates)
                question = _generate_question(template, skill, difficulty)
                questions.append(question)
                total_points += question.max_points

    # Create question set
    question_set = QuestionSet(
        set_id=set_id,
        candidate_id=candidate_id,
        candidate_name=candidate_name,
        jd_id=jd_id,
        level=screening_level,
        questions=questions,
        total_points=total_points,
        time_limit_minutes=config.interview.l1_time_limit
        if screening_level == ScreeningLevel.L1
        else config.interview.l2_time_limit
        if screening_level == ScreeningLevel.L2
        else config.interview.l3_time_limit,
    )

    # Store question set
    sets = _get_question_sets(session_id)
    sets[set_id] = question_set

    # Format output (without answers for security)
    output = f"""## Interview Question Set Generated

**Set ID**: {set_id}
**Candidate**: {candidate_name} ({candidate_id})
**Level**: {screening_level.value}
**Questions**: {len(questions)}
**Total Points**: {total_points}
**Time Limit**: {question_set.time_limit_minutes} minutes

---

### Question Breakdown by Difficulty

"""

    difficulty_counts = {}
    type_counts = {}
    for q in questions:
        diff = q.difficulty.value
        qtype = q.question_type
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    for diff, count in sorted(difficulty_counts.items()):
        output += f"- **{diff.title()}**: {count} questions\n"

    output += "\n### Question Types\n\n"
    for qtype, count in sorted(type_counts.items()):
        output += f"- **{qtype.upper()}**: {count} questions\n"

    output += f"""
---

*Question set is ready. Use `export_question_set` to generate candidate-facing document.*
*Use `submit_candidate_answers` to record responses.*
"""

    return output


@tool
def export_question_set(
    set_id: str,
    include_answers: bool = False,
    session_id: str = "default",
) -> str:
    """Export question set as formatted document.

    Use this to generate a document that can be saved to SharePoint
    for the candidate to answer.

    Args:
        set_id: Question set ID.
        include_answers: Whether to include correct answers (for internal use).
        session_id: Session identifier.

    Returns:
        Formatted question document.
    """
    sets = _get_question_sets(session_id)

    if set_id not in sets:
        return f"Question set not found: {set_id}"

    qs = sets[set_id]

    output = f"""# Technical Assessment

**Candidate**: {qs.candidate_name}
**Level**: {qs.level.value}
**Total Questions**: {len(qs.questions)}
**Total Points**: {qs.total_points}
**Time Limit**: {qs.time_limit_minutes} minutes

**Instructions**:
1. Read each question carefully before answering
2. For multiple choice, select the best answer
3. For coding questions, provide working code
4. For scenario questions, explain your reasoning
5. Manage your time wisely

---

"""

    for i, q in enumerate(qs.questions, 1):
        output += f"## Question {i} [{q.max_points} points] - {q.difficulty.value.title()}\n\n"
        output += f"**Category**: {q.skill_category.title()}\n\n"
        output += f"{q.question_text}\n\n"

        if q.question_type == "mcq" and q.options:
            for opt in q.options:
                output += f"- **{opt.label}**. {opt.text}\n"
            output += "\n**Your Answer**: ___\n\n"
        elif q.question_type == "coding":
            output += "**Your Code**:\n```\n\n\n```\n\n"
        else:
            output += "**Your Answer**:\n\n___________________________________________\n\n"

        if include_answers:
            output += f"*[ANSWER KEY: {q.expected_answer}]*\n\n"

        output += "---\n\n"

    output += f"""
## Submission

**Candidate Name**: {qs.candidate_name}
**Date Completed**: _____________
**Signature**: _____________

*Please save this document with your answers and upload to the designated SharePoint folder.*
*File naming convention: {qs.candidate_name.replace(' ', '_')}_Assessment_{set_id}.docx*
"""

    return output


@tool
def submit_candidate_answers(
    set_id: str,
    answers: list[dict[str, str]],
    session_id: str = "default",
) -> str:
    """Submit candidate answers for evaluation.

    Use this to record a candidate's answers from their submitted assessment.

    Args:
        set_id: Question set ID.
        answers: List of answers with question_id and answer_text or selected_option.
        session_id: Session identifier.

    Returns:
        Submission confirmation.
    """
    sets = _get_question_sets(session_id)

    if set_id not in sets:
        return f"Question set not found: {set_id}"

    qs = sets[set_id]
    stored_answers = _get_answers(session_id)
    submitted_count = 0

    for answer_data in answers:
        question_id = answer_data.get("question_id", "")
        answer_text = answer_data.get("answer_text", "")
        selected_option = answer_data.get("selected_option", "")

        if not question_id:
            continue

        # Verify question exists in set
        question_exists = any(q.question_id == question_id for q in qs.questions)
        if not question_exists:
            continue

        answer = CandidateAnswer(
            answer_id=f"ANS-{uuid.uuid4().hex[:8].upper()}",
            question_id=question_id,
            candidate_id=qs.candidate_id,
            answer_text=answer_text,
            selected_option=selected_option,
        )

        stored_answers.append(answer)
        submitted_count += 1

    # Update question set status
    qs.status = "completed"

    output = f"""## Answers Submitted

**Set ID**: {set_id}
**Candidate**: {qs.candidate_name}
**Answers Submitted**: {submitted_count} of {len(qs.questions)}

---

*Use `evaluate_candidate_answers` to score the submission.*
"""

    return output


@tool
def evaluate_candidate_answers(
    set_id: str,
    session_id: str = "default",
) -> str:
    """Evaluate all answers for a question set.

    Use this to automatically score candidate responses.

    Args:
        set_id: Question set ID.
        session_id: Session identifier.

    Returns:
        Evaluation results with scores.
    """
    config = get_recruitment_config()
    sets = _get_question_sets(session_id)
    stored_answers = _get_answers(session_id)
    evaluations = _get_evaluations(session_id)
    scores = _get_scores(session_id)

    if set_id not in sets:
        return f"Question set not found: {set_id}"

    qs = sets[set_id]

    # Get answers for this candidate
    candidate_answers = [a for a in stored_answers if a.candidate_id == qs.candidate_id]

    total_points = 0
    max_points = 0
    correct_count = 0
    category_scores: dict[str, dict[str, Any]] = {}

    for question in qs.questions:
        # Find answer for this question
        answer = next(
            (a for a in candidate_answers if a.question_id == question.question_id),
            None,
        )

        points = 0
        is_correct = False
        feedback = ""

        if not answer:
            feedback = "No answer submitted"
        elif question.question_type == "mcq":
            # Check MCQ answer
            correct_option = next(
                (o for o in question.options if o.is_correct),
                None,
            )
            if correct_option and answer.selected_option.upper() == correct_option.label:
                points = question.max_points
                is_correct = True
                feedback = "Correct!"
            else:
                feedback = f"Incorrect. Correct answer: {correct_option.label if correct_option else 'N/A'}"
        else:
            # For coding/scenario, use basic keyword matching (would use LLM in production)
            expected_lower = question.expected_answer.lower()
            answer_lower = answer.answer_text.lower()

            # Simple scoring based on keyword overlap
            keywords = set(expected_lower.split())
            answer_words = set(answer_lower.split())
            overlap = len(keywords.intersection(answer_words))

            if overlap >= len(keywords) * 0.7:
                points = question.max_points
                is_correct = True
                feedback = "Excellent answer covering key concepts"
            elif overlap >= len(keywords) * 0.4:
                points = int(question.max_points * 0.6)
                feedback = "Partial credit - some key concepts addressed"
            elif overlap > 0:
                points = int(question.max_points * 0.3)
                feedback = "Limited understanding demonstrated"
            else:
                feedback = "Answer does not address the question adequately"

        # Create evaluation
        evaluation = AnswerEvaluation(
            evaluation_id=f"EVAL-{uuid.uuid4().hex[:8].upper()}",
            answer_id=answer.answer_id if answer else "",
            question_id=question.question_id,
            candidate_id=qs.candidate_id,
            points_awarded=points,
            max_points=question.max_points,
            is_correct=is_correct,
            feedback=feedback,
        )
        evaluations.append(evaluation)

        total_points += points
        max_points += question.max_points
        if is_correct:
            correct_count += 1

        # Track category scores
        category = question.skill_category
        if category not in category_scores:
            category_scores[category] = {"points": 0, "max": 0, "questions": 0}
        category_scores[category]["points"] += points
        category_scores[category]["max"] += question.max_points
        category_scores[category]["questions"] += 1

    # Calculate percentage
    percentage = (total_points / max_points * 100) if max_points > 0 else 0

    # Determine if passed
    passing_score = config.scoring.get_passing_score(qs.level)
    passed = percentage >= passing_score

    # Generate recommendation
    if passed and percentage >= 80:
        recommendation = f"STRONGLY RECOMMEND for {qs.level.value} position - Advance to L2 interview"
    elif passed:
        recommendation = f"RECOMMEND for {qs.level.value} position - Consider for next round"
    else:
        recommendation = f"NOT RECOMMENDED - Score {percentage:.1f}% below {passing_score}% threshold"

    # Create score record
    score = CandidateScore(
        candidate_id=qs.candidate_id,
        candidate_name=qs.candidate_name,
        set_id=set_id,
        level=qs.level,
        total_points=total_points,
        max_points=max_points,
        percentage_score=percentage,
        passed=passed,
        questions_attempted=len(candidate_answers),
        questions_correct=correct_count,
        category_scores=category_scores,
        recommendation=recommendation,
    )
    scores.append(score)

    # Format output
    status_icon = "✅" if passed else "❌"

    output = f"""## Evaluation Results {status_icon}

**Candidate**: {qs.candidate_name} ({qs.candidate_id})
**Set ID**: {set_id}
**Level**: {qs.level.value}

---

### Overall Score

| Metric | Value |
|--------|-------|
| Total Points | {total_points} / {max_points} |
| Percentage | {percentage:.1f}% |
| Passing Score | {passing_score}% |
| Status | {'PASSED' if passed else 'FAILED'} |
| Questions Correct | {correct_count} / {len(qs.questions)} |

---

### Scores by Category

| Category | Points | Percentage |
|----------|--------|------------|
"""

    for cat, data in category_scores.items():
        cat_pct = (data["points"] / data["max"] * 100) if data["max"] > 0 else 0
        output += f"| {cat.title()} | {data['points']}/{data['max']} | {cat_pct:.1f}% |\n"

    output += f"""
---

### Recommendation

**{recommendation}**

---

*Evaluation complete. Use `get_candidate_score` to retrieve detailed breakdown.*
*Use scoring tools to export results to Excel.*
"""

    return output


@tool
def get_candidate_score(
    candidate_id: str,
    session_id: str = "default",
) -> str:
    """Get detailed score for a candidate.

    Args:
        candidate_id: Candidate identifier.
        session_id: Session identifier.

    Returns:
        Detailed score information.
    """
    scores = _get_scores(session_id)

    candidate_scores = [s for s in scores if s.candidate_id == candidate_id]

    if not candidate_scores:
        return f"No scores found for candidate: {candidate_id}"

    # Get most recent score
    score = candidate_scores[-1]

    output = f"""## Candidate Score: {score.candidate_name}

**Candidate ID**: {score.candidate_id}
**Assessment**: {score.set_id}
**Level**: {score.level.value}

---

### Results

| Metric | Value |
|--------|-------|
| Total Score | {score.percentage_score:.1f}% |
| Points | {score.total_points} / {score.max_points} |
| Questions Correct | {score.questions_correct} / {score.questions_attempted} |
| Passed | {'Yes' if score.passed else 'No'} |

### Category Breakdown

"""

    for cat, data in score.category_scores.items():
        pct = (data["points"] / data["max"] * 100) if data["max"] > 0 else 0
        output += f"- **{cat.title()}**: {data['points']}/{data['max']} ({pct:.1f}%)\n"

    output += f"""
---

### Recommendation

{score.recommendation}

*Scored: {score.scored_at}*
"""

    return output


@tool
def list_question_sets(session_id: str = "default") -> str:
    """List all generated question sets.

    Args:
        session_id: Session identifier.

    Returns:
        List of question sets.
    """
    sets = _get_question_sets(session_id)

    if not sets:
        return "No question sets found. Generate questions first."

    output = "## Question Sets\n\n"
    output += "| Set ID | Candidate | Level | Questions | Status |\n"
    output += "|--------|-----------|-------|-----------|--------|\n"

    for set_id, qs in sets.items():
        output += f"| {set_id} | {qs.candidate_name} | {qs.level.value} | {len(qs.questions)} | {qs.status} |\n"

    return output


__all__ = [
    "InterviewQuestion",
    "QuestionSet",
    "CandidateAnswer",
    "AnswerEvaluation",
    "CandidateScore",
    "generate_interview_questions",
    "export_question_set",
    "submit_candidate_answers",
    "evaluate_candidate_answers",
    "get_candidate_score",
    "list_question_sets",
]
