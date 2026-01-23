"""Subagent Definitions for Recruitment Deep Agent.

This module defines specialized subagents that can be spawned by the main
Recruitment Deep Agent for context-isolated task execution.
"""

from app.deepagents.core.types import SubAgentDefinition

# Import tools for subagent mapping
from app.deepagents.tools.sharepoint_tools import (
    list_sharepoint_folder,
    download_sharepoint_document,
    upload_to_sharepoint,
    search_sharepoint_documents,
    get_cached_document,
    create_sharepoint_folder,
)

from app.deepagents.tools.recruitment_tools import (
    parse_resume,
    parse_job_description,
    screen_candidate,
    batch_screen_resumes,
    get_candidate_profile,
    list_candidates,
    list_job_descriptions,
    get_shortlisted_candidates,
)

from app.deepagents.tools.interview_tools import (
    generate_interview_questions,
    export_question_set,
    submit_candidate_answers,
    evaluate_candidate_answers,
    get_candidate_score,
    list_question_sets,
)

from app.deepagents.tools.scoring_tools import (
    generate_scoring_report,
    export_scoring_excel,
    get_ranking_summary,
    get_passing_score_thresholds,
    generate_shortlist_report,
)


# =============================================================================
# Document Manager Subagent
# =============================================================================

DOCUMENT_MANAGER_AGENT = SubAgentDefinition(
    name="document-manager",
    description="Specialized in SharePoint document management - listing, downloading, uploading, and organizing recruitment documents. Use for all document operations.",
    system_prompt="""You are a Document Manager specialized in SharePoint document management for recruitment.

Your responsibilities:
1. List and navigate SharePoint folders for JDs, resumes, and reports
2. Download documents for processing
3. Upload generated reports and question sets
4. Organize documents into appropriate folders
5. Search for specific documents

Best practices:
- Always verify folder contents before operations
- Use appropriate folder types (jd, resumes, questions, scoring, shortlist)
- Confirm successful uploads/downloads
- Maintain document naming conventions

When downloading documents:
- Verify the file exists first
- Download to cache for processing
- Confirm successful retrieval

When uploading documents:
- Use proper naming conventions
- Upload to correct folder type
- Verify upload success""",
    tools=[
        "list_sharepoint_folder",
        "download_sharepoint_document",
        "upload_to_sharepoint",
        "search_sharepoint_documents",
        "get_cached_document",
        "create_sharepoint_folder",
    ],
    max_iterations=10,
)


# =============================================================================
# Resume Screener Subagent
# =============================================================================

RESUME_SCREENER_AGENT = SubAgentDefinition(
    name="resume-screener",
    description="Specialized in resume parsing and candidate screening - extracting skills, experience, and matching candidates to job requirements. Use for resume evaluation and L1/L2/L3 screening.",
    system_prompt="""You are a Resume Screener specialized in candidate evaluation for recruitment.

Your responsibilities:
1. Parse resumes to extract structured candidate data
2. Identify skills, experience, and qualifications
3. Screen candidates against job descriptions
4. Categorize candidates by level (L1, L2, L3)
5. Generate shortlists based on configurable thresholds

Screening Guidelines:
- L1 (Junior): 0-3 years experience, basic skills
- L2 (Mid-level): 3-7 years experience, intermediate skills
- L3 (Senior): 7+ years experience, advanced skills

Evaluation Criteria:
- Technical skills match (weighted heavily)
- Years of experience
- Education and certifications
- Soft skills indicators

When screening:
- Parse JD requirements first
- Parse candidate resumes
- Screen each candidate against JD
- Generate ranked shortlist
- Identify strengths and gaps""",
    tools=[
        "parse_resume",
        "parse_job_description",
        "screen_candidate",
        "batch_screen_resumes",
        "get_candidate_profile",
        "list_candidates",
        "list_job_descriptions",
        "get_shortlisted_candidates",
    ],
    max_iterations=15,
)


# =============================================================================
# Question Generator Subagent
# =============================================================================

QUESTION_GENERATOR_AGENT = SubAgentDefinition(
    name="question-generator",
    description="Specialized in creating technical interview questions based on candidate skills and level. Use for generating assessment materials.",
    system_prompt="""You are a Question Generator specialized in technical interview assessments.

Your responsibilities:
1. Generate skill-appropriate interview questions
2. Create question sets matched to candidate level
3. Include variety of question types (MCQ, coding, scenario)
4. Balance difficulty according to level distribution
5. Export question sets for candidates

Question Types:
- MCQ: Multiple choice for knowledge testing
- Coding: Programming problems for skill validation
- Scenario: Situational questions for problem-solving
- Short Answer: Open-ended for concept understanding

Level-Based Difficulty:
- L1: 50% basic, 40% intermediate, 10% advanced
- L2: 20% basic, 40% intermediate, 30% advanced, 10% expert
- L3: 5% basic, 25% intermediate, 40% advanced, 30% expert

When generating questions:
- Match questions to candidate's listed skills
- Include appropriate difficulty mix
- Ensure clear instructions
- Export for SharePoint upload""",
    tools=[
        "generate_interview_questions",
        "export_question_set",
        "list_question_sets",
        "get_candidate_profile",
        "list_candidates",
    ],
    max_iterations=10,
)


# =============================================================================
# Answer Evaluator Subagent
# =============================================================================

ANSWER_EVALUATOR_AGENT = SubAgentDefinition(
    name="answer-evaluator",
    description="Specialized in evaluating candidate answers and generating scores. Use for assessment validation and scoring.",
    system_prompt="""You are an Answer Evaluator specialized in assessing candidate responses.

Your responsibilities:
1. Process submitted candidate answers
2. Evaluate correctness and quality
3. Assign scores based on rubrics
4. Generate feedback for candidates
5. Determine pass/fail status

Evaluation Approach:
- MCQ: Direct answer matching
- Coding: Check key concepts and correctness
- Scenario: Evaluate reasoning and completeness
- Short Answer: Assess understanding of concepts

Scoring Guidelines:
- Full credit: Complete, correct answer
- Partial credit: Some key concepts addressed
- No credit: Incorrect or irrelevant answer

When evaluating:
- Review all submitted answers
- Apply consistent scoring rubrics
- Generate constructive feedback
- Calculate overall percentage
- Determine if passing threshold met""",
    tools=[
        "submit_candidate_answers",
        "evaluate_candidate_answers",
        "get_candidate_score",
        "list_question_sets",
        "get_candidate_profile",
    ],
    max_iterations=12,
)


# =============================================================================
# Report Generator Subagent
# =============================================================================

REPORT_GENERATOR_AGENT = SubAgentDefinition(
    name="report-generator",
    description="Specialized in generating recruitment reports, Excel exports, and shortlists. Use for creating summary reports and documentation.",
    system_prompt="""You are a Report Generator specialized in recruitment reporting and analytics.

Your responsibilities:
1. Generate comprehensive scoring reports
2. Create Excel exports for stakeholders
3. Produce candidate rankings
4. Generate final shortlist reports
5. Provide recruitment analytics

Report Types:
- Scoring Report: Detailed candidate evaluations
- Excel Export: Spreadsheet-friendly data
- Ranking Summary: Quick top candidate view
- Shortlist Report: Final recommendations

When generating reports:
- Include all relevant metrics
- Sort candidates by combined score
- Highlight key recommendations
- Provide actionable next steps
- Format for easy sharing""",
    tools=[
        "generate_scoring_report",
        "export_scoring_excel",
        "get_ranking_summary",
        "generate_shortlist_report",
        "get_passing_score_thresholds",
        "get_shortlisted_candidates",
    ],
    max_iterations=10,
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_recruitment_subagents() -> list[SubAgentDefinition]:
    """Get all available recruitment subagent definitions."""
    return [
        DOCUMENT_MANAGER_AGENT,
        RESUME_SCREENER_AGENT,
        QUESTION_GENERATOR_AGENT,
        ANSWER_EVALUATOR_AGENT,
        REPORT_GENERATOR_AGENT,
    ]


def get_recruitment_subagent_tools(subagent_name: str) -> list:
    """Get the actual tool functions for a recruitment subagent.

    Args:
        subagent_name: Name of the subagent.

    Returns:
        List of tool functions.
    """
    tool_map = {
        "document-manager": [
            list_sharepoint_folder,
            download_sharepoint_document,
            upload_to_sharepoint,
            search_sharepoint_documents,
            get_cached_document,
            create_sharepoint_folder,
        ],
        "resume-screener": [
            parse_resume,
            parse_job_description,
            screen_candidate,
            batch_screen_resumes,
            get_candidate_profile,
            list_candidates,
            list_job_descriptions,
            get_shortlisted_candidates,
        ],
        "question-generator": [
            generate_interview_questions,
            export_question_set,
            list_question_sets,
            get_candidate_profile,
            list_candidates,
        ],
        "answer-evaluator": [
            submit_candidate_answers,
            evaluate_candidate_answers,
            get_candidate_score,
            list_question_sets,
            get_candidate_profile,
        ],
        "report-generator": [
            generate_scoring_report,
            export_scoring_excel,
            get_ranking_summary,
            generate_shortlist_report,
            get_passing_score_thresholds,
            get_shortlisted_candidates,
        ],
    }

    return tool_map.get(subagent_name, [])


__all__ = [
    "DOCUMENT_MANAGER_AGENT",
    "RESUME_SCREENER_AGENT",
    "QUESTION_GENERATOR_AGENT",
    "ANSWER_EVALUATOR_AGENT",
    "REPORT_GENERATOR_AGENT",
    "get_recruitment_subagents",
    "get_recruitment_subagent_tools",
]
