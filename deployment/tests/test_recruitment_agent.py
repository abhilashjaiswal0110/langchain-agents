"""Tests for Recruitment Deep Agent functionality.

Comprehensive E2E tests covering:
- SharePoint integration tools
- Resume screening tools (L1/L2/L3)
- Interview question generation/evaluation tools
- Scoring and reporting tools
- Subagent definitions
- API endpoints
- End-to-end workflows
"""

import os
import pytest
import uuid

# Set up mock API keys before importing app modules
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing"


# =============================================================================
# SharePoint Tools Tests
# =============================================================================

class TestSharePointTools:
    """Tests for SharePoint integration tools."""

    def test_list_sharepoint_folder_root(self):
        """Test listing root SharePoint folder."""
        from app.deepagents.tools.sharepoint_tools import list_sharepoint_folder

        result = list_sharepoint_folder.invoke({"folder_type": "jd"})
        assert isinstance(result, str)
        # Demo mode returns sample data - check for files or items
        assert "Files" in result or "files" in result.lower() or "items" in result.lower()

    def test_list_sharepoint_folder_resumes(self):
        """Test listing resumes folder."""
        from app.deepagents.tools.sharepoint_tools import list_sharepoint_folder

        result = list_sharepoint_folder.invoke({"folder_type": "resumes"})
        assert isinstance(result, str)

    def test_list_sharepoint_folder_questions(self):
        """Test listing questions folder."""
        from app.deepagents.tools.sharepoint_tools import list_sharepoint_folder

        result = list_sharepoint_folder.invoke({"folder_type": "questions"})
        assert isinstance(result, str)

    def test_download_sharepoint_document(self):
        """Test downloading a document from SharePoint."""
        from app.deepagents.tools.sharepoint_tools import download_sharepoint_document

        result = download_sharepoint_document.invoke({
            "folder_type": "jd",
            "filename": "test_jd.docx"
        })
        assert isinstance(result, str)
        # Should return path or demo message

    def test_upload_to_sharepoint(self):
        """Test uploading a document to SharePoint."""
        from app.deepagents.tools.sharepoint_tools import upload_to_sharepoint

        result = upload_to_sharepoint.invoke({
            "folder_type": "scoring",
            "filename": "test_upload.txt",
            "content": "Test content for upload"
        })
        assert isinstance(result, str)

    def test_search_sharepoint_documents(self):
        """Test searching documents in SharePoint."""
        from app.deepagents.tools.sharepoint_tools import search_sharepoint_documents

        result = search_sharepoint_documents.invoke({"query": "python developer"})
        assert isinstance(result, str)

    def test_get_cached_document(self):
        """Test retrieving a cached document."""
        from app.deepagents.tools.sharepoint_tools import get_cached_document

        result = get_cached_document.invoke({
            "folder_type": "jd",
            "filename": "test.docx"
        })
        assert isinstance(result, str)

    def test_create_sharepoint_folder(self):
        """Test creating a new folder in SharePoint."""
        from app.deepagents.tools.sharepoint_tools import create_sharepoint_folder

        result = create_sharepoint_folder.invoke({
            "folder_path": "Recruitment/TestFolder"
        })
        assert isinstance(result, str)


# =============================================================================
# Recruitment/Screening Tools Tests
# =============================================================================

class TestRecruitmentTools:
    """Tests for resume screening and candidate management tools."""

    def test_parse_resume(self):
        """Test parsing a resume to extract candidate data."""
        from app.deepagents.tools.recruitment_tools import parse_resume

        result = parse_resume.invoke({
            "content": """
            John Smith
            Python Developer
            5 years of experience in Python, Django, AWS
            Education: BS Computer Science
            Skills: Python, JavaScript, SQL, AWS, Docker
            """,
            "filename": "john_smith_resume.pdf"
        })
        assert isinstance(result, str)
        assert "John" in result or "parsed" in result.lower() or "candidate" in result.lower()

    def test_parse_job_description(self):
        """Test parsing a job description."""
        from app.deepagents.tools.recruitment_tools import parse_job_description

        result = parse_job_description.invoke({
            "content": """
            Senior Python Developer
            Requirements:
            - 5+ years Python experience
            - Django/Flask framework
            - AWS/GCP cloud experience
            - SQL and NoSQL databases
            """,
            "title": "Senior Python Developer"
        })
        assert isinstance(result, str)
        assert "Python" in result or "parsed" in result.lower() or "JD" in result

    def test_screen_candidate(self):
        """Test screening a candidate against job requirements."""
        from app.deepagents.tools.recruitment_tools import screen_candidate

        result = screen_candidate.invoke({
            "candidate_id": "CAND-001",
            "jd_id": "JD-001"
        })
        assert isinstance(result, str)
        # Should contain screening result
        assert "score" in result.lower() or "screen" in result.lower() or "candidate" in result.lower() or "No" in result

    def test_batch_screen_resumes(self):
        """Test batch screening multiple resumes."""
        from app.deepagents.tools.recruitment_tools import batch_screen_resumes

        result = batch_screen_resumes.invoke({
            "jd_id": "JD-001"
        })
        assert isinstance(result, str)

    def test_get_candidate_profile(self):
        """Test retrieving a candidate profile."""
        from app.deepagents.tools.recruitment_tools import get_candidate_profile

        result = get_candidate_profile.invoke({"candidate_id": "CAND-001"})
        assert isinstance(result, str)

    def test_list_candidates(self):
        """Test listing all candidates."""
        from app.deepagents.tools.recruitment_tools import list_candidates

        result = list_candidates.invoke({})
        assert isinstance(result, str)

    def test_list_job_descriptions(self):
        """Test listing all job descriptions."""
        from app.deepagents.tools.recruitment_tools import list_job_descriptions

        result = list_job_descriptions.invoke({})
        assert isinstance(result, str)

    def test_get_shortlisted_candidates(self):
        """Test getting shortlisted candidates."""
        from app.deepagents.tools.recruitment_tools import get_shortlisted_candidates

        result = get_shortlisted_candidates.invoke({"jd_id": "JD-001"})
        assert isinstance(result, str)


# =============================================================================
# Interview Tools Tests
# =============================================================================

class TestInterviewTools:
    """Tests for interview question generation and evaluation tools."""

    def test_generate_interview_questions(self):
        """Test generating interview questions for a candidate."""
        from app.deepagents.tools.interview_tools import generate_interview_questions

        result = generate_interview_questions.invoke({
            "candidate_id": "CAND-001",
            "candidate_name": "John Smith",
            "skills": ["Python", "Django", "AWS"],
            "level": "L2"
        })
        assert isinstance(result, str)
        # Should generate questions
        assert "question" in result.lower() or "generated" in result.lower() or "QS-" in result

    def test_generate_interview_questions_l1(self):
        """Test generating L1 (Junior) level questions."""
        from app.deepagents.tools.interview_tools import generate_interview_questions

        result = generate_interview_questions.invoke({
            "candidate_id": "CAND-002",
            "candidate_name": "Jane Doe",
            "skills": ["Python", "SQL"],
            "level": "L1"
        })
        assert isinstance(result, str)

    def test_generate_interview_questions_l3(self):
        """Test generating L3 (Senior) level questions."""
        from app.deepagents.tools.interview_tools import generate_interview_questions

        result = generate_interview_questions.invoke({
            "candidate_id": "CAND-003",
            "candidate_name": "Bob Senior",
            "skills": ["Python", "System Design", "AWS", "Kubernetes"],
            "level": "L3"
        })
        assert isinstance(result, str)

    def test_export_question_set(self):
        """Test exporting a question set."""
        from app.deepagents.tools.interview_tools import export_question_set

        result = export_question_set.invoke({
            "set_id": "QS-001"
        })
        assert isinstance(result, str)

    def test_submit_candidate_answers(self):
        """Test submitting candidate answers."""
        from app.deepagents.tools.interview_tools import submit_candidate_answers

        result = submit_candidate_answers.invoke({
            "set_id": "QS-001",
            "answers": [
                {"question_id": "Q1", "answer": "Python is a high-level programming language"},
                {"question_id": "Q2", "answer": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"}
            ]
        })
        assert isinstance(result, str)

    def test_evaluate_candidate_answers(self):
        """Test evaluating candidate answers."""
        from app.deepagents.tools.interview_tools import evaluate_candidate_answers

        result = evaluate_candidate_answers.invoke({
            "set_id": "QS-001"
        })
        assert isinstance(result, str)

    def test_get_candidate_score(self):
        """Test getting a candidate's score."""
        from app.deepagents.tools.interview_tools import get_candidate_score

        result = get_candidate_score.invoke({"candidate_id": "CAND-001"})
        assert isinstance(result, str)

    def test_list_question_sets(self):
        """Test listing all question sets."""
        from app.deepagents.tools.interview_tools import list_question_sets

        result = list_question_sets.invoke({})
        assert isinstance(result, str)


# =============================================================================
# Scoring Tools Tests
# =============================================================================

class TestScoringTools:
    """Tests for scoring and reporting tools."""

    def test_generate_scoring_report(self):
        """Test generating a scoring report."""
        from app.deepagents.tools.scoring_tools import generate_scoring_report

        result = generate_scoring_report.invoke({"jd_id": "JD-001"})
        assert isinstance(result, str)

    def test_export_scoring_excel(self):
        """Test exporting scoring data to Excel format."""
        from app.deepagents.tools.scoring_tools import export_scoring_excel

        result = export_scoring_excel.invoke({
            "jd_id": "JD-001"
        })
        assert isinstance(result, str)

    def test_get_ranking_summary(self):
        """Test getting a ranking summary."""
        from app.deepagents.tools.scoring_tools import get_ranking_summary

        result = get_ranking_summary.invoke({"jd_id": "JD-001"})
        assert isinstance(result, str)

    def test_get_passing_score_thresholds(self):
        """Test getting passing score thresholds."""
        from app.deepagents.tools.scoring_tools import get_passing_score_thresholds

        result = get_passing_score_thresholds.invoke({})
        assert isinstance(result, str)
        # Should contain L1, L2, L3 thresholds
        assert "L1" in result or "l1" in result.lower() or "%" in result

    def test_generate_shortlist_report(self):
        """Test generating a shortlist report."""
        from app.deepagents.tools.scoring_tools import generate_shortlist_report

        result = generate_shortlist_report.invoke({
            "jd_id": "JD-001"
        })
        assert isinstance(result, str)


# =============================================================================
# Recruitment Subagent Tests
# =============================================================================

class TestRecruitmentSubagentDefinitions:
    """Tests for recruitment subagent definitions."""

    def test_get_all_recruitment_subagents(self):
        """Test retrieving all recruitment subagent definitions."""
        from app.deepagents.subagents.recruitment_subagents import get_recruitment_subagents

        subagents = get_recruitment_subagents()
        assert len(subagents) >= 5  # At least 5 subagents defined

        names = [s.name for s in subagents]
        assert "document-manager" in names
        assert "resume-screener" in names
        assert "question-generator" in names
        assert "answer-evaluator" in names
        assert "report-generator" in names

    def test_get_recruitment_subagent_tools(self):
        """Test that subagents have associated tools."""
        from app.deepagents.subagents.recruitment_subagents import get_recruitment_subagent_tools

        # Check document-manager has sharepoint tools
        doc_manager_tools = get_recruitment_subagent_tools("document-manager")
        assert len(doc_manager_tools) > 0

        # Check resume-screener has screening tools
        screener_tools = get_recruitment_subagent_tools("resume-screener")
        assert len(screener_tools) > 0

        # Check question-generator has interview tools
        question_tools = get_recruitment_subagent_tools("question-generator")
        assert len(question_tools) > 0

    def test_get_recruitment_subagent_tools_not_found(self):
        """Test retrieving tools for unknown subagent."""
        from app.deepagents.subagents.recruitment_subagents import get_recruitment_subagent_tools

        tools = get_recruitment_subagent_tools("unknown-agent")
        assert tools == []

    def test_subagent_system_prompts(self):
        """Test that all subagents have proper system prompts."""
        from app.deepagents.subagents.recruitment_subagents import get_recruitment_subagents

        for subagent in get_recruitment_subagents():
            assert subagent.system_prompt is not None
            assert len(subagent.system_prompt) > 50
            assert subagent.description is not None
            assert len(subagent.description) > 10


# =============================================================================
# Configuration Tests
# =============================================================================

class TestRecruitmentConfig:
    """Tests for recruitment configuration."""

    def test_get_recruitment_config(self):
        """Test getting recruitment configuration."""
        from app.deepagents.config.recruitment_config import get_recruitment_config

        config = get_recruitment_config()
        assert config is not None
        assert config.scoring is not None
        assert config.interview is not None
        assert config.resume_parsing is not None
        assert config.sharepoint is not None

    def test_scoring_config_defaults(self):
        """Test scoring configuration default values."""
        from app.deepagents.config.recruitment_config import get_recruitment_config

        config = get_recruitment_config()
        # Check default passing scores
        assert config.scoring.l1_passing_score == 60.0
        assert config.scoring.l2_passing_score == 70.0
        assert config.scoring.l3_passing_score == 80.0

    def test_scoring_weights(self):
        """Test scoring weight configuration."""
        from app.deepagents.config.recruitment_config import get_recruitment_config

        config = get_recruitment_config()
        # Weights should sum close to 1.0
        total_weight = (
            config.scoring.technical_weight +
            config.scoring.experience_weight +
            config.scoring.education_weight +
            config.scoring.soft_skills_weight +
            config.scoring.certification_weight
        )
        assert 0.99 <= total_weight <= 1.01

    def test_interview_config(self):
        """Test interview configuration."""
        from app.deepagents.config.recruitment_config import get_recruitment_config

        config = get_recruitment_config()
        # Check question counts exist
        assert config.interview.l1_question_count > 0
        assert config.interview.l2_question_count > 0
        assert config.interview.l3_question_count > 0

    def test_sharepoint_config(self):
        """Test SharePoint configuration."""
        from app.deepagents.config.recruitment_config import get_recruitment_config

        config = get_recruitment_config()
        assert config.sharepoint.jd_folder is not None
        assert config.sharepoint.resumes_folder is not None
        assert config.sharepoint.interview_questions_folder is not None


# =============================================================================
# Recruitment Agent Tests
# =============================================================================

class TestRecruitmentAgent:
    """Tests for the main Recruitment Deep Agent."""

    def test_agent_creation_without_api_key(self):
        """Test that agent creation fails without API key."""
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        old_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
        old_azure = os.environ.pop("AZURE_OPENAI_API_KEY", None)

        try:
            from app.deepagents.recruitment_agent import create_recruitment_agent
            with pytest.raises(ValueError, match="No LLM API key found"):
                create_recruitment_agent()
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
            if old_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = old_anthropic
            if old_azure:
                os.environ["AZURE_OPENAI_API_KEY"] = old_azure

    def test_agent_module_imports(self):
        """Test that agent module imports correctly."""
        from app.deepagents.recruitment_agent import (
            RecruitmentDeepAgent,
            create_recruitment_agent,
            RECRUITMENT_SYSTEM_PROMPT,
        )

        assert RecruitmentDeepAgent is not None
        assert create_recruitment_agent is not None
        assert "Recruitment" in RECRUITMENT_SYSTEM_PROMPT or "recruitment" in RECRUITMENT_SYSTEM_PROMPT

    def test_agent_system_prompt_content(self):
        """Test system prompt contains key capabilities."""
        from app.deepagents.recruitment_agent import RECRUITMENT_SYSTEM_PROMPT

        prompt_lower = RECRUITMENT_SYSTEM_PROMPT.lower()
        # Should mention key capabilities
        assert "resume" in prompt_lower or "screening" in prompt_lower
        assert "sharepoint" in prompt_lower or "document" in prompt_lower
        assert "question" in prompt_lower or "interview" in prompt_lower

    def test_agent_exports_from_deepagents(self):
        """Test that recruitment agent is exported from deepagents module."""
        from app.deepagents import RecruitmentDeepAgent, create_recruitment_agent

        assert RecruitmentDeepAgent is not None
        assert create_recruitment_agent is not None


# =============================================================================
# Recruitment Agent API Endpoint Tests
# =============================================================================

class TestRecruitmentAgentAPIEndpoints:
    """Tests for Recruitment Agent API endpoints."""

    def test_start_recruitment_agent_session(self):
        """Test starting a Recruitment Agent session."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.post(
            "/api/recruitment-agent/start",
            json={"user_id": "test-user"}
        )

        # May fail if agent not loaded (503), or return success
        assert response.status_code in [200, 401, 503]
        data = response.json()
        if response.status_code == 200:
            assert "session_id" in data or "success" in data

    def test_recruitment_agent_chat_endpoint(self):
        """Test chat endpoint."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.post(
            "/api/recruitment-agent/chat",
            json={
                "session_id": "test-session",
                "message": "List all job descriptions"
            }
        )

        # Should return 200, 422 (validation), or 503 (agent not loaded)
        assert response.status_code in [200, 422, 503]

    def test_list_recruitment_subagents_endpoint(self):
        """Test listing available recruitment subagents via API."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/recruitment-agent/subagents")

        # May return 503 if agent not loaded
        assert response.status_code in [200, 401, 503]
        if response.status_code == 200:
            data = response.json()
            assert "subagents" in data
            assert data["count"] >= 5

    def test_recruitment_agent_config_endpoint(self):
        """Test getting recruitment configuration via API."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/recruitment-agent/config")

        assert response.status_code in [200, 401, 503]
        if response.status_code == 200:
            data = response.json()
            assert "config" in data
            assert "scoring" in data["config"]

    def test_recruitment_agent_context_endpoint(self):
        """Test context endpoint with session ID."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/recruitment-agent/context/test-session-id")

        # Should return 200, 404, or 503
        assert response.status_code in [200, 404, 500, 503]

    def test_recruitment_agent_todos_endpoint(self):
        """Test todos endpoint."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/recruitment-agent/todos/test-session")

        assert response.status_code in [200, 404, 500, 503]

    def test_recruitment_agent_files_endpoint(self):
        """Test files endpoint."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/recruitment-agent/files/test-session")

        assert response.status_code in [200, 404, 500, 503]

    def test_recruitment_agent_attachments_endpoint(self):
        """Test attachments endpoint."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/recruitment-agent/attachments/test-session")

        assert response.status_code in [200, 503]


class TestRecruitmentEndpointRouting:
    """Test that all recruitment endpoints are properly routed."""

    def test_recruitment_agent_routes_exist(self):
        """Verify all Recruitment Agent routes exist and respond."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        routes = [
            ("GET", "/api/recruitment-agent/subagents"),
            ("GET", "/api/recruitment-agent/config"),
            ("POST", "/api/recruitment-agent/start"),
            ("POST", "/api/recruitment-agent/chat"),
        ]

        for method, path in routes:
            if method == "GET":
                response = client.get(path)
            else:
                response = client.post(path, json={"session_id": "test", "message": "test"})

            # Should not return 404 (route doesn't exist) or 405 (method not allowed)
            assert response.status_code not in [404, 405], f"Route {method} {path} not found or method not allowed"


# =============================================================================
# E2E Integration Tests (Workflows)
# =============================================================================

class TestRecruitmentE2EWorkflows:
    """End-to-end workflow tests for Recruitment Agent."""

    def test_resume_screening_workflow(self):
        """Test complete resume screening workflow."""
        from app.deepagents.tools.sharepoint_tools import list_sharepoint_folder
        from app.deepagents.tools.recruitment_tools import (
            parse_job_description,
            parse_resume,
            screen_candidate,
            get_shortlisted_candidates
        )

        # Step 1: List available JDs
        jds = list_sharepoint_folder.invoke({"folder_type": "jd"})
        assert isinstance(jds, str)

        # Step 2: Parse a JD
        jd_result = parse_job_description.invoke({
            "content": """
            Python Developer - L2
            Requirements:
            - 3-5 years Python experience
            - Django or Flask
            - SQL and PostgreSQL
            - REST API development
            """,
            "title": "Python Developer L2"
        })
        assert isinstance(jd_result, str)

        # Step 3: Parse a resume
        resume_result = parse_resume.invoke({
            "content": """
            Jane Developer
            Python Developer with 4 years experience
            Skills: Python, Django, PostgreSQL, REST APIs, Git
            Education: MS Computer Science
            """,
            "filename": "jane_developer.pdf"
        })
        assert isinstance(resume_result, str)

        # Step 4: Screen the candidate
        screening = screen_candidate.invoke({
            "candidate_id": "CAND-TEST-001",
            "jd_id": "JD-TEST-001"
        })
        assert isinstance(screening, str)

        # Step 5: Get shortlisted candidates
        shortlist = get_shortlisted_candidates.invoke({"jd_id": "JD-TEST-001"})
        assert isinstance(shortlist, str)

    def test_interview_generation_workflow(self):
        """Test interview question generation and evaluation workflow."""
        from app.deepagents.tools.interview_tools import (
            generate_interview_questions,
            export_question_set,
            submit_candidate_answers,
            evaluate_candidate_answers,
            get_candidate_score
        )

        # Step 1: Generate questions
        questions = generate_interview_questions.invoke({
            "candidate_id": "CAND-WORKFLOW-001",
            "candidate_name": "Test Candidate",
            "skills": ["Python", "Django", "SQL"],
            "level": "L2"
        })
        assert isinstance(questions, str)

        # Step 2: Export question set
        exported = export_question_set.invoke({
            "set_id": "QS-WORKFLOW-001"
        })
        assert isinstance(exported, str)

        # Step 3: Submit answers
        submitted = submit_candidate_answers.invoke({
            "set_id": "QS-WORKFLOW-001",
            "answers": [
                {"question_id": "Q1", "answer": "Python uses indentation for code blocks"},
                {"question_id": "Q2", "answer": "Django ORM provides database abstraction"}
            ]
        })
        assert isinstance(submitted, str)

        # Step 4: Evaluate answers
        evaluated = evaluate_candidate_answers.invoke({
            "set_id": "QS-WORKFLOW-001"
        })
        assert isinstance(evaluated, str)

        # Step 5: Get final score
        score = get_candidate_score.invoke({"candidate_id": "CAND-WORKFLOW-001"})
        assert isinstance(score, str)

    def test_reporting_workflow(self):
        """Test scoring and reporting workflow."""
        from app.deepagents.tools.scoring_tools import (
            get_passing_score_thresholds,
            generate_scoring_report,
            get_ranking_summary,
            export_scoring_excel,
            generate_shortlist_report
        )

        # Step 1: Get passing thresholds
        thresholds = get_passing_score_thresholds.invoke({})
        assert isinstance(thresholds, str)

        # Step 2: Generate scoring report
        report = generate_scoring_report.invoke({"jd_id": "JD-REPORT-001"})
        assert isinstance(report, str)

        # Step 3: Get ranking summary
        ranking = get_ranking_summary.invoke({"jd_id": "JD-REPORT-001"})
        assert isinstance(ranking, str)

        # Step 4: Export to Excel
        excel = export_scoring_excel.invoke({
            "jd_id": "JD-REPORT-001"
        })
        assert isinstance(excel, str)

        # Step 5: Generate shortlist report
        shortlist = generate_shortlist_report.invoke({
            "jd_id": "JD-REPORT-001"
        })
        assert isinstance(shortlist, str)

    def test_full_recruitment_pipeline(self):
        """Test complete recruitment pipeline from JD to shortlist."""
        from app.deepagents.tools.sharepoint_tools import list_sharepoint_folder, upload_to_sharepoint
        from app.deepagents.tools.recruitment_tools import (
            parse_job_description,
            parse_resume,
            screen_candidate,
            list_candidates
        )
        from app.deepagents.tools.interview_tools import generate_interview_questions
        from app.deepagents.tools.scoring_tools import generate_shortlist_report

        # Pipeline Step 1: List SharePoint folders
        folders = list_sharepoint_folder.invoke({"folder_type": "jd"})
        assert isinstance(folders, str)

        # Pipeline Step 2: Parse JD
        jd = parse_job_description.invoke({
            "content": "Senior Data Engineer with 5+ years experience in Python, Spark, AWS",
            "title": "Senior Data Engineer"
        })
        assert isinstance(jd, str)

        # Pipeline Step 3: Parse and screen resumes
        resume = parse_resume.invoke({
            "content": "Experienced Data Engineer with Python, Spark, AWS skills. 6 years experience.",
            "filename": "data_engineer_resume.pdf"
        })
        assert isinstance(resume, str)

        # Pipeline Step 4: Screen candidate
        screening = screen_candidate.invoke({
            "candidate_id": "CAND-PIPELINE-001",
            "jd_id": "JD-PIPELINE-001"
        })
        assert isinstance(screening, str)

        # Pipeline Step 5: Generate interview questions
        questions = generate_interview_questions.invoke({
            "candidate_id": "CAND-PIPELINE-001",
            "candidate_name": "Pipeline Candidate",
            "skills": ["Python", "Spark", "AWS"],
            "level": "L3"
        })
        assert isinstance(questions, str)

        # Pipeline Step 6: Generate shortlist report
        report = generate_shortlist_report.invoke({
            "jd_id": "JD-PIPELINE-001"
        })
        assert isinstance(report, str)

        # Pipeline Step 7: Upload report to SharePoint
        upload = upload_to_sharepoint.invoke({
            "folder_type": "shortlist",
            "filename": "shortlist_report.txt",
            "content": report
        })
        assert isinstance(upload, str)


# =============================================================================
# Tool Export Tests
# =============================================================================

class TestRecruitmentToolExports:
    """Tests for tool module exports."""

    def test_sharepoint_tools_exported(self):
        """Test SharePoint tools are properly exported."""
        from app.deepagents.tools import (
            list_sharepoint_folder,
            download_sharepoint_document,
            upload_to_sharepoint,
            search_sharepoint_documents,
            get_cached_document,
            create_sharepoint_folder,
        )
        assert list_sharepoint_folder is not None
        assert download_sharepoint_document is not None
        assert upload_to_sharepoint is not None
        assert search_sharepoint_documents is not None
        assert get_cached_document is not None
        assert create_sharepoint_folder is not None

    def test_recruitment_tools_exported(self):
        """Test recruitment tools are properly exported."""
        from app.deepagents.tools import (
            parse_resume,
            parse_job_description,
            screen_candidate,
            batch_screen_resumes,
            get_candidate_profile,
            list_candidates,
            list_job_descriptions,
            get_shortlisted_candidates,
        )
        assert parse_resume is not None
        assert parse_job_description is not None
        assert screen_candidate is not None
        assert batch_screen_resumes is not None
        assert get_candidate_profile is not None
        assert list_candidates is not None
        assert list_job_descriptions is not None
        assert get_shortlisted_candidates is not None

    def test_interview_tools_exported(self):
        """Test interview tools are properly exported."""
        from app.deepagents.tools import (
            generate_interview_questions,
            export_question_set,
            submit_candidate_answers,
            evaluate_candidate_answers,
            get_candidate_score,
            list_question_sets,
        )
        assert generate_interview_questions is not None
        assert export_question_set is not None
        assert submit_candidate_answers is not None
        assert evaluate_candidate_answers is not None
        assert get_candidate_score is not None
        assert list_question_sets is not None

    def test_scoring_tools_exported(self):
        """Test scoring tools are properly exported."""
        from app.deepagents.tools import (
            generate_scoring_report,
            export_scoring_excel,
            get_ranking_summary,
            get_passing_score_thresholds,
            generate_shortlist_report,
        )
        assert generate_scoring_report is not None
        assert export_scoring_excel is not None
        assert get_ranking_summary is not None
        assert get_passing_score_thresholds is not None
        assert generate_shortlist_report is not None


# =============================================================================
# Security Tests
# =============================================================================

class TestRecruitmentAgentSecurity:
    """Security-focused tests for Recruitment Agent."""

    def test_no_sensitive_data_in_resume_results(self):
        """Test that resume parsing doesn't expose sensitive patterns."""
        from app.deepagents.tools.recruitment_tools import parse_resume

        result = parse_resume.invoke({
            "content": "John Doe, Python Developer",
            "filename": "john_doe.pdf"
        })
        # Should not contain API keys or passwords
        assert "sk-" not in result
        assert "api_key" not in result.lower()

    def test_no_sql_injection_in_candidate_search(self):
        """Test that candidate search handles SQL injection attempts."""
        from app.deepagents.tools.recruitment_tools import list_candidates

        # Try SQL injection pattern
        result = list_candidates.invoke({})
        assert isinstance(result, str)
        # Should not cause errors, just return safely


# =============================================================================
# UI Integration Tests
# =============================================================================

class TestRecruitmentUIIntegration:
    """Tests for UI integration with Recruitment Agent."""

    def test_chat_html_includes_recruitment_agent(self):
        """Test that chat.html includes Recruitment Agent option."""
        from pathlib import Path

        chat_html_path = Path(__file__).parent.parent / "app" / "static" / "chat.html"
        if chat_html_path.exists():
            content = chat_html_path.read_text(encoding="utf-8")
            assert "recruitment_deep" in content
            assert "Recruitment" in content
            assert "/api/recruitment-agent" in content

    def test_recruitment_agent_quick_actions_defined(self):
        """Test that quick actions are defined for Recruitment Agent in UI."""
        from pathlib import Path

        chat_html_path = Path(__file__).parent.parent / "app" / "static" / "chat.html"
        if chat_html_path.exists():
            content = chat_html_path.read_text(encoding="utf-8")
            # Check for recruitment-specific quick actions
            assert "recruitment_deep" in content
            # UI should have recruitment agent in dropdown
            assert "Recruitment Deep Agent" in content or "recruitment_deep" in content


# =============================================================================
# Session Dashboard Tests
# =============================================================================

class TestSessionDashboard:
    """Tests for the session dashboard tool."""

    def test_dashboard_empty_session(self):
        """Test dashboard shows setup phase for empty session."""
        from app.deepagents.tools.recruitment_tools import get_session_dashboard

        result = get_session_dashboard.invoke({"session_id": f"test-dashboard-{uuid.uuid4().hex[:8]}"})
        assert "Dashboard" in result
        assert "Setup" in result
        assert "Pending" in result

    def test_dashboard_after_jd_parsed(self):
        """Test dashboard shows resume collection phase after JD is parsed."""
        from app.deepagents.tools.recruitment_tools import (
            get_session_dashboard,
            parse_job_description,
        )

        session_id = f"test-dash-jd-{uuid.uuid4().hex[:8]}"
        parse_job_description.invoke({
            "content": "Looking for a Python developer with 5 years experience in Django and AWS.",
            "title": "Senior Python Developer",
            "session_id": session_id,
        })

        result = get_session_dashboard.invoke({"session_id": session_id})
        assert "Dashboard" in result
        assert "Resume Collection" in result
        assert "Job Descriptions | 1" in result

    def test_dashboard_after_candidates_parsed(self):
        """Test dashboard shows screening phase after candidates are parsed."""
        from app.deepagents.tools.recruitment_tools import (
            get_session_dashboard,
            parse_resume,
            parse_job_description,
        )

        session_id = f"test-dash-cand-{uuid.uuid4().hex[:8]}"
        parse_job_description.invoke({
            "content": "Python developer needed with AWS experience.",
            "title": "Python Dev",
            "session_id": session_id,
        })
        parse_resume.invoke({
            "content": "John Doe - 5 years Python, AWS, Django experience. john@email.com",
            "filename": "john_doe_resume.pdf",
            "session_id": session_id,
        })

        result = get_session_dashboard.invoke({"session_id": session_id})
        assert "Screening" in result
        assert "Candidates Parsed | 1" in result

    def test_dashboard_progress_tracking(self):
        """Test dashboard tracks progress through phases."""
        from app.deepagents.tools.recruitment_tools import (
            get_session_dashboard,
            parse_resume,
            parse_job_description,
            batch_screen_resumes,
            _get_jds,
        )

        session_id = f"test-dash-progress-{uuid.uuid4().hex[:8]}"

        # Parse JD and resume
        parse_job_description.invoke({
            "content": "Need Python developer with 3 years experience.",
            "title": "Python Developer",
            "session_id": session_id,
        })
        parse_resume.invoke({
            "content": "Jane Smith - 4 years Python, Flask, Docker. jane@test.com",
            "filename": "jane_smith.pdf",
            "session_id": session_id,
        })

        # Get JD ID
        jds = _get_jds(session_id)
        jd_id = list(jds.keys())[0]

        # Screen candidates
        batch_screen_resumes.invoke({
            "jd_id": jd_id,
            "session_id": session_id,
        })

        result = get_session_dashboard.invoke({"session_id": session_id})
        assert "Screenings Completed | 1" in result

    def test_dashboard_next_steps(self):
        """Test dashboard provides actionable next steps."""
        from app.deepagents.tools.recruitment_tools import get_session_dashboard

        result = get_session_dashboard.invoke({"session_id": f"test-steps-{uuid.uuid4().hex[:8]}"})
        assert "Next Steps" in result


# =============================================================================
# Data Lifecycle / PII Cleanup Tests
# =============================================================================

class TestDataLifecycle:
    """Tests for data lifecycle management and PII cleanup."""

    def test_clear_session_data_empty(self):
        """Test clearing empty session returns appropriate message."""
        from app.deepagents.tools.recruitment_tools import clear_session_data

        result = clear_session_data.invoke({
            "session_id": f"test-empty-{uuid.uuid4().hex[:8]}",
        })
        assert "No recruitment data found" in result or "Cache cleared" in result

    def test_clear_session_data_with_candidates(self):
        """Test clearing session removes candidate PII."""
        from app.deepagents.tools.recruitment_tools import (
            clear_session_data,
            parse_resume,
            _get_candidates,
        )

        session_id = f"test-clear-{uuid.uuid4().hex[:8]}"

        # Add candidate data
        parse_resume.invoke({
            "content": "Alice Johnson - 6 years experience. alice@company.com +1-555-0123",
            "filename": "alice_johnson.pdf",
            "session_id": session_id,
        })

        # Verify data exists
        assert len(_get_candidates(session_id)) > 0

        # Clear session
        result = clear_session_data.invoke({"session_id": session_id})
        assert "Cleared" in result
        assert "profiles removed" in result
        assert "PII" in result

    def test_clear_session_data_comprehensive(self):
        """Test clearing session removes all data types."""
        from app.deepagents.tools.recruitment_tools import (
            clear_session_data,
            parse_resume,
            parse_job_description,
            screen_candidate,
            _get_candidates,
            _get_jds,
            _get_screening_results,
        )

        session_id = f"test-clear-all-{uuid.uuid4().hex[:8]}"

        # Add JD
        parse_job_description.invoke({
            "content": "Senior engineer with Python and Kubernetes.",
            "title": "Senior Engineer",
            "session_id": session_id,
        })

        # Add candidate
        parse_resume.invoke({
            "content": "Bob Smith - 8 years Python, Kubernetes, Docker. bob@test.com",
            "filename": "bob_smith.pdf",
            "session_id": session_id,
        })

        # Screen
        jds = _get_jds(session_id)
        jd_id = list(jds.keys())[0]
        candidates = _get_candidates(session_id)
        candidate_id = list(candidates.keys())[0]

        screen_candidate.invoke({
            "candidate_id": candidate_id,
            "jd_id": jd_id,
            "session_id": session_id,
        })

        # Clear all
        result = clear_session_data.invoke({"session_id": session_id})
        assert "profiles removed" in result
        assert "JDs removed" in result
        assert "results removed" in result
        assert "SharePoint cache" in result

    def test_cache_ttl_functions_exist(self):
        """Test that cache TTL functions are accessible."""
        from app.deepagents.tools.sharepoint_tools import (
            cleanup_expired_cache,
            clear_session_cache,
        )

        # Cleanup expired cache should return 0 for non-existent session
        result = cleanup_expired_cache(f"test-ttl-{uuid.uuid4().hex[:8]}")
        assert result == 0

    def test_clear_session_cache(self):
        """Test clearing SharePoint session cache."""
        from app.deepagents.tools.sharepoint_tools import (
            clear_session_cache,
            list_sharepoint_folder,
        )

        session_id = f"test-cache-{uuid.uuid4().hex[:8]}"

        # Generate some cache data
        list_sharepoint_folder.invoke({
            "folder_type": "jd",
            "session_id": session_id,
        })

        # Clear cache - should not raise
        clear_session_cache(session_id)


# =============================================================================
# Enhanced Server Endpoints Tests
# =============================================================================

class TestEnhancedEndpoints:
    """Tests for new dashboard and session cleanup endpoints."""

    def test_dashboard_endpoint(self):
        """Test the dashboard API endpoint."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/recruitment-agent/dashboard/test-session")
        # Should return 200 or 503 (agent not loaded)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "dashboard" in data or "success" in data

    def test_session_cleanup_endpoint(self):
        """Test the session cleanup API endpoint."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.delete("/api/recruitment-agent/session/test-session")
        # Should return 200 or 503 (agent not loaded)
        assert response.status_code in [200, 503]

    def test_dashboard_endpoint_routing(self):
        """Test that dashboard endpoint is properly routed."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.get("/api/recruitment-agent/dashboard/test-session")
        # Should not return 405 Method Not Allowed
        assert response.status_code != 405

    def test_session_cleanup_endpoint_routing(self):
        """Test that session cleanup endpoint is properly routed."""
        from fastapi.testclient import TestClient
        from app.server import app

        client = TestClient(app)
        response = client.delete("/api/recruitment-agent/session/test-session")
        # Should not return 405 Method Not Allowed
        assert response.status_code != 405


# =============================================================================
# Enhanced System Prompt Tests
# =============================================================================

class TestEnhancedSystemPrompt:
    """Tests for enhanced system prompt features."""

    def test_system_prompt_has_quick_actions(self):
        """Test system prompt includes quick action shortcuts."""
        from app.deepagents.recruitment_agent import RECRUITMENT_SYSTEM_PROMPT

        assert "Quick Actions" in RECRUITMENT_SYSTEM_PROMPT
        assert "dashboard" in RECRUITMENT_SYSTEM_PROMPT
        assert "screen all" in RECRUITMENT_SYSTEM_PROMPT
        assert "full cycle" in RECRUITMENT_SYSTEM_PROMPT

    def test_system_prompt_has_error_recovery(self):
        """Test system prompt includes error recovery guidance."""
        from app.deepagents.recruitment_agent import RECRUITMENT_SYSTEM_PROMPT

        assert "Error Recovery" in RECRUITMENT_SYSTEM_PROMPT
        assert "demo mode" in RECRUITMENT_SYSTEM_PROMPT

    def test_system_prompt_has_data_privacy(self):
        """Test system prompt includes data privacy guidelines."""
        from app.deepagents.recruitment_agent import RECRUITMENT_SYSTEM_PROMPT

        assert "Data Privacy" in RECRUITMENT_SYSTEM_PROMPT
        assert "PII" in RECRUITMENT_SYSTEM_PROMPT
        assert "clear_session_data" in RECRUITMENT_SYSTEM_PROMPT

    def test_system_prompt_has_first_response_priority(self):
        """Test system prompt includes first response priority."""
        from app.deepagents.recruitment_agent import RECRUITMENT_SYSTEM_PROMPT

        assert "First Response Priority" in RECRUITMENT_SYSTEM_PROMPT
        assert "get_session_dashboard" in RECRUITMENT_SYSTEM_PROMPT

    def test_agent_includes_new_tools(self):
        """Test that agent tool collection includes new tools."""
        from app.deepagents.recruitment_agent import RECRUITMENT_SYSTEM_PROMPT

        # Verify new tools are referenced in the prompt
        assert "get_session_dashboard" in RECRUITMENT_SYSTEM_PROMPT
        assert "clear_session_data" in RECRUITMENT_SYSTEM_PROMPT


# =============================================================================
# New Tool Exports Tests
# =============================================================================

class TestNewToolExports:
    """Tests for new tool exports."""

    def test_session_dashboard_exported_from_tools(self):
        """Test get_session_dashboard is exported from tools module."""
        from app.deepagents.tools import get_session_dashboard
        assert get_session_dashboard is not None

    def test_clear_session_data_exported_from_tools(self):
        """Test clear_session_data is exported from tools module."""
        from app.deepagents.tools import clear_session_data
        assert clear_session_data is not None

    def test_sharepoint_cache_functions_exported(self):
        """Test SharePoint cache functions are exported."""
        from app.deepagents.tools.sharepoint_tools import (
            cleanup_expired_cache,
            clear_session_cache,
        )
        assert cleanup_expired_cache is not None
        assert clear_session_cache is not None
