"""Recruitment Deep Agent Configuration.

This module provides configurable parameters for the Recruitment Deep Agent,
including passing scores, SharePoint paths, skill weights, and evaluation criteria.

Following Enterprise Development Standards:
- Software Architect: Modular, extensible configuration
- Security Architect: No hardcoded secrets, environment-based settings
- Data Architect: Structured configuration with validation
- Software Engineer: Type-safe with Pydantic validation
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ScreeningLevel(str, Enum):
    """Candidate screening levels."""

    L1 = "L1"  # Entry-level / Junior
    L2 = "L2"  # Mid-level / Experienced
    L3 = "L3"  # Senior / Expert


class EvaluationCriteria(str, Enum):
    """Evaluation criteria for candidates."""

    TECHNICAL_SKILLS = "technical_skills"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    CERTIFICATIONS = "certifications"
    SOFT_SKILLS = "soft_skills"
    COMMUNICATION = "communication"
    PROBLEM_SOLVING = "problem_solving"
    LEADERSHIP = "leadership"


class QuestionDifficulty(str, Enum):
    """Question difficulty levels."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SharePointConfig(BaseModel):
    """SharePoint integration configuration."""

    # SharePoint site and paths
    site_url: str = Field(
        default_factory=lambda: os.getenv("SHAREPOINT_SITE_URL", ""),
        description="SharePoint site URL",
    )
    tenant_id: str = Field(
        default_factory=lambda: os.getenv("SHAREPOINT_TENANT_ID", ""),
        description="Azure AD tenant ID",
    )
    client_id: str = Field(
        default_factory=lambda: os.getenv("SHAREPOINT_CLIENT_ID", ""),
        description="Azure AD application client ID",
    )
    client_secret: str = Field(
        default_factory=lambda: os.getenv("SHAREPOINT_CLIENT_SECRET", ""),
        description="Azure AD application client secret",
    )

    # Folder structure
    jd_folder: str = Field(
        default="Recruitment/JobDescriptions",
        description="SharePoint folder for Job Descriptions",
    )
    resumes_folder: str = Field(
        default="Recruitment/Resumes",
        description="SharePoint folder for candidate resumes",
    )
    roles_folder: str = Field(
        default="Recruitment/RolesResponsibilities",
        description="SharePoint folder for roles and responsibilities",
    )
    interview_questions_folder: str = Field(
        default="Recruitment/InterviewQuestions",
        description="SharePoint folder for generated interview questions",
    )
    candidate_answers_folder: str = Field(
        default="Recruitment/CandidateAnswers",
        description="SharePoint folder for candidate answer submissions",
    )
    scoring_folder: str = Field(
        default="Recruitment/Scoring",
        description="SharePoint folder for scoring Excel files",
    )
    shortlist_folder: str = Field(
        default="Recruitment/Shortlisted",
        description="SharePoint folder for shortlisted candidates",
    )


class ScoringConfig(BaseModel):
    """Scoring and passing criteria configuration."""

    # Passing scores by level (percentage)
    l1_passing_score: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Minimum passing score for L1 (Junior) candidates",
    )
    l2_passing_score: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Minimum passing score for L2 (Mid-level) candidates",
    )
    l3_passing_score: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Minimum passing score for L3 (Senior) candidates",
    )

    # Resume screening thresholds
    resume_screening_threshold: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Minimum resume match score to proceed to screening",
    )

    # Interview score weights
    technical_weight: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Weight for technical assessment score",
    )
    experience_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for experience match score",
    )
    education_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for education match score",
    )
    soft_skills_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for soft skills assessment",
    )
    certification_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for certifications match score",
    )

    def get_passing_score(self, level: ScreeningLevel) -> float:
        """Get passing score for a specific level.

        Args:
            level: The screening level.

        Returns:
            The passing score percentage for that level.
        """
        scores = {
            ScreeningLevel.L1: self.l1_passing_score,
            ScreeningLevel.L2: self.l2_passing_score,
            ScreeningLevel.L3: self.l3_passing_score,
        }
        return scores.get(level, self.l1_passing_score)


class InterviewConfig(BaseModel):
    """Interview question configuration."""

    # Question counts per level
    l1_question_count: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of questions for L1 candidates",
    )
    l2_question_count: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Number of questions for L2 candidates",
    )
    l3_question_count: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of questions for L3 candidates",
    )

    # Question type distribution (percentages)
    mcq_percentage: int = Field(
        default=40,
        ge=0,
        le=100,
        description="Percentage of multiple choice questions",
    )
    coding_percentage: int = Field(
        default=30,
        ge=0,
        le=100,
        description="Percentage of coding questions",
    )
    scenario_percentage: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Percentage of scenario-based questions",
    )
    short_answer_percentage: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Percentage of short answer questions",
    )

    # Difficulty distribution per level
    l1_difficulty_distribution: dict[str, float] = Field(
        default={
            "basic": 0.50,
            "intermediate": 0.40,
            "advanced": 0.10,
            "expert": 0.00,
        },
        description="Question difficulty distribution for L1",
    )
    l2_difficulty_distribution: dict[str, float] = Field(
        default={
            "basic": 0.20,
            "intermediate": 0.40,
            "advanced": 0.30,
            "expert": 0.10,
        },
        description="Question difficulty distribution for L2",
    )
    l3_difficulty_distribution: dict[str, float] = Field(
        default={
            "basic": 0.05,
            "intermediate": 0.25,
            "advanced": 0.40,
            "expert": 0.30,
        },
        description="Question difficulty distribution for L3",
    )

    # Time limits (minutes)
    l1_time_limit: int = Field(default=45, description="Time limit for L1 assessment")
    l2_time_limit: int = Field(default=60, description="Time limit for L2 assessment")
    l3_time_limit: int = Field(default=90, description="Time limit for L3 assessment")

    def get_question_count(self, level: ScreeningLevel) -> int:
        """Get question count for a specific level."""
        counts = {
            ScreeningLevel.L1: self.l1_question_count,
            ScreeningLevel.L2: self.l2_question_count,
            ScreeningLevel.L3: self.l3_question_count,
        }
        return counts.get(level, self.l1_question_count)

    def get_difficulty_distribution(self, level: ScreeningLevel) -> dict[str, float]:
        """Get difficulty distribution for a specific level."""
        distributions = {
            ScreeningLevel.L1: self.l1_difficulty_distribution,
            ScreeningLevel.L2: self.l2_difficulty_distribution,
            ScreeningLevel.L3: self.l3_difficulty_distribution,
        }
        return distributions.get(level, self.l1_difficulty_distribution)


class ResumeParsingConfig(BaseModel):
    """Resume parsing configuration."""

    # Supported file formats
    supported_formats: list[str] = Field(
        default=[".pdf", ".docx", ".doc", ".txt", ".rtf"],
        description="Supported resume file formats",
    )

    # Skill extraction
    min_skill_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for skill extraction",
    )

    # Experience parsing
    min_experience_years: int = Field(
        default=0,
        ge=0,
        description="Minimum years of experience for L1",
    )
    l2_min_experience: int = Field(
        default=3,
        ge=0,
        description="Minimum years of experience for L2",
    )
    l3_min_experience: int = Field(
        default=7,
        ge=0,
        description="Minimum years of experience for L3",
    )


class NotificationConfig(BaseModel):
    """Notification and email configuration."""

    send_email_notifications: bool = Field(
        default=True,
        description="Enable email notifications for candidates and recruiters",
    )
    email_sender: str = Field(
        default_factory=lambda: os.getenv("RECRUITMENT_EMAIL_SENDER", ""),
        description="Email sender address",
    )
    recruiter_emails: list[str] = Field(
        default_factory=list,
        description="List of recruiter email addresses for notifications",
    )
    email_templates_folder: str = Field(
        default="Recruitment/EmailTemplates",
        description="SharePoint folder for email templates",
    )


class RecruitmentAgentConfig(BaseModel):
    """Complete configuration for Recruitment Deep Agent.

    This configuration can be loaded from environment variables,
    a JSON/YAML file, or passed directly.
    """

    # Agent metadata
    agent_name: str = Field(
        default="Recruitment Deep Agent",
        description="Name of the recruitment agent",
    )
    version: str = Field(default="1.0.0", description="Configuration version")

    # Sub-configurations
    sharepoint: SharePointConfig = Field(default_factory=SharePointConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    interview: InterviewConfig = Field(default_factory=InterviewConfig)
    resume_parsing: ResumeParsingConfig = Field(default_factory=ResumeParsingConfig)
    notification: NotificationConfig = Field(default_factory=NotificationConfig)

    # Processing settings
    max_concurrent_evaluations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent resume evaluations",
    )
    auto_shortlist: bool = Field(
        default=True,
        description="Automatically shortlist candidates meeting passing criteria",
    )
    require_human_approval: bool = Field(
        default=True,
        description="Require human approval before finalizing shortlist",
    )

    # Storage settings
    local_storage_path: str = Field(
        default="./data/recruitment",
        description="Local storage path for cached documents",
    )

    @classmethod
    def from_file(cls, file_path: str | Path) -> "RecruitmentAgentConfig":
        """Load configuration from a JSON or YAML file.

        Args:
            file_path: Path to the configuration file.

        Returns:
            RecruitmentAgentConfig instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is unsupported.
        """
        import json

        path = Path(file_path)
        if not path.exists():
            msg = f"Configuration file not found: {path}"
            raise FileNotFoundError(msg)

        with open(path) as f:
            if path.suffix in [".json"]:
                data = json.load(f)
            elif path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    msg = "PyYAML required for YAML config files: pip install pyyaml"
                    raise ValueError(msg)
            else:
                msg = f"Unsupported config format: {path.suffix}"
                raise ValueError(msg)

        return cls(**data)

    def to_file(self, file_path: str | Path) -> None:
        """Save configuration to a JSON or YAML file.

        Args:
            file_path: Path to save the configuration.
        """
        import json

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            if path.suffix in [".json"]:
                json.dump(self.model_dump(), f, indent=2)
            elif path.suffix in [".yaml", ".yml"]:
                try:
                    import yaml
                    yaml.dump(self.model_dump(), f, default_flow_style=False)
                except ImportError:
                    msg = "PyYAML required for YAML config files: pip install pyyaml"
                    raise ValueError(msg)

    def get_level_for_experience(self, years: int) -> ScreeningLevel:
        """Determine screening level based on years of experience.

        Args:
            years: Years of experience.

        Returns:
            Appropriate screening level.
        """
        if years >= self.resume_parsing.l3_min_experience:
            return ScreeningLevel.L3
        elif years >= self.resume_parsing.l2_min_experience:
            return ScreeningLevel.L2
        else:
            return ScreeningLevel.L1


# Default configuration instance
_default_config: RecruitmentAgentConfig | None = None


def get_recruitment_config() -> RecruitmentAgentConfig:
    """Get the recruitment agent configuration.

    Loads from file if RECRUITMENT_CONFIG_PATH is set,
    otherwise uses default configuration.

    Returns:
        RecruitmentAgentConfig instance.
    """
    global _default_config

    if _default_config is None:
        config_path = os.getenv("RECRUITMENT_CONFIG_PATH")
        if config_path and Path(config_path).exists():
            _default_config = RecruitmentAgentConfig.from_file(config_path)
        else:
            _default_config = RecruitmentAgentConfig()

    return _default_config


def set_recruitment_config(config: RecruitmentAgentConfig) -> None:
    """Set the recruitment agent configuration.

    Args:
        config: Configuration to set as default.
    """
    global _default_config
    _default_config = config


def update_recruitment_config(**kwargs: Any) -> RecruitmentAgentConfig:
    """Update specific configuration values.

    Args:
        **kwargs: Configuration values to update.

    Returns:
        Updated configuration instance.
    """
    global _default_config

    current = get_recruitment_config()
    updated_data = current.model_dump()

    # Update nested configurations
    for key, value in kwargs.items():
        if "." in key:
            # Handle nested keys like "scoring.l1_passing_score"
            parts = key.split(".")
            target = updated_data
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        else:
            updated_data[key] = value

    _default_config = RecruitmentAgentConfig(**updated_data)
    return _default_config


__all__ = [
    "ScreeningLevel",
    "EvaluationCriteria",
    "QuestionDifficulty",
    "SharePointConfig",
    "ScoringConfig",
    "InterviewConfig",
    "ResumeParsingConfig",
    "NotificationConfig",
    "RecruitmentAgentConfig",
    "get_recruitment_config",
    "set_recruitment_config",
    "update_recruitment_config",
]
