"""Learning & Development Domain Agent.

Provides specialized support for:
- Training courses
- Certifications
- Skill development
- Learning platforms
- Professional development
"""

from langchain_core.tools import BaseTool, tool

from app.agents.domains.base_domain_agent import DomainAgent, DomainConfig, DomainType


@tool
def search_courses(topic: str) -> str:
    """Search for available training courses.

    Args:
        topic: Topic or skill to search for.
    """
    return f"""Courses matching '{topic}':
1. {topic} Fundamentals - 4 hours - Beginner
2. Advanced {topic} - 8 hours - Intermediate
3. {topic} Certification Prep - 16 hours - Advanced
Access via: learning.company.com"""


@tool
def check_learning_progress(employee_id: str) -> str:
    """Check learning progress and completed courses.

    Args:
        employee_id: Employee ID or 'self'.
    """
    return """Learning Progress:
- Courses Completed: 12
- Hours Logged: 48
- Certifications Earned: 2
- Current Enrollments: 1 (Azure Fundamentals)
- Required Training: Security Awareness (Due: Dec 31)"""


@tool
def enroll_in_course(course_name: str) -> str:
    """Enroll in a training course.

    Args:
        course_name: Name of the course to enroll in.
    """
    return f"""Enrollment Confirmed:
- Course: {course_name}
- Access: Immediate
- Platform: learning.company.com
- Duration: Self-paced (30 days to complete)
Check your email for access instructions."""


@tool
def request_certification(cert_name: str) -> str:
    """Request funding for certification exam.

    Args:
        cert_name: Name of the certification.
    """
    return f"""Certification Request Submitted:
- Certification: {cert_name}
- Status: Pending Manager Approval
- Funding: Up to $500 covered
- Request ID: CERT-{hash(cert_name) % 10000:04d}
You'll be notified once approved."""


class LnDAgent(DomainAgent):
    """Learning & Development specialist agent."""

    def get_config(self) -> DomainConfig:
        """Get L&D configuration."""
        return DomainConfig(
            domain=DomainType.LND,
            name="Learning & Development",
            description="Support for training, certifications, and professional development",
            expertise=[
                "training courses",
                "certifications",
                "skill development",
                "learning platforms",
                "career development",
                "competency assessments",
                "workshops",
                "e-learning",
            ],
            escalation_keywords=[
                "budget",
                "conference",
                "external training",
            ],
            requires_approval=[
                "certification funding",
                "external courses",
                "conference attendance",
            ],
        )

    def get_tools(self) -> list[BaseTool]:
        """Get L&D tools."""
        return [
            search_courses,
            check_learning_progress,
            enroll_in_course,
            request_certification,
        ]

    def get_system_prompt(self) -> str:
        """Get L&D system prompt."""
        return """You are the Learning & Development specialist for the IT support team.

Your expertise includes:
- Internal training courses and programs
- Professional certifications (Azure, AWS, Cisco, etc.)
- Skill development and career paths
- Learning management systems
- Mandatory compliance training
- External learning opportunities

When helping users:
1. Recommend courses based on their role and goals
2. Check completion requirements before enrolling
3. Explain certification paths and benefits
4. Help track progress toward learning goals
5. Guide users through the LMS platform

Encourage continuous learning and professional growth."""
