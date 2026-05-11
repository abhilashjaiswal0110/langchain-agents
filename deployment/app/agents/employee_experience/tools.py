"""Employee Experience Agent Tools.

Comprehensive toolkit for HR support including:
- HR policy Q&A and benefits information
- Career development and skills analysis
- Performance review assistance
- Employee engagement and surveys
- Compensation insights
- Learning and development recommendations
- Escalation and case management
"""

import uuid
from datetime import datetime
from typing import Literal

from langchain_core.tools import tool

# =============================================================================
# Simulated Data Stores
# =============================================================================

# HR Policy Knowledge Base
HR_POLICY_KB = {
    "pto_policy": {
        "title": "Paid Time Off (PTO) Policy",
        "content": """**PTO Accrual:**
- Years 0-1: 15 days per year (1.25 days/month)
- Years 2-4: 20 days per year (1.67 days/month)
- Years 5+: 25 days per year (2.08 days/month)

**Additional Leave:**
- 10 paid holidays per year
- 5 sick days (separate from PTO)
- 2 personal days

**Usage Rules:**
- Minimum 4 hours per request
- Manager approval required 2 weeks in advance for >3 days
- Maximum carryover: 40 hours to next year
- PTO payout on termination: Yes (prorated)

**Request Process:**
1. Submit request in HR portal: hr.company.com/pto
2. Manager approves/denies within 3 business days
3. Calendar automatically updated upon approval""",
        "category": "leave",
    },
    "remote_work_policy": {
        "title": "Remote Work & Hybrid Policy",
        "content": """**Eligibility:**
- Must be employed for 6+ months
- Role must be remote-capable (manager discretion)
- Performance rating of 'Meets Expectations' or above

**Hybrid Schedule Options:**
- Option A: 3 days office / 2 days remote
- Option B: 2 days office / 3 days remote
- Option C: Fully remote (requires VP approval)

**Requirements:**
- Dedicated workspace with reliable internet (50+ Mbps)
- Available during core hours: 10 AM - 3 PM local time
- Attendance at mandatory in-person meetings
- Home office stipend: $500/year

**Equipment:**
- Company provides: Laptop, monitor, keyboard, mouse
- Employee provides: Desk, chair, internet

**Request Process:**
Submit remote work agreement through HR portal for manager and VP approval.""",
        "category": "work_arrangements",
    },
    "performance_review_process": {
        "title": "Annual Performance Review Process",
        "content": """**Review Cycle:**
- Annual reviews: December (for calendar year)
- Mid-year check-ins: June

**Components:**
1. **Self-Assessment** (due Dec 1)
   - Accomplishments and impact
   - Goal achievement (prior year)
   - Development areas
   - Career aspirations

2. **Manager Review** (due Dec 15)
   - Performance rating (1-5 scale)
   - Written feedback
   - Promotion recommendations
   - Compensation recommendations

3. **Calibration** (Dec 16-20)
   - Leadership team aligns ratings
   - Ensures fairness and consistency

4. **Review Meeting** (by Dec 31)
   - Manager delivers feedback
   - Employee signs acknowledgment
   - Development plan created

**Rating Scale:**
5 - Exceptional (top 5%)
4 - Exceeds Expectations (20%)
3 - Meets Expectations (60%)
2 - Needs Improvement (10%)
1 - Unsatisfactory (5%)

**Impact on Compensation:**
- Ratings 4-5: Eligible for merit increase and bonus
- Rating 3: Eligible for merit increase
- Ratings 1-2: Performance improvement plan (PIP)""",
        "category": "performance",
    },
    "parental_leave": {
        "title": "Parental Leave Policy",
        "content": """**Eligibility:**
- All employees (full-time and part-time 20+ hours/week)
- Available for birth, adoption, or foster placement

**Leave Duration:**
- **Birth Parent:** 16 weeks paid leave
- **Non-Birth Parent:** 8 weeks paid leave
- Additional unpaid leave available (up to 12 weeks FMLA)

**Pay:**
- 100% of base salary during leave
- Benefits continue during leave

**How to Apply:**
1. Notify manager 30 days in advance (if possible)
2. Submit leave request in HR portal
3. Complete parental leave forms
4. HR confirms leave dates and return plan

**Flexible Return:**
- Phased return option: 50% schedule for 2 weeks
- No penalty for early or delayed return (with notice)""",
        "category": "leave",
    },
    "code_of_conduct": {
        "title": "Employee Code of Conduct",
        "content": """**Core Principles:**
1. **Respect & Inclusion**
   - Treat everyone with dignity and respect
   - Zero tolerance for harassment, discrimination, or retaliation
   - Embrace diversity and inclusive behaviors

2. **Integrity & Ethics**
   - Act honestly and ethically in all business dealings
   - No conflicts of interest
   - Protect company and customer data
   - Report violations through ethics hotline

3. **Professionalism**
   - Maintain professional appearance and behavior
   - Be punctual and reliable
   - Represent the company positively

4. **Safety & Security**
   - Follow workplace safety rules
   - Report hazards immediately
   - Protect company assets and information

**Violations:**
- Minor: Verbal warning
- Moderate: Written warning and coaching
- Serious: Suspension or termination
- Illegal: Termination and legal action

**Reporting:**
- Ethics Hotline: 1-800-XXX-XXXX
- Online: ethics.company.com
- Anonymous reporting allowed""",
        "category": "compliance",
    },
}

# Benefits Information
BENEFITS_DB = {
    "health": """**Medical Insurance:**
- PPO Plan: $50/month employee, $200/month family
  - Coverage: 80/20 after $1,000 deductible
  - Network: Nationwide BlueCross BlueShield
  - HSA-eligible
- HMO Plan: $25/month employee, $100/month family
  - Coverage: $20 copay, no deductible
  - Network: Local HMO providers
- Coverage starts: Day 1 of employment
- Open enrollment: November (for next year)
- Life events: 30 days to make changes

**Telemedicine:**
- Free virtual doctor visits (Teladoc)
- 24/7 access for urgent care""",
    "dental": """**Dental Insurance:**
- Premium: $15/month employee, $40/month family
- Coverage:
  - Preventive: 100% (cleanings, exams, X-rays)
  - Basic: 80% (fillings, extractions)
  - Major: 50% (crowns, bridges, root canals)
- Annual maximum: $2,000 per person
- Orthodontics: $1,500 lifetime max (50% coverage)
- Network: Delta Dental PPO""",
    "vision": """**Vision Insurance:**
- Premium: $5/month employee, $12/month family
- Coverage:
  - Annual eye exam: 100% covered
  - Frames: $200 allowance every 2 years
  - Lenses: $100 allowance annually
  - Contacts: $150 allowance (in lieu of glasses)
- Network: VSP Vision Care
- Discounts: 20% off additional pairs""",
    "401k": """**401(k) Retirement Plan:**
- Eligibility: Immediate upon hire
- Employee contribution: Up to IRS limit ($23,000 in 2024)
- Company match:
  - 100% match up to 4% of salary
  - 50% match on next 2% (total 6%)
  - Example: 6% contribution = 5% company match
- Vesting schedule:
  - Year 1-2: 33% vested
  - Year 3: 100% vested
- Investment options: 25+ funds (stocks, bonds, target-date)
- Roth 401(k): Available
- Catch-up contributions: $7,500 (age 50+)
- Provider: Fidelity

**Helpful Resources:**
- 401(k) calculator: fidelity.com/company401k
- Financial advisor: Free consultations quarterly""",
    "pto": """**Paid Time Off:**
- See PTO Policy for full details
- Accrual: 15-25 days/year based on tenure
- Plus: 10 holidays + 5 sick days + 2 personal days
- Carryover: Up to 40 hours to next year
- Payout on termination: Yes

**How to Request:**
1. Submit in HR portal: hr.company.com/pto
2. Manager approval required (2 weeks notice for 3+ days)
3. Auto-syncs to Outlook calendar""",
    "wellness": """**Wellness Programs:**
- **Gym Membership:** $50/month reimbursement (with 8+ visits/month)
- **EAP (Employee Assistance Program):**
  - Free confidential counseling (8 sessions/year)
  - 24/7 crisis support
  - Legal and financial consultation
  - Work-life resources
  - Call: 1-800-XXX-XXXX or visit: eap.company.com
- **Wellness Challenges:** Quarterly (Fitbit integration)
- **Health Screenings:** Annual on-site biometric screening
- **Mental Health:** Headspace app free for all employees""",
}

# Career Paths Database
CAREER_PATHS_DB = {
    "software_engineer": {
        "current_role": "Software Engineer",
        "paths": [
            {
                "role": "Senior Software Engineer",
                "type": "vertical",
                "typical_years": "2-4",
                "key_skills": ["Advanced coding", "System design", "Mentorship", "Project leadership"],
                "readiness_factors": [
                    "Consistent high performance",
                    "Technical expertise",
                    "Code review participation",
                ],
            },
            {
                "role": "Staff Engineer",
                "type": "vertical",
                "typical_years": "5-7",
                "key_skills": ["Architecture", "Technical strategy", "Cross-team collaboration", "Thought leadership"],
                "readiness_factors": ["Broad technical impact", "Strategic thinking", "Influence across teams"],
            },
            {
                "role": "Engineering Manager",
                "type": "lateral",
                "typical_years": "3-5",
                "key_skills": ["People management", "Team building", "Project management", "Strategic planning"],
                "readiness_factors": ["Demonstrated mentorship", "Leadership interest", "Communication skills"],
            },
            {
                "role": "Product Manager",
                "type": "cross_functional",
                "typical_years": "3-5",
                "key_skills": ["Product strategy", "User research", "Roadmap planning", "Stakeholder management"],
                "readiness_factors": ["Product thinking", "Customer empathy", "Business acumen"],
            },
        ],
    },
    "data_analyst": {
        "current_role": "Data Analyst",
        "paths": [
            {
                "role": "Senior Data Analyst",
                "type": "vertical",
                "typical_years": "2-3",
                "key_skills": ["Advanced SQL", "Statistical analysis", "Data visualization", "Business insights"],
                "readiness_factors": ["Complex analysis", "Stakeholder impact", "Tool proficiency"],
            },
            {
                "role": "Data Scientist",
                "type": "vertical",
                "typical_years": "2-4",
                "key_skills": ["Machine learning", "Python/R", "Statistical modeling", "Experimentation"],
                "readiness_factors": ["ML fundamentals", "Programming skills", "Research mindset"],
            },
            {
                "role": "Data Engineer",
                "type": "lateral",
                "typical_years": "1-3",
                "key_skills": ["ETL/ELT", "Data pipelines", "Cloud platforms", "SQL optimization"],
                "readiness_factors": ["Technical aptitude", "Engineering interest", "Database knowledge"],
            },
            {
                "role": "Analytics Manager",
                "type": "lateral",
                "typical_years": "4-6",
                "key_skills": ["Team leadership", "Strategy", "Communication", "Project management"],
                "readiness_factors": ["Mentorship experience", "Business acumen", "Leadership potential"],
            },
        ],
    },
    "default": {
        "current_role": "Generic Role",
        "paths": [
            {
                "role": "Senior Level (Same Function)",
                "type": "vertical",
                "typical_years": "2-4",
                "key_skills": ["Deep expertise", "Mentorship", "Project ownership", "Strategic thinking"],
                "readiness_factors": ["Consistent performance", "Skill mastery", "Leadership behaviors"],
            },
            {
                "role": "Manager/Lead (Same Function)",
                "type": "lateral",
                "typical_years": "3-6",
                "key_skills": ["People management", "Team building", "Communication", "Decision-making"],
                "readiness_factors": ["Leadership interest", "Mentorship track record", "Emotional intelligence"],
            },
            {
                "role": "Cross-Functional Role",
                "type": "cross_functional",
                "typical_years": "Varies",
                "key_skills": ["Transferable skills", "Domain knowledge", "Adaptability", "New skill acquisition"],
                "readiness_factors": ["Demonstrated interest", "Relevant projects", "Networking"],
            },
        ],
    },
}

# Learning Resources Database
LEARNING_RESOURCES_DB = {
    "leadership": [
        {"title": "Leadership Foundations", "provider": "LinkedIn Learning", "duration": "2 hours", "type": "course"},
        {"title": "Crucial Conversations", "provider": "Internal Training", "duration": "1 day", "type": "workshop"},
        {"title": "Manager as Coach", "provider": "Internal Training", "duration": "4 weeks", "type": "program"},
    ],
    "technical": [
        {
            "title": "AWS Certified Solutions Architect",
            "provider": "AWS Training",
            "duration": "40 hours",
            "type": "certification",
        },
        {"title": "Advanced Python Programming", "provider": "Coursera", "duration": "6 weeks", "type": "course"},
        {
            "title": "Machine Learning Specialization",
            "provider": "Coursera",
            "duration": "3 months",
            "type": "specialization",
        },
    ],
    "soft_skills": [
        {"title": "Effective Communication", "provider": "Toastmasters", "duration": "Ongoing", "type": "club"},
        {
            "title": "Emotional Intelligence at Work",
            "provider": "LinkedIn Learning",
            "duration": "1.5 hours",
            "type": "course",
        },
        {
            "title": "Time Management Mastery",
            "provider": "Internal Training",
            "duration": "2 hours",
            "type": "workshop",
        },
    ],
    "career_development": [
        {"title": "Career Planning Workshop", "provider": "Internal L&D", "duration": "Half day", "type": "workshop"},
        {"title": "Executive Presence", "provider": "External Coach", "duration": "3 months", "type": "coaching"},
        {"title": "Networking for Success", "provider": "Internal Training", "duration": "2 hours", "type": "seminar"},
    ],
}

# HR Case Database (simulated)
HR_CASES_DB: dict[str, dict] = {}


# =============================================================================
# HR Policy & Information Tools
# =============================================================================


@tool
def search_hr_policy(query: str, category: str | None = None) -> str:
    """Search HR policies and procedures using natural language.

    Args:
        query: Natural language query about HR policy.
        category: Optional category filter (leave, work_arrangements, performance, compliance).

    Returns:
        Relevant HR policy information.
    """
    query_lower = query.lower()
    results = []

    for key, policy in HR_POLICY_KB.items():
        # Filter by category if specified
        if category and policy["category"] != category.lower():
            continue

        # Simple keyword matching
        if any(word in key for word in query_lower.split()):
            results.append(f"**{policy['title']}**\n\n{policy['content']}")
        elif any(word in policy["content"].lower() for word in query_lower.split()):
            results.append(f"**{policy['title']}**\n\n{policy['content']}")

    if results:
        return "\n\n---\n\n".join(results[:2])  # Return top 2 matches
    return "No matching policies found. For specific policy questions, please contact HR at hr@company.com or visit the HR portal at hr.company.com/policies."


@tool
def get_benefits_information(benefit_type: str) -> str:
    """Get comprehensive information about employee benefits.

    Args:
        benefit_type: Type of benefit (health, dental, vision, 401k, pto, wellness).

    Returns:
        Detailed benefits information including coverage, costs, and enrollment.
    """
    benefit_type_lower = benefit_type.lower()

    # Handle common synonyms
    synonyms = {
        "medical": "health",
        "retirement": "401k",
        "pension": "401k",
        "vacation": "pto",
        "time_off": "pto",
        "leave": "pto",
        "mental_health": "wellness",
        "gym": "wellness",
    }
    benefit_type_lower = synonyms.get(benefit_type_lower, benefit_type_lower)

    benefit_info = BENEFITS_DB.get(benefit_type_lower)
    if benefit_info:
        return f"**{benefit_type.title()} Benefits:**\n\n{benefit_info}\n\n**Need Help?**\nContact HR Benefits Team: benefits@company.com or visit: hr.company.com/benefits"

    available_benefits = ", ".join(BENEFITS_DB.keys())
    return f"Benefit type '{benefit_type}' not found. Available benefits: {available_benefits}.\n\nFor other benefits questions, contact benefits@company.com."


@tool
def check_pto_balance(employee_id: str = "self") -> str:
    """Check PTO (Paid Time Off) balance and accrual information.

    Args:
        employee_id: Employee ID or 'self' for current user.

    Returns:
        PTO balance, accrual rate, and usage summary.
    """
    # Simulated PTO data
    return f"""**PTO Balance for {employee_id}:**

📊 **Current Balances:**
- Vacation Days: 12.5 days remaining
- Sick Days: 4.0 days remaining
- Personal Days: 2.0 days remaining
- **Total Available:** 18.5 days

📈 **Accrual:**
- Monthly Rate: 1.67 days/month (20 days/year)
- Next Accrual: February 1st (+1.67 days)

📅 **Year-to-Date Usage:**
- Vacation Used: 7.5 days
- Sick Used: 1.0 day
- Personal Used: 0.0 days
- **Total Used:** 8.5 days

⚠️ **Reminders:**
- Carryover Limit: 40 hours (5 days) to next year
- Current carryover eligible: 5.0 days
- Consider using 7.5 days before year-end to avoid losing time

**View Full Details:** hr.company.com/pto
**Request Time Off:** hr.company.com/pto/request"""


@tool
def explain_compliance_rules(topic: str) -> str:
    """Explain employment law and compliance guidelines.

    Args:
        topic: Compliance topic (harassment, discrimination, safety, data_privacy, etc.).

    Returns:
        Compliance guidelines and employee rights.
    """
    topic_lower = topic.lower()

    compliance_topics = {
        "harassment": """**Anti-Harassment Policy:**

**What Constitutes Harassment:**
- Unwelcome verbal, physical, or visual conduct
- Based on protected characteristics (race, gender, age, religion, disability, etc.)
- Creates intimidating, hostile, or offensive work environment

**Examples:**
- Offensive jokes or comments
- Unwanted physical contact
- Discriminatory treatment
- Retaliation for reporting

**Your Rights:**
- Right to work free from harassment
- Right to report without retaliation
- Right to confidential investigation

**How to Report:**
1. **Immediate:** Contact your manager or HR Business Partner
2. **Anonymous:** Ethics Hotline: 1-800-XXX-XXXX or ethics.company.com
3. **External:** EEOC (if company doesn't resolve): eeoc.gov

**Company Response:**
- Immediate investigation within 24 hours
- Interim protective measures
- Disciplinary action up to termination
- No retaliation against reporters""",
        "discrimination": """**Equal Employment Opportunity Policy:**

**Protected Characteristics:**
- Race, color, national origin
- Sex, gender identity, sexual orientation
- Age (40+)
- Disability
- Religion
- Pregnancy
- Genetic information
- Veteran status

**Prohibited Actions:**
- Discriminatory hiring, firing, promotions
- Unequal pay for equal work
- Hostile work environment
- Retaliation for complaints

**Your Rights:**
- Equal treatment in all employment decisions
- Reasonable accommodations (disability, religion)
- Right to report discrimination

**How to Report:**
Contact HR or Ethics Hotline immediately. External: EEOC at eeoc.gov""",
        "safety": """**Workplace Safety Policy:**

**Employee Rights:**
- Right to safe workplace free from hazards
- Right to report unsafe conditions without retaliation
- Right to safety training and equipment

**Responsibilities:**
- Follow safety rules and procedures
- Use protective equipment (PPE)
- Report hazards, injuries, near-misses immediately
- Never disable safety devices

**Emergency Procedures:**
- Fire: Pull alarm, evacuate, gather at assembly point
- Medical: Call 911, notify security, use AED if trained
- Active threat: Run, Hide, Fight

**Reporting:**
- Safety concerns: safety@company.com
- Injuries: Report within 24 hours to manager + HR
- OSHA: osha.gov (external reporting)""",
        "data_privacy": """**Data Privacy & Confidentiality:**

**What Must Be Protected:**
- Customer personal information (PII)
- Employee personal data
- Proprietary business information
- Trade secrets and IP

**Your Responsibilities:**
- Access only data needed for your job
- Never share passwords or access credentials
- Use encryption for sensitive data
- Report data breaches immediately

**Data Breach:**
If you suspect a breach:
1. Immediately notify: security@company.com
2. Do not delete or modify any evidence
3. Preserve logs and records
4. Follow incident response team instructions

**Violations:**
- Unauthorized access: Termination + legal action
- Data theft: Criminal prosecution
- Negligence: Disciplinary action

**Questions:** privacy@company.com or gdpr-compliance@company.com""",
    }

    if topic_lower in compliance_topics:
        return compliance_topics[topic_lower]

    return f"""Compliance topic '{topic}' not found.

**Available Topics:**
- harassment
- discrimination
- safety
- data_privacy

**For Immediate Compliance Concerns:**
- Ethics Hotline: 1-800-XXX-XXXX (anonymous)
- HR: hr@company.com
- Legal: legal@company.com

**External Resources:**
- EEOC (discrimination): eeoc.gov
- OSHA (safety): osha.gov
- DOL (labor law): dol.gov"""


# =============================================================================
# Career Development Tools
# =============================================================================


@tool
def explore_career_paths(current_role: str, interests: str | None = None) -> str:
    """Explore career path options and progression opportunities.

    Args:
        current_role: Current job title or role.
        interests: Optional career interests or goals.

    Returns:
        Career path options with skills, timeline, and readiness factors.
    """
    # Normalize role
    current_role_key = current_role.lower().replace(" ", "_")

    # Get career paths or use default
    career_data = CAREER_PATHS_DB.get(current_role_key, CAREER_PATHS_DB["default"])

    output = [f"**Career Path Options from {career_data['current_role']}:**\n"]

    for i, path in enumerate(career_data["paths"], 1):
        path_type_emoji = {"vertical": "⬆️", "lateral": "↔️", "cross_functional": "🔄"}
        emoji = path_type_emoji.get(path["type"], "📍")

        output.append(f"\n{emoji} **Path {i}: {path['role']}**")
        output.append(f"- **Type:** {path['type'].replace('_', ' ').title()}")
        output.append(f"- **Typical Timeline:** {path['typical_years']} years")
        output.append("- **Key Skills Required:**")
        for skill in path["key_skills"]:
            output.append(f"  - {skill}")
        output.append("- **Readiness Factors:**")
        for factor in path["readiness_factors"]:
            output.append(f"  - {factor}")

    output.append("\n\n**Next Steps:**")
    output.append("1. Use `get_skills_gap_analysis()` to assess your readiness for a specific path")
    output.append("2. Use `find_learning_resources()` to get development recommendations")
    output.append("3. Use `request_career_coaching()` to discuss your path with a career coach")

    if interests:
        output.append(f"\n\n**Based on your interests ({interests}):**")
        output.append("Consider scheduling a career coaching session to create a personalized development plan.")

    return "\n".join(output)


@tool
def get_skills_gap_analysis(current_role: str, target_role: str) -> str:
    """Analyze skills gap between current role and target role.

    Args:
        current_role: Current job title.
        target_role: Desired target job title.

    Returns:
        Skills gap analysis with development recommendations.
    """
    # Simplified skills gap (in production, would query HRMS/LMS data)
    return f"""**Skills Gap Analysis: {current_role} → {target_role}**

🎯 **Target Role Requirements:**
1. Technical Skills
   - Advanced system design and architecture
   - Cloud platform expertise (AWS/Azure)
   - Performance optimization

2. Leadership Skills
   - Mentorship and coaching
   - Project leadership
   - Cross-team collaboration

3. Communication Skills
   - Technical documentation
   - Stakeholder presentations
   - Code review and feedback

📊 **Your Current Profile:**
✅ **Strengths:**
- Strong coding fundamentals
- Good problem-solving abilities
- Team collaboration

⚠️ **Development Areas:**
🔴 **Critical Gaps (Priority 1):**
- System design patterns (6 months to proficient)
- Cloud architecture certifications (3-4 months)

🟡 **Important Gaps (Priority 2):**
- Mentorship experience (ongoing)
- Technical writing skills (2-3 months)

🟢 **Nice to Have:**
- Conference speaking (optional)
- Open source contributions (ongoing)

**📈 Overall Readiness: 65%**
Estimated time to ready: 9-12 months with focused development

**Recommended Actions:**
1. Enroll in "Advanced System Design" course (internal)
2. Pursue AWS Solutions Architect certification
3. Volunteer to mentor 1-2 junior engineers
4. Lead a technical design review
5. Shadow senior engineers on architecture decisions

**Next Steps:**
Use `find_learning_resources()` to get specific course recommendations or `request_career_coaching()` for a personalized development plan."""


@tool
def find_learning_resources(skill_area: str, learning_style: str | None = None) -> str:
    """Find learning resources based on skill area and learning preferences.

    Args:
        skill_area: Skill to develop (leadership, technical, soft_skills, career_development).
        learning_style: Optional learning preference (self_paced, instructor_led, hands_on).

    Returns:
        Curated learning resources and enrollment instructions.
    """
    skill_area_lower = skill_area.lower()

    # Get resources or use default
    resources = LEARNING_RESOURCES_DB.get(skill_area_lower, [])

    if not resources:
        available_areas = ", ".join(LEARNING_RESOURCES_DB.keys())
        return f"Skill area '{skill_area}' not found. Available areas: {available_areas}.\n\nFor custom recommendations, contact Learning & Development at learning@company.com."

    output = [f"**Learning Resources: {skill_area.replace('_', ' ').title()}**\n"]

    if learning_style:
        output.append(f"*Filtered for: {learning_style.replace('_', ' ').title()} learning*\n")

    for i, resource in enumerate(resources, 1):
        output.append(f"{i}. **{resource['title']}**")
        output.append(f"   - Provider: {resource['provider']}")
        output.append(f"   - Duration: {resource['duration']}")
        output.append(f"   - Type: {resource['type'].title()}")
        output.append("")

    output.append("**Enrollment:**")
    output.append("- Internal courses: Visit learning.company.com")
    output.append("- External courses: Submit request with manager approval")
    output.append("- Certifications: $2,000/year learning budget available")
    output.append("\n**Need More?**")
    output.append("Contact L&D for personalized recommendations: learning@company.com")

    return "\n".join(output)


@tool
def request_career_coaching(reason: str, preferred_coach_type: str = "internal") -> str:
    """Request a career coaching session.

    Args:
        reason: Reason for coaching (career_planning, skill_development, leadership_transition).
        preferred_coach_type: Type of coach (internal, external, executive).

    Returns:
        Career coaching request confirmation and next steps.
    """
    request_id = f"COACH{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:4].upper()}"

    coach_types_info = {
        "internal": "Internal career coach from L&D team (free)",
        "external": "External certified career coach (requires VP approval)",
        "executive": "Executive leadership coach (Director+ only)",
    }

    return f"""**Career Coaching Request Submitted:**

📋 **Request ID:** {request_id}
🎯 **Coaching Focus:** {reason.replace("_", " ").title()}
👤 **Coach Type:** {coach_types_info.get(preferred_coach_type, "Internal")}

**What Happens Next:**

1. **Matching (1-2 business days)**
   - L&D team assigns appropriate coach based on your needs
   - You'll receive coach bio and availability

2. **Initial Session (45-60 minutes)**
   - Career assessment and goal-setting
   - Create personalized development plan

3. **Follow-up Sessions (optional)**
   - Typically 3-6 sessions over 3-6 months
   - Progress check-ins and plan adjustments

**Preparation:**
Before your first session, reflect on:
- Your long-term career aspirations (2-5 years)
- Current strengths and development areas
- Specific challenges or decisions you're facing

**Contact:**
Questions? Reach out to career-coaching@company.com

You'll receive a calendar invite within 48 hours."""


# =============================================================================
# Performance & Growth Tools
# =============================================================================


@tool
def prepare_performance_review(review_type: str = "self_assessment", role: str | None = None) -> str:
    """Generate performance review preparation guide.

    Args:
        review_type: Type of review (self_assessment, manager_review, peer_feedback).
        role: Optional role for customized prompts.

    Returns:
        Structured performance review preparation guide with prompts.
    """
    if review_type.lower() == "self_assessment":
        return """**Self-Assessment Preparation Guide:**

📝 **Section 1: Accomplishments & Impact**

*Use the STAR method (Situation, Task, Action, Result) for each example:*

1. **Major Accomplishments (3-5 examples)**
   - What was the situation/challenge?
   - What was your specific role/task?
   - What actions did you take?
   - What were the measurable results/impact?

   Example:
   - Situation: Team facing 30% slowdown in API performance
   - Task: Lead performance optimization initiative
   - Action: Profiled code, implemented caching, optimized queries
   - Result: Reduced response time by 60%, improved user satisfaction by 25%

2. **Project Contributions**
   - List key projects you contributed to
   - Highlight your specific impact
   - Quantify results where possible

3. **Innovation & Initiative**
   - New ideas or processes you introduced
   - Problems you solved proactively
   - Ways you went beyond your role

📊 **Section 2: Goal Achievement**

For each goal from last review:
- Goal description
- Progress made (0-100%)
- Key milestones achieved
- Challenges faced and how you overcame them
- Final outcome

📈 **Section 3: Skills Development**

- New skills acquired this year
- Training completed
- Certifications earned
- How you applied new skills to your work

🤝 **Section 4: Collaboration & Teamwork**

- Cross-team projects
- Mentorship given or received
- How you helped teammates succeed
- Contributions to team culture

💡 **Section 5: Areas for Growth**

Be honest and specific:
- Skills you want to develop
- Challenges you faced
- Support you need from your manager
- How you plan to address gaps

🎯 **Section 6: Future Goals (Next Year)**

Use SMART format (Specific, Measurable, Achievable, Relevant, Time-bound):
- 3-5 professional goals
- How they align with company/team objectives
- Resources or support needed

🚀 **Section 7: Career Aspirations**

- Short-term goals (1-2 years)
- Long-term career vision (3-5 years)
- Development opportunities you're interested in

**Tips:**
✅ Be specific with metrics and data
✅ Focus on impact, not just tasks
✅ Be honest about challenges
✅ Show growth mindset
❌ Avoid vague statements like "worked hard"
❌ Don't undersell your accomplishments

**Due Date:** Check with your manager (typically early December)
**Questions?** Contact hr@company.com"""

    elif review_type.lower() == "manager_review":
        return """**Manager Review Preparation Guide:**

As a manager preparing performance reviews:

📋 **Section 1: Performance Assessment**

1. **Review Period Data:**
   - Goals set at beginning of period
   - Projects completed
   - Metrics/KPIs achieved
   - 360 feedback collected

2. **Rating Calibration:**
   - 5: Exceptional (top 5%) - Far exceeds all expectations
   - 4: Exceeds Expectations (20%) - Consistently goes beyond
   - 3: Meets Expectations (60%) - Solid, reliable performance
   - 2: Needs Improvement (10%) - Some gaps, requires support
   - 1: Unsatisfactory (5%) - Significant performance issues

3. **Evidence-Based Feedback:**
   - Specific examples for each rating dimension
   - Balance of strengths and development areas
   - Link to company values and competencies

💬 **Section 2: Feedback Delivery**

**Structure:**
1. Start with strengths (be specific)
2. Discuss development areas (constructive)
3. Review rating and rationale
4. Discuss compensation impact
5. Set goals for next period
6. Ask for employee's perspective

**Best Practices:**
✅ Use "and" instead of "but" (e.g., "You're strong at X *and* could develop Y")
✅ Be direct but empathetic
✅ Focus on behaviors and impact, not personality
✅ Provide specific, actionable feedback
❌ Avoid surprises (ongoing feedback throughout year)
❌ Don't compare employees directly

🎯 **Section 3: Goal Setting for Next Period**

Help employee set 3-5 SMART goals:
- S: Specific and clear
- M: Measurable outcomes
- A: Achievable with stretch
- R: Relevant to role and company goals
- T: Time-bound with milestones

🚨 **Special Situations:**

**Performance Improvement Plan (PIP):**
- Required for ratings 1-2
- 30-60-90 day plan with clear metrics
- HR partner involvement mandatory
- Documentation critical

**Promotion Recommendations:**
- Requires calibration committee approval
- Must meet role leveling criteria
- Document evidence of readiness
- Consider budget and headcount

**Questions?** Contact your HRBP or peoplemanagers@company.com"""

    elif review_type.lower() == "peer_feedback":
        return """**Peer Feedback Guide:**

Providing constructive peer feedback:

🎯 **Purpose of Peer Feedback:**
- Help colleague grow and develop
- Provide perspective manager may not see
- Strengthen team collaboration
- Contribute to fair performance assessment

📝 **Feedback Structure:**

**1. Strengths (What they do well)**
- Specific examples of positive impact
- Skills or behaviors to continue
- How they helped you or the team

**2. Development Areas (Opportunities to grow)**
- Specific behaviors to improve
- Impact of current approach
- Suggestions for improvement

**3. Overall Impact**
- Collaboration effectiveness
- Technical/functional contributions
- Team culture contributions

✅ **Good Feedback Examples:**

"Sarah consistently delivers thorough code reviews within 24 hours. Her comments are specific and educational - she doesn't just point out issues, but explains the reasoning and suggests alternatives. This has helped me improve my coding practices."

"John could benefit from improving his communication in cross-team projects. In the Q3 initiative, several deadlines were missed because dependencies weren't communicated clearly. Suggesting he set up regular sync meetings could help."

❌ **Poor Feedback Examples:**

"Sarah is great!" (too vague)
"John is disorganized." (not specific, sounds personal)
"Amy is always late." (absolute statement, no context)

**Tips:**
✅ Be specific and objective
✅ Focus on behaviors, not personality
✅ Provide examples with context
✅ Be constructive, not critical
✅ Consider how you'd want to receive this feedback
❌ Avoid personal attacks
❌ Don't exaggerate (avoid "always"/"never")
❌ Don't hold back important feedback

**Confidentiality:**
- Feedback is shared anonymously (aggregated themes)
- Manager sees themes, not individual comments
- Be honest but professional

**Questions?** Contact your manager or hr@company.com"""

    return (
        f"Review type '{review_type}' not recognized. Available types: self_assessment, manager_review, peer_feedback."
    )


@tool
def get_goal_setting_framework(goal_type: str = "smart") -> str:
    """Get goal-setting framework and templates.

    Args:
        goal_type: Framework type (smart, okr, bhag).

    Returns:
        Goal-setting framework with examples and templates.
    """
    if goal_type.lower() == "smart":
        return """**SMART Goal Framework:**

SMART goals are:
- **S**pecific: Clearly defined, not vague
- **M**easurable: Quantifiable outcomes or milestones
- **A**chievable: Challenging but realistic
- **R**elevant: Aligned with role, team, company objectives
- **T**ime-bound: Clear deadline or timeline

📋 **SMART Goal Template:**

**Goal:** [One clear sentence describing what you want to achieve]

**Specific:** What exactly will you accomplish? Who is involved? Where? Why is it important?

**Measurable:** How will you know when you've achieved it? What metrics will you track?

**Achievable:** Do you have the resources, skills, and time? What challenges might you face?

**Relevant:** How does this align with your role, team goals, and company objectives?

**Time-bound:** What's the deadline? What are the milestones along the way?

**Action Plan:**
- Step 1: [with date]
- Step 2: [with date]
- Step 3: [with date]

---

**✅ Good SMART Goal Examples:**

**Example 1 (Engineering):**
"Improve API response time by 40% (from 500ms to 300ms average) by implementing caching and query optimization, to be completed by Q2 end. Success measured by New Relic monitoring and user satisfaction scores."

Breaking it down:
- Specific: Improve API response time through caching and optimization
- Measurable: 40% improvement (500ms → 300ms)
- Achievable: Using established techniques (caching, optimization)
- Relevant: Improves user experience, aligns with performance OKRs
- Time-bound: By Q2 end (~3 months)

**Example 2 (Career Development):**
"Earn AWS Solutions Architect Associate certification by December 31st by studying 5 hours/week and taking 2 practice exams, to qualify for cloud architecture projects."

**Example 3 (Leadership):**
"Mentor 2 junior team members throughout the year with bi-weekly 30-minute 1:1s, helping them each complete one stretch project, as measured by their performance reviews and project completion."

❌ **Poor Goal Examples:**

"Be a better developer" - Not specific, measurable, or time-bound
"Learn cloud technologies" - Vague, no clear success criteria
"Help the team more" - Not measurable

**Tips:**
- Start with 3-5 major goals (don't overcommit)
- Review progress monthly
- Adjust as needed (goals can evolve)
- Celebrate milestones along the way"""

    elif goal_type.lower() == "okr":
        return """**OKR (Objectives & Key Results) Framework:**

Used company-wide for alignment and focus.

📋 **Structure:**

**Objective:** What you want to achieve (inspiring, qualitative)
**Key Results:** How you'll measure success (3-5, quantitative)

---

**OKR Template:**

**Objective:** [Inspiring, qualitative goal]

**Key Results:**
1. [Measurable outcome 1] - from [baseline] to [target]
2. [Measurable outcome 2] - from [baseline] to [target]
3. [Measurable outcome 3] - from [baseline] to [target]

**Timeline:** [Quarter or year]
**Owner:** [Your name]
**Status:** [On track / At risk / Off track]

---

**✅ Good OKR Examples:**

**Example 1 (Product Team):**
Objective: Become the #1 choice for small business analytics

Key Results:
1. Increase SMB signups from 500/month to 1,200/month
2. Improve SMB user retention from 70% to 85%
3. Achieve NPS score of 50+ for SMB segment
4. Launch 3 SMB-specific features by Q4

**Example 2 (Engineering):**
Objective: Deliver world-class reliability and performance

Key Results:
1. Reduce P1 incidents from 12/quarter to 3/quarter
2. Achieve 99.95% uptime (from current 99.8%)
3. Decrease average response time from 450ms to 250ms
4. Ship 100% of releases with zero rollbacks

**Example 3 (Personal Development):**
Objective: Become a technical leader

Key Results:
1. Mentor 3 engineers to successful project completion
2. Deliver 2 technical talks (1 internal, 1 conference)
3. Lead architecture design for 2 major features
4. Earn AWS certification with score >850

**OKR Best Practices:**
- Objectives are inspiring and qualitative
- Key Results are specific and measurable
- Aim for 70% achievement (stretch goals)
- Review progress weekly
- Update status transparently
- Celebrate progress, learn from misses"""

    return f"Goal type '{goal_type}' not recognized. Available types: smart, okr."


@tool
def request_feedback_survey(survey_type: str, recipients: str) -> str:
    """Request 360-degree feedback survey.

    Args:
        survey_type: Type of feedback (performance, leadership, skills).
        recipients: Recipients (peers, manager, directs, all).

    Returns:
        Feedback survey request confirmation.
    """
    request_id = f"FEEDBACK{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:4].upper()}"

    return f"""**360-Degree Feedback Request Submitted:**

📋 **Request ID:** {request_id}
📊 **Survey Type:** {survey_type.title()}
👥 **Recipients:** {recipients.title()}

**What Happens Next:**

1. **Survey Distribution (within 24 hours)**
   - Recipients receive anonymous survey link
   - 7-day window to complete

2. **Survey Structure:**
   - Strengths (what you do well)
   - Development areas (opportunities to improve)
   - Specific behavioral examples
   - Overall impact assessment

3. **Results (10 days after close)**
   - Aggregated themes (maintains anonymity)
   - Actionable insights
   - Development recommendations

**Tips for Recipients:**
- Be specific and constructive
- Focus on behaviors, not personality
- Provide examples
- Suggest improvements

**Your Action:**
After receiving feedback:
1. Review with open mind
2. Discuss with your manager
3. Create development plan
4. Follow up with feedback providers (anonymously via themes)

**Questions?** Contact feedback-team@company.com"""


# =============================================================================
# Sentiment & Wellbeing Tools
# =============================================================================


@tool
def get_wellbeing_resources(resource_type: str = "general") -> str:
    """Get employee wellbeing resources and support programs.

    Args:
        resource_type: Type of resource (mental_health, physical_health, financial, work_life_balance).

    Returns:
        Wellbeing resources, programs, and contact information.
    """
    resources = {
        "general": """**General Wellbeing Resources Overview:**

🌟 **Quick Links:**
- **EAP (Employee Assistance Program):** 1-800-XXX-XXXX (24/7, confidential)
- **Mental Health Support:** See `mental_health` resources
- **Physical Wellness:** See `physical_health` resources
- **Financial Wellness:** See `financial` resources
- **Work-Life Balance:** See `work_life_balance` resources

💙 **Immediate Support:**
- Crisis hotline: 988 (Suicide & Crisis Lifeline)
- EAP counseling: Free, confidential, available 24/7
- Manager 1:1: Schedule via Outlook

📋 **How to Get Started:**
1. Identify your wellbeing need (mental health, physical, financial, work-life)
2. Use the specific resource type for detailed information
3. Contact HR or EAP if you're unsure where to start

**Remember:** Your wellbeing is a priority. All resources are confidential and available to support you.""",
        "mental_health": """**Mental Health & EAP Resources:**

🧠 **Employee Assistance Program (EAP):**
- **Free confidential counseling:** 8 sessions per year per issue
- **24/7 crisis support:** 1-800-XXX-XXXX
- **Services:**
  - Individual and family counseling
  - Stress management
  - Grief and loss support
  - Substance abuse support
  - Work-life challenges
- **Access:** Call 1-800-XXX-XXXX or visit eap.company.com

💙 **Mental Health Apps:**
- **Headspace:** Free premium subscription for all employees
  - Meditation and mindfulness
  - Sleep support
  - Stress reduction
- **Calm:** Also available via EAP
- **Talkspace:** Text-based therapy (covered by health plan)

🏥 **Mental Health Coverage:**
- Therapy/Counseling: $20 copay per session (PPO)
- Psychiatry: Covered same as primary care
- In-network: Search at bcbs.com/findprovider
- Virtual visits: Available 24/7 via Teladoc

🆘 **Crisis Resources:**
- **988 Suicide & Crisis Lifeline:** Call or text 988 (24/7)
- **Crisis Text Line:** Text HOME to 741741
- **Company EAP:** 1-800-XXX-XXXX (24/7)

💪 **Additional Support:**
- Mental Health First Aid training (quarterly workshops)
- Peer support groups (monthly, confidential)
- Manager mental health training
- Flexible work arrangements for mental health needs

**Remember:** Taking care of your mental health is just as important as physical health. All resources are confidential.""",
        "physical_health": """**Physical Health & Wellness Programs:**

💪 **Fitness Benefits:**
- **Gym Reimbursement:** $50/month (requires 8+ visits)
  - Submit receipts monthly via hr.company.com/wellness
  - Covers gym memberships, fitness classes, personal training
- **On-site Fitness Center:** Free for employees (Building A, Floor 1)
  - Open 6 AM - 8 PM weekdays
  - Equipment: Cardio, weights, yoga studio
  - Free classes: Yoga (Mon/Wed 12PM), HIIT (Tue/Thu 6AM)

🏃 **Wellness Challenges:**
- Quarterly challenges (step challenges, healthy eating, meditation)
- Team-based competitions with prizes
- Fitbit/Apple Watch integration
- Current challenge: company.com/wellness/challenges

🏥 **Preventive Care:**
- **Annual Physical:** 100% covered (no copay)
- **Biometric Screening:** On-site annually (receive $100 wellness incentive)
- **Flu Shots:** Free on-site clinics (October-November)
- **Health Coaching:** Free via health plan

🍎 **Nutrition:**
- Healthy snacks in break rooms (fresh fruit, nuts, yogurt)
- Nutrition counseling via EAP
- Meal prep workshops (quarterly)

💤 **Sleep & Rest:**
- Ergonomic assessments (free, request via facilities@company.com)
- Standing desks available
- Quiet rooms for rest/meditation (Book via Outlook)

📊 **Wellness Incentive Program:**
- Earn up to $500/year in rewards:
  - Annual physical: $100
  - Biometric screening: $100
  - Preventive dental: $50
  - Complete health assessment: $50
  - Participate in wellness challenge: $100
  - Fitness goal achievement: $100

**Get Started:** Visit company.com/wellness or email wellness@company.com""",
        "financial": """**Financial Wellness Resources:**

💰 **Financial Planning:**
- **Free Financial Advisor Sessions:**
  - Quarterly 1:1 consultations with certified advisors
  - Topics: Retirement, debt, savings, investments, taxes
  - Book at: financialadvisor.company.com
- **Financial Workshops (Monthly):**
  - Budgeting 101
  - Investing basics
  - Retirement planning
  - Home buying
  - Estate planning

📊 **Retirement Planning:**
- 401(k) plan with company match (up to 5%)
- Free retirement calculator: fidelity.com/company401k
- Target-date fund recommendations
- Catch-up contributions (age 50+)
- Roth 401(k) option available

💳 **Debt Management:**
- Through EAP: Free debt counseling and planning
- Student loan resources and refinancing guidance
- Negotiation support for medical bills

🏦 **Banking Benefits:**
- Preferred banking rates through company partners
- No-fee checking and savings accounts
- Mortgage discount programs
- Identity theft protection

📚 **Education:**
- Financial literacy courses (LinkedIn Learning)
- Lunch & Learn sessions (monthly)
- One-on-one coaching via EAP

💵 **Emergency Assistance:**
- Employee Assistance Fund (for hardship situations)
- Flexible payment plans for medical expenses
- Salary advances (in emergency situations, manager approval required)

**Get Started:** Contact EAP at 1-800-XXX-XXXX or email financial-wellness@company.com""",
        "work_life_balance": """**Work-Life Balance Resources:**

⏰ **Flexible Work Arrangements:**
- **Hybrid Options:** 2-3 days remote (manager approval)
- **Flexible Hours:** Core hours 10 AM - 3 PM, flex outside
- **Compressed Workweek:** 4x10 schedule (role-dependent)
- **Job Sharing:** Available for certain roles
- Request via: hr.company.com/flexible-work

👶 **Family Support:**
- **Parental Leave:** 16 weeks paid (birth parent), 8 weeks (non-birth)
- **Childcare:**
  - Backup childcare: 10 days/year (through Bright Horizons)
  - Childcare FSA: Up to $5,000 pre-tax
  - On-site childcare: Waitlist available
- **Eldercare:** Resources and support via EAP
- **Family Sick Leave:** Use sick days for family members

🏖️ **Time Off:**
- 15-25 PTO days (based on tenure)
- 10 paid holidays
- 5 sick days
- 2 personal days
- Volunteer time off: 2 days/year
- Sabbatical: 4 weeks after 5 years (unpaid or PTO)

🎯 **Productivity & Boundaries:**
- **Meeting-Free Fridays:** No internal meetings after 3 PM
- **Email Boundaries:** No expectation to respond after 6 PM or weekends
- **Right to Disconnect:** Company policy supports unplugging
- **Focus Time:** Block calendar for deep work (respect signals)

🌱 **Personal Development:**
- Learning days: 5 days/year for professional development
- Conference attendance (budget + time off)
- Side project time: 10% time for innovation (role-dependent)

🧘 **Wellness Time:**
- Wellness hour: 1 hour/week for health activities
- Mental health days: Take PTO when needed, no questions
- Quiet rooms available for meditation/prayer

📞 **Work-Life Support:**
- Concierge services via EAP (errands, event planning, travel booking)
- Discounts: Gym, tickets, travel, shopping (company.com/perks)
- Commuter benefits: Pre-tax transit and parking

**Need Help?** Work-life coaches available through EAP: 1-800-XXX-XXXX""",
    }

    resource_type_lower = resource_type.lower()
    if resource_type_lower in resources:
        return resources[resource_type_lower]

    return f"""Resource type '{resource_type}' not found.

**Available Wellbeing Resources:**
- mental_health
- physical_health
- financial
- work_life_balance

For general wellbeing support, contact:
- EAP: 1-800-XXX-XXXX (24/7)
- Wellness Team: wellness@company.com
- HR: hr@company.com

**Remember:** Your wellbeing is a priority. Don't hesitate to use these resources."""


@tool
def schedule_wellbeing_check(reason: str, preferred_contact: str = "confidential_eap") -> str:
    """Schedule a proactive wellbeing check-in with HR or EAP.

    Args:
        reason: Reason for check-in (stress, burnout, personal_issues, career_concerns).
        preferred_contact: Contact method (confidential_eap, hr_business_partner, manager).

    Returns:
        Wellbeing check-in confirmation and next steps.
    """
    check_id = f"WELLBEING{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:4].upper()}"

    contact_info = {
        "confidential_eap": """**EAP Counselor (100% Confidential)**
- Call 1-800-XXX-XXXX to schedule
- Available 24/7
- No information shared with company
- 8 free sessions per issue
- Virtual or in-person options""",
        "hr_business_partner": """**HR Business Partner**
- Confidential (shared only with your consent)
- Will contact you within 24 hours
- Can help with workplace accommodations
- Connect you to additional resources
- Email: your-hrbp@company.com""",
        "manager": """**Your Manager**
- Will reach out within 24 hours
- Can discuss workload, flexibility, support needs
- Maintains confidentiality
- Works with HR if needed for accommodations""",
    }

    return f"""**Wellbeing Check-In Requested:**

🆔 **Check-In ID:** {check_id}
🎯 **Focus Area:** {reason.replace("_", " ").title()}
📞 **Contact Method:** {preferred_contact.replace("_", " ").title()}

**What Happens Next:**

{contact_info.get(preferred_contact, contact_info["confidential_eap"])}

**You're Not Alone:**
Taking care of your wellbeing is important, and we're here to support you. Whether you're dealing with stress, burnout, personal challenges, or career concerns, there are resources and people ready to help.

**Additional Immediate Resources:**
- **Crisis Support:** 988 Suicide & Crisis Lifeline (24/7)
- **EAP:** 1-800-XXX-XXXX (24/7, completely confidential)
- **Headspace App:** Free meditation and mental health support

**Remember:**
- All conversations are confidential (unless you're in danger)
- No judgment - everyone faces challenges
- Seeking support is a sign of strength
- Your job is secure while you focus on wellbeing

You'll receive a confirmation email shortly."""


# =============================================================================
# HR Operations Tools
# =============================================================================


@tool
def submit_hr_request(
    request_type: str,
    description: str,
    urgency: Literal["routine", "urgent"] = "routine",
) -> str:
    """Submit an HR request or case.

    Args:
        request_type: Type of request (leave, transfer, accommodation, name_change, verification).
        description: Detailed description of the request.
        urgency: Request urgency.

    Returns:
        HR request confirmation with case ID and expected timeline.
    """
    case_id = f"HR{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"

    case_data = {
        "id": case_id,
        "type": request_type,
        "description": description,
        "urgency": urgency,
        "status": "submitted",
        "created_at": datetime.now().isoformat(),
        "expected_resolution": "3-5 business days" if urgency == "routine" else "24 hours",
    }

    HR_CASES_DB[case_id] = case_data

    return f"""**HR Request Submitted Successfully:**

📋 **Case ID:** {case_id}
📂 **Request Type:** {request_type.replace("_", " ").title()}
⏱️ **Urgency:** {urgency.title()}
📅 **Submitted:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

**Description:**
{description}

**What Happens Next:**

1. **Acknowledgment:** You'll receive an email confirmation within 1 hour
2. **Assignment:** Case assigned to HR specialist within 4 hours
3. **Review:** HR will review and may request additional information
4. **Resolution:** Expected within {case_data["expected_resolution"]}

**Expected Timeline by Request Type:**
- Leave requests: 1-2 business days
- Accommodation requests: 3-5 business days
- Transfer requests: 1-2 weeks
- Name/address changes: 1-2 business days
- Employment verification: 24 hours

**Track Your Case:**
- Check status: hr.company.com/cases/{case_id}
- Email updates sent automatically
- Contact assigned HR specialist directly (in confirmation email)

**Need Immediate Help?**
- Urgent matters: Call HR directly at ext. 5000
- After hours: Contact EAP at 1-800-XXX-XXXX

You'll receive a detailed email confirmation shortly."""


@tool
def check_request_status(case_id: str) -> str:
    """Check the status of an HR request or case.

    Args:
        case_id: The HR case ID to check.

    Returns:
        Current case status and details.
    """
    case = HR_CASES_DB.get(case_id)

    if not case:
        return f"""**Case {case_id} not found in current session.**

To check existing HR cases:
1. Visit: hr.company.com/cases
2. Log in with your credentials
3. View "My Cases" dashboard

You can also:
- Email: hr@company.com with your case ID
- Call HR Service Center: ext. 5000

If you just submitted a request, it may take a few minutes to appear in the system."""

    # Simulate case progress
    status_emoji = {
        "submitted": "📝",
        "in_review": "👀",
        "pending_info": "⏳",
        "approved": "✅",
        "resolved": "✅",
        "closed": "📁",
    }

    emoji = status_emoji.get(case["status"], "📋")

    return f"""{emoji} **HR Case Status: {case_id}**

**Request Type:** {case["type"].replace("_", " ").title()}
**Current Status:** {case["status"].replace("_", " ").title()}
**Submitted:** {case["created_at"][:10]}
**Urgency:** {case["urgency"].title()}

**Description:**
{case["description"]}

**Progress:**
✅ Case submitted
✅ Acknowledgment sent
🔄 Under review by HR specialist
⏳ Expected resolution: {case["expected_resolution"]}

**Next Steps:**
- HR specialist reviewing your request
- May contact you for additional information
- You'll receive email updates on progress

**Need to Update Your Request?**
Reply to the confirmation email or call HR at ext. 5000

**Questions?**
Contact your assigned HR specialist (see confirmation email) or hr@company.com"""


@tool
def get_onboarding_checklist(employee_type: str = "new_hire") -> str:
    """Get onboarding checklist for new hires or role transitions.

    Args:
        employee_type: Type of onboarding (new_hire, internal_transfer, returning_employee).

    Returns:
        Comprehensive onboarding checklist with timelines.
    """
    if employee_type.lower() == "new_hire":
        return """**New Hire Onboarding Checklist:**

🎉 **Welcome to the Team!**

**📋 Before Day 1 (Preboarding):**
- [ ] Complete pre-employment paperwork (DocuSign link sent via email)
  - I-9 verification
  - Tax withholding (W-4)
  - Direct deposit information
  - Emergency contacts
  - Benefits enrollment forms
- [ ] Review welcome email from HR
- [ ] Set up home office (if remote)
- [ ] Review employee handbook: hr.company.com/handbook
- [ ] Prepare questions for Day 1

**🚀 Day 1:**
- [ ] Arrive at 9 AM (or log in for remote)
- [ ] Meet with HR for orientation (9-11 AM)
  - Company overview and culture
  - Benefits overview
  - IT equipment setup
  - Badge and access cards
- [ ] Lunch with your team (12-1 PM)
- [ ] IT setup and training (1-3 PM)
  - Laptop configuration
  - Email and calendar setup
  - VPN and security tools
  - Required software installation
- [ ] Meet your manager and buddy (3-4 PM)
- [ ] Review first week schedule

**📅 Week 1:**
- [ ] Complete required training modules (HR portal)
  - Code of Conduct
  - Information Security
  - Anti-Harassment
  - Safety Training
- [ ] 1:1 with manager to discuss:
  - 30-60-90 day goals
  - Team structure and dynamics
  - Communication norms
  - Immediate priorities
- [ ] Set up 1:1s with key stakeholders (10-15 people)
- [ ] Join team meetings and standups
- [ ] Review team documentation and resources
- [ ] Enroll in benefits (deadline: 30 days from start)

**📊 Month 1 (30 Days):**
- [ ] Complete all required training (100%)
- [ ] Finalize benefits enrollment
- [ ] Complete 1:1 intro meetings with stakeholders
- [ ] Deliver first small project or contribution
- [ ] 30-day check-in with manager
  - Discuss progress on goals
  - Address any questions or concerns
  - Adjust onboarding plan if needed
- [ ] Complete employee engagement survey

**🎯 Month 2 (60 Days):**
- [ ] Take on more significant project work
- [ ] Begin participating in team ceremonies
- [ ] Identify development opportunities
- [ ] 60-day check-in with manager
- [ ] Start contributing to code reviews / team reviews

**🏆 Month 3 (90 Days):**
- [ ] Fully ramped and productive
- [ ] Leading or co-leading projects
- [ ] 90-day performance review with manager
  - Review accomplishments
  - Set goals for next quarter
  - Discuss career development
  - Confirm successful onboarding
- [ ] Complete onboarding feedback survey

**📚 Resources:**
- Employee Handbook: hr.company.com/handbook
- IT Support: it-support@company.com or ext. 5555
- HR Support: hr@company.com or ext. 5000
- Your Onboarding Buddy: [Name] - [email]
- Manager: [Name] - [email]

**Questions Anytime?**
Don't hesitate to ask! Your manager, buddy, and HR team are here to help you succeed."""

    elif employee_type.lower() == "internal_transfer":
        return """**Internal Transfer Onboarding Checklist:**

🔄 **Congratulations on Your New Role!**

**📋 Before Transfer Date:**
- [ ] Complete handover in current role
  - Document ongoing projects
  - Transfer knowledge to team
  - Close out action items
- [ ] Review new role expectations with new manager
- [ ] Meet with new team (if possible)
- [ ] Update HR systems with new role info (HR will help)

**🚀 First Week in New Role:**
- [ ] Attend orientation with new team
- [ ] Set up 1:1s with new manager (weekly for first month)
- [ ] Meet key stakeholders in new function
- [ ] Review team processes and workflows
- [ ] Access new systems and tools (request via IT if needed)
- [ ] Discuss 30-60-90 day goals with manager

**📊 First 30 Days:**
- [ ] Complete any role-specific training
- [ ] Shadow team members
- [ ] Contribute to first project in new role
- [ ] 30-day check-in with new manager
- [ ] Stay connected with previous team (maintain relationships)

**🎯 60-90 Days:**
- [ ] Fully ramped in new responsibilities
- [ ] Regular contributions to team goals
- [ ] 90-day review with new manager
- [ ] Celebrate transition!

**Resources:**
- HR Business Partner: [Name] - [email]
- New Manager: [Name] - [email]
- Transfer FAQ: hr.company.com/internal-transfers"""

    return f"Employee type '{employee_type}' not recognized. Available types: new_hire, internal_transfer."


@tool
def initiate_exit_process(exit_type: str, last_day: str, reason: str | None = None) -> str:
    """Initiate employee exit process (resignation, retirement, etc.).

    Args:
        exit_type: Type of exit (resignation, retirement, end_of_contract).
        last_day: Proposed last day of employment (YYYY-MM-DD).
        reason: Optional reason for leaving (for feedback purposes).

    Returns:
        Exit process confirmation and offboarding checklist.
    """
    exit_id = f"EXIT{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:4].upper()}"

    return f"""**Exit Process Initiated:**

📋 **Exit ID:** {exit_id}
📅 **Proposed Last Day:** {last_day}
📂 **Exit Type:** {exit_type.replace("_", " ").title()}

**What Happens Next:**

**1. Manager Notification (Immediate)**
   - HR will coordinate with your manager
   - Discuss transition plan and knowledge transfer

**2. HR Meeting (Within 48 hours)**
   - Exit interview (voluntary but encouraged)
   - Discuss benefits continuation (COBRA, 401k)
   - Return of company property
   - Final paycheck and PTO payout details

**3. Offboarding Checklist:**

📦 **Before Last Day:**
- [ ] Provide written resignation (if not already done)
- [ ] Work with manager on transition plan
- [ ] Document ongoing projects and responsibilities
- [ ] Train replacement or team members
- [ ] Complete exit interview with HR
- [ ] Transfer knowledge and files

💼 **Last Day Activities:**
- [ ] Return all company property:
  - Laptop, monitors, keyboard, mouse
  - Phone and accessories
  - Access badge and keys
  - Company credit card
  - Any other company equipment
- [ ] Final 1:1 with manager
- [ ] Say goodbyes to team
- [ ] Complete exit paperwork

💰 **Financial & Benefits:**
- **Final Paycheck:**
  - Includes: Salary through last day + accrued PTO payout
  - Paid: Next regular payroll cycle after last day
  - Delivery: Direct deposit or paper check (your choice)

- **Benefits Continuation:**
  - Health insurance: Ends last day of month
  - COBRA: Eligible for 18 months (info mailed within 14 days)
  - Life insurance: Portable option available
  - FSA: Use by last day or file claims within 90 days

- **401(k):**
  - Remains in current account (can keep at Fidelity)
  - Options: Keep, roll over, or cash out
  - Vested amount: [View at fidelity.com/company401k]
  - Financial advisor consultation available

**🔐 Access & Accounts:**
- Email access: Ends at 5 PM on last day
- VPN/Systems: Disabled at 5 PM on last day
- Personal files: Download before last day
- Slack/Teams: Access ends on last day

**📧 After Exit:**
- W-2 form: Mailed in January (for prior year)
- Employment verification: hrverify@company.com
- 401(k) inquiries: Fidelity at 1-800-XXX-XXXX
- COBRA questions: benefits-cobra@company.com
- General questions: hr@company.com

**Exit Interview:**
Your feedback helps us improve! Topics covered:
- Reason for leaving
- Experience at company
- Suggestions for improvement
- Career plans
(100% confidential, voluntary but appreciated)

**Stay Connected:**
- Alumni network: alumni.company.com
- LinkedIn company page: Follow for updates
- Boomerang policy: Returning employees welcome!

**Thank You for Your Contributions!**

We appreciate your time with us and wish you the best in your next chapter. Please reach out to HR at hr@company.com with any questions.

Confirmation email with full details sent to your work email."""


# =============================================================================
# Engagement & Surveys Tools
# =============================================================================


@tool
def send_pulse_survey(survey_topic: str, target: str = "self") -> str:
    """Send a quick pulse survey to gather employee feedback.

    Args:
        survey_topic: Topic for pulse survey (engagement, workload, team_dynamics, wellbeing).
        target: Survey target (self, team, department).

    Returns:
        Pulse survey confirmation and link.
    """
    survey_id = f"PULSE{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:4].upper()}"

    survey_topics_info = {
        "engagement": "How engaged and motivated do you feel at work?",
        "workload": "How manageable is your current workload?",
        "team_dynamics": "How well is your team collaborating and communicating?",
        "wellbeing": "How are you feeling in terms of stress and work-life balance?",
    }

    return f"""**Pulse Survey Initiated:**

📊 **Survey ID:** {survey_id}
🎯 **Topic:** {survey_topic.replace("_", " ").title()}
👥 **Target:** {target.title()}
🔍 **Question Focus:** {survey_topics_info.get(survey_topic.lower(), "General feedback")}

**Survey Details:**

**Quick Questions (2 minutes):**
1. Rating scale (1-5): {survey_topics_info.get(survey_topic.lower(), "How are things going?")}
2. Open feedback: What's working well?
3. Open feedback: What could be better?
4. Priority improvement: What would have the biggest impact?

**Survey Link:**
Complete your pulse survey: company.com/pulse/{survey_id}

**Why Pulse Surveys Matter:**
- Quick feedback (2 min) drives meaningful change
- Your responses are aggregated and anonymous
- Results shared with leadership monthly
- Action plans created based on themes

**What Happens with Results:**

1. **Aggregation (1 week):**
   - Individual responses remain anonymous
   - Themes and trends identified
   - Scores calculated by dimension

2. **Analysis (2 weeks):**
   - Leadership reviews results
   - Compares to baseline and trends
   - Identifies priority areas

3. **Action Planning (3 weeks):**
   - Teams create improvement plans
   - Quick wins implemented immediately
   - Longer-term initiatives planned

4. **Communication (4 weeks):**
   - Results shared transparently
   - Action plans communicated
   - Accountability established

**Previous Pulse Survey Results:**
- Overall engagement score: 4.2/5 (up from 4.0 last quarter)
- Top strength: "Team collaboration" (4.6/5)
- Top improvement area: "Work-life balance" (3.7/5)
- Actions taken: Implemented Meeting-Free Fridays, expanded remote work policy

**Your Voice Matters!**
Complete the survey to help shape our workplace culture and practices.

Survey closes in 7 days. Reminder emails sent at 3 days and 1 day remaining."""


@tool
def get_engagement_insights(timeframe: str = "current_quarter", segment: str = "company") -> str:
    """Get employee engagement insights and trends.

    Args:
        timeframe: Time period (current_quarter, last_quarter, year_to_date).
        segment: Organizational segment (company, department, team).

    Returns:
        Engagement insights, trends, and action plans.
    """
    return f"""**Employee Engagement Insights: {timeframe.replace("_", " ").title()}**

📊 **Overall Engagement Score: 4.2 / 5.0** ⬆️ (+0.2 from last quarter)

**Score Breakdown:**

🌟 **Engagement Dimensions:**

1. **Meaningful Work:** 4.5/5 ⬆️
   - "I find my work purposeful and impactful"
   - Highest score in 12 months
   - 89% of employees rate 4 or 5

2. **Growth Opportunities:** 4.0/5 ➡️
   - "I have opportunities to learn and grow"
   - Stable quarter-over-quarter
   - Action: Expanded L&D budget by 25%

3. **Manager Support:** 4.3/5 ⬆️
   - "My manager supports my development"
   - Improved after manager training initiative

4. **Work-Life Balance:** 3.7/5 ⬇️
   - "I can balance work and personal life"
   - **Priority Area for Improvement**
   - Actions: Meeting-Free Fridays, flexible hours

5. **Recognition:** 3.9/5 ⬆️
   - "I feel valued for my contributions"
   - Peer recognition program launched

6. **Team Collaboration:** 4.6/5 ⬆️
   - "My team works well together"
   - Highest scoring dimension

7. **Company Direction:** 4.1/5 ➡️
   - "I understand and believe in company strategy"
   - Quarterly all-hands improving transparency

**📈 Trends:**

**Positive Momentum:**
- Engagement up 0.2 points from Q3
- 85% participation rate in pulse surveys (target: 80%)
- eNPS (Employee Net Promoter Score): +42 (Industry benchmark: +30)

**Areas of Focus:**
- Work-life balance remains below target (3.7 vs 4.0 goal)
- 15% report feeling overworked (down from 22% last quarter)
- Career advancement clarity requested by 28% in feedback

**💡 Top Employee Feedback Themes:**

**What's Working Well:**
1. "Great team culture and supportive colleagues"
2. "Meaningful work that makes a difference"
3. "Flexible remote work policy"
4. "Strong leadership transparency"
5. "Good benefits package"

**What Could Be Better:**
1. "Need clearer career paths" (28% mention)
2. "More manageable workload" (15% mention)
3. "Faster decision-making" (12% mention)
4. "Better cross-team coordination" (10% mention)
5. "More recognition for achievements" (9% mention)

**🎯 Action Plans in Progress:**

**Immediate (This Quarter):**
✅ Meeting-Free Fridays (launched)
✅ Manager coaching on workload management (in progress)
🔄 Career framework rollout (next month)

**Short-term (Next Quarter):**
- Launch career pathing tool
- Expand recognition programs
- Improve sprint planning to reduce overcommitment
- Cross-functional collaboration workshops

**Long-term (6-12 months):**
- Leadership development program expansion
- Workload planning and capacity management tools
- Enhanced wellness benefits

**📢 Recent Actions Based on Feedback:**

| Quarter | Feedback | Action Taken | Impact |
|---------|----------|--------------|--------|
| Q3 | "Too many meetings" | Meeting-Free Fridays | +0.3 on work-life balance |
| Q2 | "Unclear career paths" | Career framework design | In progress |
| Q1 | "Need more recognition" | Peer recognition program | +0.2 on recognition score |

**🗣️ Your Manager's Role:**

Your manager receives:
- Team-specific engagement scores (anonymized if < 5 people)
- Action planning resources
- Coaching on improvement areas

**Get Involved:**
- Participate in monthly pulse surveys
- Share feedback in 1:1s with your manager
- Join employee resource groups
- Volunteer for workplace committees

**Questions or Ideas?**
Contact Employee Experience team: employee-experience@company.com

*Note: Segment-specific insights available at: insights.company.com/engagement*
*Next engagement survey: End of quarter (in 6 weeks)*"""


# =============================================================================
# Compensation Tools
# =============================================================================


@tool
def get_compensation_insights(insight_type: str = "market_data") -> str:
    """Get compensation benchmarking insights and information.

    Args:
        insight_type: Type of insight (market_data, pay_equity, total_rewards, raise_process).

    Returns:
        Compensation insights within appropriate boundaries.
    """
    insights = {
        "market_data": """**Market Compensation Data & Benchmarking:**

💰 **How We Determine Compensation:**

**1. Market Positioning:**
   - Target: 50th-75th percentile of market
   - Benchmarked against 3 peer companies annually
   - Data sources: Radford, Mercer, Glassdoor, Payscale

**2. Factors That Influence Pay:**
   - **Role & Level:** Job family, seniority, scope
   - **Skills & Experience:** Years of experience, specialized skills
   - **Performance:** Performance rating history
   - **Location:** Geographic cost-of-labor adjustments
   - **Market Conditions:** Supply and demand for skills

**3. Compensation Philosophy:**
   - Pay for performance: Top performers earn significantly more
   - Equity and fairness: Equal pay for equal work
   - Transparency: Salary ranges published internally
   - Competitiveness: Regular market reviews

**📊 Typical Compensation Components:**

**Base Salary:**
- Fixed annual pay
- Reviewed annually (merit increase cycle)
- Mid-year adjustments for promotions or equity corrections

**Variable Pay:**
- Annual bonus (10-20% of base for most roles)
- Based on company and individual performance
- Paid in March for prior calendar year

**Equity (RSUs/Stock Options):**
- Eligibility varies by level (typically Senior+ roles)
- Vests over 4 years (25% per year)
- Refresh grants for high performers

**Benefits & Perks:**
- Health, dental, vision (company pays 80%)
- 401(k) match (up to 5%)
- PTO, holidays, parental leave
- Professional development ($2,000/year)
- See total value: hr.company.com/total-rewards

**🔍 Salary Range Transparency:**

To view salary ranges for all roles:
1. Visit: hr.company.com/salary-ranges
2. Search by job family and level
3. View: Min, Midpoint, Max for your location

Example:
- **Software Engineer II (US - San Francisco Bay Area)**
  - Min: $120,000
  - Midpoint: $145,000
  - Max: $170,000
  - Typical total comp: $145k base + $20k bonus + $40k equity = $205k

**Your Compensation:**
- View your total rewards statement: hr.company.com/total-rewards
- Shows: Base + bonus + equity + benefits value
- Updated annually

**📈 Market Trends (2024):**
- Tech industry: 3-5% merit increase average
- High demand skills: AI/ML, Cloud, Security (premium of 10-20%)
- Location trends: Remote roles reducing geo adjustments
- Equity: More companies using RSUs vs stock options

**Questions About Market Data?**
- Compensation team: compensation@company.com
- Your HRBP: [email]
- Annual compensation review: Coming in Q4""",
        "pay_equity": """**Pay Equity & Fairness:**

⚖️ **Our Commitment to Pay Equity:**

**1. Equal Pay for Equal Work:**
   - Compensation based on role, level, performance, and market data
   - NOT based on: Gender, race, age, or other protected characteristics
   - Regular pay equity audits (annually)

**2. Pay Equity Analysis:**
   - **Last Audit:** Q4 2023
   - **Result:** 99.2% pay equity (within 5% for similar roles)
   - **Action Taken:** 23 proactive adjustments made
   - **Investment:** $450,000 in equity corrections

**3. Factors We Control For:**
   - Job family and level
   - Performance rating
   - Years of experience
   - Location (cost of labor)
   - Skills and expertise
   - Time in role

**📊 Transparency Measures:**

**Salary Ranges:**
- Published internally for all roles
- Updated annually based on market data
- View at: hr.company.com/salary-ranges

**Promotion Criteria:**
- Clear leveling guidelines for each role
- Consistent evaluation process
- Calibration committees for fairness

**Performance Reviews:**
- Standardized rating process
- Calibration across teams
- Merit increase tied to performance

**🔍 How to Check Your Pay:**

**1. View Your Compensation:**
   - Total rewards statement: hr.company.com/total-rewards
   - Includes: Base, bonus, equity, benefits

**2. Compare to Salary Range:**
   - Find your role at: hr.company.com/salary-ranges
   - See where you fall (Min-Mid-Max)
   - Target: Midpoint at "fully performing" in role

**3. Understand Your Position:**
   - Below midpoint: Typically newer to role or level
   - At midpoint: Meeting expectations consistently
   - Above midpoint: Exceeding expectations, high tenure

**💬 If You Have Concerns:**

**When to Raise Pay Equity Concerns:**
- You believe your pay is not aligned with peers
- You have data showing market discrepancy
- You suspect bias in compensation decisions

**How to Raise Concerns:**
1. **Start with your manager:** Discuss your compensation
2. **Contact your HRBP:** For confidential discussion
3. **Compensation team:** compensation@company.com
4. **Ethics hotline:** For serious concerns: 1-800-XXX-XXXX

**What Happens:**
- Confidential review of your compensation
- Comparison to similar roles and peers
- Market data analysis
- Adjustment if warranted (effective immediately)
- No retaliation for raising concerns

**📈 Our Track Record:**
- 99.2% pay equity maintained
- Annual proactive adjustments
- Transparent pay ranges
- Third-party audit validation

**Resources:**
- Pay Equity FAQ: hr.company.com/pay-equity
- Compensation Philosophy: hr.company.com/compensation
- Contact: compensation@company.com""",
        "total_rewards": """**Total Rewards Statement:**

💼 **Your Total Compensation Package:**

**Understanding Your Full Rewards:**
Your compensation is more than just your salary. Here's how to view your complete package:

**📊 Components of Total Rewards:**

**1. Cash Compensation:**
   - Base Salary: [View at hr.company.com/total-rewards]
   - Annual Bonus Target: 10-20% of base (based on level)
   - Merit Increase: Annual (typically 3-5% for meeting expectations)

**2. Equity Compensation (if applicable):**
   - RSUs (Restricted Stock Units): Vest over 4 years
   - Current value: [View at equity.company.com]
   - Vesting schedule: 25% per year
   - Refresh grants: Considered annually for high performers

**3. Health & Insurance Benefits:**
   - **Medical Insurance:** Company pays 80% of premium (~$9,600/year value)
   - **Dental Insurance:** Company pays 75% (~$600/year value)
   - **Vision Insurance:** Company pays 80% (~$240/year value)
   - **Life Insurance:** 2x salary (company-paid)
   - **Disability Insurance:** Short & long-term (company-paid)

**4. Retirement Benefits:**
   - **401(k) Match:** Up to 5% of salary (~$7,000/year avg value)
   - **Vesting:** 3-year schedule
   - **Investment Options:** 25+ funds

**5. Time Off:**
   - **PTO:** 15-25 days/year ($11,500/year value at avg salary)
   - **Holidays:** 10 paid days ($3,800/year value)
   - **Sick Leave:** 5 days ($1,900/year value)
   - **Parental Leave:** 8-16 weeks paid (~$12,000-$24,000 value)

**6. Professional Development:**
   - **Learning Budget:** $2,000/year per employee
   - **Conference Budget:** $3,000/year (role-dependent)
   - **Certification Reimbursement:** 100% of costs
   - **Tuition Assistance:** Up to $5,250/year (tax-free)

**7. Wellbeing & Perks:**
   - **EAP (Employee Assistance Program):** Free counseling (value: $1,500/year)
   - **Gym Reimbursement:** $600/year
   - **Headspace Premium:** Free ($70/year value)
   - **Commuter Benefits:** Pre-tax up to $3,000/year
   - **Home Office Stipend:** $500/year (remote employees)

**8. Additional Perks:**
   - Free snacks and beverages (~$500/year value)
   - Company events and team outings (~$200/year value)
   - Employee discount programs (~$300/year avg savings)
   - Referral bonuses ($2,000 per successful hire)

**💰 Example Total Rewards (Software Engineer II):**

| Component | Annual Value |
|-----------|--------------|
| Base Salary | $145,000 |
| Bonus (15% target) | $21,750 |
| Equity (RSU grant) | $40,000 |
| 401(k) Match (5%) | $7,250 |
| Health Benefits | $10,440 |
| PTO Value | $11,500 |
| Learning Budget | $2,000 |
| Other Perks | $3,000 |
| **Total Value** | **$240,940** |

**📈 View Your Personal Statement:**

Access your customized total rewards statement:
1. Visit: hr.company.com/total-rewards
2. Log in with your credentials
3. View breakdown by category
4. Download PDF for personal records

**Statement Includes:**
- Your specific compensation and benefits
- Employer contributions and subsidies
- Value of time off and leave policies
- Comparison to prior year
- Benefit utilization summary

**💡 Maximizing Your Benefits:**

**Optimize Your Rewards:**
- **401(k):** Contribute at least 6% to maximize match
- **FSA:** Use pre-tax dollars for medical expenses
- **Learning:** Take advantage of full $2,000 budget
- **Wellness:** Use gym reimbursement and EAP
- **PTO:** Take your full allocation (it's part of your comp!)

**Questions?**
- Benefits: benefits@company.com
- Compensation: compensation@company.com
- Total Rewards: hr@company.com""",
        "raise_process": """**Merit Increase & Raise Process:**

📈 **How Compensation Reviews Work:**

**🗓️ Annual Compensation Cycle:**

**Timeline:**
- **September - October:** Performance reviews completed
- **November:** Compensation planning and calibration
- **December:** Manager communicates decisions
- **January:** New compensation effective (first paycheck of year)
- **March:** Prior year bonuses paid

**📊 Merit Increase Process:**

**1. Performance Rating (September):**
   - Performance reviews completed
   - Ratings: 1 (Unsatisfactory) to 5 (Exceptional)
   - Calibration across teams for fairness

**2. Market Data Review (October):**
   - Annual market benchmark refresh
   - Salary range adjustments for all roles
   - Cost of living adjustments by location

**3. Budget Allocation (November):**
   - Company sets overall merit increase budget (typically 3-5%)
   - Allocated by division, then department
   - Performance distribution determines individual increases

**4. Individual Decisions (November-December):**
   - Managers propose increases for each employee
   - Calibration committees review for fairness
   - HR and Finance approve
   - Final decisions locked

**5. Communication (December):**
   - Manager 1:1s to communicate new compensation
   - Total rewards statements updated
   - Effective date: January 1

**💰 Typical Merit Increases by Performance:**

| Performance Rating | Typical Increase | Example ($100k salary) |
|-------------------|------------------|----------------------|
| 5 - Exceptional | 7-10% | $107,000 - $110,000 |
| 4 - Exceeds | 5-7% | $105,000 - $107,000 |
| 3 - Meets | 3-4% | $103,000 - $104,000 |
| 2 - Needs Improvement | 0-2% | $100,000 - $102,000 |
| 1 - Unsatisfactory | 0% | $100,000 (PIP required) |

*Note: Ranges vary based on position in salary range and market conditions*

**🚀 Promotion Increases:**

**Promotion Process:**
- Typically aligned with annual cycle (effective January 1)
- Mid-year promotions possible for exceptional cases
- Requires: Manager nomination, performance evidence, level criteria met

**Promotion Increases:**
- Larger than merit increases (typically 8-15%)
- Equity grants for promotions to senior+ levels
- Adjustment to new salary range

**💼 Off-Cycle Adjustments:**

**When Off-Cycle Increases Happen:**
- **Equity Corrections:** Pay equity issues identified
- **Market Adjustments:** Role significantly under market
- **Retention:** Counter-offers or retention risk
- **Role Changes:** Significant expansion of scope

**How to Request:**
1. Discuss with your manager
2. Provide market data or rationale
3. Manager works with HRBP and Compensation team
4. Review and decision (typically 4-6 weeks)

**🗣️ Discussing Compensation with Your Manager:**

**Best Practices:**
✅ **Do:**
- Schedule a dedicated 1:1 (don't surprise them)
- Bring specific examples of your impact and performance
- Reference market data for your role and location
- Express your career goals and development
- Be open to feedback

❌ **Don't:**
- Demand or make ultimatums
- Compare yourself to specific colleagues
- Make it emotional or personal
- Threaten to leave (unless you genuinely have an offer)

**What to Prepare:**
1. **Your Accomplishments:**
   - STAR format examples
   - Quantified impact and results
   - Projects led or major contributions

2. **Market Research:**
   - Salary data from Glassdoor, Levels.fyi, Payscale
   - Job postings for similar roles
   - Industry reports

3. **Your Value:**
   - Skills and expertise gained
   - Additional responsibilities taken on
   - Ways you've gone beyond your role

**📝 Sample Conversation:**

*"I'd like to discuss my compensation. I've been in this role for X years and have consistently exceeded expectations, as shown in my performance reviews. I've taken on [specific responsibilities] and delivered [specific results]. Based on my research of market data for [role] in [location], I believe I'm below market. I'd like to understand the path to a compensation adjustment."*

**🤔 If You're Not Satisfied:**

**Options:**
1. **Understand the why:** Ask for specific feedback and development areas
2. **Create a plan:** Work with manager on goals to earn future increase
3. **Request review:** Ask HRBP to review your compensation vs peers
4. **Escalate:** If concerns about fairness, contact compensation@company.com

**Resources:**
- Compensation Philosophy: hr.company.com/compensation
- Salary Ranges: hr.company.com/salary-ranges
- Contact: compensation@company.com
- Your HRBP: [email]""",
    }

    insight_type_lower = insight_type.lower()
    if insight_type_lower in insights:
        return insights[insight_type_lower]

    return f"""Insight type '{insight_type}' not found.

**Available Compensation Insights:**
- market_data: Industry benchmarks and how pay is determined
- pay_equity: Our commitment to equal pay and fairness
- total_rewards: Complete view of your compensation package
- raise_process: How merit increases and promotions work

**For Specific Compensation Questions:**
Contact compensation@company.com or your HR Business Partner

**View Your Compensation:**
- Total rewards statement: hr.company.com/total-rewards
- Salary ranges: hr.company.com/salary-ranges"""


@tool
def request_compensation_review(reason: str, supporting_data: str) -> str:
    """Request a compensation review.

    Args:
        reason: Reason for review (market_adjustment, pay_equity, role_change, retention).
        supporting_data: Supporting information (market data, role changes, accomplishments).

    Returns:
        Compensation review request confirmation and process details.
    """
    review_id = f"COMPREVIEW{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:4].upper()}"

    return f"""**Compensation Review Request Submitted:**

📋 **Review ID:** {review_id}
🎯 **Reason:** {reason.replace("_", " ").title()}
📅 **Submitted:** {datetime.now().strftime("%Y-%m-%d")}

**Supporting Information Provided:**
{supporting_data}

**What Happens Next:**

**1. Initial Review (1-2 weeks):**
   - HR Business Partner reviews your request
   - Manager consultation
   - Initial assessment of merit

**2. Data Analysis (2-3 weeks):**
   - Compensation team analyzes:
     - Your current compensation vs salary range
     - Comparison to similar roles (peers)
     - Market data for your role and location
     - Performance history
     - Time since last increase

**3. Decision (3-4 weeks):**
   - Recommendation developed
   - Approvals obtained (Manager, Director, HR, Finance)
   - Decision communicated

**4. Communication (4 weeks):**
   - Manager schedules 1:1 to discuss outcome
   - If approved: Effective date and new compensation
   - If not approved: Rationale and development plan

**Possible Outcomes:**

✅ **Approved:**
- Compensation adjustment (typical range: 5-15%)
- Effective date (typically next pay period)
- Updated total rewards statement

📝 **Approved with Conditions:**
- Future adjustment pending goal achievement
- Clear criteria and timeline provided
- Re-evaluation in X months

❌ **Not Approved:**
- Explanation of rationale
- Development plan to earn future increase
- Next review opportunity (typically annual cycle)

**⏱️ Expected Timeline:**
- Standard request: 4-6 weeks
- Urgent retention case: 1-2 weeks
- Annual cycle review: Per company timeline

**📊 Success Factors:**

Reviews are more likely to be approved if:
- Clear gap to market data (10%+ below market)
- Role has significantly expanded
- Performance consistently exceeds expectations
- Equity concerns (backed by data)
- Retention risk (competing offer)

**💡 What You Can Do While Waiting:**

1. **Continue Strong Performance:**
   - Document ongoing achievements
   - Take on stretch projects
   - Demonstrate value

2. **Develop Skills:**
   - Close skill gaps identified
   - Pursue relevant certifications
   - Expand expertise

3. **Prepare for Discussion:**
   - Think about your career goals
   - Consider feedback you may receive
   - Be open to development plan

**🗣️ Meeting with Your Manager:**

When your manager reaches out for the discussion:
- **Listen:** Understand the full rationale
- **Ask Questions:** Clarify anything unclear
- **Be Professional:** Even if disappointed
- **Discuss Next Steps:** Create plan for future

**If Not Approved:**
- Ask: "What would it take to earn an increase?"
- Request: Clear goals and timeline
- Follow-up: Regular check-ins on progress

**📧 Status Updates:**

You'll receive email updates at:
- Initial review started (1 week)
- Analysis in progress (2 weeks)
- Decision pending (3 weeks)
- Ready to discuss (4 weeks)

**Questions?**
- Contact your HRBP: [email]
- Compensation team: compensation@company.com
- HR: hr@company.com

**Confirmation email with case details sent to your work email.**"""


# =============================================================================
# Learning & Development Tools
# =============================================================================


@tool
def get_learning_path(role_goal: str, current_skills: str | None = None) -> str:
    """Get personalized learning path recommendations.

    Args:
        role_goal: Target role or career goal.
        current_skills: Optional current skills assessment.

    Returns:
        Personalized learning path with courses, timelines, and resources.
    """
    return f"""**Personalized Learning Path: Path to {role_goal}**

🎯 **Career Goal:** {role_goal}

**📚 Recommended Learning Journey:**

**Phase 1: Foundation (Months 1-3)**

**Core Technical Skills:**
1. **Advanced System Design**
   - Course: "System Design Fundamentals" (Internal L&D)
   - Duration: 6 weeks, 5 hours/week
   - Format: Self-paced with weekly live Q&A
   - Certification: Internal

2. **Cloud Architecture**
   - Course: AWS Solutions Architect Associate
   - Provider: AWS Training
   - Duration: 8 weeks, 6 hours/week
   - Certification: AWS Certified (industry-recognized)

3. **Performance Optimization**
   - Course: "High-Performance Systems" (LinkedIn Learning)
   - Duration: 12 hours
   - Format: Self-paced video

**Phase 2: Leadership Foundations (Months 4-6)**

**People & Communication:**
1. **Technical Leadership**
   - Course: "Leading Without Authority" (Internal L&D)
   - Duration: 4-week cohort program
   - Format: Weekly 2-hour sessions + homework
   - Includes: Coaching, peer feedback

2. **Mentorship Skills**
   - Workshop: "Effective Mentoring" (Internal L&D)
   - Duration: 1-day workshop + 6-month mentorship assignment
   - Practice: Mentor 1-2 junior engineers

3. **Communication for Impact**
   - Course: "Technical Communication" (Coursera)
   - Duration: 4 weeks
   - Focus: Documentation, presentations, stakeholder management

**Phase 3: Advanced Expertise (Months 7-12)**

**Strategic & Architectural:**
1. **Distributed Systems**
   - Course: MIT OpenCourseWare "Distributed Systems"
   - Duration: 12 weeks, self-paced
   - Includes: Papers, assignments, projects

2. **Architecture Patterns**
   - Book Club: "Designing Data-Intensive Applications" (Martin Kleppmann)
   - Duration: 12 weeks, weekly discussions
   - Internal group: #architecture-book-club

3. **Business Acumen**
   - Course: "Tech for Non-Tech Leaders" (Internal L&D)
   - Duration: 4 weeks
   - Learn: Business strategy, financial planning, ROI analysis

**Phase 4: Application & Practice (Ongoing)**

**Hands-On Experience:**
1. **Lead a Design Review**
   - Opportunity: Lead 2-3 architectural design reviews
   - Mentorship: Shadow senior architects first
   - Timeline: Months 6-12

2. **Cross-Team Project**
   - Opportunity: Lead a cross-team technical initiative
   - Skills: Collaboration, influence, technical depth
   - Timeline: 3-6 month project

3. **Tech Talks & Knowledge Sharing**
   - Present: 2 internal tech talks
   - Optional: Submit to external conference
   - Timeline: Months 9-12

**📊 Progress Milestones:**

✅ **Month 3:**
- Complete foundation courses
- AWS certification earned
- Actively mentoring 1 engineer

✅ **Month 6:**
- Leadership cohort completed
- Leading regular design reviews
- Cross-team project initiated

✅ **Month 9:**
- Advanced courses in progress
- Tech talk delivered
- Consistently demonstrating senior-level impact

✅ **Month 12:**
- All courses completed
- Promotion-ready portfolio assembled
- Ready for {role_goal} role

**💰 Budget & Support:**

**Learning Budget:**
- $2,000/year individual budget
- Additional budget for certifications (AWS ~$300)
- Conference budget: $3,000 (if presenting)

**Time Off:**
- 5 learning days/year
- Use for workshops, conferences, intensive study

**Manager Support:**
- Discuss this plan in your next 1:1
- Request 5 hours/week for learning activities
- Regular progress check-ins

**📈 Tracking Your Progress:**

**Use Learning Management System:**
1. Add this learning path: learning.company.com/my-path
2. Track course completions
3. Share progress with manager
4. Request feedback and adjustments

**Quarterly Check-ins:**
- Review progress with manager
- Adjust plan based on changing needs
- Celebrate milestones

**🎓 Completion & Next Steps:**

**After 12 Months:**
- Schedule promotion discussion with manager
- Submit portfolio of work
- Request feedback from senior engineers you've worked with
- Consider applying for {role_goal} openings

**Certificate & Recognition:**
- Earn "Leadership Development" internal certificate
- Add certifications to LinkedIn and internal profile
- Share your journey to inspire others

**🤝 Additional Support:**

**Mentorship:**
- Request mentor in {role_goal} role
- Join #career-development Slack channel
- Participate in architecture office hours

**Resources:**
- Career coaching: Request via `request_career_coaching()`
- Learning resources: learning@company.com
- Mentorship program: mentorship@company.com

**Questions?**
Contact Learning & Development: learning@company.com

**Ready to Start?**
Use `enroll_in_course()` to register for your first course!"""


@tool
def enroll_in_course(course_name: str, start_date: str | None = None) -> str:
    """Enroll in a learning course or program.

    Args:
        course_name: Name of the course or program.
        start_date: Optional preferred start date (YYYY-MM-DD).

    Returns:
        Course enrollment confirmation and access details.
    """
    enrollment_id = f"ENROLL{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:4].upper()}"

    return f"""**Course Enrollment Confirmation:**

📚 **Enrollment ID:** {enrollment_id}
📖 **Course:** {course_name}
📅 **Requested Start Date:** {start_date or "Next available session"}

**What Happens Next:**

**1. Enrollment Processing (24 hours):**
   - L&D team reviews your request
   - Checks seat availability
   - Verifies learning budget
   - Sends confirmation email

**2. Course Access:**
   - **Internal Courses:** Access link sent within 24 hours
   - **External Courses:** Registration completed on your behalf
   - **Cohort Programs:** Assigned to next available cohort
   - **Self-Paced:** Immediate access after processing

**3. Getting Started:**
   - Log in to learning platform: learning.company.com
   - Access course materials
   - Review syllabus and schedule
   - Join course Slack channel (if applicable)

**📋 Course Information:**

**What to Expect:**
- **Format:** [Online/In-person/Hybrid - will be in confirmation email]
- **Duration:** [Will be specified in confirmation]
- **Time Commitment:** [Hours per week]
- **Prerequisites:** [If any]
- **Certification:** [If applicable]

**Your Responsibilities:**
- Complete course within timeframe
- Participate in discussions/activities
- Submit assignments (if applicable)
- Provide feedback at completion

**💰 Cost & Budget:**

**Learning Budget:**
- Your annual budget: $2,000
- This course cost: [Will be in confirmation email]
- Remaining budget: [Updated after enrollment]

**Free Courses:**
- Internal L&D courses: Free
- LinkedIn Learning: Free (company subscription)
- Some external courses: Covered by department budget

**Manager Approval:**
- Courses < $500: No approval needed
- Courses > $500: Manager approval required
- Sent for approval: [If needed]

**⏰ Time Off for Learning:**

**Learning Time:**
- Use your 5 learning days/year
- Or work with manager for dedicated weekly hours
- Block calendar for course time

**Tips:**
- Schedule recurring "Learning Time" blocks
- Treat it like any other work commitment
- Update manager on progress

**📊 Tracking Progress:**

**In Learning Platform:**
- Track course progress
- Complete quizzes/assignments
- Earn certificates
- Add to your learning transcript

**Share with Manager:**
- Discuss progress in 1:1s
- Share key learnings
- Apply skills to work projects

**🎓 Completion & Recognition:**

**Upon Completion:**
- Certificate issued (internal platform)
- Updated skills profile
- LinkedIn-shareable certificates (external courses)
- Add to resume and internal profile

**Recognition:**
- Monthly "Learning Spotlight" (top learners featured)
- Earn learning badges
- Contribute to team learning goals

**🆘 Need Help?**

**Technical Issues:**
- Platform support: learning-tech@company.com
- Can't access course: Check email for access link

**Course Content:**
- Questions about material: Instructor or course forum
- Falling behind: Contact instructor for extension

**Administrative:**
- Budget questions: learning@company.com
- Enrollment status: learning@company.com
- Manager approval: Your manager + learning@company.com

**📧 Confirmation Email:**

You'll receive a detailed email within 24 hours with:
- Course access link
- Schedule and syllabus
- Instructor contact info
- Budget deduction details
- Getting started guide

**🚀 Pro Tips:**

1. **Start Strong:** Complete first module within first week
2. **Schedule It:** Block calendar time for learning
3. **Engage:** Participate in discussions and forums
4. **Apply:** Use learnings in real projects immediately
5. **Share:** Teach others what you've learned

**Questions?**
Contact Learning & Development: learning@company.com or Slack: #learning-and-development

**Happy Learning! 🎓**"""


# =============================================================================
# Escalation Tools
# =============================================================================


@tool
def escalate_to_hr_business_partner(
    issue_type: str,
    urgency: Literal["routine", "urgent", "critical"] = "routine",
    details: str = "",
) -> str:
    """Escalate issue to HR Business Partner.

    Args:
        issue_type: Type of issue (harassment, discrimination, accommodation, performance, compensation, other).
        urgency: Issue urgency level.
        details: Detailed description of the issue.

    Returns:
        Escalation confirmation and HRBP contact information.
    """
    escalation_id = f"HRBP{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:6].upper()}"

    urgency_response = {
        "routine": "HR Business Partner will contact you within 2 business days",
        "urgent": "HR Business Partner will contact you within 4 hours",
        "critical": "HR Business Partner will contact you immediately (within 1 hour)",
    }

    # Sensitive issues get immediate attention
    sensitive_keywords = ["harassment", "discrimination", "retaliation", "unsafe", "legal", "threat"]
    is_sensitive = any(keyword in issue_type.lower() or keyword in details.lower() for keyword in sensitive_keywords)

    if is_sensitive:
        urgency = "critical"

    return f"""**Escalation to HR Business Partner:**

🆔 **Escalation ID:** {escalation_id}
🚨 **Urgency Level:** {urgency.upper()}
📂 **Issue Type:** {issue_type.replace("_", " ").title()}
📅 **Escalated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

**Issue Summary:**
{details}

**Immediate Response:**

{urgency_response[urgency]}

**Your HR Business Partner:**
- Name: [Will be provided in confirmation email based on your department]
- Email: [Provided in confirmation]
- Phone: [Provided in confirmation]
- Office Hours: Mon-Fri, 8 AM - 5 PM

**What Happens Next:**

**1. Acknowledgment (Immediate):**
   - You'll receive a confirmation email
   - Case opened in confidential HR system
   - HRBP notified

**2. Initial Contact ({urgency_response[urgency]}):**
   - HRBP will reach out to you
   - Schedule confidential discussion
   - Gather additional context
   - Assess situation and needs

**3. Action Plan (After initial discussion):**
   - HRBP creates action plan
   - May involve: Investigation, accommodation, mediation, policy review
   - Timeline established
   - Regular updates provided

**4. Resolution & Follow-up:**
   - Issue addressed according to plan
   - Follow-up to ensure resolution
   - Case documented (confidentially)
   - Ongoing support as needed

**🔒 Confidentiality:**

**What's Protected:**
- Your conversation with HRBP is confidential
- Information shared only on need-to-know basis
- Your consent required before sharing (with exceptions below)

**Exceptions (Legal/Safety Requirements):**
- Threat to safety (self or others)
- Illegal activity
- Subpoena or legal requirement

**No Retaliation:**
- You're protected from retaliation for raising concerns
- Retaliation is a terminable offense
- Report any retaliation immediately

**🆘 If This Is an Emergency:**

**Immediate Danger:**
- **Call 911** if anyone is in immediate danger
- **Security:** ext. 9999 (on-site security)
- **EAP Crisis Line:** 1-800-XXX-XXXX (24/7)

**Crisis Resources:**
- Suicide Prevention: 988 (call or text)
- Domestic Violence: 1-800-799-7233
- Sexual Assault Hotline: 1-800-656-4673

**📋 Types of Issues HR Business Partners Handle:**

**Employee Relations:**
- Performance concerns
- Interpersonal conflicts
- Team dynamics
- Manager concerns

**Compliance & Legal:**
- Harassment or discrimination
- Accommodation requests
- FMLA and leave questions
- Policy violations

**Compensation & Career:**
- Pay equity concerns
- Promotion questions
- Career development
- Job classification

**Personal Situations:**
- Family/medical leave
- Reasonable accommodations
- Personal hardship
- Confidential matters

**💬 How to Prepare for HRBP Discussion:**

**Gather:**
- Specific examples (dates, times, witnesses)
- Relevant documents or communications
- Timeline of events
- Impact on you and your work

**Think About:**
- What outcome you're seeking
- What support you need
- Questions you have
- Any concerns about the process

**Remember:**
- Be honest and specific
- Stick to facts, not assumptions
- It's okay to be emotional - this is a safe space
- You can bring notes or documentation

**🤝 Alternative Resources:**

**If You Prefer:**
- **Ethics Hotline:** 1-800-XXX-XXXX (anonymous reporting)
- **EAP Counselor:** 1-800-XXX-XXXX (confidential support)
- **Your Manager:** If comfortable (HRBP can also loop in manager with your consent)
- **Legal:** For legal concerns: legal@company.com

**📧 Confirmation:**

You'll receive a confirmation email within 15 minutes with:
- Your HRBP contact information
- Case ID for reference
- Next steps and timeline
- Resources and support options

**🙏 Thank You for Reaching Out:**

We take all concerns seriously. Your HRBP is here to support you and help resolve this matter confidentially and professionally.

**Questions or Need Immediate Help?**
- Call HR immediately: ext. 5000
- After-hours urgent: Call EAP crisis line: 1-800-XXX-XXXX
- Email (non-urgent): hr@company.com

**Your wellbeing and the integrity of our workplace are our top priorities.**"""


@tool
def schedule_hr_meeting(meeting_topic: str, preferred_times: str | None = None) -> str:
    """Schedule a meeting with HR specialist.

    Args:
        meeting_topic: Topic for meeting (benefits, career, policy, general).
        preferred_times: Optional preferred meeting times.

    Returns:
        Meeting scheduling confirmation and calendar invite.
    """
    meeting_id = f"HRMEET{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:4].upper()}"

    return f"""**HR Meeting Scheduled:**

📅 **Meeting ID:** {meeting_id}
💼 **Topic:** {meeting_topic.replace("_", " ").title()}
🕐 **Preferred Times:** {preferred_times or "Will coordinate via email"}

**What Happens Next:**

**1. HR Specialist Assignment (Within 4 hours):**
   - Based on your topic, you'll be matched with the right specialist:
     - **Benefits:** Benefits Coordinator
     - **Career:** Career Development Advisor
     - **Policy:** HR Generalist
     - **General:** Your HR Business Partner

**2. Calendar Invite (Within 24 hours):**
   - HR specialist will send Outlook invite
   - Meeting: 30-60 minutes (depending on topic)
   - Format: Virtual (Teams) or In-person (your choice)
   - Includes: Dial-in info, agenda, any prep materials

**3. Pre-Meeting Prep:**
   - Review any materials sent in advance
   - Prepare questions you want to discuss
   - Gather relevant documents (if applicable)

**📋 What to Expect in Your Meeting:**

**Meeting Format:**
- Introduction and agenda review (5 min)
- Discussion of your topic/questions (40-50 min)
- Next steps and action items (5 min)
- Q&A throughout

**Come Prepared With:**
- Specific questions you have
- Any relevant background or context
- Documentation (if applicable)
- Your goals for the meeting

**🔒 Confidentiality:**
Your discussion with HR is confidential. Information will only be shared with your consent, except when required by law or policy (safety, legal compliance).

**💡 Meeting Topics We Handle:**

**Benefits:**
- Health insurance questions
- 401(k) and retirement planning
- Life insurance and disability
- Open enrollment assistance
- Claims issues

**Career Development:**
- Career path planning
- Skills development
- Promotion readiness
- Performance improvement
- Job search (internal mobility)

**Policy Questions:**
- Leave policies (PTO, parental, medical)
- Remote work arrangements
- Time and attendance
- Code of conduct
- Any company policy

**General HR:**
- Onboarding questions
- Payroll and compensation
- Employment verification
- Personal changes (name, address)
- Other HR topics

**📧 Confirmation Email:**

You'll receive an email within 4 hours with:
- Name of your HR specialist
- Their contact information
- Proposed meeting times
- Agenda and any prep materials
- How to confirm or reschedule

**📞 Need to Reschedule?**

Life happens! To reschedule:
- Reply to the calendar invite with new times
- Email the HR specialist directly
- Call HR Service Center: ext. 5000

**🆘 If This Is Urgent:**

**Can't Wait for a Meeting?**
- Call HR Service Center: ext. 5000 (Mon-Fri, 8 AM - 5 PM)
- Email: hr@company.com (24-hour response)
- EAP (confidential support): 1-800-XXX-XXXX (24/7)
- Ethics Hotline (anonymous): 1-800-XXX-XXXX (24/7)

**💬 Virtual Meeting Tips:**

If meeting via Teams:
- Test your audio/video beforehand
- Find a quiet, private space
- Have a pen and paper for notes
- Keep phone handy as backup

**🙏 We're Here to Help:**

Thank you for reaching out. Our goal is to provide you with the information and support you need to succeed and thrive at the company.

**Questions?**
Contact HR Service Center: hr@company.com or ext. 5000

**We look forward to meeting with you!**"""
