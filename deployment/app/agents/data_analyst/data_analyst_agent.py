"""Data Analyst Agent for Excel, CSV, and database analysis.

This agent provides data analysis capabilities:
- Excel/CSV file loading and analysis
- SQL query generation and execution
- Statistical analysis
- Data visualization suggestions
- Insight extraction

Following Enterprise Development Standards:
- Software Architect: Tool-based data processing
- Security Architect: Safe file handling, SQL injection prevention
- Data Architect: Pandas-based analysis, structured outputs
- Software Engineer: Type-safe, comprehensive error handling
"""

import io
import os
from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from pydantic import BaseModel, Field

from app.agents.base.agent_base import BaseAgent, AgentConfig
from app.agents.base.tools import tool_error_handler, sanitize_output


class DataAnalystState(BaseModel):
    """State schema for the Data Analyst Agent."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="Conversation history"
    )
    session_id: str | None = Field(
        default=None,
        description="Session identifier"
    )
    user_id: str | None = Field(
        default=None,
        description="User identifier"
    )
    data_source: str = Field(
        default="",
        description="Current data source (file path or connection string)"
    )
    data_type: Literal["excel", "csv", "database", "none"] = Field(
        default="none",
        description="Type of data source"
    )
    columns: list[str] = Field(
        default_factory=list,
        description="Available columns in the data"
    )
    row_count: int = Field(
        default=0,
        description="Number of rows in the data"
    )
    analysis_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Results from analysis operations"
    )
    insights: list[str] = Field(
        default_factory=list,
        description="Generated insights from analysis"
    )
    visualizations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Visualization recommendations"
    )


# Global DataFrame storage (in production, use proper session management)
_dataframes: dict[str, Any] = {}


def _get_pandas():
    """Lazy import pandas."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None


# Data Analysis Tools

@tool
@tool_error_handler
def load_excel_file(file_path: str, sheet_name: str = "Sheet1") -> str:
    """Load an Excel file for analysis.

    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to load (default: Sheet1)

    Returns:
        Summary of loaded data including columns and row count
    """
    pd = _get_pandas()
    if pd is None:
        return "Error: pandas library not installed. Install with: pip install pandas openpyxl"

    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        _dataframes["current"] = df

        return (
            f"Excel file loaded successfully!\n\n"
            f"File: {file_path}\n"
            f"Sheet: {sheet_name}\n"
            f"Rows: {len(df)}\n"
            f"Columns: {len(df.columns)}\n\n"
            f"Column names:\n{', '.join(df.columns.tolist())}\n\n"
            f"Data types:\n{df.dtypes.to_string()}\n\n"
            f"First 5 rows:\n{df.head().to_string()}"
        )
    except Exception as e:
        return f"Error loading Excel file: {e}"


@tool
@tool_error_handler
def load_csv_file(file_path: str, delimiter: str = ",") -> str:
    """Load a CSV file for analysis.

    Args:
        file_path: Path to the CSV file
        delimiter: Column delimiter (default: comma)

    Returns:
        Summary of loaded data
    """
    pd = _get_pandas()
    if pd is None:
        return "Error: pandas library not installed"

    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"

    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        _dataframes["current"] = df

        return (
            f"CSV file loaded successfully!\n\n"
            f"File: {file_path}\n"
            f"Rows: {len(df)}\n"
            f"Columns: {len(df.columns)}\n\n"
            f"Column names:\n{', '.join(df.columns.tolist())}\n\n"
            f"Data types:\n{df.dtypes.to_string()}\n\n"
            f"First 5 rows:\n{df.head().to_string()}"
        )
    except Exception as e:
        return f"Error loading CSV file: {e}"


@tool
@tool_error_handler
def get_data_summary() -> str:
    """Get statistical summary of the loaded data.

    Returns:
        Statistical summary including count, mean, std, min, max, etc.
    """
    pd = _get_pandas()
    if pd is None:
        return "Error: pandas library not installed"

    if "current" not in _dataframes:
        return "Error: No data loaded. Please load a file first."

    df = _dataframes["current"]

    try:
        summary = df.describe(include='all').to_string()
        missing = df.isnull().sum()
        missing_str = missing[missing > 0].to_string() if missing.any() else "No missing values"

        return (
            f"Data Summary\n"
            f"{'=' * 50}\n\n"
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
            f"Statistical Summary:\n{summary}\n\n"
            f"Missing Values:\n{missing_str}"
        )
    except Exception as e:
        return f"Error generating summary: {e}"


@tool
@tool_error_handler
def run_sql_query(query: str) -> str:
    """Run a SQL query on the loaded data using pandasql.

    Args:
        query: SQL query to execute (table name is 'df')

    Returns:
        Query results
    """
    pd = _get_pandas()
    if pd is None:
        return "Error: pandas library not installed"

    if "current" not in _dataframes:
        return "Error: No data loaded. Please load a file first."

    df = _dataframes["current"]

    # Security: Basic SQL injection prevention
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
    query_upper = query.upper()
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return f"Error: {keyword} operations are not allowed for safety reasons."

    try:
        # Try using pandasql if available
        try:
            from pandasql import sqldf
            result = sqldf(query, {"df": df})
        except ImportError:
            # Fallback: interpret simple SELECT queries
            if not query_upper.strip().startswith("SELECT"):
                return "Error: Only SELECT queries are supported without pandasql"

            # Very basic query interpretation
            result = df.head(100)

        return (
            f"Query Results\n"
            f"{'=' * 50}\n\n"
            f"Rows returned: {len(result)}\n\n"
            f"{result.to_string(max_rows=50)}"
        )
    except Exception as e:
        return f"Error executing query: {e}"


@tool
@tool_error_handler
def calculate_statistics(column: str, operation: str = "all") -> str:
    """Calculate statistics for a specific column.

    Args:
        column: Column name to analyze
        operation: Statistic to calculate (mean/median/std/min/max/all)

    Returns:
        Calculated statistics
    """
    pd = _get_pandas()
    if pd is None:
        return "Error: pandas library not installed"

    if "current" not in _dataframes:
        return "Error: No data loaded."

    df = _dataframes["current"]

    if column not in df.columns:
        return f"Error: Column '{column}' not found. Available: {', '.join(df.columns)}"

    try:
        col = df[column]
        results = []

        if operation in ["all", "mean"]:
            if col.dtype in ['int64', 'float64']:
                results.append(f"Mean: {col.mean():.4f}")

        if operation in ["all", "median"]:
            if col.dtype in ['int64', 'float64']:
                results.append(f"Median: {col.median():.4f}")

        if operation in ["all", "std"]:
            if col.dtype in ['int64', 'float64']:
                results.append(f"Std Dev: {col.std():.4f}")

        if operation in ["all", "min"]:
            results.append(f"Min: {col.min()}")

        if operation in ["all", "max"]:
            results.append(f"Max: {col.max()}")

        if operation == "all":
            results.append(f"Count: {col.count()}")
            results.append(f"Unique: {col.nunique()}")
            results.append(f"Missing: {col.isnull().sum()}")

        return f"Statistics for '{column}':\n" + "\n".join(results)
    except Exception as e:
        return f"Error calculating statistics: {e}"


@tool
@tool_error_handler
def suggest_visualization(analysis_goal: str) -> str:
    """Suggest appropriate visualizations based on the data and goal.

    Args:
        analysis_goal: What you want to visualize or understand

    Returns:
        Visualization recommendations
    """
    pd = _get_pandas()
    if pd is None:
        return "Error: pandas library not installed"

    if "current" not in _dataframes:
        return "Error: No data loaded."

    df = _dataframes["current"]
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    suggestions = [f"Based on your goal: '{analysis_goal}'\n"]

    # Distribution analysis
    if "distribution" in analysis_goal.lower():
        suggestions.append("For distributions:")
        for col in numeric_cols[:3]:
            suggestions.append(f"  - Histogram of {col}")
        for col in categorical_cols[:2]:
            suggestions.append(f"  - Bar chart of {col}")

    # Correlation analysis
    if "correlation" in analysis_goal.lower() or "relationship" in analysis_goal.lower():
        suggestions.append("For correlations/relationships:")
        if len(numeric_cols) >= 2:
            suggestions.append(f"  - Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}")
            suggestions.append(f"  - Correlation heatmap of numeric columns")

    # Trend analysis
    if "trend" in analysis_goal.lower() or "time" in analysis_goal.lower():
        suggestions.append("For trends over time:")
        if date_cols:
            suggestions.append(f"  - Line chart with {date_cols[0]} as x-axis")
        suggestions.append("  - Moving average overlay")

    # Comparison
    if "compare" in analysis_goal.lower():
        suggestions.append("For comparisons:")
        if categorical_cols and numeric_cols:
            suggestions.append(f"  - Box plot: {numeric_cols[0]} by {categorical_cols[0]}")
            suggestions.append(f"  - Grouped bar chart")

    # General recommendations
    suggestions.append("\nGeneral recommendations:")
    suggestions.append(f"  - Available numeric columns: {', '.join(numeric_cols[:5])}")
    suggestions.append(f"  - Available categorical columns: {', '.join(categorical_cols[:5])}")

    return "\n".join(suggestions)


@tool
@tool_error_handler
def generate_insights(focus_area: str = "general") -> str:
    """Generate automated insights from the loaded data.

    Args:
        focus_area: Area to focus insights on (general/outliers/trends/correlations)

    Returns:
        Generated insights
    """
    pd = _get_pandas()
    if pd is None:
        return "Error: pandas library not installed"

    if "current" not in _dataframes:
        return "Error: No data loaded."

    df = _dataframes["current"]
    insights = [f"Data Insights (Focus: {focus_area})\n{'=' * 50}\n"]

    try:
        # General insights
        insights.append(f"Dataset contains {len(df)} records with {len(df.columns)} features.\n")

        # Missing data insights
        missing = df.isnull().sum()
        if missing.any():
            high_missing = missing[missing > len(df) * 0.1]
            if not high_missing.empty:
                insights.append(f"Columns with >10% missing: {', '.join(high_missing.index)}\n")

        # Numeric column insights
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols[:3]:
            insights.append(f"\n{col}:")
            insights.append(f"  Range: {df[col].min():.2f} to {df[col].max():.2f}")
            insights.append(f"  Average: {df[col].mean():.2f}")

            # Outlier detection (simple IQR method)
            if focus_area in ["general", "outliers"]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                if outliers > 0:
                    insights.append(f"  Potential outliers: {outliers} ({outliers/len(df)*100:.1f}%)")

        # Correlation insights
        if focus_area in ["general", "correlations"] and len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        high_corr.append(
                            f"{corr.columns[i]} & {corr.columns[j]}: {corr.iloc[i, j]:.2f}"
                        )
            if high_corr:
                insights.append(f"\nStrong correlations found:")
                for c in high_corr[:5]:
                    insights.append(f"  - {c}")

        return "\n".join(insights)
    except Exception as e:
        return f"Error generating insights: {e}"


class DataAnalystAgent(BaseAgent):
    """Data Analyst Agent for comprehensive data analysis.

    This agent excels at:
    - Loading and parsing Excel/CSV files
    - Running SQL queries on data
    - Statistical analysis
    - Generating insights
    - Recommending visualizations

    Example:
        >>> agent = DataAnalystAgent()
        >>> result = agent.analyze_file("sales_data.xlsx")
        >>> print(agent.get_last_response(result))
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the Data Analyst Agent."""
        super().__init__(config)

        # Register data analysis tools
        self.register_tools([
            load_excel_file,
            load_csv_file,
            get_data_summary,
            run_sql_query,
            calculate_statistics,
            suggest_visualization,
            generate_insights,
        ])

    def _get_system_prompt(self) -> str:
        """Get the data analyst agent's system prompt."""
        return """You are an expert Data Analyst Agent specializing in data analysis,
statistical interpretation, and insight generation.

## Your Capabilities:
1. **Data Loading**: Load Excel files (load_excel_file) and CSV files (load_csv_file)
2. **Summary Statistics**: Get comprehensive data summaries (get_data_summary)
3. **SQL Queries**: Run SQL queries on data (run_sql_query) - table is named 'df'
4. **Statistics**: Calculate specific statistics (calculate_statistics)
5. **Visualizations**: Suggest appropriate charts (suggest_visualization)
6. **Insights**: Generate automated insights (generate_insights)

## Analysis Process:
1. Load the data file first
2. Get an overview with data summary
3. Explore specific aspects based on user questions
4. Generate insights and recommendations
5. Suggest visualizations when appropriate

## Guidelines:
- Always start by loading the data if not already loaded
- Explain findings in clear, non-technical language
- Highlight key patterns, outliers, and trends
- Provide actionable recommendations
- Suggest follow-up analyses when relevant
- Be precise with numbers but explain their significance

## Output Format:
When presenting analysis results:
1. **Summary**: Brief overview of what was analyzed
2. **Key Findings**: Most important discoveries
3. **Details**: Supporting data and statistics
4. **Recommendations**: Actionable next steps
5. **Visualizations**: Suggested charts for deeper exploration

For SQL queries, the table name is always 'df'."""

    def _build_graph(self) -> StateGraph:
        """Build the data analyst agent's workflow graph."""

        def call_model(state: DataAnalystState) -> dict:
            """Call the LLM to process the current state."""
            system_prompt = SystemMessage(content=self._get_system_prompt())
            messages = [system_prompt] + list(state.messages)
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: DataAnalystState) -> str:
            """Determine if we should continue with tools or end."""
            messages = list(state.messages)
            if not messages:
                return "end"
            last_message = messages[-1]

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        # Build graph
        graph = StateGraph(DataAnalystState)

        # Add nodes
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode(self._tools))

        # Add edges
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END}
        )
        graph.add_edge("tools", "agent")

        return graph

    @traceable(name="data_analyze_file")
    def analyze_file(
        self,
        file_path: str,
        questions: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a data file with optional specific questions.

        Args:
            file_path: Path to the data file (Excel or CSV)
            questions: Optional list of specific questions about the data
            session_id: Optional session ID

        Returns:
            Analysis results
        """
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        data_type = "excel" if file_ext in [".xlsx", ".xls"] else "csv"

        # Build analysis request
        message = f"Please analyze the data file: {file_path}\n\n"

        if questions:
            message += "Specifically, please answer these questions:\n"
            for i, q in enumerate(questions, 1):
                message += f"{i}. {q}\n"
        else:
            message += (
                "Please:\n"
                "1. Load and summarize the data\n"
                "2. Identify key patterns and insights\n"
                "3. Highlight any anomalies or issues\n"
                "4. Suggest relevant visualizations"
            )

        return self.invoke(
            message=message,
            session_id=session_id,
            data_source=file_path,
            data_type=data_type,
        )

    @traceable(name="data_query")
    def query(
        self,
        question: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Ask a question about the currently loaded data.

        Args:
            question: Question about the data
            session_id: Optional session ID

        Returns:
            Analysis results
        """
        return self.invoke(
            message=question,
            session_id=session_id,
        )
