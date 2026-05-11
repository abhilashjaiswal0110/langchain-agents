"""Simple chat chain using LangChain."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.agents.base.llm_factory import get_llm

# Simple chat prompt template (no history for simplicity)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Answer questions accurately and concisely.",
        ),
        ("human", "{input}"),
    ]
)

# Initialize the LLM (uses factory with Azure OpenAI as primary)
llm = get_llm(temperature=0.7)

# Build the chat chain
chat_chain = prompt | llm | StrOutputParser()
