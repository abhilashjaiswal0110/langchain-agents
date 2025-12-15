"""Simple chat chain using LangChain."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Simple chat prompt template (no history for simplicity)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. Answer questions accurately and concisely.",
    ),
    ("human", "{input}"),
])

# Initialize the LLM (uses OPENAI_API_KEY from environment)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)

# Build the chat chain
chat_chain = prompt | llm | StrOutputParser()
