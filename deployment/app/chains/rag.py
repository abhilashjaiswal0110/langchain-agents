"""RAG (Retrieval Augmented Generation) chain."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = InMemoryVectorStore(embeddings)

# Add some sample documents for demonstration
sample_docs = [
    "LangChain is a framework for developing applications powered by language models.",
    "LangChain provides tools for building chains, agents, and RAG applications.",
    "RAG stands for Retrieval Augmented Generation, which combines retrieval with generation.",
    "LangServe helps deploy LangChain applications as REST APIs.",
    "The LangChain ecosystem includes LangSmith for debugging and monitoring.",
]

# Add documents to the vector store
vectorstore.add_texts(sample_docs)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG prompt template
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant. Answer the question based only on the following context:

{context}

If you cannot answer the question based on the context, say so.""",
    ),
    ("human", "{question}"),
])

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def format_docs(docs: list) -> str:
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


# Build the RAG chain
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)
