"""Document RAG Chain with PDF, Word, and TXT support.

This module provides a document-based Retrieval-Augmented Generation (RAG) system
that can process and query documents in various formats.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable


class DocumentRAGChain:
    """Document RAG Chain for processing and querying documents.

    Supports PDF, Word (.docx), and plain text (.txt) files.
    Uses FAISS for vector storage and OpenAI for embeddings and LLM.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
    ) -> None:
        """Initialize the Document RAG Chain.

        Args:
            chunk_size: Size of text chunks for splitting documents.
            chunk_overlap: Overlap between chunks to maintain context.
            model: OpenAI model to use for generation.
            temperature: Temperature for LLM responses.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.temperature = temperature

        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # Vector store (initialized when documents are loaded)
        self.vector_store: FAISS | None = None
        self.documents: list[Document] = []
        self.document_metadata: dict[str, Any] = {}

        # RAG prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful assistant that answers questions based on the provided document context.

Instructions:
- Answer questions accurately based ONLY on the provided context
- If the answer is not in the context, say "I couldn't find this information in the document"
- Cite specific parts of the document when possible
- Be concise but thorough""",
            ),
            (
                "human",
                """Context from document:
{context}

Question: {question}

Answer:""",
            ),
        ])

    def _get_loader(self, file_path: str) -> PyPDFLoader | TextLoader | Docx2txtLoader:
        """Get the appropriate document loader based on file extension.

        Args:
            file_path: Path to the document file.

        Returns:
            Document loader instance.

        Raises:
            ValueError: If file type is not supported.
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            return PyPDFLoader(file_path)
        elif ext == ".txt":
            return TextLoader(file_path, encoding="utf-8")
        elif ext in [".docx", ".doc"]:
            return Docx2txtLoader(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {ext}. Supported types: .pdf, .txt, .docx"
            )

    @traceable(name="load_document", tags=["doc-rag", "ingestion"])
    def load_document(self, file_path: str) -> dict[str, Any]:
        """Load and process a document into the vector store.

        Args:
            file_path: Path to the document file.

        Returns:
            Dictionary with loading status and document info.
        """
        try:
            # Get appropriate loader
            loader = self._get_loader(file_path)

            # Load document
            raw_documents = loader.load()

            # Split into chunks
            chunks = self.text_splitter.split_documents(raw_documents)

            # Add metadata
            file_name = Path(file_path).name
            for i, chunk in enumerate(chunks):
                chunk.metadata["source_file"] = file_name
                chunk.metadata["chunk_index"] = i

            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vector_store.add_documents(chunks)

            # Store documents
            self.documents.extend(chunks)
            self.document_metadata[file_name] = {
                "num_chunks": len(chunks),
                "total_characters": sum(len(c.page_content) for c in chunks),
                "file_type": Path(file_path).suffix.lower(),
            }

            return {
                "status": "success",
                "file_name": file_name,
                "chunks_created": len(chunks),
                "total_documents": len(self.documents),
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    @traceable(name="load_from_bytes", tags=["doc-rag", "ingestion"])
    def load_from_bytes(
        self, content: bytes, filename: str
    ) -> dict[str, Any]:
        """Load a document from bytes (for file uploads).

        Args:
            content: File content as bytes.
            filename: Original filename with extension.

        Returns:
            Dictionary with loading status and document info.
        """
        # Create temporary file
        ext = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            result = self.load_document(tmp_path)
            result["original_filename"] = filename
            return result
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

    def _format_docs(self, docs: list[Document]) -> str:
        """Format retrieved documents into a single context string.

        Args:
            docs: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "unknown")
            formatted.append(f"[Source {i}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    @traceable(name="query_document", tags=["doc-rag", "query"])
    def query(self, question: str, k: int = 4) -> dict[str, Any]:
        """Query the loaded documents.

        Args:
            question: Question to ask about the documents.
            k: Number of relevant chunks to retrieve.

        Returns:
            Dictionary with answer and source information.
        """
        if self.vector_store is None:
            return {
                "status": "error",
                "error": "No documents loaded. Please upload a document first.",
            }

        try:
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            )

            # Build RAG chain
            rag_chain = (
                {
                    "context": retriever | self._format_docs,
                    "question": RunnablePassthrough(),
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

            # Execute query
            answer = rag_chain.invoke(question)

            # Get source documents for reference
            source_docs = retriever.invoke(question)
            sources = [
                {
                    "source": doc.metadata.get("source_file", "unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", -1),
                    "preview": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content,
                }
                for doc in source_docs
            ]

            return {
                "status": "success",
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources),
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    def get_document_info(self) -> dict[str, Any]:
        """Get information about loaded documents.

        Returns:
            Dictionary with document statistics.
        """
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.documents),
            "documents": self.document_metadata,
            "vector_store_initialized": self.vector_store is not None,
        }

    def clear_documents(self) -> dict[str, str]:
        """Clear all loaded documents from memory.

        Returns:
            Status dictionary.
        """
        self.vector_store = None
        self.documents = []
        self.document_metadata = {}
        return {"status": "success", "message": "All documents cleared"}


# Global instance for the application
doc_rag_chain = DocumentRAGChain()
