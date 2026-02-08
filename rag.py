"""RAG pipeline for medical conversations."""

import os
from typing import List
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import Config
from llm import LLMManager


class RAGPipeline:
    """RAG pipeline for retrieving relevant medical conversations."""

    def __init__(self, csv_path: str = None):
        self.csv_path = csv_path or Config.CSV_PATH
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

    def load_documents(self) -> List[Document]:
        """Load documents from CSV file."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        loader = CSVLoader(
            file_path=self.csv_path,
            encoding="utf-8",
        )
        documents = loader.load()
        return documents

    def create_vector_store(self, documents: List[Document] = None) -> FAISS:
        """Create vector store from documents."""
        if documents is None:
            documents = self.load_documents()

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")

        # Create vector store
        embeddings = LLMManager.get_embeddings()
        self.vector_store = FAISS.from_documents(chunks, embeddings)

        return self.vector_store

    def save_vector_store(self, path: str = None):
        """Save vector store to disk."""
        path = path or Config.VECTOR_STORE_PATH
        if self.vector_store:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.vector_store.save_local(path)
            print(f"Vector store saved to {path}")

    def load_vector_store(self, path: str = None) -> FAISS:
        """Load vector store from disk."""
        path = path or Config.VECTOR_STORE_PATH
        if os.path.exists(path):
            embeddings = LLMManager.get_embeddings()
            self.vector_store = FAISS.load_local(
                path, embeddings, allow_dangerous_deserialization=True
            )
            print(f"Vector store loaded from {path}")
            return self.vector_store
        return None

    def get_retriever(self, k: int = None):
        """Get retriever from vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")

        k = k or Config.TOP_K_RESULTS
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant documents for a query."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")

        k = k or Config.TOP_K_RESULTS
        return self.vector_store.similarity_search(query, k=k)
