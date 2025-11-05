from fastapi import APIRouter, HTTPException
import faiss
import os
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from dotenv import load_dotenv
from typing import List, Dict
import pandas as pd
from pathlib import Path
#from langchain.text_splitter import MarkdownTextSplitter

router = APIRouter(
    prefix="/embedding",
    tags=["Embedding"],
    responses={404: {"description": "Not found"}},
)

load_dotenv(override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define the path where the vector store will be saved
VECTOR_STORE_PATH = "app/faiss/"
FILES_DIRECTORY = "files/"

def vector_store_exists():
    """Check if the vector store exists locally"""
    return os.path.exists(VECTOR_STORE_PATH)

def create_new_vector_store(embeddings):
    """Create a new vector store"""
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    return vector_store

def load_vector_store(embeddings):
    """Load the vector store from local storage"""
    try:
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded from local storage")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

def save_vector_store(vector_store):
    """Save the vector store to local storage"""
    try:
        vector_store.save_local(VECTOR_STORE_PATH)
        print("Vector store saved to local storage")
    except Exception as e:
        print(f"Error saving vector store: {e}")

def load_files_from_directory():
    """Load and process XLSX files from the files/ directory"""
    
    documents = []
    
    # Get all XLSX files from the directory
    csv_files = list(Path(FILES_DIRECTORY).glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {FILES_DIRECTORY}")
        return documents

    for file_path in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, encoding='utf-8', sep=';')
                        
            # Extract only the 'Title' column and create chunks from it
            if 'Title' in df.columns:
                chunks = df['Title'].dropna().astype(str).tolist()
            else:
                print(f"Warning: 'Title' column not found in {file_path.name}")
                chunks = []
            
            # Create Document objects for each chunk
            for idx, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty chunks
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "response": df["Content"][idx] if 'Content' in df.columns else "",
                            "school": df["Écoles"][idx] if 'Écoles' in df.columns else "",
                        }
                    )
                    documents.append(doc)
            
            print(f"Loaded {len(chunks)} chunks from {file_path.name}")
            
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            continue
    
    return documents


@router.post("/embed-files")
async def embed_files():
    """Embed all markdown files from the files/ directory into the vector store"""
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
            transport="rest"
        )
        
        # Load  files from directory first to know what we're working with
        documents = load_files_from_directory()
        
        if not documents:
            return {
                "message": f"No files found in {FILES_DIRECTORY}",
                "chunks_created": 0,
                "files_processed": 0
            }
        
        # Create a new vector store (this ensures no duplicates)
        # may be improved later with more complex logic (ex: checking existing IDs)
        print("Creating new vector store to ensure no duplicates...")
        vector_store = create_new_vector_store(embeddings)
        
        # Generate UUIDs for the documents
        uuids = [str(uuid4()) for _ in range(len(documents))]

        # Add all documents to the vector store
        print(f"Adding {len(documents)} chunks to vector store...")
        vector_store.add_documents(documents=documents, ids=uuids)

        # Save the vector store
        save_vector_store(vector_store)
        
        # Collect statistics for the response
        files_processed = set([doc.metadata["filename"] for doc in documents])
        chunks_by_file = {}
        
        print(f"Successfully embedded {len(documents)} chunks from {len(files_processed)} file(s)")
        
        return {
            "message": f"Successfully embedded all markdown files from {FILES_DIRECTORY}",
            "chunks_created": len(documents),
            "files_processed": len(files_processed),
            "vector_store_recreated": True,
            "files_details": {
                filename: {
                    "total_chunks": len(chunks),
                    "chunks": chunks
                } for filename, chunks in chunks_by_file.items()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error embedding markdown files: {str(e)}")

@router.get("/inspect-vector-store")
async def inspect_vector_store():
    """Inspect the vector store content and metadata"""
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
            transport="rest"
        )
        
        # Check if vector store exists
        if not vector_store_exists():
            return {
                "message": "Vector store does not exist",
                "total_documents": 0,
                "files": {}
            }
        
        # Load vector store
        vector_store = load_vector_store(embeddings)
        if vector_store is None:
            return {
                "message": "Failed to load vector store",
                "total_documents": 0,
                "files": {}
            }
        
        # Get all documents from the vector store
        docs = vector_store.similarity_search("", k=1000)  # Get up to 1000 docs
        
        if not docs:
            return {
                "message": "Vector store is empty",
                "total_documents": 0,
                "files": {}
            }
        
        # Organize documents by file
        files_info = {}
        for doc in docs:
            filename = doc.metadata.get("filename", "unknown")
            if filename not in files_info:
                files_info[filename] = {
                    "chunks": [],
                    "file_size": doc.metadata.get("file_size", "unknown")
                }
            
            files_info[filename]["chunks"].append({
                "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "metadata": {
                    "response": doc.metadata.get("response", ""),
                    "school": doc.metadata.get("school", ""),
                    "chunk_id": doc.metadata.get("chunk_id", "unknown")
                }
            })
        
        # Sort chunks by chunk_id for each file
        for filename in files_info:
            files_info[filename]["chunks"].sort(key=lambda x: x.get("chunk_id", 0))
            files_info[filename]["actual_chunks_count"] = len(files_info[filename]["chunks"])
        
        return {
            "message": "Vector store inspection completed",
            "total_documents": len(docs),
            "total_files": len(files_info),
            "files": files_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inspecting vector store: {str(e)}")

