from fastapi import APIRouter, Query, HTTPException
import os
from dotenv import load_dotenv
from typing import Optional, List
from fastapi import Query
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
    responses={404: {"description": "Not found"}},
)

load_dotenv(override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

VECTOR_STORE_PATH = "app/faiss/"
PROMPT_PATH = "prompts/rag_prompt.txt"
FILES_DIRECTORY = "files/"

# Sentence Window Retrieval configuration
SENTENCE_WINDOW_SIZE = 3  # Number of sentences before and after to include
SENTENCE_CHUNK_SIZE = 2   # Number of sentences per chunk for embedding


def load_prompt():
    """Load the RAG prompt template"""
    try:
        with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return None
    
def vector_store_exists():
    """Check if the vector store exists locally"""
    return os.path.exists(VECTOR_STORE_PATH)

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
    
def format_context_with_basic_chunks(results):
    """Format the vector store search results into context for the prompt"""
    context_parts = []
    for i, doc in enumerate(results, 1):
        source_type = doc.metadata.get('source', 'unknown')
        context_parts.append(f"{i}. [{source_type.upper()}] {doc.page_content}")
    
    return "\n".join(context_parts) if context_parts else "Aucune information pertinente trouvée."
    
def format_context_with_sentence_windows(docs: List[Document]) -> str:
    """Format documents with sentence window information for better context"""
    context_parts = []
    
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get('filename', 'unknown')
        source_type = doc.metadata.get('source', 'unknown')
        
        if doc.metadata.get("expanded_context"):
            expansion_info = doc.metadata.get("expansion_info", "")
            context_parts.append(
                f"{i}. [{source_type.upper()}] {filename}\n"
                f"   Context: {expansion_info}\n"
                f"   Content: {doc.page_content}\n"
            )
        else:
            context_parts.append(
                f"{i}. [{source_type.upper()}] {filename}\n"
                f"   Content: {doc.page_content}\n"
            )
    
    return "\n".join(context_parts) if context_parts else "Aucune information pertinente trouvée."


def retrieve_with_sentence_window(vector_store, query: str, k: int = 3, window_size: int = SENTENCE_WINDOW_SIZE) -> List[Document]:
    """
    Retrieve documents with expanded sentence window context
    
    Args:
        vector_store: The FAISS vector store
        query: Search query
        k: Number of documents to retrieve
        window_size: Number of sentences before and after to include in context
    
    Returns:
        List of documents with expanded context
    """
    # First, do the standard similarity search
    retrieved_docs = vector_store.similarity_search(query, k=k)
    
    expanded_docs = []
    
    for doc in retrieved_docs:
        if doc.metadata.get("chunk_type") == "sentence_window":
            # Get the sentence indices and windows from metadata
            sentence_indices = doc.metadata.get("sentence_indices", [])
            sentence_windows = doc.metadata.get("sentence_windows", {})
            all_sentences = doc.metadata.get("all_sentences", [])
            
            if sentence_indices and sentence_windows:
                # Find the range of sentences to include with window
                start_sentence = min(sentence_indices)
                end_sentence = max(sentence_indices)
                
                # Expand the window 
                expanded_start = max(0, start_sentence - window_size)
                expanded_end = min(len(all_sentences), end_sentence + window_size + 1)
                
                # Get the expanded context
                expanded_sentences = all_sentences[expanded_start:expanded_end]
                expanded_text = ' '.join(expanded_sentences)
                
                # Create new document with expanded context
                expanded_doc = Document(
                    page_content=expanded_text,
                    metadata={
                        **doc.metadata,
                        "original_chunk": doc.page_content,
                        "expanded_context": True,
                        "expanded_start_sentence": expanded_start,
                        "expanded_end_sentence": expanded_end - 1,
                        "window_size_used": window_size,
                    }
                )
                expanded_docs.append(expanded_doc)
            else:
                # Fallback to original document if no sentence info
                expanded_docs.append(doc)
        else:
            # For non-sentence-window chunks, use original
            expanded_docs.append(doc)
    
    return expanded_docs

def generate_rag_response(question: str, context: str, llm):
    """Generate a response using the RAG prompt"""
    prompt_template = load_prompt()
    if not prompt_template:
        return "Erreur : Impossible de charger le template de prompt."
    
    # Format the prompt with the question and context
    formatted_prompt = prompt_template.format(question=question, context=context)
    
    try:
        response = llm.invoke(formatted_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Erreur lors de la génération de la réponse : {e}"


@router.post("/search-with-basic-rag")
async def use_rag(question: str = Query(..., description="Question à poser au système RAG avec Sentence Window")):
    """
    Make a RAG query using basic chunk retrieval
    """
    # Set the API key for GoogleGenerativeAIEmbeddings 
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
        transport="rest"  # Use REST instead of gRPC to avoid ALTS
    )
    
    # Initialize the LLM for generating responses
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1
    )

    # Check if vector store exists
    if not vector_store_exists():
        raise HTTPException(status_code=404, detail="Vector store not found. Please embed documents first.")
    
    # Load vector store
    vector_store = load_vector_store(embeddings)
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Failed to load vector store")

    # Perform similarity search with the user's question
    results = vector_store.similarity_search(
        question,
        k=3,  # Get top 3 results
        # filter={"source": "tweet"},  
    )
    
    # Format the context from search results (basic chunks)
    context = format_context_with_basic_chunks(results)

    # Generate RAG response
    rag_response = generate_rag_response(question, context, llm)

    return {
        "question": question,
        "documents_found": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get('source', 'unknown')
            } for doc in results
        ],
        "context": context,
        "rag_response": rag_response,
        "vector_store_loaded": vector_store_exists()
    }

@router.post("/search-with-window")
async def use_rag_with_sentence_window(
    question: str = Query(..., description="Question à poser au système RAG avec Sentence Window"),
    k: int = Query(default=3, description="Nombre de chunks à récupérer"),
    window_size: int = Query(default=SENTENCE_WINDOW_SIZE, description="Taille de la fenêtre de contexte (nombre de phrases avant/après)")
):
    """
    Make a RAG query using Sentence Window Retrieval
    """
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
            transport="rest"
        )

        # Initialize the LLM for generating responses
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
        
        # Check if vector store exists
        if not vector_store_exists():
            raise HTTPException(status_code=404, detail="Vector store not found. Please embed documents first.")
        
        # Load vector store
        vector_store = load_vector_store(embeddings)
        if vector_store is None:
            raise HTTPException(status_code=500, detail="Failed to load vector store")
        
        # Perform sentence window retrieval
        retrieved_docs = retrieve_with_sentence_window(
            vector_store=vector_store,
            query=question,
            k=k,
            window_size=window_size
        )

        # Format context from retrieved chunks (sentences)
        expanded_context = format_context_with_sentence_windows(retrieved_docs)
        
        # Generate RAG response
        rag_response = generate_rag_response(question, expanded_context, llm)

        return {
                "question": question,
                "message": f"Recherche terminée avec Sentence Window Retrieval",
                "expanded_context": expanded_context,
                "total_documents": len(retrieved_docs),
                "window_config": {
                    "window_size": window_size,
                    "chunks_requested": k,
                    "sentence_chunk_size": SENTENCE_CHUNK_SIZE
                },
                "rag_response" : rag_response
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during sentence window search: {str(e)}")

