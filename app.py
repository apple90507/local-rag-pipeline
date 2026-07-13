import os
import hashlib
import json
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    get_response_synthesizer,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import PyMuPDFReader
from sentence_transformers import CrossEncoder

def init_settings():
    """Initialize global settings for LLM, Embedding, and Node Parser."""
    Settings.llm = Ollama(
        model="llama3",
        request_timeout=120.0,
        system_prompt="你是一個專業的繁體中文助理。請務必完全使用『繁體中文』(Taiwan) 回答所有問題，絕對不要使用英文或簡體中文。"
    )
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=30)

def calculate_dir_md5(data_dir="data"):
    """Calculate a single MD5 hash representing all files in the data directory."""
    hasher = hashlib.md5()
    if not os.path.exists(data_dir):
        return ""
    
    # Sort file names to ensure deterministic hash order
    for file_name in sorted(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                # Read in chunks to prevent memory overflow for large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
    return hasher.hexdigest()

def verify_cache_integrity(persist_dir, current_md5):
    """Verify if the local vector cache matches the current data MD5."""
    manifest_path = os.path.join(persist_dir, "cache_manifest.json")
    docstore_path = os.path.join(persist_dir, "docstore.json")
    
    # Check if necessary cache files exist
    if not (os.path.exists(persist_dir) and os.path.exists(docstore_path) and os.path.exists(manifest_path)):
        return False
        
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest.get("data_md5") == current_md5
    except (json.JSONDecodeError, IOError):
        return False

def get_vector_index(persist_dir="./storage", data_dir="data"):
    """Load index from local storage if valid, otherwise rebuild and cache it."""
    current_md5 = calculate_dir_md5(data_dir)
    
    if verify_cache_integrity(persist_dir, current_md5):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)
    
    # Cache miss or corrupted: Trigger re-indexing
    documents = SimpleDirectoryReader(
        data_dir, 
        file_extractor={".pdf": PyMuPDFReader()}
    ).load_data()
    
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
    
    # Write the current MD5 hash into the manifest file
    manifest_path = os.path.join(persist_dir, "cache_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"data_md5": current_md5}, f, ensure_ascii=False, indent=4)
        
    return index

def main():
    init_settings()
    index = get_vector_index()
    
    query_str = "請根據這份文件，用繁體中文條列出三個最重要的核心重點。"
    
    # Stage 1: Vector Retrieval (Bi-Encoder)
    retriever = index.as_retriever(similarity_top_k=5)
    retrieved_nodes = retriever.retrieve(query_str)
    
    # Stage 2: Two-Stage Reranking (Cross-Encoder)
    rerank_model = CrossEncoder("BAAI/bge-reranker-base")
    pairs = [[query_str, node.node.get_content()] for node in retrieved_nodes]
    rerank_scores = rerank_model.predict(pairs)
    
    for node, score in zip(retrieved_nodes, rerank_scores):
        node.score = float(score)
        
    retrieved_nodes.sort(key=lambda x: x.score, reverse=True)
    reranked_nodes = retrieved_nodes[:2]
    
    # Stage 3: Response Synthesis
    synthesizer = get_response_synthesizer()
    response = synthesizer.synthesize(query_str, nodes=reranked_nodes)
    
    print("\n=== Result ===")
    print(response)

if __name__ == "__main__":
    main()