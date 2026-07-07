import os
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

def get_vector_index(persist_dir="./storage", data_dir="data"):
    """Load index from local storage if exists, otherwise build a new one."""
    docstore_path = os.path.join(persist_dir, "docstore.json")
    
    if os.path.exists(persist_dir) and os.path.exists(docstore_path):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)
    
    # Ingest documents using PyMuPDFReader for accurate PDF parsing
    documents = SimpleDirectoryReader(
        data_dir, 
        file_extractor={".pdf": PyMuPDFReader()}
    ).load_data()
    
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
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