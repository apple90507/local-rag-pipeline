import os
import hashlib
import json
import sqlite3
import streamlit as st
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

DB_PATH = "chat_history.db"

# ==================== 1. Database Layer (SQLite3) ====================

def init_db():
    """Initialize SQLite database and create chat history table if not exists."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def save_message_to_db(session_id, role, content):
    """Persist a single chat message into the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_logs (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()

def load_chat_history_from_db(session_id, limit=None):
    """Fetch chat history from database. Supports sliding window via LIMIT."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if limit:
            cursor.execute(
                "SELECT role, content FROM chat_logs WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        else:
            cursor.execute(
                "SELECT role, content FROM chat_logs WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,)
            )
            rows = cursor.fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in rows]

# ==================== 2. Resource Caching Layer ====================

@st.cache_resource
def init_global_settings():
    """Initialize core LLM and Embedding components inside global cache."""
    Settings.llm = Ollama(
        model="llama3",
        request_timeout=120.0,
        system_prompt="你是一個專業的繁體中文助理。請務必完全使用『繁體中文』(Taiwan) 回答所有問題，絕對不要使用英文或簡體中文。"
    )
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=30)

@st.cache_resource
def load_rerank_model():
    """Load Cross-Encoder model into resource cache to prevent memory leaks."""
    return CrossEncoder("BAAI/bge-reranker-base")

def calculate_dir_md5(data_dir="data"):
    """Calculate MD5 fingerprint of the target directory with chunking."""
    hasher = hashlib.md5()
    if not os.path.exists(data_dir):
        return ""
    for file_name in sorted(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
    return hasher.hexdigest()

def verify_cache_integrity(persist_dir, current_md5):
    """Verify metadata manifest for vector storage validation."""
    manifest_path = os.path.join(persist_dir, "cache_manifest.json")
    docstore_path = os.path.join(persist_dir, "docstore.json")
    if not (os.path.exists(persist_dir) and os.path.exists(docstore_path) and os.path.exists(manifest_path)):
        return False, "Cache Miss: Index not found"
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("data_md5") == current_md5:
            return True, "Cache Hit: Integrity verified via MD5"
        return False, "Cache Miss: Data modification detected"
    except (json.JSONDecodeError, IOError):
        return False, "Cache Corrupted: Manifest error"

@st.cache_resource
def get_vector_index(current_md5, persist_dir="./storage", data_dir="data"):
    """Fetch or rebuild the Vector Store Index."""
    is_valid, _ = verify_cache_integrity(persist_dir, current_md5)
    if is_valid:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)
    documents = SimpleDirectoryReader(data_dir, file_extractor={".pdf": PyMuPDFReader()}).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
    manifest_path = os.path.join(persist_dir, "cache_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"data_md5": current_md5}, f, ensure_ascii=False, indent=4)
    return index

# ==================== 3. Contextual Query Rewriting Engine ====================

def rewrite_query_with_history(current_query, db_history):
    """Refactor sequential sub-queries into a standalone query based on database context."""
    if not db_history:
        return current_query
        
    history_str = ""
    for msg in db_history:
        role_label = "User" if msg["role"] == "user" else "AI"
        history_str += f"{role_label}: {msg['content']}\n"
        
    prompt = f"""請根據以下對話紀錄以及一個後續問題，將該後續問題重寫成一個「完全獨立、不需要上下文也能看懂」的繁體中文查詢語句，以便進行向量資料庫檢索。

[對話紀錄]
{history_str}

[後續問題]
{current_query}

[嚴格限制]
1. 請直接輸出重寫後的獨立查詢語句，絕對不要夾帶任何解釋、開場白、備註或回答問題。
2. 保持專業的繁體中文語意。
3. 如果該後續問題已經很完整，不需要上下文就能理解，請直接原樣輸出。

獨立查詢語句："""

    return Settings.llm.complete(prompt).text.strip()

# ==================== 4. UI Layer & Orchestration ====================

def main():
    st.set_page_config(page_title="Enterprise RAG System", page_icon="⚙️", layout="wide")
    st.title("Enterprise RAG Engine")
    st.caption("Two-Stage Retrieval Pipeline with SQLite Session Persistence")
    
    init_db()
    init_global_settings()
    rerank_model = load_rerank_model()
    
    CURRENT_SESSION = "default_user_session"
    
    with st.sidebar:
        st.subheader("System Status")
        current_md5 = calculate_dir_md5("data")
        is_cache_hit, status_msg = verify_cache_integrity("./storage", current_md5)
        
        if is_cache_hit:
            st.success(status_msg)
        else:
            st.warning(status_msg)
            
        st.text_input("Repository Hash (MD5)", value=current_md5, disabled=True)
        st.caption("Database Engine: SQLite3 (Local)")

    index = get_vector_index(current_md5)
    ui_messages = load_chat_history_from_db(CURRENT_SESSION)

    for message in ui_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query_str := st.chat_input("Ask a question about the repository..."):
        save_message_to_db(CURRENT_SESSION, "user", query_str)
        with st.chat_message("user"):
            st.markdown(query_str)

        with st.chat_message("assistant"):
            with st.spinner("Processing context from database..."):
                sliding_window_history = load_chat_history_from_db(CURRENT_SESSION, limit=4)
                search_query = rewrite_query_with_history(query_str, sliding_window_history[:-1])
            
            st.caption(f"Optimized Search Query: `{search_query}`")
            
            with st.spinner("Executing two-stage retrieval..."):
                # Stage 1: Vector Retrieval
                retriever = index.as_retriever(similarity_top_k=5)
                retrieved_nodes = retriever.retrieve(search_query)
                
                # Stage 2: Cross-Encoder Reranking
                pairs = [[search_query, node.node.get_content()] for node in retrieved_nodes]
                rerank_scores = rerank_model.predict(pairs)
                
                for node, score in zip(retrieved_nodes, rerank_scores):
                    node.score = float(score)
                    
                retrieved_nodes.sort(key=lambda x: x.score, reverse=True)
                reranked_nodes = retrieved_nodes[:2]
                
                # Stage 3: Response Synthesis
                synthesizer = get_response_synthesizer()
                response = synthesizer.synthesize(query_str, nodes=reranked_nodes)
                
                st.markdown(response)
                save_message_to_db(CURRENT_SESSION, "assistant", str(response))
                
                with st.expander("Retrieval & Reranking Diagnostics"):
                    for i, node_with_score in enumerate(reranked_nodes):
                        st.markdown(f"**Rank {i+1} Node** | Score: `{node_with_score.score:.4f}` | Source: `{node_with_score.node.metadata.get('file_name')}`")
                        st.code(node_with_score.node.get_content()[:200] + "...", language="text")

if __name__ == "__main__":
    main()