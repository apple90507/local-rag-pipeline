import os
import time
import hashlib
import json
import re
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# ==================== 1. Resource Initialization Layer ====================

def init_global_settings():
    """Initialize core embedding and optimized node parsing configurations."""
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

def calculate_dir_md5(data_dir="data"):
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
    manifest_path = os.path.join(persist_dir, "cache_manifest.json")
    docstore_path = os.path.join(persist_dir, "docstore.json")
    if not (os.path.exists(persist_dir) and os.path.exists(docstore_path) and os.path.exists(manifest_path)):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest.get("data_md5") == current_md5
    except (json.JSONDecodeError, IOError):
        return False

def get_vector_index(persist_dir="./storage", data_dir="data"):
    current_md5 = calculate_dir_md5(data_dir)
    
    if verify_cache_integrity(persist_dir, current_md5):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)
    
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        raise FileNotFoundError(f"Source directory '{data_dir}' is empty or does not exist.")
        
    documents = SimpleDirectoryReader(data_dir, file_extractor={".pdf": PyMuPDFReader()}).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
    
    manifest_path = os.path.join(persist_dir, "cache_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"data_md5": current_md5}, f, ensure_ascii=False, indent=4)
    return index

# ==================== 2. Advanced Search Algorithms ====================

def tokenize_chinese(text):
    """Character-level tokenization for CJK, word-level for alphanumeric."""
    text = text.lower()
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text)

def get_raw_node(item):
    """Helper to extract raw node from NodeWithScore or Node directly."""
    if hasattr(item, "node"):
        return item.node
    return item

def get_node_id(item):
    if hasattr(item, "node"):
        return item.node.node_id
    return item.node_id

def reciprocal_rank_fusion(vector_nodes, bm25_nodes, k=60):
    """
    Reciprocal Rank Fusion (RRF) Algorithm.
    Fuses rankings from different retrieval paradigms without normalization issues.
    """
    rrf_scores = {}
    node_map = {}
    
    # Process vector rankings
    for rank, item in enumerate(vector_nodes):
        nid = get_node_id(item)
        node_map[nid] = get_raw_node(item)
        rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (k + (rank + 1))
        
    # Process BM25 rankings
    for rank, item in enumerate(bm25_nodes):
        nid = get_node_id(item)
        node_map[nid] = get_raw_node(item)
        rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (k + (rank + 1))
        
    # Sort descending by RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Wrap back to NodeWithScore for Cross-Encoder compatibility
    fused_nodes = []
    for nid, score in sorted_items:
        fused_nodes.append(NodeWithScore(node=node_map[nid], score=score))
    return fused_nodes

# ==================== 3. Evaluation Core Pipeline ====================

def get_golden_dataset():
    return [
        {
            "query": "DiCE 反事實解釋與特徵重要性分析之價值不同點",
            "ground_truth_keyword": "反事實式之決策邊界探索"
        },
        {
            "query": "法律文本分類之可解釋性研究提出的核心方法名稱",
            "ground_truth_keyword": "基於語意保持式擾動"
        },
        {
            "query": "本篇論文主要探討之機器學習與人工智慧研究領域",
            "ground_truth_keyword": "可解釋人工智慧"
        },
        {
            "query": "後驗式模型無關型可解釋性研究包含之決策探索範疇",
            "ground_truth_keyword": "決策邊界探索"
        }
    ]

def clean_text(text):
    return re.sub(r'\s+', '', text)

def evaluate_pipeline(
    index, 
    golden_data, 
    rerank_model=None, 
    use_hybrid=False, 
    all_nodes=None, 
    bm25_engine=None, 
    top_k_retrieval=10, 
    top_k_final=2
):
    total_queries = len(golden_data)
    hits = 0
    rr_sum = 0.0
    latencies = []

    vector_retriever = index.as_retriever(similarity_top_k=top_k_retrieval)

    for item in golden_data:
        query_str = item["query"]
        target_keyword = clean_text(item["ground_truth_keyword"])

        start_time = time.time()
        
        # --- Stage 1: Retrieval ---
        if use_hybrid:
            # Vector Retrieval
            vector_results = vector_retriever.retrieve(query_str)
            # BM25 Retrieval
            query_tokens = tokenize_chinese(query_str)
            bm25_scores = bm25_engine.get_scores(query_tokens)
            # Fetch Top-K BM25 nodes
            indexed_scores = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
            bm25_results = []
            for idx, score in indexed_scores[:top_k_retrieval]:
                if score > 0:
                    bm25_results.append(all_nodes[idx])
            
            # --- Stage 1.5: Reciprocal Rank Fusion ---
            retrieved_nodes = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
        else:
            retrieved_nodes = vector_retriever.retrieve(query_str)
        
        # --- Stage 2: Cross-Encoder Reranking ---
        if rerank_model:
            pairs = [[query_str, node.node.get_content()] for node in retrieved_nodes]
            rerank_scores = rerank_model.predict(pairs)
            for node, score in zip(retrieved_nodes, rerank_scores):
                node.score = float(score)
            retrieved_nodes.sort(key=lambda x: x.score, reverse=True)
        
        final_nodes = retrieved_nodes[:top_k_final]
        latency = time.time() - start_time
        latencies.append(latency)

        hit_found = False
        rank_position = 0
        
        for index_idx, node in enumerate(final_nodes):
            processed_content = clean_text(node.node.get_content())
            if target_keyword in processed_content:
                hit_found = True
                rank_position = index_idx + 1
                break

        if hit_found:
            hits += 1
            rr_sum += 1.0 / rank_position

    avg_hit_rate = (hits / total_queries) * 100
    avg_mrr = rr_sum / total_queries
    avg_latency = sum(latencies) / total_queries

    return avg_hit_rate, avg_mrr, avg_latency

# ==================== 4. Execution & Performance Reporting ====================

def main():
    print("=" * 60)
    print("RAG System Benchmarking: Ablation Study Pipeline")
    print("=" * 60)
    
    init_global_settings()
    
    try:
        index = get_vector_index()
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return

    golden_data = get_golden_dataset()
    
    # 建立 BM25 語料庫
    all_nodes = list(index.docstore.docs.values())
    corpus = [tokenize_chinese(node.get_content()) for node in all_nodes]
    bm25_engine = BM25Okapi(corpus)
    
    # 1. Pipeline Alpha: Base Vector Search Only (No Rerank)
    print("\n[Evaluating] Pipeline Alpha: Base Vector Retrieval...")
    alpha_hit, alpha_mrr, alpha_lat = evaluate_pipeline(
        index, golden_data, rerank_model=None, use_hybrid=False, top_k_retrieval=10, top_k_final=2
    )
    
    # 2. Pipeline Beta: Two-Stage (Bi + Cross-Encoder)
    print("[Evaluating] Pipeline Beta: Two-Stage Retrieval (No Hybrid)...")
    rerank_model = CrossEncoder("BAAI/bge-reranker-base")
    beta_hit, beta_mrr, beta_lat = evaluate_pipeline(
        index, golden_data, rerank_model=rerank_model, use_hybrid=False, top_k_retrieval=10, top_k_final=2
    )
    
    # 3. Pipeline Gamma: Hybrid Two-Stage (Vector & BM25 + RRF + Cross-Encoder)
    print("[Evaluating] Pipeline Gamma: Hybrid Two-Stage with RRF Fusion...")
    gamma_hit, gamma_mrr, gamma_lat = evaluate_pipeline(
        index, 
        golden_data, 
        rerank_model=rerank_model, 
        use_hybrid=True, 
        all_nodes=all_nodes, 
        bm25_engine=bm25_engine, 
        top_k_retrieval=10, 
        top_k_final=2
    )
    
    # 4. Print Quantitative Comparison Report Table
    print("\n" + "=" * 80)
    print(f"{'Retrieval Pipeline Architecture':<45} | {'Hit Rate':<8} | {'MRR':<5} | {'Latency':<7}")
    print("-" * 80)
    print(f"{'Alpha: Base Vector Search (Bi-Encoder Only)':<45} | {alpha_hit:>7.2f}% | {alpha_mrr:.3f} | {alpha_lat:.3f}s")
    print(f"{'Beta: Two-Stage (Bi + Cross-Encoder)':<45} | {beta_hit:>7.2f}% | {beta_mrr:.3f} | {beta_lat:.3f}s")
    print(f"{'Gamma: Hybrid Two-Stage (BM25 + Vector + RRF + CE)':<45} | {gamma_hit:>7.2f}% | {gamma_mrr:.3f} | {gamma_lat:.3f}s")
    print("=" * 80)

    # 5. Persist Results to Disk
    report_path = "evaluation_report.json"
    report_data = {
        "benchmark_version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "configuration": {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k_retrieval": 10,
            "top_k_final": 2,
            "rrf_k_constant": 60
        },
        "results": {
            "pipeline_alpha": {"hit_rate_pct": round(alpha_hit, 2), "mrr": round(alpha_mrr, 3), "avg_latency_sec": round(alpha_lat, 3)},
            "pipeline_beta": {"hit_rate_pct": round(beta_hit, 2), "mrr": round(beta_mrr, 3), "avg_latency_sec": round(beta_lat, 3)},
            "pipeline_gamma": {"hit_rate_pct": round(gamma_hit, 2), "mrr": round(gamma_mrr, 3), "avg_latency_sec": round(gamma_lat, 3)}
        }
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n[Success] 混合檢索數據報表已更新：'{report_path}'")

if __name__ == "__main__":
    main()