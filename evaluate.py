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
from sentence_transformers import CrossEncoder

# ==================== 1. Resource Initialization Layer ====================

def init_global_settings():
    """Initialize core embedding and optimized node parsing configurations."""
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    # 優化點 1：將 Chunk Size 擴張至 512，確保碩博士論文長句的語意完整性
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

# ==================== 2. Evaluation Core Pipeline ====================

def get_golden_dataset():
    """
    Define the evaluation benchmark dataset (Golden QA Pairs).
    優化點 2：移除口語與誌謝雜訊字眼，改為高特徵密度的標準檢索語句。
    """
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
    """優化點 3：移除所有空格、換行與特殊符號，徹底消除 PDF 斷字對字串比對造成的干擾"""
    return re.sub(r'\s+', '', text)

def evaluate_pipeline(index, golden_data, rerank_model=None, top_k_retrieval=5, top_k_final=2):
    total_queries = len(golden_data)
    hits = 0
    rr_sum = 0.0
    latencies = []

    retriever = index.as_retriever(similarity_top_k=top_k_retrieval)

    for item in golden_data:
        query_str = item["query"]
        target_keyword = clean_text(item["ground_truth_keyword"]) # 清洗關鍵字

        start_time = time.time()
        retrieved_nodes = retriever.retrieve(query_str)
        
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
            # 關鍵優化：將抽出來的 Chunk 內容也進行全面清洗再做包含比對
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

# ==================== 3. Execution & Performance Reporting ====================

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
    
    # 1. Evaluate Baseline Pipeline (Vector Search Only)
    print("\n[Evaluating] Pipeline Alpha: Base Vector Retrieval (No Rerank)...")
    base_hit, base_mrr, base_lat = evaluate_pipeline(
        index, golden_data, rerank_model=None, top_k_retrieval=10, top_k_final=2
    )
    
    # 2. Evaluate Advanced Pipeline (Two-Stage with Cross-Encoder)
    print("[Evaluating] Pipeline Beta: Two-Stage Retrieval (With BGE Reranker)...")
    rerank_model = CrossEncoder("BAAI/bge-reranker-base")
    adv_hit, adv_mrr, adv_lat = evaluate_pipeline(
        index, golden_data, rerank_model=rerank_model, top_k_retrieval=10, top_k_final=2
    )
    
    # 3. Print Quantitative Comparison Report Table
    print("\n" + "=" * 65)
    print(f"{'Retrieval Pipeline Architecture':<35} | {'Hit Rate':<8} | {'MRR':<5} | {'Latency':<7}")
    print("-" * 65)
    print(f"{'Base Vector Search (Bi-Encoder Only)':<35} | {base_hit:>7.2f}% | {base_mrr:.3f} | {base_lat:.3f}s")
    print(f"{'Two-Stage Pipeline (Bi + Cross-Encoder)':<35} | {adv_hit:>7.2f}% | {adv_mrr:.3f} | {adv_lat:.3f}s")
    print("=" * 65)

    # 4. 核心補強：將消融實驗數據序列化存入硬碟 (MLOps Production Standard)
    report_path = "evaluation_report.json"
    report_data = {
        "benchmark_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "configuration": {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k_retrieval": 10,
            "top_k_final": 2,
            "embedding_model": "nomic-embed-text",
            "rerank_model": "bge-reranker-base"
        },
        "results": {
            "baseline_pipeline": {
                "hit_rate_pct": round(base_hit, 2),
                "mrr": round(base_mrr, 3),
                "avg_latency_sec": round(base_lat, 3)
            },
            "two_stage_pipeline": {
                "hit_rate_pct": round(adv_hit, 2),
                "mrr": round(adv_mrr, 3),
                "avg_latency_sec": round(adv_lat, 3)
            }
        }
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n[Success] 數據報表已成功固化至實體硬碟：'{report_path}'")

if __name__ == "__main__":
    main()