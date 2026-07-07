from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# 1. 調整 LLM 設定：加入 system_prompt 強制約束語言
Settings.llm = Ollama(
    model="llama3", 
    request_timeout=120.0,
    system_prompt="你是一個專業的繁體中文助理。請務必完全使用『繁體中文』(Taiwan) 回答所有問題，絕對不要使用英文或簡體中文。"
)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

custom_node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=30)
Settings.node_parser = custom_node_parser

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

# 發問
query_str = "請根據這份文件，用繁體中文條列出三個最重要的核心重點。"
response = query_engine.query(query_str)

print("\n=== 本地 AI 的回答 ===")
print(response)

print("\n" + "="*20 + " 資工專用：檢索 Debug 資訊 " + "="*20)
# 印出 RAG 到底幫 LLM 撈了哪些小抄
for i, node_with_score in enumerate(response.source_nodes):
    print(f"\n[路徑] {node_with_score.node.metadata.get('file_name')}")
    print(f"[相似度分數 Score]: {node_with_score.score:.4f}")
    print(f"[撈出來的小抄內容 (前150字)]: {node_with_score.node.get_content()[:150]}...")
print("="*60)