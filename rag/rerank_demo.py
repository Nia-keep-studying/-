from settings import MODEL_CACHE_DIR

# 在导入模型相关库之前指定缓存，供嵌入模型和重排序模型共同使用。
from sentence_transformers import CrossEncoder

def load_rerank_model():
    reranker = CrossEncoder(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        device = "cpu",
        cache_folder=str(MODEL_CACHE_DIR),
    )
    return reranker

def get_rerank_score(result):
    return result["score"]

def rerank_chunks(question,chunks,reranker,top_k=2):
    if not chunks:
        return []
    pairs = []
    for chunk in chunks:
        text = f"章节:{chunk['section']}\n正文:{chunk['content']}"
        pairs.append([question,text])

    scores = reranker.predict(pairs)

    rerank_results = []

    for index,score in enumerate(scores):
        rerank_results.append(
            {
                "chunk":chunks[index],
                "score":float(score),
            }
        )

    rerank_results.sort(key=get_rerank_score,reverse=True)

    selected_results = rerank_results[:top_k]

    return selected_results

if __name__ == "__main__":
    from .rag_loader import load_documents, split_documents

    question = input("请输入重排序问题：").strip()
    chunks = split_documents(load_documents())
    reranker = load_rerank_model()
    for result in rerank_chunks(question, chunks, reranker, top_k=3):
        print(result["chunk"]["source"], result["chunk"]["section"], result["score"])
