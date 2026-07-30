import torch
from sentence_transformers import SentenceTransformer
from parser import get_clean_articles

print("Загрузка модели энкодера...")
encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
print("Энкодер готов!")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    words = text.split()
    chunks = []
    
    if len(words) < chunk_size:
        return [text]
        
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
    return chunks

def get_top_chunks(query: str, articles: list, top_k: int = 10) -> list:
    print("Нарезка статей на чанки...")
    all_chunks = []
    for article in articles:
        all_chunks.extend(chunk_text(article['text']))
        
    if not all_chunks:
        print("Нет текста для фильтрации.")
        return []

    print(f"Векторизация и поиск лучшего из {len(all_chunks)} чанков...")
    
    query_embedding = encoder.encode(query, convert_to_tensor=True)

    chunk_embeddings = encoder.encode(all_chunks, convert_to_tensor=True)

    cos_scores = torch.nn.functional.cosine_similarity(query_embedding, chunk_embeddings)
    
    top_results = torch.topk(cos_scores, k=min(top_k, len(all_chunks)))
    
    best_chunks = []
    for score, idx in zip(top_results.values, top_results.indices):
        best_chunks.append(all_chunks[idx])
        print(f"Косинусное расстояние: {score:.4f} | Текст: {all_chunks[idx][:60]}...")
        
    print("Семантическая фильтрация завершена!")
    return best_chunks

if __name__ == "__main__":
    query = "Какие уязвимости и атаки на LLM-системы актуальны в 2025-2026 годах?"

    articles = get_clean_articles(query, max_results=10)
    top_facts = get_top_chunks(query, articles, top_k=10)

    context_for_llm = "\n\n---\n\n".join(top_facts)