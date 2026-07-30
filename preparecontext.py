import json
from parser import get_clean_articles 
from semanticfilter import get_top_chunks

TEST_QUERIES = [
    "Методы защиты RAG-систем от prompt injection",
    "Сравнение архитектур Transformer и Mamba"
]

dataset = {}

for query in TEST_QUERIES:
    articles = get_clean_articles(query, max_results=10)
    
    if articles:
        top_facts = get_top_chunks(query, articles, top_k=10)
        
        context_text = "\n\n---\n\n".join(top_facts)
        
        dataset[query] = context_text
        print(f"Контекст сохранен (Символов: {len(context_text)})\n")

with open("research/datasets/context.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)