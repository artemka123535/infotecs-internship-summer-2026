import trafilatura
from ddgs import DDGS

def get_clean_articles(query: str, max_results: int = 15) -> list:
    print(f"Запрос: '{query}'")
    
    urls = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results, region='ru-ru')
            if results:
                urls = [r['href'] for r in results]
    except Exception as e:
        print(f"Ошибка поиска: {e}")
        return []

    print(f"Найдено {len(urls)} ссылок.\n")
    
    articles = []
    for url in urls:
        print(f"Анализ: {url}")
            
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                print("     [Пропуск] Не удалось загрузить страницу.")
                continue
            
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            
            if text and len(text) > 800:
                query_words = [w.lower() for w in query.split() if len(w) > 3]
                text_lower = text.lower()
                
                if not any(w in text_lower for w in query_words):
                    print("     [Пропуск] Статья не относится к теме (нет ключевых слов).")
                    continue

                articles.append({
                    "url": url,
                    "text": text
                })
                print(f"    [Успех] Извлечено {len(text)} символов.")
            else:
                print("     [Пропуск] Слишком мало текста.")

        except Exception as e:
            print(f"    [Ошибка] {e}")
            
    print(f"\nИтого собрано статей: {len(articles)}")
    return articles

if __name__ == "__main__":
    test_query = "Какие уязвимости и атаки на LLM-системы актуальны в 2025-2026 годах?"
    
    data = get_clean_articles(test_query, max_results=10)
    
    if data:
        print(f"\nПример начала первой статьи ({data[0]['url']}):")
        print(data[0]['text'][:300] + "...")
    else:
        print("\nПолезных статей не найдено.")