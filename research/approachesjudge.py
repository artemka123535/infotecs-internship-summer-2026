import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
JUDGE_MODEL = "qwen3.5:27b"

print("Инициализация судьи для сравнения подходов...\n")

try:
    with open("research/datasets/approachresults.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    with open("research/datasets/context.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    query = list(dataset.keys())[0]
    context = dataset[query]
except Exception as e:
    print(f"Ошибка загрузки файлов: {e}")
    exit()

text_1 = results.get("Тест 1 (чистая модель)", "Текст отсутствует")
text_2 = results.get("Тест 2 (контекст)", "Текст отсутствует")
text_3 = results.get("Тест 3 (контекст + план)", "Текст отсутствует")

judge_system_prompt = (
    "Ты — ведущий ML-исследователь и строгий редактор. Тебе предоставлены исходные факты и "
    "три варианта отчета, сгенерированных разными архитектурными пайплайнами.\n"
    "Твоя задача — проанализировать их и расставить по местам (1, 2 и 3 место).\n\n"
    "Критерии оценки:\n"
    "1. Отсутствие галлюцинаций (опора СТРОГО на исходные факты, без выдумывания лишнего).\n"
    "2. Структура и логика (наличие четкого введения, разделов, академического стиля).\n"
    "3. Полнота раскрытия темы.\n\n"
    "Выведи ответ в строгом формате:\n"
    "1 МЕСТО: [Название теста]\n"
    "ОБОСНОВАНИЕ: [Почему он лучший]\n\n"
    "2 МЕСТО: [Название теста]\n"
    "ОБОСНОВАНИЕ: [В чем его недостатки по сравнению с первым]\n\n"
    "3 МЕСТО: [Название теста]\n"
    "ОБОСНОВАНИЕ: [Почему он худший (например, галлюцинации или плохая структура)]"
)

judge_user_prompt = (
    f"ТЕМА ОТЧЕТА: {query}\n\n"
    f"ИСХОДНЫЕ ФАКТЫ (RAG-контекст):\n{context}\n\n"
    f"--- Тест 1 (чистая модель) ---\n{text_1}\n\n"
    f"--- Тест 2 (контекст) ---\n{text_2}\n\n"
    f"--- Тест 3 (контекст + план) ---\n{text_3}\n\n"
    "Оцени тексты и расставь их по местам."
)

print(f"Судья анализирует подходы для темы: '{query}'...")

try:
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": judge_user_prompt}
        ],
        temperature=0.0 
    )
    
    evaluation = response.choices[0].message.content
    print("\nИтоговое решение:\n")
    print(evaluation)
        
except Exception as e:
    print(f"Ошибка при оценке: {e}")