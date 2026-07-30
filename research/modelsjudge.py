import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
JUDGE_MODEL = "qwen3.5:27b"

print("Инициализация судьи для сравнения моделей...")

with open("research/datasets/modelresults.json", "r", encoding="utf-8") as f:
    results = json.load(f)
with open("research/datasets/context.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

grouped_data = {}
for item in results:
    q = item["query"]
    if q not in grouped_data:
        grouped_data[q] = {}
    grouped_data[q][item["model"]] = item["text"]

judge_system_prompt = (
    "Ты — строгий эксперт по оценке качества текстов, написанных ИИ. "
    "Тебе предоставлены исходные факты и два отчета (Отчет А и Отчет Б). "
    "Сравни их и выбери ЛУЧШИЙ отчет. Критерии: отсутствие воды, глубина анализа, логика, структурированность. "
    "Если оба отчета одинаково хороши или плохи, можешь объявить НИЧЬЮ. "
    "Выведи ответ СТРОГО в таком формате:\n"
    "ПОБЕДИТЕЛЬ: [Отчет А / Отчет Б / НИЧЬЯ]\n"
    "ОБОСНОВАНИЕ: [Одно предложение с главной причиной выбора]"
)

for query, models_dict in grouped_data.items():
    models_list = list(models_dict.keys())
    
    if len(models_list) < 2:
        continue
        
    model_a = models_list[0]
    model_b = models_list[1]
    text_a = models_dict[model_a]
    text_b = models_dict[model_b]
    
    original_context = dataset[query]
    
    print(f"\nСравнение: {model_a} [Отчет А] и {model_b} [Отчет Б]")
    print(f"Тема: '{query}'")
    
    judge_user_prompt = (
        f"ИСХОДНЫЕ ФАКТЫ:\n{original_context}\n\n"
        f"--- ОТЧЕТ А ---\n{text_a}\n\n"
        f"--- ОТЧЕТ Б ---\n{text_b}\n\n"
        "Кто написал отчет лучше?"
    )
    
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
        print(f"Итоговое решение:\n{evaluation}\n")
        
    except Exception as e:
        print(f"Ошибка при оценке: {e}")