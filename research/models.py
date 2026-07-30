import time
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

MODELS_TO_TEST = ["qwen3.5:27b", "qwen3.5:4b"]

with open("research/datasets/context.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []
print("Запуск тестирования моделей...\n")

for model in MODELS_TO_TEST:
    print(f"Модель: {model}")

    for query, context in dataset.items():
        print(f"Запрос: {query}")
        
        start_time = time.time()
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Ты ИИ-исследователь. Напиши детальный отчет на основе фактов."},
                    {"role": "user", "content": f"Тема: {query}\n\nФакты: {context}"}
                ],
                temperature=0.2
            )
            report_text = response.choices[0].message.content
            
            execution_time = round(time.time() - start_time, 2)
            length = len(report_text)
            
            chars_per_sec = round(length / execution_time, 1)
            
            print(f"Отчет готов за {execution_time} сек | Длина: {length} симв | Скорость: {chars_per_sec} симв/сек")
            
            results.append({
                "model": model,
                "query": query,
                "time": execution_time,
                "length": length,
                "speed": chars_per_sec,
                "text": report_text
            })
            
        except Exception as e:
            print(f"Ошибка: {e}")
            
with open("research/datasets/modelresults.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("\nРезультаты успешно сохранены!")