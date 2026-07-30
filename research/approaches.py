import time
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen3.5:27b"

try:
    with open("research/datasets/context.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    query = list(dataset.keys())[0]
    context = dataset[query]
except Exception as e:
    print("Ошибка загрузки контекста")
    exit()

print(f"Запуск исследования для запроса: '{query}'\n")
results = {}

print("Генерация отчёта без плана и контекста...")
start = time.time()
resp_no_rag = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Ты ИИ-исследователь. Напиши детальный технический отчет."},
        {"role": "user", "content": f"Тема: {query}"}
    ],
    temperature=0.2
).choices[0].message.content

print(f"Готово за {round(time.time() - start, 2)} сек.\n")
results["Тест 1 (чистая модель)"] = resp_no_rag


print("Генерация отчёта с контекстом, но без плана...")
start = time.time()
resp_rag_direct = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Ты ИИ-исследователь. Напиши технический отчет строго на основе фактов."},
        {"role": "user", "content": f"Тема: {query}\n\nФакты:\n{context}"}
    ],
    temperature=0.2
).choices[0].message.content

print(f"Готово за {round(time.time() - start, 2)} сек.\n")
results["Тест 2 (контекст)"] = resp_rag_direct


print("Генерация отчёта с контекстом и планом...")
start = time.time()
plan = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Ты ИИ-исследователь. Напиши только план отчета на основе фактов."},
        {"role": "user", "content": f"Тема: {query}\n\nФакты:\n{context}"}
    ],
    temperature=0.4
).choices[0].message.content

resp_rag_cot = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Ты ИИ-исследователь. Напиши отчет строго по плану и фактам."},
        {"role": "user", "content": f"Тема: {query}\n\nПлан:\n{plan}\n\nФакты:\n{context}"}
    ],
    temperature=0.2
).choices[0].message.content

print(f"Готово за {round(time.time() - start, 2)} сек.\n")
results["Тест 3 (контекст + план)"] = resp_rag_cot

with open("research/datasets/approachresults.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("Результаты разных подходов сохранены!")