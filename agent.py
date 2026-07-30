from openai import OpenAI
from parser import get_clean_articles
from semanticfilter import get_top_chunks
import time

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

MODEL_NAME = "qwen3.5:27b"

def generate_report(query: str, context: str, urls_list: list, report_mode: str) -> str:
    urls_text = "\n".join([f"- {url}" for url in urls_list])

    if report_mode == "Краткий пересказ":
        print("Генерация краткого пересказа...")
        summary_prompt = (
            "Ты — профессиональный ИИ-ассистент. Твоя задача — сделать очень КРАТКУЮ и емкую выжимку"
            "по запросу пользователя на основе предоставленных фактов. "
            "Не пиши длинных введений. Выдели только самую суть в 2-3 абзацах или маркированном списке.\n\n"
            f"Тема: {query}\n\nФакты:\n{context}\n\n"
            "Напиши краткую выжимку и в конце добавь раздел '## Источники'."
        )
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.2 
        )
        print("Краткий пересказ готов!\n")
        return response.choices[0].message.content

    else:
        print("ЭТАП 1: Анализ фактов и составление плана...")
        
        planner_system_prompt = (
            "Ты — профессиональный ИИ-ассистент. Твоя главная задача — помогать специалистам "
            "в проведении исследовательских задач в области Machine Learning (ML) и Data Science. "
            "Составь строгий, логичный и детализированный план научно-технического отчета на основе "
            "предоставленных фактов. План должен включать введение, глубокий разбор проблематики, "
            "технические детали (алгоритмы, архитектуры, векторы) и обоснованное заключение."
        )
        
        planner_user_prompt = f"Тема отчета: {query}\n\nНайденные факты:\n{context}\n\nНапиши только план отчета."

        try:
            plan_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": planner_system_prompt},
                    {"role": "user", "content": planner_user_prompt}
                ],
                temperature=0.4
            )
            report_plan = plan_response.choices[0].message.content
            print("План успешно сгенерирован!\n")
            print(f"--- Утвержденный план ---\n{report_plan}\n-------------------------\n")

        except Exception as e:
            print(f"Ошибка при генерации плана: {e}")
            return None

        print("ЭТАП 2: Написание итогового отчета...")
        
        writer_system_prompt = (
            "Ты — профессиональный ИИ-ассистент, помогающий в проведении исследовательских задач в области ML. "
            "Твоя задача — написать глубокий, развернутый и детализированный исследовательский отчет по утвержденному плану.\n\n"
            "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
            "1. Глубина раскрытия: Каждый раздел и подраздел должен быть объемным и состоять как минимум из 2-3 подробных абзацев. КАТЕГОРИЧЕСКИ ЗАПРЕЩАЕТСЯ создавать короткие подразделы из 1-2 предложений. Объясняй механизмы, причины и следствия.\n"
            "2. Достоверность: Опирайся строго на факты из контекста.\n"
            "3. Стиль: Академический, с использованием профессиональной ML-терминологии.\n"
            "4. Оформление: Markdown, списки, выделение терминов жирным шрифтом.\n"
            "5. Источники: В самом конце отчета ОБЯЗАТЕЛЬНО создай раздел '## Источники' и перечисли все предоставленные ссылки."
        )
        
        writer_user_prompt = (
            f"Тема: {query}\n\n"
            f"Утвержденный план отчета:\n{report_plan}\n\n"
            f"Факты для использования:\n{context}\n\n"
            f"Доступные источники для добавления в конец отчета:\n{urls_text}\n\n"
            "Напиши итоговый, развернутый отчет."
        )

        try:
            report_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": writer_system_prompt},
                    {"role": "user", "content": writer_user_prompt}
                ],
                temperature=0.2 
            )
            final_report = report_response.choices[0].message.content
            print("Итоговый отчет готов!")
            return final_report
        except Exception as e:
            print(f"Ошибка при написании отчета: {e}")
            return None
    
def run_deep_research(query: str, file_text: str, use_web: bool, max_queries: int, report_mode: str):
    start_time = time.time()

    print(f"Запуск пайплайна | Режим: {report_mode} | Поиск: {use_web}")
    
    articles = []
    used_urls = []
    
    if use_web:
        print("Выполняем поиск в интернете...")
        articles = get_clean_articles(query, max_results=max_queries)
        used_urls = list(set([a['url'] for a in articles]))
    
    if file_text.strip():
        print("Добавляем контекст из файла...")
        articles.append({"url": "Локальный файл", "text": file_text})
        used_urls.append("Прикрепленный документ")
        
    if not articles:
        return "Нет данных для анализа."

    top_facts = get_top_chunks(query, articles, top_k=12)
    context = "\n\n".join(top_facts)
    
    report = generate_report(query, context, used_urls, report_mode)

    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    time_badge = f"> **Время выполнения пайплайна:** {execution_time} сек.\n\n"
    report = time_badge + report
    
    return report

if __name__ == "__main__":
    final_text = run_deep_research("Какие уязвимости и атаки на LLM-системы актуальны в 2025-2026 годах?",
                                    file_text="",
                                    use_web=True, 
                                    max_queries=15, 
                                    report_mode="Стандартный отчёт")
    
    if final_text:
        with open("reports/llmvulnerabilities.md", "w", encoding="utf-8") as f:
            f.write(final_text)
        print("Отчет сохранен!")