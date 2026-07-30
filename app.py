import streamlit as st
import PyPDF2
from parser import get_clean_articles
from semanticfilter import get_top_chunks
from agent import generate_report
import time

st.set_page_config(page_title="LLM-ассистент", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Настройки исследования")
    st.write("---")
    
    report_structure = st.selectbox(
        "Выберите структуру отчёта",
        ["Стандартный отчёт", "Краткий пересказ"]
    )
    
    max_queries = st.number_input(
        "Максимальное количество поисковых запросов",
        min_value=1, max_value=20, value=4, step=1
    )
    
    enable_web_search = st.checkbox("Поиск в сети", value=True)
    
    uploaded_files = st.file_uploader(
        "Загрузка новых файлов",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'csv', 'md']
    )

col1, col2 = st.columns([5, 1])
with col1:
    st.header("Ассистент для исследователя")
with col2:
    if st.button("Очистить чат", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.write("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Спросить..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Инициализация пайплайна...", expanded=True)

        start_time = time.time()
        
        with status:
            file_context = ""
            if uploaded_files:
                st.write("Чтение загруженных файлов...")
                for file in uploaded_files:
                    if file.name.endswith('.pdf'):
                        reader = PyPDF2.PdfReader(file)
                        for page in reader.pages:
                            file_context += page.extract_text() + "\n"
                    else:
                        file_context += file.getvalue().decode("utf-8") + "\n\n"
                st.write(f"Файлы прочитаны ({len(file_context)} символов).")

            try:
                articles = []
                used_urls = []
                
                if enable_web_search:
                    st.write(f"Поиск в сети (до {max_queries} источников)...")
                    articles = get_clean_articles(prompt, max_results=max_queries)
                    used_urls = list(set([a['url'] for a in articles]))

                if file_context.strip():
                    articles.append({"url": "Локальный файл", "text": file_context})
                    used_urls.append("Прикрепленный документ")
                    
                if not articles:
                    st.error("Не удалось найти статьи.")
                    st.stop()

                st.write(f"Найдено и скачано источников: {len(articles)}")

                st.write("Семантическая фильтрация чанков...")
                top_facts = get_top_chunks(prompt, articles, top_k=12)
                context = "\n\n".join(top_facts)
                st.write(f"Отобрано {len(top_facts)} самых релевантных абзацев.")
                    
                st.write("Анализ фактов и написание отчета...")
                final_report = generate_report(prompt, context, used_urls, report_structure)

                end_time = time.time()
                execution_time = round(end_time - start_time, 2)

            except Exception as e:
                st.error(f"Ошибка: {e}")
                st.stop()

        status.update(label=f"Исследование завершено за {execution_time} сек.!", state="complete", expanded=False)
        
        st.markdown(final_report)
        
        st.session_state.messages.append({"role": "assistant", "content": final_report})