# -*- coding: utf-8 -*-
import re

with open('creditlens/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacements = {
    'page_icon="✨"': 'page_icon=""',
    '<h1>✨ CreditLens</h1>': '<h1>CreditLens</h1>',
    'st.header("📝 Анкета клиента")': 'st.header("О клиенте")',
    'st.markdown("### 📌 Основные параметры")': 'st.markdown("### Основная информация")',
    'with st.expander("⚙️ Расширенные параметры", expanded=False):': 'with st.expander("Дополнительные сведения", expanded=False):',
    'run_button = st.button("🚀 Оценить заявку", type="primary", use_container_width=True)': 'run_button = st.button("Узнать результат", type="primary", use_container_width=True)',
    'decision = "✅ ОДОБРЕНО"': 'decision = "Одобрено"',
    'decision = "✅ ОДОБРЕНО" if approved else "⛔ ОТКАЗ"': 'decision = "Одобрено" if approved else "Отказ"',
    'tab1, tab2, tab3 = st.tabs(["📊 Главный дашборд", "🔍 Инсайты и Факторы (SHAP)", "🎛 Моделирование (What-If)"])': 'tab1, tab2, tab3 = st.tabs(["Общая картина", "Детали решения", "Подбор вариантов"])',
    'st.markdown("### 📝 Объяснение инференса")': 'st.markdown("### Взгляд изнутри: как система пришла к такому выводу?")',
    'st.markdown("### 🎯 Краткое резюме")': 'st.markdown("### Самое важное для клиента")',
    'st.markdown("### 🤖 Персональные рекомендации")': 'st.markdown("### Как улучшить ситуацию? Наши советы")',
    "<div class='info-panel'>💡 {rec}</div>": "<div class='info-panel'>{rec}</div>",
    'st.markdown("### 👥 Сегментный профиль")': 'st.markdown("### Скрытые закономерности (сегментный профиль)")',
    'with st.expander("❓ Как интерпретировать эти данные?", expanded=False):': 'with st.expander("Как правильно трактовать этот график?", expanded=False):',
    'st.markdown("### 🔍 Влияние каждого фактора кредитного скоринга")': 'st.markdown("### Какие факторы оказались решающими?")',
    'st.markdown("### 🎛 Регулировка параметров (What-if симуляция)")': 'st.markdown("### Что будет, если немного изменить запрос?")',
    'st.info("👈 Подберите параметры анкеты в левой панели и нажмите «Оценить заявку» для начала работы.", icon="✨")': 'st.markdown("<div style=\\"padding:20px; background:var(--secondary-background-color); border-radius:10px; text-align:center; font-size:1.1rem;\\">Пожалуйста, заполните небольшую анкету слева, чтобы мы могли подготовить для вас точный результат.</div>", unsafe_allow_html=True)',
    'st.info("Анализ принадлежности к демографическим группам")': 'st.markdown("<p style=\\"opacity:0.8; margin-bottom:15px;\\">Немного статистики о людях с похожим социально-профессиональным опытом.</p>", unsafe_allow_html=True)'
}

for old, new in replacements.items():
    text = text.replace(old, new)
    
with open('creditlens/app.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Done app.py")
