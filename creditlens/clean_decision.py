# -*- coding: utf-8 -*-
with open('creditlens/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('✅ ОДОБРЕНО', 'Одобрено')
text = text.replace('⛔ ОТКАЗ', 'Отказ')

with open('creditlens/app.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Emojis removed')
