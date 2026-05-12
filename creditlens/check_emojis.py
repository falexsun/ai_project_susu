import unicodedata

with open('creditlens/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

emojis = set([c for c in text if unicodedata.category(c) in ('So', 'Sk') and not c.isascii()])
print('Symbols in app.py:', emojis)

text = text.replace('⚙', '').replace('️', '')

with open('creditlens/app.py', 'w', encoding='utf-8') as f:
    f.write(text)

with open('creditlens/src/text_generator.py', 'r', encoding='utf-8') as f:
    text2 = f.read()

emojis2 = set([c for c in text2 if unicodedata.category(c) in ('So', 'Sk') and not c.isascii()])
print('Symbols in text_generator.py:', emojis2)
text2 = text2.replace('⚙', '').replace('️', '')

with open('creditlens/src/text_generator.py', 'w', encoding='utf-8') as f:
    f.write(text2)
