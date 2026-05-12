import re

with open('creditlens/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

emojis = set(re.findall(r'[\U00010000-\U0010ffff]', text))
print('Emojis in app.py:', emojis)

with open('creditlens/src/text_generator.py', 'r', encoding='utf-8') as f:
    text2 = f.read()

emojis2 = set(re.findall(r'[\U00010000-\U0010ffff]', text2))
print('Emojis in text:', emojis2)

for e in emojis:
    text = text.replace(e, '')
for e in emojis2:
    text2 = text2.replace(e, '')

with open('creditlens/app.py', 'w', encoding='utf-8') as f:
    f.write(text)

with open('creditlens/src/text_generator.py', 'w', encoding='utf-8') as f:
    f.write(text2)
    
print("Emoji purgers finished")
