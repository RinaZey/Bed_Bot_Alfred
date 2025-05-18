import json
import os

def load_emo_dict(path='emo_dict.json'):
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

emo_dict = load_emo_dict()

def get_sentiment(text):
    # Очень примитивно: считаем сумму весов из словаря эмоций
    s = 0
    words = text.lower().split()
    for w in words:
        if w in emo_dict:
            s += emo_dict[w]
    # -1...0...+1 шкала
    if s > 0.5:
        return 'positive'
    elif s < -0.5:
        return 'negative'
    else:
        return 'neutral'
