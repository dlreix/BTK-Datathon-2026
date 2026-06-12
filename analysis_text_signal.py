# -*- coding: utf-8 -*-
"""
KİLİT ANALİZ 2 — mentor_feedback_text içindeki sinyal

Bu analiz NLP'ye geçme kararının gerekçesidir.
Soru: Metin hedefle ilişkili mi? İlişki uzunlukta mı içerikte mi?
"""
import pandas as pd
import numpy as np
from collections import Counter
import re

train = pd.read_csv('train.csv')
y = train['career_success_score']
col = 'mentor_feedback_text'

# ── 1) Metin var mı, dolu mu, uzunluk hedefle ilişkili mi? ─────────────────────
test = pd.read_csv('test_x.csv')
print("Metin test'te de var mı:", col in test.columns, "| train dolu:",
      train[col].notna().mean(), "| test dolu:", test[col].notna().mean())
length = train[col].fillna('').str.len()
print(f"Uzunluk → hedef korelasyonu: r={length.corr(y):.3f}")
# BULGU: Metin hem train hem test'te %100 dolu (kullanılabilir).
#        Ama UZUNLUK hedefle ilişkisiz (r~0.01) → basit meta-feature işe yaramaz.
#        Sinyal varsa İÇERİKTE (hangi kelimeler) olmalı.

# ── 2) Yüksek vs düşük skorlu metinlerde ayırt edici kelimeler ────────────────
hi = train[y >= y.quantile(0.85)][col]
lo = train[y <= y.quantile(0.15)][col]
def word_freq(series):
    c = Counter()
    for t in series.fillna(''):
        for w in re.findall(r'[a-zçğıöşüâî]+', t.lower()):
            if len(w) > 3:
                c[w] += 1
    return c
ch, cl = word_freq(hi), word_freq(lo)
nh, nl = sum(ch.values()), sum(cl.values())
diff = {w: ch[w]/nh - cl[w]/nl for w in set(ch) | set(cl) if ch[w] + cl[w] > 20}
top = sorted(diff.items(), key=lambda x: -x[1])
print("\nYÜKSEK skorda baskın:", ', '.join(w for w, _ in top[:10]))
print("DÜŞÜK skorda baskın:  ", ', '.join(w for w, _ in top[-10:]))
# BULGU: Yüksek skor → "mükemmel, başarı, etkileyici, güçlü" (övgü)
#        Düşük skor → "gelişim, gerekiyor, ancak, çalışması" (eleştiri/gelişim)
# KARAR: Metnin TONU hedefle güçlü ilişkili. Mentor düşük skorluya
#        "X iyi ANCAK Y geliştir" diyor. → Sözlük (lexicon) tabanlı ton skoru
#        ile başla; sonra embeddings ile anlamsal sinyali yakala.

print("\nÖZET: Sinyal uzunlukta değil İÇERİKTE/tonda → lexicon + embeddings yolu")
