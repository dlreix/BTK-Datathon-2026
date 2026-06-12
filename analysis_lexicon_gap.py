# -*- coding: utf-8 -*-
"""
KİLİT ANALİZ 3 — Lexicon doygunluğu + CV-LB gap'inin kaynağı

İki soruyu cevaplar:
  (a) Lexicon'da "ama" gibi kelimeler eksik mi? Liste genişletilebilir mi?
  (b) Lexicon eklenince CV-LB gap'i neden büyüdü? (drift mi, overfit mi?)
"""
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
y = train['career_success_score'].values
col = 'mentor_feedback_text'
tl = train[col].fillna('').str.lower()

# ── 1) "ama" tuzağı: substring vs kelime-sınırı ───────────────────────────────
sub = tl.str.count('ama').sum()              # substring (yanlış)
wb  = tl.str.count(r'\bama\b').sum()          # word-boundary (doğru)
print(f'"ama" substring={sub}  word-boundary={wb}  → {sub-wb} YANLIŞ pozitif')
# BULGU: "ama" substring olarak 2745 kez geçiyor ama gerçek "ama" sadece 75.
#        Kalanı çal-IŞMA, yaş-AMA gibi kelimelerin içinde (yanlış eşleşme).
# KARAR: Kısa bağlaçları substring ile eklemek gürültü katar. "ancak" zaten
#        listede ve aynı sinyali çok daha geniş (5831 metin) yakalıyor.
#        → Manuel liste genişletmek marjinal; lexicon DOYGUN.

# ── 2) Gap kaynağı: lexicon sinyali test döneminde zayıflıyor mu? (drift testi)
POS = ['mükemmel','başarıl','etkiley','güçlü','yüksek','sektör','sahip','büyük','harika']
NEG = ['gelişim','geliştir','gerekiyor','ancak','fakat','eksik','zayıf','yetersiz','olacaktır']
tp = sum(tl.str.count(w) for w in POS)
tn = sum(tl.str.count(w) for w in NEG)
ratio = (tp - tn) / (tp + tn + 1)
print("\ntext_sent_ratio → hedef korelasyonu, yıl gruplarına göre:")
for a, b in [(2019, 2021), (2022, 2023), (2024, 2026)]:
    m = (train['application_year'] >= a) & (train['application_year'] <= b)
    print(f"  {a}-{b}: r={np.corrcoef(ratio[m], y[m])[0,1]:.3f}")
# BULGU: Korelasyon test döneminde (2024-26) ZAYIFLAMIYOR, hatta güçleniyor (0.48).
# KARAR: Gap temporal drift'ten DEĞİL. Sebep: güçlü feature eklendikçe model
#        train kombinasyonlarına overfit eder, random CV bunu fazla ödüllendirir.
#        → CV kazancını LB'ye çevirirken iskonto uygula (lexicon: ~%58 yansıdı).
#        NOT: embeddings overfit YAPMADI (genel model) — LB kazancı CV'den büyüktü.

print("\nÖZET: lexicon doygun + gap drift değil model-overfit → embeddings daha güvenli")
