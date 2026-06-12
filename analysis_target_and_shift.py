# -*- coding: utf-8 -*-
"""
KİLİT ANALİZ 1 — Hedef sinyali + Train/Test temporal kayması

Bu analiz iki kritik kararın gerekçesidir:
  (a) Hedef değişken zayıf açıklanıyor → büyük kazanç metin/dış sinyalden gelir
  (b) Test seti train'den FARKLI bir zaman dilimine ait → CV stratejisi temporal olmalı

Çıktılar projenin dönüm noktalarını açıklar; sonuçlar yorum bloklarında özetlidir.
"""
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test  = pd.read_csv('test_x.csv')
y = train['career_success_score']

# ── 1) Feature → hedef korelasyonları ─────────────────────────────────────────
# Soru: mevcut sayısal feature'lar hedefi ne kadar açıklıyor?
num_cols = [c for c in train.select_dtypes(include=[np.number]).columns
            if c not in ['student_id', 'career_success_score']]
corr = train[num_cols].corrwith(y).abs().sort_values(ascending=False)
print("=== Feature → hedef korelasyonu (en güçlü 8) ===")
print(corr.head(8).round(3).to_string())
# BULGU: En güçlü feature project_quality_score = 0.54; gerisi 0.1-0.34 arası.
# cgpa, attendance, english, age neredeyse SIFIR korelasyon.
# KARAR: Hedef tablo verisiyle zayıf açıklanıyor → asıl kazanç metinde (NLP) olabilir.

# ── 2) Train vs Test zaman dağılımı (temporal shift) ──────────────────────────
# Soru: train ve test aynı dağılımdan mı geliyor?
print("\n=== application_year dağılımı (oran %) ===")
tr_pct = (train['application_year'].value_counts(normalize=True).sort_index() * 100).round(1)
te_pct = (test['application_year'].value_counts(normalize=True).sort_index() * 100).round(1)
cmp = pd.DataFrame({'train_%': tr_pct, 'test_%': te_pct})
print(cmp.to_string())
# BULGU: Train yıllara ~eşit dağılmış; TEST 2024-2026'da yoğun (~%62).
# Yani test, train'in görece az gördüğü "gelecek" döneme ait.
# KARAR: Random KFold yanıltıcı (interpolation ölçer). Test'i taklit için
#        TEMPORAL CV gerekir (geçmişle eğit, geleceği tahmin et) — değerlendirme için.
#        ANCAK submission'lar TÜM veriyle üretilmeli (temporal fold modelleri
#        eksik veriyle eğitilip LB'yi bozuyor — bkz. sub_08 = LB 101).

print("\nÖZET: (1) hedef zayıf açıklanıyor → NLP potansiyeli yüksek")
print("      (2) test 2024-26 ağırlıklı → değerlendir=temporal, üret=tüm-veri")
