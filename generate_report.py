# -*- coding: utf-8 -*-
"""Datathon ilerleme raporunu PDF olarak üretir."""
import os, matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 HRFlowable, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Türkçe karakter destekli font (DejaVuSans) ────────────────────────────────
FP = os.path.join(matplotlib.get_data_path(), 'fonts/ttf')
pdfmetrics.registerFont(TTFont('DejaVu',      os.path.join(FP, 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', os.path.join(FP, 'DejaVuSans-Bold.ttf')))

# ── Renk paleti ───────────────────────────────────────────────────────────────
NAVY  = colors.HexColor('#1a3a5c')
BLUE  = colors.HexColor('#2c6fbb')
LIGHT = colors.HexColor('#eaf2fb')
GREEN = colors.HexColor('#1e8449')
RED   = colors.HexColor('#c0392b')
GREY  = colors.HexColor('#555555')
LGREY = colors.HexColor('#f4f4f4')

# ── Stiller ───────────────────────────────────────────────────────────────────
ss = getSampleStyleSheet()
def style(name, **kw):
    kw.setdefault('fontName', 'DejaVu')
    return ParagraphStyle(name, **kw)

st_title = style('t', fontName='DejaVu-Bold', fontSize=20, textColor=NAVY, alignment=TA_CENTER, leading=24)
st_sub   = style('s', fontSize=11, textColor=GREY, alignment=TA_CENTER, leading=15)
st_h1    = style('h1', fontName='DejaVu-Bold', fontSize=14, textColor=NAVY, spaceBefore=16, spaceAfter=7, leading=17)
st_h2    = style('h2', fontName='DejaVu-Bold', fontSize=11.5, textColor=BLUE, spaceBefore=9, spaceAfter=4, leading=14)
st_body  = style('b', fontSize=9.7, textColor=colors.black, leading=14, alignment=TA_LEFT, spaceAfter=5)
st_bullet= style('bu', fontSize=9.7, textColor=colors.black, leading=14, leftIndent=12, spaceAfter=2)
st_small = style('sm', fontSize=8.3, textColor=GREY, leading=11)
st_cell  = style('c', fontSize=8.6, leading=11)
st_cellb = style('cb', fontName='DejaVu-Bold', fontSize=8.6, leading=11)
st_cellw = style('cw', fontSize=8.6, leading=11, textColor=colors.white)

def P(t, s=st_body): return Paragraph(t, s)
def rule(c=BLUE, w=1.2): return HRFlowable(width='100%', thickness=w, color=c, spaceBefore=2, spaceAfter=8)

story = []

# ══ BAŞLIK ════════════════════════════════════════════════════════════════════
story += [Spacer(1, 0.3*cm),
          P('Kaggle Datathon — İlerleme Raporu', st_title),
          Spacer(1, 0.15*cm),
          P('Kariyer Başarı Skoru Tahmini · Regresyon Problemi', st_sub),
          P('Hazırlanma tarihi: 11 Haziran 2026', st_sub),
          Spacer(1, 0.25*cm), rule(NAVY, 2)]

# ══ 1. PROJE ÖZETİ ════════════════════════════════════════════════════════════
story += [P('1. Proje Özeti', st_h1)]
ov = [
    [P('<b>Görev</b>', st_cellb), P('<b>career_success_score</b> tahmini (0–100 arası sürekli değer)', st_cell)],
    [P('<b>Veri</b>', st_cellb),  P('10.000 eğitim + 10.000 test satırı, ~46 öznitelik', st_cell)],
    [P('<b>Metrik</b>', st_cellb),P('MSE (Kaggle Public Leaderboard üzerinde havuzlanmış/pooled)', st_cell)],
    [P('<b>Başlangıç</b>', st_cellb), P('Baseline LB ≈ 93.7  ·  Hedef ≈ 80', st_cell)],
    [P('<b>Şu anki en iyi</b>', st_cellb), P('<b>LB = 91.65</b>  (submission_10_fulldata_blend)', st_cellb)],
]
t = Table(ov, colWidths=[3.2*cm, 13.3*cm])
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(0,-1),LIGHT), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#cfd8e3')),
    ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
    ('LEFTPADDING',(0,0),(-1,-1),7),('BACKGROUND',(0,4),(-1,4),colors.HexColor('#e8f6ee')),
]))
story += [t, Spacer(1, 0.2*cm)]

# ══ 2. PROBLEM TEŞHİSİ ════════════════════════════════════════════════════════
story += [P('2. Ana Problemin Teşhisi: CV–LB Uçurumu', st_h1),
          P('Projenin başında çapraz doğrulama (CV) skoru ile gerçek leaderboard (LB) skoru arasında '
            'büyük bir uçurum vardı: <b>CV ≈ 85, LB ≈ 94</b>. Her "iyileştirme" CV\'yi düzeltirken LB\'yi '
            'kötüleştiriyordu. Sistematik araştırma kök nedeni ortaya çıkardı:', st_body),
          P('<b>Zamansal dağılım kayması (temporal distribution shift)</b>', st_h2)]
dist = [
    [P('<b>application_year</b>', st_cellw), P('2019–2022', st_cellw), P('2023', st_cellw), P('2024', st_cellw), P('2025', st_cellw), P('2026', st_cellw)],
    [P('Eğitim (train)', st_cell), P('~%52', st_cell), P('%13', st_cell), P('%13', st_cell), P('%12', st_cell), P('%10', st_cell)],
    [P('Test', st_cell), P('%25', st_cell), P('%13', st_cell), P('%20', st_cell), P('%22', st_cell), P('%20', st_cell)],
]
t = Table(dist, colWidths=[4.3*cm]+[2.44*cm]*5)
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY), ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#cfd8e3')),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(1,0),(-1,-1),'CENTER'),
    ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, LGREY]),
]))
story += [t, Spacer(1, 0.15*cm),
          P('Test setinin <b>%62\'si 2024–2026</b> dönemine ait, eğitim seti ise yıllara eşit dağılmış. '
            'Rastgele (random) çapraz doğrulama bunu gizliyordu: model 2025 verisini görüp 2026\'yı tahmin '
            'ediyor, gerçek görevse hiç görmediği bir geleceği tahmin etmek. Bu yüzden CV iyimser yalan söylüyordu.', st_body)]

# ══ 3. DENENEN YAKLAŞIMLAR ════════════════════════════════════════════════════
story += [PageBreak()]
story += [P('3. Denenen Yaklaşımlar ve Sonuçları', st_h1)]
def row(approach, result, verdict, color):
    return [P(approach, st_cell), P(result, st_cell), Paragraph(verdict, style('v', fontName='DejaVu-Bold', fontSize=8.6, textColor=color, leading=11))]
hdr = [P('<b>Yaklaşım</b>', st_cellw), P('<b>Sonuç / Bulgu</b>', st_cellw), P('<b>Karar</b>', st_cellw)]
rows = [hdr,
    row('Veri sızıntısı (leakage) kontrolü', 'İmputasyonu CV içine taşıdık; CV neredeyse değişmedi', 'Sorun değildi', GREY),
    row('_missing bayrakları + log dönüşümleri', 'Sıfır/çok düşük öznitelik önemi, gürültü yaratıyor', 'Kaldırıldı', RED),
    row('Native kategorik + imputasyonsuz', 'One-hot yerine modelin kendi split\'i; LB 94.8 → 93.6', 'Benimsendi', GREEN),
    row('Temporal CV keşfi', 'Gerçek görevi yansıtan pusula; gap\'in kaynağını gösterdi', 'Pusula oldu', GREEN),
    row('CatBoost vs LightGBM', 'CatBoost temporal drift\'e daha dayanıklı (ordered boosting)', 'CatBoost öne', GREEN),
    row('Hyperparameter tuning (Optuna, 40 deneme)', 'Kazanç ±0.4 MSE — fold varyansı içinde, tekrar üretilemedi', 'Gürültü', RED),
    row('XGBoost ekleme', 'LightGBM\'e çok benzer (korelasyon 0.98), katkı ~0', 'İşe yaramadı', RED),
    row('Temporal fold ile submission üretimi', 'Modeller 2025–26\'yı görmeden tahmin etti; LB 101 (felaket)', 'Reddedildi', RED),
    row('Tüm-veri + random fold + blend', 'Tüm yılları gören modeller + CatBoost katkısı; LB 91.65', 'EN İYİ', GREEN),
]
t = Table(rows, colWidths=[5.3*cm, 8.2*cm, 3.0*cm])
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY), ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#cfd8e3')),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
    ('LEFTPADDING',(0,0),(-1,-1),6),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, LGREY]),
    ('BACKGROUND',(0,9),(-1,9),colors.HexColor('#e8f6ee')),
]))
story += [t, PageBreak()]

# ══ 4. SUBMISSION GEÇMİŞİ ═════════════════════════════════════════════════════
story += [P('4. Leaderboard Geçmişi', st_h1)]
subs = [
    [P('<b>Submission</b>', st_cellw), P('<b>Açıklama</b>', st_cellw), P('<b>Public LB (MSE)</b>', st_cellw)],
    [P('submission_00', st_cell), P('İlk deneme (native cat, FE yok)', st_cell), P('93.71', st_cell)],
    [P('submission_01 / 02', st_cell), P('One-hot + imputasyon + feature engineering', st_cell), P('94.72 / 94.80', st_cell)],
    [P('submission_04', st_cell), P('_missing bayrakları kaldırıldı', st_cell), P('94.77', st_cell)],
    [P('submission_06', st_cell), P('Native cat + baseline param + regularizasyon', st_cell), P('93.57', st_cell)],
    [P('submission_08_blend', st_cell), P('LGB+CB blend — TEMPORAL fold üretimi (hatalı)', st_cell), P('101.16', st_cell)],
    [P('submission_10_fulldata', st_cell), P('LGB+CB blend — TÜM-VERİ random fold üretimi', st_cell), P('91.65', st_cell)],
]
t = Table(subs, colWidths=[4.0*cm, 8.7*cm, 3.8*cm])
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY), ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#cfd8e3')),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(2,0),(2,-1),'CENTER'),
    ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),6),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, LGREY]),
    ('BACKGROUND',(0,6),(-1,6),colors.HexColor('#e8f6ee')),
    ('TEXTCOLOR',(2,5),(2,5),RED), ('FONTNAME',(2,5),(2,5),'DejaVu-Bold'),
    ('TEXTCOLOR',(2,6),(2,6),GREEN), ('FONTNAME',(2,6),(2,6),'DejaVu-Bold'),
]))
story += [t, Spacer(1, 0.15*cm),
          P('<i>Not: submission_08 (101.16) ile submission_10 (91.65) aynı modeli kullanır — fark yalnızca '
            'submission\'ın nasıl üretildiğindedir. Bu, raporun en önemli dersine işaret eder.</i>', st_small)]

# ══ 5. ÖNEMLİ DERSLER ═════════════════════════════════════════════════════════
story += [P('5. Çıkarılan Önemli Dersler', st_h1)]
def lesson(title, body):
    return [P(f'<b>{title}</b>', st_h2), P(body, st_body)]
story += lesson('① Değerlendirme ≠ Üretim',
    'Temporal CV, modelin gelecek-genelleme yeteneğini doğru ölçer (LB\'yi 104→101 ile yakın tahmin etti). '
    'Ancak o temporal fold modelleri eksik veriyle eğitildiği için <b>submission üretimi için kullanılmamalı</b>. '
    'Doğru reçete: temporal CV ile <b>değerlendir</b>, ama final submission\'ı <b>tüm veriyle üret</b>.')
story += lesson('② CV pusulasını kalibre et',
    'Bilinen bir LB noktası (submission_06 = 93.57) kullanılarak farklı CV şemaları test edildi. '
    'Random CV ile LB arasındaki fark tutarlı (~+9 MSE). Bu sayede artık <b>submit etmeden</b> iyileştirmeleri '
    'random CV ile ölçebiliyoruz — kısıtlı submission hakkı boşa harcanmıyor.')
story += lesson('③ Hyperparameter tuning her zaman kazandırmaz',
    '40 denemelik Optuna araması yalnızca ±0.4 MSE oynattı; bu fark fold varyansı içinde kaldı ve bağımsız '
    'çalıştırmada tekrar üretilemedi. Asıl darboğaz model parametreleri değil, <b>zayıf sinyaldi</b> '
    '(en güçlü öznitelik korelasyonu yalnızca 0.54).')
story += lesson('④ Ensemble için çeşitlilik şart',
    'XGBoost eklemek işe yaramadı çünkü LightGBM\'e çok benziyordu (korelasyon 0.98). Blend ancak modeller '
    '<b>farklı hatalar</b> yaptığında kazandırır. CatBoost değer kattı çünkü farklı bir algoritma ailesinden geliyor.')

story += [PageBreak()]

# ══ 6. GELDİĞİMİZ NOKTA ═══════════════════════════════════════════════════════
story += [P('6. Geldiğimiz Nokta', st_h1)]
prog = [
    [P('<b>Aşama</b>', st_cellw), P('<b>Public LB (MSE)</b>', st_cellw)],
    [P('Başlangıç (submission_00)', st_cell), P('93.71', st_cell)],
    [P('Önceki en iyi (submission_06)', st_cell), P('93.57', st_cell)],
    [P('Mevcut en iyi (submission_10)', st_cellb), P('91.65', st_cellb)],
    [P('Hedef', st_cell), P('≈ 80', st_cell)],
]
t = Table(prog, colWidths=[10.0*cm, 6.5*cm])
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY), ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#cfd8e3')),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(1,0),(1,-1),'CENTER'),
    ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),7),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, LGREY]),
    ('BACKGROUND',(0,3),(-1,3),colors.HexColor('#e8f6ee')),
]))
story += [t, Spacer(1, 0.2*cm),
          P('Model tarafı (tuning, ek model, blend) büyük ölçüde tüketildi; ~91.65 bir plato. '
            '91.65 → 80 yolculuğu artık bir <b>sinyal meselesi</b>: mevcut öznitelikler hedefi zayıf açıklıyor. '
            'Kalan kaldıraçlar veri/öznitelik tarafında.', st_body)]

# ══ 7. SONRAKİ ADIMLAR ════════════════════════════════════════════════════════
story += [P('7. Planlanan Sonraki Adımlar', st_h1)]
nxt = [
    [P('<b>#</b>', st_cellw), P('<b>Kaldıraç</b>', st_cellw), P('<b>Gerekçe</b>', st_cellw)],
    [P('1', st_cell), P('Öznitelik mühendisliği', st_cell), P('Etkileşim/oran öznitelikleri: mülakat dönüşüm oranı, hackathon başarı oranı, GitHub verimliliği', st_cell)],
    [P('2', st_cell), P('Harici veri', st_cell), P('İzin mevcut: üniversite sıralamaları, rol bazlı piyasa talebi verileri', st_cell)],
    [P('3', st_cell), P('NLP', st_cell), P('mentor_feedback_text — tek dokunulmamış sinyal kaynağı (en sona bırakıldı)', st_cell)],
]
t = Table(nxt, colWidths=[1.0*cm, 4.3*cm, 11.2*cm])
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),NAVY), ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#cfd8e3')),
    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(0,0),(0,-1),'CENTER'),
    ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),6),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, LGREY]),
]))
story += [t, Spacer(1, 0.4*cm), rule(NAVY, 1.5),
          P('Bu rapor, datathon çalışmasının mevcut durumunu özetler. Çalışan ana pipeline: '
            '<b>full_data_blend.py</b> (LightGBM + CatBoost, tüm-veri random 5-fold blend).', st_small)]

# ── PDF üret ──────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate('Datathon_Ilerleme_Raporu.pdf', pagesize=A4,
                        topMargin=1.4*cm, bottomMargin=1.4*cm,
                        leftMargin=2.0*cm, rightMargin=2.0*cm,
                        title='Datathon İlerleme Raporu')

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('DejaVu', 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(2.0*cm, 1.0*cm, 'Kaggle Datathon — Kariyer Başarı Skoru Tahmini')
    canvas.drawRightString(A4[0]-2.0*cm, 1.0*cm, f'Sayfa {doc.page}')
    canvas.setStrokeColor(colors.HexColor('#cfd8e3'))
    canvas.line(2.0*cm, 1.3*cm, A4[0]-2.0*cm, 1.3*cm)
    canvas.restoreState()

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('Datathon_Ilerleme_Raporu.pdf üretildi.')
