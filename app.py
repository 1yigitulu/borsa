import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from fpdf import FPDF

# === 0. GÜVENLİK (LOGIN EKRANI) ===
def sifre_kontrol():
    if st.session_state.get("password_correct", False):
        return True

    st.set_page_config(page_title="Giriş Yap", page_icon="🔒")
    st.header("🔒 Korumalı Alan")
    
    SIFRE = "1239" # Şifren burada

    password = st.text_input("Lütfen erişim şifresini girin:", type="password")
    
    if st.button("Giriş Yap"):
        if password == SIFRE:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("❌ Hatalı şifre!")
    return False

if not sifre_kontrol():
    st.stop()

# Sayfa Yapılandırması
st.set_page_config(page_title="AI Borsa Asistanı v7", page_icon="🦅", layout="wide")

# === 1. YARDIMCI FONKSİYONLAR (PDF & KARAKTER) ===
def tr_to_en(metin):
    """PDF'in çökmesini engellemek için Türkçe karakter temizliği yapar."""
    mapping = {
        "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G", "ı": "i", "İ": "I",
        "ö": "o", "Ö": "O", "ş": "s", "Ş": "S", "ü": "u", "Ü": "U"
    }
    for tr, en in mapping.items():
        metin = metin.replace(tr, en)
    return metin

def pdf_olustur(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(190, 10, tr_to_en("BIST TARAMA RAPORU"), ln=True, align="C")
    
    pdf.set_font("helvetica", "", 10)
    tarih = tr_to_en(f"Rapor Tarihi: {pd.Timestamp.now().strftime('%d-%m-%Y %H:%M')}")
    pdf.cell(190, 10, tarih, ln=True, align="C")
    pdf.ln(10)
    
    # Tablo Başlıkları
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(60, 10, "Hisse", 1, 0, "C", True)
    pdf.cell(60, 10, "Fiyat (TL)", 1, 0, "C", True)
    pdf.cell(70, 10, "Guven Skoru", 1, 1, "C", True)
    
    # Veriler
    pdf.set_font("helvetica", "", 12)
    for i, row in df.iterrows():
        pdf.cell(60, 10, tr_to_en(str(row['Hisse'])), 1, 0, "C")
        pdf.cell(60, 10, tr_to_en(str(row['Fiyat'])), 1, 0, "C")
        pdf.cell(70, 10, tr_to_en(str(row['Güven'])), 1, 1, "C")
    
    # HATA ÇÖZÜMÜ: pdf.output() çıktısını bytes() ile sarmalıyoruz
    return bytes(pdf.output())

# === 2. MODEL VE VERİ MOTORU ===
@st.cache_resource
def model_yukle():
    try:
        model = joblib.load("model_v7_final.pkl")
        cols = joblib.load("ozellik_sutunlari_v7.pkl")
        return model, cols
    except: return None, None

model, beklenen_sutunlar = model_yukle()

@st.cache_data(ttl=900)
def veri_getir(sembol):
    try:
        xu100 = yf.download("XU100.IS", period="2y", progress=False)
        if isinstance(xu100.columns, pd.MultiIndex): xu100.columns = xu100.columns.get_level_values(0)
        hisse = yf.download(sembol, period="2y", progress=False)
        if isinstance(hisse.columns, pd.MultiIndex): hisse.columns = hisse.columns.get_level_values(0)
        return hisse, xu100, None
    except Exception as e: return None, None, str(e)

# (Teknik özellik hesaplama ve Yorum motoru kodları buraya gelecek - Eski kodunla aynı olduğu için kısalttım)
def hesapla_teknik_ozellikler_final(df, xu100_df):
    f = pd.DataFrame(index=df.index)
    f['Kapanis'] = df['Close']
    close = df['Close']; high = df['High']; low = df['Low']; volume = df['Volume']
    delta = close.diff(); gain = delta.where(delta > 0, 0).rolling(14).mean(); loss = -delta.where(delta < 0, 0).rolling(14).mean()
    f['rsi_norm'] = (100 - (100 / (1 + (gain / loss))) - 50) / 50
    f['mesafe_ma50'] = (close - close.rolling(50).mean()) / close.rolling(50).mean()
    f['mesafe_ma200'] = (close - close.rolling(200).mean()) / close.rolling(200).mean()
    f['trend_guc'] = (close.rolling(50).mean() - close.rolling(200).mean()) / close.rolling(200).mean()
    ma20 = close.rolling(20).mean(); std20 = close.rolling(20).std(); upper = ma20 + (std20 * 2); lower = ma20 - (std20 * 2)
    f['bb_width'] = (upper - lower) / ma20
    f['bb_pozisyon'] = (close - lower) / (upper - lower)
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1); atr = tr.rolling(14).mean()
    up_move = high - high.shift(1); down_move = low.shift(1) - low
    plus_di = 100 * (pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0)).rolling(14).mean() / atr)
    minus_di = 100 * (pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0)).rolling(14).mean() / atr)
    f['adx'] = (100 * abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(14).mean().fillna(25)
    f['hacim_zscore'] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    f['volatilite'] = (np.log(close / close.shift(1))).rolling(20).std()
    if xu100_df is not None: f['relative_strength'] = (close.pct_change().rolling(20).mean() - xu100_df['Close'].pct_change().rolling(20).mean()) * 100
    else: f['relative_strength'] = 0
    return f

def yorum_olustur(sembol, hisse_df, tahmin, guven, features):
    son_fiyat = hisse_df['Close'].iloc[-1]
    degisim = ((son_fiyat - hisse_df['Close'].iloc[-2]) / hisse_df['Close'].iloc[-2]) * 100
    st.markdown("---")
    st.subheader(f"🧠 AI {sembol} Strateji Raporu")
    col1, col2 = st.columns(2)
    col1.metric("Fiyat", f"{son_fiyat:.2f} TL", f"%{degisim:.2f}")
    col2.metric("Güven", f"%{guven*100:.1f}")
    
    if tahmin == 2:
        st.success(f"**🚀 AL SİNYALİ:** {sembol} için önümüzdeki 1 hafta olumlu görünüyor.")
        st.info("🧘 Sabırlı ol, 1 hafta bekle, %5 karda çıkmayı düşün. Stop: %5 altı.")
    elif tahmin == 0: st.error("🛑 **SAT/UZAK DUR:** Teknik göstergeler zayıf.")
    else: st.warning("⚪ **NOTR:** Belirsiz piyasa, izlemede kal.")

# === 3. ARAYÜZ ===
st.title("🦅 AI Borsa Asistanı v7.2")
tab1, tab2 = st.tabs(["🔍 Tek Hisse", "🔭 BIST 100 Tarama"])

with tab1:
    sembol_giris = st.text_input("Hisse Kodu:", "").upper()
    if st.button("Analiz"):
        sembol = sembol_giris + ".IS" if not sembol_giris.endswith(".IS") else sembol_giris
        hisse, xu, err = veri_getir(sembol)
        if hisse is not None:
            feat = hesapla_teknik_ozellikler_final(hisse, xu)
            row = feat.iloc[[-1]][beklenen_sutunlar].fillna(0)
            tahmin = model.predict(row)[0]; guven = model.predict_proba(row)[0][tahmin]
            yorum_olustur(sembol, hisse, tahmin, guven, feat)

with tab2:
    if st.button("Taramayı Başlat"):
        TARAMA_LISTESI = ['AKBNK.IS', 'AKSEN.IS', 'ASELS.IS', 'BIMAS.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'KCHOL.IS', 'PETKM.IS', 'PGSUS.IS', 'SISE.IS', 'TCELL.IS', 'THYAO.IS', 'TTKOM.IS', 'TUPRS.IS', 'YKBNK.IS'] # Örnek liste
        sonuclar = []
        progress = st.progress(0)
        for i, s in enumerate(TARAMA_LISTESI):
            progress.progress((i+1)/len(TARAMA_LISTESI))
            h, xu, err = veri_getir(s)
            if h is not None and len(h) > 200:
                feat = hesapla_teknik_ozellikler_final(h, xu)
                row = feat.iloc[[-1]][beklenen_sutunlar].fillna(0)
                tahmin = model.predict(row)[0]; guven = model.predict_proba(row)[0][tahmin]
                if tahmin == 2 and guven >= 0.45:
                    sonuclar.append({"Hisse": s.replace(".IS",""), "Fiyat": f"{h['Close'].iloc[-1]:.2f}", "Güven": f"%{guven*100:.1f}", "Skor": guven})
        
        if sonuclar:
            df_res = pd.DataFrame(sonuclar).sort_values("Skor", ascending=False)
            st.dataframe(df_res[['Hisse', 'Fiyat', 'Güven']], use_container_width=True)
            
            # PDF İNDİRME BUTONU
            pdf_bytes = pdf_olustur(df_res)
            st.download_button(label="📄 Raporu PDF Olarak İndir", data=pdf_bytes, file_name="BIST_Tarama.pdf", mime="application/pdf")
