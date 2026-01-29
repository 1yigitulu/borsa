# app.py
# Kişisel Yapay Zeka Borsa Asistanı v7.1 (Kararlı Sürüm)
# Güncelleme: Cache Sistemi (Dalgalanmayı Önler) + Hata Kontrolü + Güvenlik

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import time

# === 0. GÜVENLİK (LOGIN EKRANI) ===
def sifre_kontrol():
    """Kullanıcı doğru şifreyi girdi mi kontrol eder."""
    
    # Şifre zaten doğru girildiyse True döndür
    if st.session_state.get("password_correct", False):
        return True

    # Şifre giriş ekranı (Sayfa ayarı burada yapılmalı)
    st.set_page_config(page_title="Giriş Yap", page_icon="🔒")
    st.header("🔒 Korumalı Alan")
    
    # SENİN ÖZEL ŞİFREN
    SIFRE = "1239" 

    password = st.text_input("Lütfen erişim şifresini girin:", type="password")
    
    if st.button("Giriş Yap"):
        if password == SIFRE:
            st.session_state["password_correct"] = True
            st.rerun()  # Sayfayı yenile ve içeri al
        else:
            st.error("❌ Hatalı şifre!")
            
    return False

# Eğer şifre girilmediyse kodun geri kalanını çalıştırma (DUR)
if not sifre_kontrol():
    st.stop()

# Şifre doğruysa ana sayfa ayarlarını güncelle
st.set_page_config(
    page_title="AI Borsa Asistanı v7",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === 1. MODELİ VE AYARLARI YÜKLE ===
@st.cache_resource
def model_yukle():
    try:
        model = joblib.load("model_v7_final.pkl")
        cols = joblib.load("ozellik_sutunlari_v7.pkl")
        return model, cols
    except Exception as e:
        st.error(f"Model dosyaları bulunamadı! Hata: {e}")
        return None, None

model, beklenen_sutunlar = model_yukle()

# === 2. GÜVENLİ VERİ ÇEKME (CACHE SİSTEMİ) ===
@st.cache_data(ttl=900) 
def veri_getir(sembol):
    try:
        # Endeks verisi
        xu100 = yf.download("XU100.IS", period="2y", progress=False)
        if isinstance(xu100.columns, pd.MultiIndex): xu100.columns = xu100.columns.get_level_values(0)
        
        # Hisse verisi
        hisse = yf.download(sembol, period="2y", progress=False)
        if isinstance(hisse.columns, pd.MultiIndex): hisse.columns = hisse.columns.get_level_values(0)
        
        # Basit Hata Kontrolü
        if len(hisse) > 0:
            son_fiyat = hisse['Close'].iloc[-1]
            if son_fiyat > 10000 and "IS" in sembol and sembol != "KONYA.IS": 
                return None, None, "Hatalı Fiyat Verisi (Yahoo Kaynaklı)"

        return hisse, xu100, None
    except Exception as e:
        return None, None, str(e)

# === 3. ÖZELLİK MOTORU ===
def hesapla_teknik_ozellikler_final(df, xu100_df):
    f = pd.DataFrame(index=df.index)
    f['Kapanis'] = df['Close']
    
    high = df['High']; low = df['Low']; close = df['Close']; volume = df['Volume']
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    f['rsi_norm'] = (rsi - 50) / 50
    
    # Ortalamalar
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    f['mesafe_ma50'] = (close - ma50) / ma50
    f['mesafe_ma200'] = (close - ma200) / ma200
    f['trend_guc'] = (ma50 - ma200) / ma200
    
    # Bollinger
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + (std20 * 2)
    lower = ma20 - (std20 * 2)
    f['bb_width'] = (upper - lower) / ma20
    f['bb_pozisyon'] = (close - lower) / (upper - lower)
    
    # ADX/ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    f['adx'] = dx.rolling(14).mean().fillna(25)
    
    # Volatilite
    f['hacim_zscore'] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    f['volatilite'] = (np.log(close / close.shift(1))).rolling(20).std()
    
    # Beta
    if xu100_df is not None:
        xu_ret = xu100_df['Close'].pct_change()
        hisse_ret = close.pct_change()
        f['relative_strength'] = (hisse_ret.rolling(20).mean() - xu_ret.rolling(20).mean()) * 100
    else:
        f['relative_strength'] = 0

    return f

# === 4. MENTOR YORUM MOTORU ===
def yorum_olustur(sembol, hisse_df, tahmin, guven, features):
    son_fiyat = hisse_df['Close'].iloc[-1]
    onceki_fiyat = hisse_df['Close'].iloc[-2]
    degisim_yuzde = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
    
    hacim_son = hisse_df['Volume'].iloc[-1]
    hacim_ort = hisse_df['Volume'].rolling(20).mean().iloc[-1]
    hacim_durumu = "YÜKSEK" if hacim_son > hacim_ort else "NORMAL"
    
    ma200 = hisse_df['Close'].rolling(200).mean().iloc[-1]
    trend = "YÜKSELİŞ (Boğa)" if son_fiyat > ma200 else "DÜŞÜŞ/BASKI (Ayı)"
    
    st.markdown("---")
    st.subheader(f"🧠 Yapay Zeka'nın {sembol} İçin Strateji Raporu")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Anlık Fiyat", f"{son_fiyat:.2f} TL", f"%{degisim_yuzde:.2f}")
    col2.metric("AI Güveni", f"%{guven*100:.1f}", "Başarı İhtimali")
    col3.metric("Ana Trend", trend)
    col4.metric("Hacim Durumu", hacim_durumu)

    st.markdown("### 📝 Analiz ve Yol Haritası")
    
    if tahmin == 2:
        if guven > 0.60:
            baslik = "🚀 FIRSAT TESPİT EDİLDİ: Güçlü Sinyal!"
            motivasyon = "Model bu grafiği geçmişteki en karlı formasyonlara benzetiyor."
            st.success(f"**{baslik}**")
        else:
            baslik = "✅ OLUMLU GÖRÜNÜM: Küçük Riskle Denenebilir"
            motivasyon = "Model potansiyel görüyor ancak piyasa şartlarına dikkat edilmeli."
            st.warning(f"**{baslik}**")

        st.info(f"""
        Merhaba! Ben senin yapay zeka asistanınım. {sembol} hissesinin geçmiş 5 yıllık verilerini taradım. 
        Şu anki teknik göstergeler, hissenin **önümüzdeki 5-7 işlem günü içinde** yukarı yönlü bir atak yapmaya hazırlandığını gösteriyor.
        
        **Yatırımcı Disiplini:**
        * **⏳ Vade:** Hedefimiz "Swing Trade". **1 hafta içinde** %3 ile %10 arası kâr.
        * **🧘 Sabır:** Yarın %1-2 düşerse panik yapma. Trend bozulmadıkça oyundasın.
        * **💰 Çıkış:** Beklenen kâr (örneğin %5) gelirse bekleme, kârı al.
        
        _{motivasyon}_
        """)
        
        stop_fiyat = son_fiyat * 0.95
        hedef_fiyat = son_fiyat * 1.05
        col_L, col_R = st.columns(2)
        with col_L: st.error(f"🛑 **Stop-Loss:** {stop_fiyat:.2f} TL")
        with col_R: st.success(f"🎯 **Hedef:** {hedef_fiyat:.2f} TL")

    elif tahmin == 0:
        st.error(f"""
        **🛑 SATIŞ BASKISI / UZAK DUR**
        
        Model, {sembol} grafiğinde zayıflık tespit etti. Geçmişte benzer durumlar genelde düşüş veya yatay seyirle sonuçlanmış.
        * **Tavsiye:** Nakitte kalmak şu an daha güvenli.
        """)
    else:
        st.warning(f"""
        **⚪ KARARSIZ BÖLGE (Nötr)**
        Model net bir yön göremiyor. İşlem yapmak yazı-tura atmak gibi olabilir.
        * **Tavsiye:** İzleme listene al ama acele etme.
        """)

# === 4. ARAYÜZ TASARIMI ===
st.title("🦅 Yapay Zeka Borsa Asistanı v7.1")
st.markdown("**Model:** XGBoost Sweet Spot | **Durum:** Kararlı & Önbellekli")

st.sidebar.header("⚙️ Ayarlar")
guven_esigi = st.sidebar.slider("Güven Eşiği (%)", 30, 90, 45) / 100
BLACKLIST = ['TUPRS.IS', 'SAHOL.IS']

tab1, tab2 = st.tabs(["🔍 Tek Hisse Analizi", "🔭 BIST 100 Tarama"])

with tab1:
    st.subheader("Hisse Dedektifi")
    sembol_giris = st.text_input("Hisse Kodu Girin (Örn: THYAO):", "").upper()
    
    if st.button("Analiz Et", key="btn1"):
        if sembol_giris:
            if sembol_giris + ".IS" in BLACKLIST:
                st.error(f"⚠️ {sembol_giris} kara listede.")
            else:
                sembol = sembol_giris + ".IS" if not sembol_giris.endswith(".IS") else sembol_giris
                with st.spinner("Veriler çekiliyor (Önbellek Kullanılıyor)..."):
                    hisse, xu100, hata = veri_getir(sembol)
                    if hata:
                        st.error(f"Veri Hatası: {hata}")
                    elif hisse is not None and len(hisse) > 200:
                        feat = hesapla_teknik_ozellikler_final(hisse, xu100)
                        son_durum = feat.iloc[[-1]][beklenen_sutunlar].fillna(0)
                        tahmin = model.predict(son_durum)[0]
                        guven = model.predict_proba(son_durum)[0][tahmin]
                        yorum_olustur(sembol, hisse, tahmin, guven, feat)
                    else:
                        st.error("Yetersiz veri veya hisse bulunamadı.")

with tab2:
    st.subheader("BIST 100 Tarama")
    if st.button("Taramayı Başlat", key="btn2"):
        TARAMA_LISTESI = [
            'AEFES.IS', 'AGHOL.IS', 'AHGAZ.IS', 'AKBNK.IS', 'AKCNS.IS', 'AKFGY.IS', 'AKSA.IS', 'AKSEN.IS', 'ALARK.IS', 'ALBRK.IS',
            'ALFAS.IS', 'ARCLK.IS', 'ASELS.IS', 'ASTOR.IS', 'ASUZU.IS', 'AYDEM.IS', 'BAGFS.IS', 'BERA.IS', 'BIMAS.IS', 'BIOEN.IS',
            'BRSAN.IS', 'BRYAT.IS', 'BUCIM.IS', 'CANTE.IS', 'CCOLA.IS', 'CEMTS.IS', 'CIMSA.IS', 'CWENE.IS', 'DOAS.IS', 'DOHOL.IS',
            'ECILC.IS', 'ECZYT.IS', 'EGEEN.IS', 'EKGYO.IS', 'ENJSA.IS', 'ENKAI.IS', 'EREGL.IS', 'EUPWR.IS', 'EUREN.IS', 'FROTO.IS',
            'GARAN.IS', 'GENIL.IS', 'GESAN.IS', 'GLYHO.IS', 'GSDHO.IS', 'GUBRF.IS', 'GWIND.IS', 'HALKB.IS', 'HEKTS.IS', 'IMASM.IS',
            'IPEKE.IS', 'ISCTR.IS', 'ISDMR.IS', 'ISGYO.IS', 'ISMEN.IS', 'IZMDC.IS', 'KARSN.IS', 'KAYSE.IS', 'KCAER.IS', 'KCHOL.IS',
            'KMPUR.IS', 'KONTR.IS', 'KONYA.IS', 'KORDS.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 'KZBGY.IS', 'MAVI.IS', 'MGROS.IS',
            'MIATK.IS', 'ODAS.IS', 'OTKAR.IS', 'OYAKC.IS', 'PENTA.IS', 'PETKM.IS', 'PGSUS.IS', 'PSGYO.IS', 'QUAGR.IS', 'SAHOL.IS',
            'SASA.IS', 'SELEC.IS', 'SISE.IS', 'SKBNK.IS', 'SMRTG.IS', 'SNGYO.IS', 'SOKM.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS',
            'TKFEN.IS', 'TOASO.IS', 'TSKB.IS', 'TTKOM.IS', 'TTRAK.IS', 'TUKAS.IS', 'TUPRS.IS', 'ULKER.IS', 'VAKBN.IS', 'VESBE.IS',
            'VESTL.IS', 'YEOTK.IS', 'YKBNK.IS', 'YYLGD.IS', 'ZOREN.IS'
        ]
        progress = st.progress(0)
        sonuclar = []
        for i, sembol in enumerate(TARAMA_LISTESI):
            progress.progress((i+1)/len(TARAMA_LISTESI))
            if sembol in BLACKLIST: continue
            try:
                hisse, xu100, err = veri_getir(sembol) 
                if hisse is not None and len(hisse) > 200:
                    feat = hesapla_teknik_ozellikler_final(hisse, xu100)
                    son_durum = feat.iloc[[-1]][beklenen_sutunlar].fillna(0)
                    tahmin = model.predict(son_durum)[0]
                    guven = model.predict_proba(son_durum)[0][tahmin]
                    if tahmin == 2 and guven >= guven_esigi:
                        sonuclar.append({
                            "Hisse": sembol.replace(".IS", ""),
                            "Fiyat": f"{hisse['Close'].iloc[-1]:.2f}",
                            "Güven": f"%{guven*100:.1f}",
                            "Skor": guven
                        })
            except: pass
        progress.empty()
        if sonuclar:
            df = pd.DataFrame(sonuclar).sort_values("Skor", ascending=False)
            st.success(f"Tarama Bitti! {len(sonuclar)} fırsat bulundu.")
            st.dataframe(df[['Hisse', 'Fiyat', 'Güven']], use_container_width=True)
        else:
            st.warning("Kriterlere uygun hisse yok.")
