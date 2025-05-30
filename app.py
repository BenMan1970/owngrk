import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import traceback
import requests
import json

st.set_page_config(page_title="Scanner Confluence Forex (OANDA Data)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données OANDA)")
st.markdown("*Utilisation de l'API OANDA pour les données de marché H1*")

# Liste des paires Forex courantes
PAIRS_OANDA = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 'NZD_USD', 'USD_CHF'
]

def ema(s, p): 
    return s.ewm(span=p, adjust=False).mean()

def rma(s, p): 
    return s.ewm(alpha=1/p, adjust=False).mean()

def hull_ma_pine(dc, p=20):
    try:
        if len(dc) < p:
            return pd.Series([np.nan] * len(dc), index=dc.index)
        
        hl = max(1, int(p/2))
        sl = max(1, int(np.sqrt(p)))
        
        def wma(series, window):
            weights = np.arange(1, window + 1)
            return series.rolling(window=window).apply(
                lambda x: np.sum(x * weights[:len(x)]) / np.sum(weights[:len(x)]) if len(x) == window else np.nan, 
                raw=True
            )
        
        wma1 = wma(dc, hl)
        wma2 = wma(dc, p)
        diff = 2 * wma1 - wma2
        return wma(diff, sl)
    except:
        return pd.Series([np.nan] * len(dc), index=dc.index)

def rsi_pine(po4, p=10): 
    try:
        d = po4.diff()
        g = d.where(d > 0, 0.0)
        l = -d.where(d < 0, 0.0)
        ag = rma(g, p)
        al = rma(l, p)
        rs = ag / al.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(po4), index=po4.index)

def adx_pine(h, l, c, p=14):
    try:
        tr1 = h - l
        tr2 = abs(h - c.shift(1))
        tr3 = abs(l - c.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = rma(tr, p)
        
        um = h.diff()
        dm = l.shift(1) - l
        
        pdm = pd.Series(np.where((um > dm) & (um > 0), um, 0.0), index=h.index)
        mdm = pd.Series(np.where((dm > um) & (dm > 0), dm, 0.0), index=h.index)
        
        satr = atr.replace(0, 1e-9)
        pdi = 100 * (rma(pdm, p) / satr)
        mdi = 100 * (rma(mdm, p) / satr)
        
        dxden = (pdi + mdi).replace(0, 1e-9)
        dx = 100 * (abs(pdi - mdi) / dxden)
        
        return rma(dx, p).fillna(0)
    except:
        return pd.Series([0] * len(h), index=h.index)

def heiken_ashi_pine(dfo):
    try:
        ha = pd.DataFrame(index=dfo.index)
        if dfo.empty:
            ha['HA_Open'] = pd.Series(dtype=float)
            ha['HA_Close'] = pd.Series(dtype=float)
            return ha['HA_Open'], ha['HA_Close']
        
        ha['HA_Close'] = (dfo['Open'] + dfo['High'] + dfo['Low'] + dfo['Close']) / 4
        ha['HA_Open'] = np.nan
        
        if not dfo.empty:
            ha.iloc[0, ha.columns.get_loc('HA_Open')] = (dfo['Open'].iloc[0] + dfo['Close'].iloc[0]) / 2
            for i in range(1, len(dfo)):
                ha.iloc[i, ha.columns.get_loc('HA_Open')] = (ha.iloc[i-1, ha.columns.get_loc('HA_Open')] + ha.iloc[i-1, ha.columns.get_loc('HA_Close')]) / 2
        
        return ha['HA_Open'], ha['HA_Close']
    except:
        empty_series = pd.Series([np.nan] * len(dfo), index=dfo.index)
        return empty_series, empty_series

def smoothed_heiken_ashi_pine(dfo, l1=10, l2=10):
    try:
        eo = ema(dfo['Open'], l1)
        eh = ema(dfo['High'], l1)
        el = ema(dfo['Low'], l1)
        ec = ema(dfo['Close'], l1)
        
        hai = pd.DataFrame({'Open': eo, 'High': eh, 'Low': el, 'Close': ec}, index=dfo.index)
        hao_i, hac_i = heiken_ashi_pine(hai)
        sho = ema(hao_i, l2)
        shc = ema(hac_i, l2)
        
        return sho, shc
    except:
        empty_series = pd.Series([np.nan] * len(dfo), index=dfo.index)
        return empty_series, empty_series

def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    try:
        min_len_req = max(tenkan_p, kijun_p, senkou_b_p)
        if len(df_high) < min_len_req or len(df_low) < min_len_req or len(df_close) < min_len_req:
            return 0
        
        ts = (df_high.rolling(window=tenkan_p).max() + df_low.rolling(window=tenkan_p).min()) / 2
        ks = (df_high.rolling(window=kijun_p).max() + df_low.rolling(window=kijun_p).min()) / 2
        sa = (ts + ks) / 2
        sb = (df_high.rolling(window=senkou_b_p).max() + df_low.rolling(window=senkou_b_p).min()) / 2
        
        if pd.isna(df_close.iloc[-1]) or pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1]):
            return 0
        
        ccl = df_close.iloc[-1]
        cssa = sa.iloc[-1]
        cssb = sb.iloc[-1]
        ctn = max(cssa, cssb)
        cbn = min(cssa, cssb)
        
        sig = 0
        if ccl > ctn:
            sig = 1
        elif ccl < cbn:
            sig = -1
        
        return sig
    except:
        return 0

@st.cache_data(ttl=300)
def get_data_oanda(symbol: str, timeframe: str = 'H1', count: int = 200):
    """
    Récupère les données OANDA via l'API REST
    """
    try:
        # Vérifier la configuration
        if 'oanda' not in st.secrets or 'API_KEY' not in st.secrets["oanda"]:
            st.error("Configuration OANDA manquante dans les secrets Streamlit")
            return None
            
        api_key = st.secrets["oanda"]["API_KEY"]
        
        # URL de l'API OANDA (practice environment)
        base_url = "https://api-fxpractice.oanda.com"
        
        # Headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Paramètres de la requête
        params = {
            "count": count,
            "granularity": timeframe
        }
        
        # Faire la requête
        url = f"{base_url}/v3/instruments/{symbol}/candles"
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code != 200:
            st.error(f"Erreur API OANDA ({response.status_code}): {response.text}")
            return None
        
        data = response.json()
        candles = data.get("candles", [])
        
        if not candles:
            st.warning(f"Aucune donnée reçue pour {symbol}")
            return None
        
        # Construire le DataFrame
        ohlc_data = []
        timestamps = []
        
        for candle in candles:
            if candle.get("complete", False):  # Seulement les bougies complètes
                mid = candle["mid"]
                ohlc_data.append({
                    'Open': float(mid['o']),
                    'High': float(mid['h']),
                    'Low': float(mid['l']),
                    'Close': float(mid['c'])
                })
                timestamps.append(pd.to_datetime(candle['time']))
        
        if not ohlc_data:
            st.warning(f"Aucune bougie complète pour {symbol}")
            return None
        
        df = pd.DataFrame(ohlc_data, index=timestamps)
        
        if len(df) < 50:
            st.warning(f"Données insuffisantes pour {symbol} ({len(df)} bougies)")
            return None
        
        return df
        
    except requests.exceptions.Timeout:
        st.error(f"Timeout lors de la récupération des données pour {symbol}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion pour {symbol}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue pour {symbol}: {str(e)}")
        return None

def calculate_all_signals_pine(data):
    if data is None or len(data) < 60:
        return None
    
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        return None
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    open_price = data['Open']
    ohlc4 = (open_price + high + low + close) / 4
    
    bull_confluences = 0
    bear_confluences = 0
    signal_details_pine = {}

    # HMA Signal
    try:
        hma_series = hull_ma_pine(close, 20)
        if len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
            hma_val = hma_series.iloc[-1]
            hma_prev = hma_series.iloc[-2]
            if hma_val > hma_prev:
                bull_confluences += 1
                signal_details_pine['HMA'] = "▲"
            elif hma_val < hma_prev:
                bear_confluences += 1
                signal_details_pine['HMA'] = "▼"
            else:
                signal_details_pine['HMA'] = "─"
        else:
            signal_details_pine['HMA'] = "N/A"
    except Exception:
        signal_details_pine['HMA'] = "Err"

    # RSI Signal
    try:
        rsi_series = rsi_pine(ohlc4, 10)
        if len(rsi_series) >= 1 and not pd.isna(rsi_series.iloc[-1]):
            rsi_val = rsi_series.iloc[-1]
            signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
            if rsi_val > 50:
                bull_confluences += 1
                signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
            elif rsi_val < 50:
                bear_confluences += 1
                signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
            else:
                signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
        else:
            signal_details_pine['RSI'] = "N/A"
            signal_details_pine['RSI_val'] = "N/A"
    except Exception:
        signal_details_pine['RSI'] = "Err"
        signal_details_pine['RSI_val'] = "N/A"

    # ADX Signal
    try:
        adx_series = adx_pine(high, low, close, 14)
        if len(adx_series) >= 1 and not pd.isna(adx_series.iloc[-1]):
            adx_val = adx_series.iloc[-1]
            signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
            if adx_val >= 20:
                bull_confluences += 1
                bear_confluences += 1
                signal_details_pine['ADX'] = f"✔({adx_val:.0f})"
            else:
                signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
        else:
            signal_details_pine['ADX'] = "N/A"
            signal_details_pine['ADX_val'] = "N/A"
    except Exception:
        signal_details_pine['ADX'] = "Err"
        signal_details_pine['ADX_val'] = "N/A"

    # Heiken Ashi Signal
    try:
        ha_open, ha_close = heiken_ashi_pine(data)
        if len(ha_open) >= 1 and len(ha_close) >= 1 and \
           not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
            if ha_close.iloc[-1] > ha_open.iloc[-1]:
                bull_confluences += 1
                signal_details_pine['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]:
                bear_confluences += 1
                signal_details_pine['HA'] = "▼"
            else:
                signal_details_pine['HA'] = "─"
        else:
            signal_details_pine['HA'] = "N/A"
    except Exception:
        signal_details_pine['HA'] = "Err"

    # Smoothed Heiken Ashi Signal
    try:
        sha_open, sha_close = smoothed_heiken_ashi_pine(data, 10, 10)
        if len(sha_open) >= 1 and len(sha_close) >= 1 and \
           not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
            if sha_close.iloc[-1] > sha_open.iloc[-1]:
                bull_confluences += 1
                signal_details_pine['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]:
                bear_confluences += 1
                signal_details_pine['SHA'] = "▼"
            else:
                signal_details_pine['SHA'] = "─"
        else:
            signal_details_pine['SHA'] = "N/A"
    except Exception:
        signal_details_pine['SHA'] = "Err"

    # Ichimoku Signal
    try:
        ichimoku_signal_val = ichimoku_pine_signal(high, low, close)
        if ichimoku_signal_val == 1:
            bull_confluences += 1
            signal_details_pine['Ichi'] = "▲"
        elif ichimoku_signal_val == -1:
            bear_confluences += 1
            signal_details_pine['Ichi'] = "▼"
        else:
            signal_details_pine['Ichi'] = "─"
    except Exception:
        signal_details_pine['Ichi'] = "Err"
    
    confluence_value = max(bull_confluences, bear_confluences)
    direction = "NEUTRE"
    if bull_confluences > bear_confluences:
        direction = "HAUSSIER"
    elif bear_confluences > bull_confluences:
        direction = "BAISSIER"
    elif bull_confluences == bear_confluences and bull_confluences > 0:
        direction = "CONFLIT"
        
    return {
        'confluence_P': confluence_value, 
        'direction_P': direction,
        'bull_P': bull_confluences, 
        'bear_P': bear_confluences,
        'rsi_P': signal_details_pine.get('RSI_val', "N/A"),
        'adx_P': signal_details_pine.get('ADX_val', "N/A"),
        'signals_P': signal_details_pine
    }

def get_stars_pine(confluence_value):
    stars_map = {6: "⭐⭐⭐⭐⭐⭐", 5: "⭐⭐⭐⭐⭐", 4: "⭐⭐⭐⭐", 
                 3: "⭐⭐⭐", 2: "⭐⭐", 1: "⭐"}
    return stars_map.get(confluence_value, "WAIT")

# Interface utilisateur
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres")
    min_conf = st.selectbox("Confluence min (0-6)", 
                           options=[0, 1, 2, 3, 4, 5, 6], 
                           index=1, 
                           format_func=lambda x: f"{x} (confluence)")
    
    show_all = st.checkbox("Voir toutes les paires (ignorer filtre)")
    
    pair_to_debug = st.selectbox("🔍 Afficher données pour:", 
                                ["Aucune"] + PAIRS_OANDA, 
                                index=0)
    
    scan_btn = st.button("🔍 Scanner (Données OANDA H1)", 
                        type="primary", 
                        use_container_width=True)

with col2:
    if scan_btn:
        # Vérifier la configuration OANDA
        if 'oanda' not in st.secrets or 'API_KEY' not in st.secrets["oanda"]:
            st.error("⚠️ Configuration OANDA manquante! Ajoutez votre clé API dans les secrets Streamlit.")
            st.info("Dans Streamlit Cloud: Settings > Secrets > Edit secrets")
            st.code("""
[oanda]
API_KEY = "votre_api_key_oanda"
            """)
        else:
            st.info(f"🔄 Scan en cours (OANDA H1)...")
            pr_res = []
            pb = st.progress(0)
            stx = st.empty()
            
            # Debug data si demandé
            if pair_to_debug != "Aucune":
                st.subheader(f"Données OHLC pour {pair_to_debug} (OANDA):")
                debug_data = get_data_oanda(pair_to_debug, timeframe="H1", count=50)
                if debug_data is not None:
                    st.dataframe(debug_data[['Open', 'High', 'Low', 'Close']].tail(10))
                else:
                    st.warning(f"N'a pas pu charger données de débogage pour {pair_to_debug}.")
                st.divider()
            
            # Scanner toutes les paires
            for i, symbol_scan in enumerate(PAIRS_OANDA):
                pnd = symbol_scan.replace('_', '/')
                cp = (i + 1) / len(PAIRS_OANDA)
                pb.progress(cp)
                stx.text(f"Analyse (OANDA H1): {pnd} ({i+1}/{len(PAIRS_OANDA)})")
                
                d_h1_oanda = get_data_oanda(symbol_scan, timeframe="H1", count=200)
                
                if d_h1_oanda is not None:
                    sigs = calculate_all_signals_pine(d_h1_oanda)
                    if sigs:
                        strs = get_stars_pine(sigs['confluence_P'])
                        rd = {
                            'Paire': pnd, 
                            'Direction': sigs['direction_P'], 
                            'Conf. (0-6)': sigs['confluence_P'],
                            'Étoiles': strs, 
                            'RSI': sigs['rsi_P'], 
                            'ADX': sigs['adx_P'],
                            'Bull': sigs['bull_P'], 
                            'Bear': sigs['bear_P'], 
                            'details': sigs['signals_P']
                        }
                        pr_res.append(rd)
                    else:
                        pr_res.append({
                            'Paire': pnd, 
                            'Direction': 'ERREUR CALCUL', 
                            'Conf. (0-6)': 0, 
                            'Étoiles': 'N/A',
                            'RSI': 'N/A', 
                            'ADX': 'N/A', 
                            'Bull': 0, 
                            'Bear': 0,
                            'details': {'Info': 'Calcul signaux échoué'}
                        })
                else:
                    pr_res.append({
                        'Paire': pnd, 
                        'Direction': 'ERREUR DONNÉES', 
                        'Conf. (0-6)': 0, 
                        'Étoiles': 'N/A',
                        'RSI': 'N/A', 
                        'ADX': 'N/A', 
                        'Bull': 0, 
                        'Bear': 0,
                        'details': {'Info': 'Données OANDA non disponibles'}
                    })
                
                time.sleep(0.2)  # Rate limiting
            
            pb.empty()
            stx.empty()
            
            # Afficher les résultats
            if pr_res:
                dfa = pd.DataFrame(pr_res)
                dfd = dfa[dfa['Conf. (0-6)'] >= min_conf].copy() if not show_all else dfa.copy()
                
                if not show_all:
                    st.success(f"🎯 {len(dfd)} paire(s) avec {min_conf}+ confluence (OANDA).")
                else:
                    st.info(f"🔍 Affichage des {len(dfd)} paires (OANDA).")
                
                if not dfd.empty:
                    dfds = dfd.sort_values('Conf. (0-6)', ascending=False)
                    vcs = [c for c in ['Paire', 'Direction', 'Conf. (0-6)', 'Étoiles', 'RSI', 'ADX', 'Bull', 'Bear'] if c in dfds.columns]
                    st.dataframe(dfds[vcs], use_container_width=True, hide_index=True)
                    
                    with st.expander("📊 Détails des signaux (OANDA)"):
                        for _, r in dfds.iterrows():
                            sm = r.get('details', {})
                            if not isinstance(sm, dict): 
                                sm = {'Info': 'Détails non disponibles'}
                            
                            st.write(f"**{r.get('Paire', 'N/A')}** - {r.get('Étoiles', 'N/A')} ({r.get('Conf. (0-6)', 'N/A')}) - Dir: {r.get('Direction', 'N/A')}")
                            
                            dc = st.columns(6)
                            so = ['HMA', 'RSI', 'ADX', 'HA', 'SHA', 'Ichi']
                            for idx, sk in enumerate(so): 
                                dc[idx].metric(label=sk, value=sm.get(sk, "N/P"))
                            st.divider()
                else:
                    st.warning(f"❌ Aucune paire avec critères de filtrage (OANDA).")
            else:
                st.error("❌ Aucune paire traitée (OANDA). Vérifiez la configuration.")

# Documentation
with st.expander("ℹ️ Comment ça marche"):
    st.markdown("""
    **6 Signaux de Confluence analysés:**
    - **HMA(20)**: Hull Moving Average sur 20 périodes
    - **RSI(10)**: Relative Strength Index sur 10 périodes  
    - **ADX(14)**: Average Directional Index (>=20 pour trend fort)
    - **HA**: Heiken Ashi candlesticks
    - **SHA(10,10)**: Smoothed Heiken Ashi
    - **Ichi(9,26,52)**: Ichimoku Cloud
    
    **Source de données:** API OANDA (environnement practice)
    
    **Configuration requise:** Ajoutez vos clés OANDA dans les secrets Streamlit
    """)

with st.expander("🔧 Configuration OANDA"):
    st.markdown("""
    Pour utiliser cette application, vous devez:
    
    1. **Créer un compte OANDA** (gratuit 7 jours)
    2. **Obtenir vos clés API** depuis votre compte OANDA
    3. **Ajouter les secrets dans Streamlit Cloud:**
    
    ```toml
    [oanda]
    API_KEY = "votre_api_key_ici"
    ```
    
    4. **Redéployer l'application**
    """)

st.caption("Scanner Forex H1 avec données OANDA • Version corrigée pour Streamlit Cloud")
