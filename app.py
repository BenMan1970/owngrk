# --- IMPORTS NECESSAIRES ---
# Ligne 1: Test de print pour les logs
print("NOUVELLE VERSION DU SCRIPT (élimination doute oandapyV20) - DÉBUT DES IMPORTS")

# Ligne 3
import streamlit as st
# Ligne 4
import pandas as pd
# Ligne 5
import numpy as np
# Ligne 6
import requests # Pour les appels API à OANDA
# Ligne 7
import time

# Ligne 9: Configuration de la page Streamlit
st.set_page_config(page_title="Scanner Confluence Forex (OANDA Data)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données OANDA)")
st.markdown("*Utilisation de l'API OANDA pour les données de marché H1*")

# Ligne 14: Liste des paires Forex courantes
PAIRS_OANDA = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 'NZD_USD', 'USD_CHF',
    'EUR_GBP', 'EUR_JPY', 'GBP_JPY', 'AUD_JPY', 'CAD_JPY', 'CHF_JPY', 'NZD_JPY',
    'EUR_AUD', 'GBP_AUD', 'EUR_CAD', 'GBP_CAD', 'EUR_CHF', 'GBP_CHF',
    'AUD_CAD', 'AUD_CHF', 'AUD_NZD', 'CAD_CHF', 'NZD_CAD', 'NZD_CHF'
]

# --- Fonctions d'indicateurs (style Pine Script) ---
def ema(s, p):
    if s is None or s.empty or p <= 0: # Ajout de p <= 0
        return pd.Series(dtype=float, index=s.index if s is not None else None)
    return s.ewm(span=p, adjust=False).mean()

def rma(s, p):
    if s is None or s.empty or p <= 0: # Ajout de p <= 0
        return pd.Series(dtype=float, index=s.index if s is not None else None)
    return s.ewm(alpha=1/p, adjust=False).mean()

def hull_ma_pine(dc, p=20):
    try:
        if dc is None or len(dc) < p or p <= 0: # Ajout de p <= 0
            return pd.Series([np.nan] * (len(dc) if dc is not None else 0), index=dc.index if dc is not None else None)

        hl = max(1, int(p/2))
        sl = max(1, int(np.sqrt(p)))

        def wma(series, window):
            if series is None or len(series) < window or window <= 0: # Ajout de window <= 0
                return pd.Series([np.nan] * (len(series) if series is not None else 0), index=series.index if series is not None else None)
            weights = np.arange(1, window + 1)
            
            # Tentative avec raw=False pour plus de robustesse si raw=True cause des soucis
            # return series.rolling(window=window).apply(
            #     lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum() if len(x) == window else np.nan,
            #     raw=False 
            # )
            # Version originale avec raw=True, plus rapide si elle fonctionne
            return series.rolling(window=window).apply(
                lambda x: np.sum(x * weights[:len(x)]) / np.sum(weights[:len(x)]) if len(x) == window else np.nan,
                raw=True
            )

        wma1 = wma(dc, hl)
        wma2 = wma(dc, p)

        if wma1 is None or wma2 is None or wma1.isna().all() or wma2.isna().all():
            return pd.Series([np.nan] * len(dc), index=dc.index)

        diff = 2 * wma1 - wma2
        if diff.isna().all(): # Si diff est tout NaN
            return pd.Series([np.nan] * len(dc), index=dc.index)
            
        return wma(diff, sl)
    except Exception as e:
        # st.warning(f"Erreur dans hull_ma_pine: {e}") # Décommentez pour débogage si nécessaire
        return pd.Series([np.nan] * (len(dc) if dc is not None else 0), index=dc.index if dc is not None else None)


def rsi_pine(po4, p=10):
    try:
        if po4 is None or len(po4) < p or p <= 0: # Ajout de p <= 0
             return pd.Series([50] * (len(po4) if po4 is not None else 0), index=po4.index if po4 is not None else None)
        d = po4.diff()
        if d.empty or d.isna().all():
            return pd.Series([50] * len(po4), index=po4.index)
        g = d.where(d > 0, 0.0)
        l = -d.where(d < 0, 0.0)
        ag = rma(g, p)
        al = rma(l, p)

        if ag is None or al is None or ag.empty or al.empty:
            return pd.Series([50] * len(po4), index=po4.index)

        rs = ag / al.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except Exception as e:
        # st.warning(f"Erreur dans rsi_pine: {e}")
        return pd.Series([50] * (len(po4) if po4 is not None else 0), index=po4.index if po4 is not None else None)

def adx_pine(h, l, c, p=14):
    try:
        if h is None or l is None or c is None or p <=0 or not all(s is not None and len(s) >= p for s in [h, l, c]): # Ajout p<=0
            return pd.Series([0] * (len(h) if h is not None else 0), index=h.index if h is not None else None)

        tr1 = h - l
        tr2 = abs(h - c.shift(1))
        tr3 = abs(l - c.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0) # fillna(0) sur TR
        atr = rma(tr, p)

        um = h.diff().fillna(0) # fillna(0)
        dm_calc = (l.shift(1) - l).fillna(0) # fillna(0)

        pdm = pd.Series(np.where((um > dm_calc) & (um > 0), um, 0.0), index=h.index)
        mdm = pd.Series(np.where((dm_calc > um) & (dm_calc > 0), dm_calc, 0.0), index=h.index)

        # Gérer les cas où rma retourne None ou vide
        rma_pdm = rma(pdm, p)
        rma_mdm = rma(mdm, p)
        if atr is None or rma_pdm is None or rma_mdm is None or atr.empty or rma_pdm.empty or rma_mdm.empty:
            return pd.Series([0] * len(h), index=h.index)

        satr = atr.replace(0, 1e-9)
        pdi = 100 * (rma_pdm / satr)
        mdi = 100 * (rma_mdm / satr)

        dxden = (pdi + mdi).replace(0, 1e-9)
        dx = 100 * (abs(pdi - mdi) / dxden)
        
        adx_result = rma(dx, p)
        if adx_result is None or adx_result.empty:
            return pd.Series([0] * len(h), index=h.index)
            
        return adx_result.fillna(0)
    except Exception as e:
        # st.warning(f"Erreur dans adx_pine: {e}")
        return pd.Series([0] * (len(h) if h is not None else 0), index=h.index if h is not None else None)

def heiken_ashi_pine(dfo):
    try:
        if dfo is None or dfo.empty or not all(col in dfo.columns for col in ['Open', 'High', 'Low', 'Close']):
            empty_len = len(dfo) if dfo is not None else 0
            empty_idx = dfo.index if dfo is not None else None
            return pd.Series(dtype=float, index=empty_idx), pd.Series(dtype=float, index=empty_idx)

        ha = pd.DataFrame(index=dfo.index)
        ha['HA_Close'] = (dfo['Open'] + dfo['High'] + dfo['Low'] + dfo['Close']) / 4
        ha['HA_Open'] = np.nan

        if not dfo.empty:
            ha.iloc[0, ha.columns.get_loc('HA_Open')] = (dfo['Open'].iloc[0] + dfo['Close'].iloc[0]) / 2
            for i in range(1, len(dfo)):
                ha_open_prev = ha.iloc[i-1, ha.columns.get_loc('HA_Open')]
                ha_close_prev = ha.iloc[i-1, ha.columns.get_loc('HA_Close')]
                if pd.isna(ha_open_prev) or pd.isna(ha_close_prev): # Si une valeur précédente est NaN
                    ha.iloc[i, ha.columns.get_loc('HA_Open')] = (dfo['Open'].iloc[i] + dfo['Close'].iloc[i]) / 2 # Fallback
                else:
                    ha.iloc[i, ha.columns.get_loc('HA_Open')] = (ha_open_prev + ha_close_prev) / 2
        else:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        return ha['HA_Open'], ha['HA_Close']
    except Exception as e:
        # st.warning(f"Erreur dans heiken_ashi_pine: {e}")
        empty_len = len(dfo) if dfo is not None else 0
        empty_idx = dfo.index if dfo is not None else None
        return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)


def smoothed_heiken_ashi_pine(dfo, l1=10, l2=10):
    try:
        if dfo is None or dfo.empty or not all(col in dfo.columns for col in ['Open', 'High', 'Low', 'Close']) or l1<=0 or l2<=0:
            empty_len = len(dfo) if dfo is not None else 0
            empty_idx = dfo.index if dfo is not None else None
            return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)

        eo = ema(dfo['Open'], l1)
        eh = ema(dfo['High'], l1)
        el = ema(dfo['Low'], l1)
        ec = ema(dfo['Close'], l1)

        if any(s is None or s.empty or s.isna().all() for s in [eo, eh, el, ec]):
            empty_len = len(dfo)
            empty_idx = dfo.index
            return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)

        hai = pd.DataFrame({'Open': eo, 'High': eh, 'Low': el, 'Close': ec}, index=dfo.index)
        if not all(col in hai.columns for col in ['Open', 'High', 'Low', 'Close']):
             empty_len = len(dfo)
             empty_idx = dfo.index
             return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)

        hao_i, hac_i = heiken_ashi_pine(hai)
        if hao_i is None or hac_i is None or hao_i.empty or hac_i.empty:
            empty_len = len(dfo)
            empty_idx = dfo.index
            return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)

        sho = ema(hao_i, l2)
        shc = ema(hac_i, l2)
        
        if sho is None or shc is None: # Vérification finale
            empty_len = len(dfo)
            empty_idx = dfo.index
            return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)

        return sho, shc
    except Exception as e:
        # st.warning(f"Erreur dans smoothed_heiken_ashi_pine: {e}")
        empty_len = len(dfo) if dfo is not None else 0
        empty_idx = dfo.index if dfo is not None else None
        return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)


def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    try:
        min_len_req = max(tenkan_p, kijun_p, senkou_b_p)
        if df_high is None or df_low is None or df_close is None or \
           not all(s is not None and len(s) >= min_len_req for s in [df_high, df_low, df_close]) or \
           any(p <= 0 for p in [tenkan_p, kijun_p, senkou_b_p]): # Ajout vérif p > 0
            return 0

        ts = (df_high.rolling(window=tenkan_p).max() + df_low.rolling(window=tenkan_p).min()) / 2
        ks = (df_high.rolling(window=kijun_p).max() + df_low.rolling(window=kijun_p).min()) / 2
        sa = (ts + ks) / 2
        sb = (df_high.rolling(window=senkou_b_p).max() + df_low.rolling(window=senkou_b_p).min()) / 2

        if df_close.empty or sa.empty or sb.empty or \
           pd.isna(df_close.iloc[-1]) or pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1]):
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
    except Exception as e:
        # st.warning(f"Erreur dans ichimoku_pine_signal: {e}")
        return 0

# --- Récupération des données OANDA ---
@st.cache_data(ttl=300)
def get_data_oanda(symbol: str, timeframe: str = 'H1', count: int = 200):
    try:
        if 'oanda' not in st.secrets or 'API_KEY' not in st.secrets.get("oanda", {}):
            st.error("Configuration OANDA manquante dans les secrets Streamlit.")
            st.code("""[oanda]\nAPI_KEY = "VOTRE_CLE_API_OANDA_ICI" """, language="toml")
            return None

        api_key = st.secrets["oanda"]["API_KEY"]
        base_url = "https://api-fxpractice.oanda.com"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Demander plus de données pour compenser les périodes de calcul des indicateurs
        # et les bougies non complètes. 60 est une marge de sécurité.
        params = {
            "count": count + 60, 
            "granularity": timeframe,
            "price": "M"
        }
        url = f"{base_url}/v3/instruments/{symbol}/candles"
        
        # print(f"OANDA Request: URL={url}, Params={params}") # Log de la requête
        response = requests.get(url, headers=headers, params=params, timeout=30)
        # print(f"OANDA Response for {symbol}: Status={response.status_code}") # Log de la réponse

        if response.status_code != 200:
            st.error(f"Erreur API OANDA ({response.status_code}) pour {symbol}: {response.text}")
            return None

        data = response.json()
        candles = data.get("candles", [])
        if not candles:
            st.warning(f"Aucune donnée de bougie reçue de OANDA pour {symbol}.")
            return None

        ohlc_data = []
        timestamps = []
        for candle in candles:
            if candle.get("complete", False): # S'assurer que la bougie est complète
                mid_prices = candle.get("mid")
                if mid_prices: # Vérifier que 'mid' existe
                    ohlc_data.append({
                        'Open': float(mid_prices['o']),
                        'High': float(mid_prices['h']),
                        'Low': float(mid_prices['l']),
                        'Close': float(mid_prices['c'])
                    })
                    timestamps.append(pd.to_datetime(candle['time']))

        if not ohlc_data:
            st.warning(f"Aucune bougie complète ('complete:true' avec section 'mid') trouvée pour {symbol}.")
            return None

        df = pd.DataFrame(ohlc_data, index=pd.DatetimeIndex(timestamps))

        # S'assurer qu'on a assez de données APRÈS avoir filtré les bougies complètes
        # et avant de prendre les `count` dernières.
        # 60 est un seuil absolu pour les calculs d'indicateurs.
        if len(df) < 60: 
            st.warning(f"Données insuffisantes pour {symbol} après filtrage ({len(df)} bougies). Min 60 requis pour calculs.")
            return None
        
        # Retourner les `count` dernières bougies, ou moins si pas assez mais plus de 60.
        return df.iloc[-count:] if len(df) >= count else df 

    except requests.exceptions.Timeout:
        st.error(f"Timeout lors de la récupération des données OANDA pour {symbol}.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion OANDA pour {symbol}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue dans get_data_oanda pour {symbol}: {str(e)}")
        # import traceback
        # st.error(traceback.format_exc()) # Pour plus de détails sur l'erreur
        return None

# --- Calcul des signaux de confluence ---
def calculate_all_signals_pine(data):
    if data is None or data.empty or len(data) < 60: # Seuil de 60 bougies pour les calculs
        return None

    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        st.warning("Colonnes OHLC manquantes pour calcul des signaux.")
        return None

    # S'assurer qu'il n'y a pas que des NaN, ce qui peut arriver si les indicateurs précédents échouent
    if data[required_cols].isna().all().all():
        st.warning("Données OHLC entièrement NaN pour calcul des signaux.")
        return None

    close = data['Close']
    high = data['High']
    low = data['Low']
    open_price = data['Open']
    ohlc4 = (open_price + high + low + close) / 4

    bull_confluences = 0
    bear_confluences = 0
    signal_details_pine = {}

    # HMA
    hma_series = hull_ma_pine(close, 20)
    if hma_series is not None and not hma_series.empty and len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
        if hma_series.iloc[-1] > hma_series.iloc[-2]: bull_confluences += 1; signal_details_pine['HMA'] = "▲"
        elif hma_series.iloc[-1] < hma_series.iloc[-2]: bear_confluences += 1; signal_details_pine['HMA'] = "▼"
        else: signal_details_pine['HMA'] = "─"
    else: signal_details_pine['HMA'] = "N/A"

    # RSI
    rsi_series = rsi_pine(ohlc4, 10)
    if rsi_series is not None and not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]):
        rsi_val = rsi_series.iloc[-1]
        signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
        if rsi_val > 50: bull_confluences += 1; signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
        elif rsi_val < 50: bear_confluences += 1; signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
        else: signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
    else: signal_details_pine['RSI'] = "N/A"; signal_details_pine['RSI_val'] = "N/A"

    # ADX
    adx_series = adx_pine(high, low, close, 14)
    if adx_series is not None and not adx_series.empty and not pd.isna(adx_series.iloc[-1]):
        adx_val = adx_series.iloc[-1]
        signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
        if adx_val >= 20: bull_confluences += 1; bear_confluences += 1; signal_details_pine['ADX'] = f"✔({adx_val:.0f})"
        else: signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
    else: signal_details_pine['ADX'] = "N/A"; signal_details_pine['ADX_val'] = "N/A"
    
    # Heiken Ashi
    ha_open, ha_close = heiken_ashi_pine(data.copy()) # .copy() pour éviter SettingWithCopyWarning potentiel
    if ha_open is not None and ha_close is not None and not ha_open.empty and not ha_close.empty and \
       not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
        if ha_close.iloc[-1] > ha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['HA'] = "▲"
        elif ha_close.iloc[-1] < ha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['HA'] = "▼"
        else: signal_details_pine['HA'] = "─"
    else: signal_details_pine['HA'] = "N/A"

    # Smoothed Heiken Ashi
    sha_open, sha_close = smoothed_heiken_ashi_pine(data.copy(), 10, 10) # .copy()
    if sha_open is not None and sha_close is not None and not sha_open.empty and not sha_close.empty and \
       not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
        if sha_close.iloc[-1] > sha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['SHA'] = "▲"
        elif sha_close.iloc[-1] < sha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['SHA'] = "▼"
        else: signal_details_pine['SHA'] = "─"
    else: signal_details_pine['SHA'] = "N/A"

    # Ichimoku
    ichimoku_signal_val = ichimoku_pine_signal(high, low, close, 9, 26, 52)
    if ichimoku_signal_val == 1: bull_confluences += 1; signal_details_pine['Ichi'] = "▲"
    elif ichimoku_signal_val == -1: bear_confluences += 1; signal_details_pine['Ichi'] = "▼"
    else: signal_details_pine['Ichi'] = "─"

    confluence_value = 0
    direction = "NEUTRE"
    if bull_confluences > bear_confluences: direction = "HAUSSIER"; confluence_value = bull_confluences
    elif bear_confluences > bull_confluences: direction = "BAISSIER"; confluence_value = bear_confluences
    elif bull_confluences == bear_confluences and bull_confluences > 0: direction = "CONFLIT"; confluence_value = bull_confluences

    return {
        'confluence_P': confluence_value, 'direction_P': direction,
        'bull_P': bull_confluences, 'bear_P': bear_confluences,
        'rsi_P': signal_details_pine.get('RSI_val', "N/A"),
        'adx_P': signal_details_pine.get('ADX_val', "N/A"),
        'signals_P': signal_details_pine
    }

# --- Fonction utilitaire pour les étoiles ---
def get_stars_pine(confluence_value):
    try:
        cv_int = int(confluence_value)
        stars_map = {6: "⭐⭐⭐⭐⭐⭐", 5: "⭐⭐⭐⭐⭐", 4: "⭐⭐⭐⭐", 3: "⭐⭐⭐", 2: "⭐⭐", 1: "⭐"}
        return stars_map.get(cv_int, "WAIT" if cv_int == 0 else "") # "" si non trouvé et pas 0
    except ValueError:
        return "" # En cas d'erreur de conversion

# --- Interface utilisateur Streamlit ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres du Scan")
    min_conf = st.selectbox("Confluence minimale (0-6):", options=list(range(7)), index=3, format_func=lambda x: f"{x} étoile(s)")
    show_all = st.checkbox("Voir toutes les paires (ignorer filtre)")
    pair_to_debug = st.selectbox("🔍 Afficher données OHLC pour:", ["Aucune"] + PAIRS_OANDA, index=0)
    scan_btn = st.button("🚀 Lancer le Scanner (OANDA H1)", type="primary", use_container_width=True)

with col2:
    if scan_btn:
        if 'oanda' not in st.secrets or 'API_KEY' not in st.secrets.get("oanda", {}):
            st.error("⚠️ **Configuration OANDA Manquante!** Vérifiez vos secrets Streamlit.")
            st.code("""[oanda]\nAPI_KEY = "VOTRE_CLE_API_OANDA_ICI" """, language="toml")
        else:
            st.info("🔄 Scan OANDA H1 en cours... Un peu de patience.")
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_list = []

            if pair_to_debug != "Aucune":
                st.subheader(f"📊 Données OHLC récentes pour {pair_to_debug} (OANDA):")
                debug_data = get_data_oanda(pair_to_debug, timeframe="H1", count=50)
                if debug_data is not None and not debug_data.empty:
                    st.dataframe(debug_data[['Open', 'High', 'Low', 'Close']].tail(10), use_container_width=True)
                else:
                    st.warning(f"Impossible de charger les données de débogage pour {pair_to_debug}.")
                st.divider()

            for i, symbol_oanda in enumerate(PAIRS_OANDA):
                pair_display_name = symbol_oanda.replace('_', '/')
                progress_bar.progress((i + 1) / len(PAIRS_OANDA))
                status_text.text(f"Analyse: {pair_display_name} ({i+1}/{len(PAIRS_OANDA)})...")

                data_h1_oanda = get_data_oanda(symbol_oanda, timeframe="H1", count=200)

                if data_h1_oanda is not None and not data_h1_oanda.empty:
                    signals_data = calculate_all_signals_pine(data_h1_oanda)
                    if signals_data:
                        results_list.append({
                            'Paire': pair_display_name, 'Direction': signals_data['direction_P'],
                            'Conf. (0-6)': signals_data['confluence_P'],
                            'Étoiles': get_stars_pine(signals_data['confluence_P']),
                            'RSI': signals_data.get('rsi_P', "N/A"), 'ADX': signals_data.get('adx_P', "N/A"),
                            'Bull Signals': signals_data['bull_P'], 'Bear Signals': signals_data['bear_P'],
                            'Détails Signaux': signals_data['signals_P']
                        })
                    else: # Erreur de calcul des signaux
                        results_list.append({'Paire': pair_display_name, 'Direction': 'CALC ERR', 'Conf. (0-6)': 0, 'Étoiles': 'N/A', 'RSI': 'N/A', 'ADX': 'N/A', 'Bull Signals': 0, 'Bear Signals': 0, 'Détails Signaux': {'Info': 'Calcul signaux a échoué'}})
                else: # Erreur de récupération des données
                    results_list.append({'Paire': pair_display_name, 'Direction': 'DATA ERR', 'Conf. (0-6)': 0, 'Étoiles': 'N/A', 'RSI': 'N/A', 'ADX': 'N/A', 'Bull Signals': 0, 'Bear Signals': 0, 'Détails Signaux': {'Info': 'Données OANDA non disponibles ou insuffisantes'}})
                
                time.sleep(0.30) # Augmenté légèrement le délai

            progress_bar.empty()
            status_text.success("✅ Scan OANDA H1 terminé!")

            if results_list:
                results_df = pd.DataFrame(results_list)
                if show_all:
                    filtered_df = results_df.copy()
                    st.info(f"🔍 Affichage des {len(filtered_df)} paires scannées (OANDA H1).")
                else:
                    filtered_df = results_df[results_df['Conf. (0-6)'] >= min_conf].copy()
                    st.success(f"🎯 {len(filtered_df)} paire(s) trouvée(s) avec au moins {min_conf} étoile(s) (OANDA H1).")

                if not filtered_df.empty:
                    sorted_df = filtered_df.sort_values(by=['Conf. (0-6)', 'Paire'], ascending=[False, True])
                    display_columns = ['Paire', 'Direction', 'Conf. (0-6)', 'Étoiles', 'RSI', 'ADX', 'Bull Signals', 'Bear Signals']
                    st.dataframe(sorted_df[display_columns], use_container_width=True, hide_index=True,
                                 column_config={"Conf. (0-6)": st.column_config.NumberColumn(format="%d")})
                    
                    with st.expander("📊 Détails des signaux pour les paires filtrées (OANDA H1)"):
                        for _, row in sorted_df.iterrows():
                            signal_details = row.get('Détails Signaux', {})
                            if not isinstance(signal_details, dict): signal_details = {'Info': 'Détails non disponibles'}
                            
                            st.write(f"**{row.get('Paire', 'N/A')}** | {row.get('Étoiles', '')} ({row.get('Conf. (0-6)', 'N/A')}) | Dir: {row.get('Direction', 'N/A')}")
                            
                            signal_order = ['HMA', 'RSI', 'ADX', 'HA', 'SHA', 'Ichi']
                            # Compter combien de colonnes on a besoin (sans les _val)
                            num_metrics = sum(1 for k in signal_order if k in signal_details)
                            if num_metrics == 0 and 'Info' in signal_details : num_metrics = 1 # Pour afficher "Info"
                            elif num_metrics == 0: num_metrics = 1 # Fallback

                            detail_cols = st.columns(num_metrics if num_metrics > 0 else 1)
                            
                            col_idx = 0
                            for key in signal_order:
                                if key in signal_details and col_idx < len(detail_cols):
                                    detail_cols[col_idx].metric(label=key, value=str(signal_details[key]))
                                    col_idx += 1
                            
                            if col_idx == 0 and 'Info' in signal_details and len(detail_cols) > 0: # Si aucun signal standard mais il y a Info
                                detail_cols[0].metric(label="Info", value=str(signal_details['Info']))
                            st.divider()
                else:
                    st.warning(f"❌ Aucune paire ne correspond à vos critères de filtrage (min {min_conf} étoile(s)) sur OANDA H1.")
            else:
                st.error("❌ Aucune paire n'a pu être traitée (OANDA H1). Vérifiez la console pour les erreurs ou la configuration de l'API.")

# --- Sections d'information en bas de page ---
with st.expander("ℹ️ Comment Fonctionne ce Scanner de Confluence?"):
    st.markdown("""
    Ce scanner analyse plusieurs indicateurs techniques sur l'horizon de temps **H1** pour identifier des signaux de trading potentiels basés sur la confluence.
    **Les 6 Indicateurs Techniques Utilisés :** HMA(20), RSI(10, ohlc4), ADX(14), Heiken Ashi, Smoothed Heiken Ashi(10,10), Ichimoku Kinko Hyo(9,26,52).
    **Source des Données :** API OANDA (compte de pratique par défaut).
    **Calcul de la Confluence :** Le nombre d'étoiles correspond au nombre de signaux dans la direction identifiée. ADX >= 20 confirme la force (compte pour bull ET bear).
    """)

with st.expander("🔧 Configuration de l'API OANDA et Secrets Streamlit"):
    st.markdown("""
    1.  **Obtenez une clé API OANDA** (compte de démonstration/pratique ou réel).
    2.  **Ajoutez la clé API aux Secrets Streamlit :** Dans les paramètres de votre application Streamlit Cloud (`Settings` > `Secrets`), ajoutez:
        ```toml
        [oanda]
        API_KEY = "VOTRE_CLE_API_OANDA_ICI"
        ```
    3.  L'application utilise par défaut l'URL de l'API de **pratique** d'OANDA.
    """)

st.caption("Scanner Forex H1 avec données OANDA • Version révisée pour Streamlit Cloud")

# Ligne de fin pour les logs
print("NOUVELLE VERSION DU SCRIPT (élimination doute oandapyV20) - FIN DU SCRIPT")
