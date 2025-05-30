import streamlit as st
import pandas as pd
import numpy as np
import requests # Pour les appels API à OANDA
import time

# Configuration de la page Streamlit
st.set_page_config(page_title="Scanner Confluence Forex (OANDA Data)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données OANDA)")
st.markdown("*Utilisation de l'API OANDA pour les données de marché H1*")

# Liste des paires Forex courantes
PAIRS_OANDA = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 'NZD_USD', 'USD_CHF',
    'EUR_GBP', 'EUR_JPY', 'GBP_JPY', 'AUD_JPY', 'CAD_JPY', 'CHF_JPY', 'NZD_JPY',
    'EUR_AUD', 'GBP_AUD', 'EUR_CAD', 'GBP_CAD', 'EUR_CHF', 'GBP_CHF',
    'AUD_CAD', 'AUD_CHF', 'AUD_NZD', 'CAD_CHF', 'NZD_CAD', 'NZD_CHF'
]

# --- Fonctions d'indicateurs (style Pine Script) ---
def ema(s, p):
    if s is None or s.empty:
        return pd.Series(dtype=float)
    return s.ewm(span=p, adjust=False).mean()

def rma(s, p):
    if s is None or s.empty:
        return pd.Series(dtype=float)
    return s.ewm(alpha=1/p, adjust=False).mean()

def hull_ma_pine(dc, p=20):
    try:
        if dc is None or len(dc) < p:
            return pd.Series([np.nan] * (len(dc) if dc is not None else 0), index=dc.index if dc is not None else None)

        hl = max(1, int(p/2))
        sl = max(1, int(np.sqrt(p)))

        def wma(series, window):
            if series is None or len(series) < window:
                return pd.Series([np.nan] * (len(series) if series is not None else 0), index=series.index if series is not None else None)
            weights = np.arange(1, window + 1)
            # Gérer le cas où raw=True ne peut pas prendre de lambda avec une longueur variable de x
            # Utiliser une approche plus simple pour wma si raw=True pose problème avec la version de pandas
            # ou désactiver raw=True si la performance n'est pas critique pour de petites fenêtres.
            # Pour la robustesse, on peut l'implémenter sans raw=True et tester.
            # Pour le moment, gardons raw=True car c'est généralement plus rapide pour les apply numériques purs.
            # Si apply ne fonctionne pas bien avec raw=True et des poids variables, il faudra ajuster.
            
            # Si series.rolling(...).apply(...) cause des soucis avec la lambda et raw=True
            # on peut faire une boucle, mais c'est moins "pandorable".
            # Alternative simple pour wma (peut être plus lente)
            # return series.rolling(window).apply(lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum() if len(x)==window else np.nan, raw=False)

            # Version actuelle:
            return series.rolling(window=window).apply(
                lambda x: np.sum(x * weights[:len(x)]) / np.sum(weights[:len(x)]) if len(x) == window else np.nan,
                raw=True # raw=True peut parfois causer des problèmes avec des types de données mixtes ou des lambdas complexes
            )

        wma1 = wma(dc, hl)
        wma2 = wma(dc, p)

        if wma1 is None or wma2 is None or wma1.isna().all() or wma2.isna().all():
            return pd.Series([np.nan] * len(dc), index=dc.index)

        diff = 2 * wma1 - wma2
        return wma(diff, sl)
    except Exception:
        return pd.Series([np.nan] * (len(dc) if dc is not None else 0), index=dc.index if dc is not None else None)


def rsi_pine(po4, p=10):
    try:
        if po4 is None or len(po4) < p:
             return pd.Series([50] * (len(po4) if po4 is not None else 0), index=po4.index if po4 is not None else None)
        d = po4.diff()
        if d.empty:
            return pd.Series([50] * len(po4), index=po4.index)
        g = d.where(d > 0, 0.0)
        l = -d.where(d < 0, 0.0)
        ag = rma(g, p)
        al = rma(l, p)
        rs = ag / al.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except Exception:
        return pd.Series([50] * (len(po4) if po4 is not None else 0), index=po4.index if po4 is not None else None)

def adx_pine(h, l, c, p=14):
    try:
        if h is None or l is None or c is None or not all(s is not None and len(s) >= p for s in [h, l, c]):
            return pd.Series([0] * (len(h) if h is not None else 0), index=h.index if h is not None else None)

        tr1 = h - l
        tr2 = abs(h - c.shift(1))
        tr3 = abs(l - c.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = rma(tr, p)

        um = h.diff()
        dm_calc = l.shift(1) - l # Pine: dm = low[1] - low if low < low[1] and high - high[1] < low[1] - low else 0

        pdm = pd.Series(np.where((um > dm_calc) & (um > 0), um, 0.0), index=h.index)
        mdm = pd.Series(np.where((dm_calc > um) & (dm_calc > 0), dm_calc, 0.0), index=h.index)

        satr = atr.replace(0, 1e-9)
        pdi = 100 * (rma(pdm, p) / satr)
        mdi = 100 * (rma(mdm, p) / satr)

        dxden = (pdi + mdi).replace(0, 1e-9)
        dx = 100 * (abs(pdi - mdi) / dxden)

        return rma(dx, p).fillna(0)
    except Exception:
        return pd.Series([0] * (len(h) if h is not None else 0), index=h.index if h is not None else None)

def heiken_ashi_pine(dfo):
    try:
        if dfo is None or dfo.empty or not all(col in dfo.columns for col in ['Open', 'Close']):
            return pd.Series(dtype=float), pd.Series(dtype=float)

        ha = pd.DataFrame(index=dfo.index)
        ha['HA_Close'] = (dfo['Open'] + dfo['High'] + dfo['Low'] + dfo['Close']) / 4
        ha['HA_Open'] = np.nan

        if not dfo.empty: # Check again to be safe before iloc
            ha.iloc[0, ha.columns.get_loc('HA_Open')] = (dfo['Open'].iloc[0] + dfo['Close'].iloc[0]) / 2
            for i in range(1, len(dfo)):
                ha.iloc[i, ha.columns.get_loc('HA_Open')] = \
                    (ha.iloc[i-1, ha.columns.get_loc('HA_Open')] + ha.iloc[i-1, ha.columns.get_loc('HA_Close')]) / 2
        else: # Should not happen if first check passed
            return pd.Series(dtype=float), pd.Series(dtype=float)

        return ha['HA_Open'], ha['HA_Close']
    except Exception:
        empty_len = len(dfo) if dfo is not None else 0
        empty_idx = dfo.index if dfo is not None else None
        return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)


def smoothed_heiken_ashi_pine(dfo, l1=10, l2=10):
    try:
        if dfo is None or dfo.empty or not all(col in dfo.columns for col in ['Open', 'High', 'Low', 'Close']):
            empty_len = len(dfo) if dfo is not None else 0
            empty_idx = dfo.index if dfo is not None else None
            return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)

        eo = ema(dfo['Open'], l1)
        eh = ema(dfo['High'], l1)
        el = ema(dfo['Low'], l1)
        ec = ema(dfo['Close'], l1)

        # Vérifier si les EMA sont vides ou entièrement NaN
        if any(s.empty or s.isna().all() for s in [eo, eh, el, ec]):
            empty_len = len(dfo)
            empty_idx = dfo.index
            return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)


        hai = pd.DataFrame({'Open': eo, 'High': eh, 'Low': el, 'Close': ec}, index=dfo.index)
        # Assurer que les colonnes pour heiken_ashi_pine sont présentes dans hai
        if not all(col in hai.columns for col in ['Open', 'High', 'Low', 'Close']): # Devrait être le cas
             empty_len = len(dfo)
             empty_idx = dfo.index
             return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)

        hao_i, hac_i = heiken_ashi_pine(hai)
        if hao_i.empty or hac_i.empty:
            empty_len = len(dfo)
            empty_idx = dfo.index
            return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)

        sho = ema(hao_i, l2)
        shc = ema(hac_i, l2)

        return sho, shc
    except Exception:
        empty_len = len(dfo) if dfo is not None else 0
        empty_idx = dfo.index if dfo is not None else None
        return pd.Series([np.nan] * empty_len, index=empty_idx), pd.Series([np.nan] * empty_len, index=empty_idx)


def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    try:
        min_len_req = max(tenkan_p, kijun_p, senkou_b_p)
        if df_high is None or df_low is None or df_close is None or \
           not all(s is not None and len(s) >= min_len_req for s in [df_high, df_low, df_close]):
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
    except Exception:
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
        base_url = "https://api-fxpractice.oanda.com" # Environnement de pratique

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        params = {
            "count": count + 60, # Demander un peu plus pour les calculs et s'assurer d'avoir 'count' bougies complètes
            "granularity": timeframe,
            "price": "M" # Midpoint prices
        }
        url = f"{base_url}/v3/instruments/{symbol}/candles"
        response = requests.get(url, headers=headers, params=params, timeout=30)

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
            if candle.get("complete", False):
                mid_prices = candle["mid"]
                ohlc_data.append({
                    'Open': float(mid_prices['o']),
                    'High': float(mid_prices['h']),
                    'Low': float(mid_prices['l']),
                    'Close': float(mid_prices['c'])
                })
                timestamps.append(pd.to_datetime(candle['time']))

        if not ohlc_data:
            st.warning(f"Aucune bougie complète trouvée pour {symbol}.")
            return None

        df = pd.DataFrame(ohlc_data, index=pd.DatetimeIndex(timestamps))

        if len(df) < count: # Vérifier si on a au moins le nombre de bougies demandé
            st.warning(f"Données insuffisantes pour {symbol} ({len(df)} bougies reçues). Minimum {count} requis après filtrage.")
            # Si on a quand même assez pour les calculs (ex: 60), on pourrait continuer, sinon retourner None
            if len(df) < 60 : # 60 est un seuil de sécurité pour la plupart des indicateurs ici
                return None
        
        return df.iloc[-count:] # Retourner les 'count' dernières bougies complètes

    except requests.exceptions.Timeout:
        st.error(f"Timeout lors de la récupération des données OANDA pour {symbol}.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion OANDA pour {symbol}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors de la récupération des données OANDA pour {symbol}: {str(e)}")
        return None

# --- Calcul des signaux de confluence ---
def calculate_all_signals_pine(data):
    if data is None or len(data) < 60: # Minimum de données pour la plupart des indicateurs
        return None

    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        st.warning("Colonnes OHLC manquantes dans les données pour le calcul des signaux.")
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
        if hma_series is not None and len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
            if hma_series.iloc[-1] > hma_series.iloc[-2]:
                bull_confluences += 1; signal_details_pine['HMA'] = "▲"
            elif hma_series.iloc[-1] < hma_series.iloc[-2]:
                bear_confluences += 1; signal_details_pine['HMA'] = "▼"
            else: signal_details_pine['HMA'] = "─"
        else: signal_details_pine['HMA'] = "N/A"
    except Exception: signal_details_pine['HMA'] = "Err"

    # RSI Signal
    try:
        rsi_series = rsi_pine(ohlc4, 10)
        if rsi_series is not None and len(rsi_series) >= 1 and not pd.isna(rsi_series.iloc[-1]):
            rsi_val = rsi_series.iloc[-1]
            signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
            if rsi_val > 50:
                bull_confluences += 1; signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
            elif rsi_val < 50:
                bear_confluences += 1; signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
            else: signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
        else:
            signal_details_pine['RSI'] = "N/A"; signal_details_pine['RSI_val'] = "N/A"
    except Exception:
        signal_details_pine['RSI'] = "Err"; signal_details_pine['RSI_val'] = "Err"

    # ADX Signal
    try:
        adx_series = adx_pine(high, low, close, 14)
        if adx_series is not None and len(adx_series) >= 1 and not pd.isna(adx_series.iloc[-1]):
            adx_val = adx_series.iloc[-1]
            signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
            if adx_val >= 20: # Tendance forte, compte pour les deux sens pour la confluence
                bull_confluences += 1; bear_confluences += 1
                signal_details_pine['ADX'] = f"✔({adx_val:.0f})"
            else: signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
        else:
            signal_details_pine['ADX'] = "N/A"; signal_details_pine['ADX_val'] = "N/A"
    except Exception:
        signal_details_pine['ADX'] = "Err"; signal_details_pine['ADX_val'] = "Err"

    # Heiken Ashi Signal
    try:
        ha_open, ha_close = heiken_ashi_pine(data)
        if ha_open is not None and ha_close is not None and \
           len(ha_open) >= 1 and len(ha_close) >= 1 and \
           not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
            if ha_close.iloc[-1] > ha_open.iloc[-1]:
                bull_confluences += 1; signal_details_pine['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]:
                bear_confluences += 1; signal_details_pine['HA'] = "▼"
            else: signal_details_pine['HA'] = "─"
        else: signal_details_pine['HA'] = "N/A"
    except Exception: signal_details_pine['HA'] = "Err"

    # Smoothed Heiken Ashi Signal
    try:
        sha_open, sha_close = smoothed_heiken_ashi_pine(data, 10, 10)
        if sha_open is not None and sha_close is not None and \
           len(sha_open) >= 1 and len(sha_close) >= 1 and \
           not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
            if sha_close.iloc[-1] > sha_open.iloc[-1]:
                bull_confluences += 1; signal_details_pine['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]:
                bear_confluences += 1; signal_details_pine['SHA'] = "▼"
            else: signal_details_pine['SHA'] = "─"
        else: signal_details_pine['SHA'] = "N/A"
    except Exception: signal_details_pine['SHA'] = "Err"

    # Ichimoku Signal
    try:
        ichimoku_signal_val = ichimoku_pine_signal(high, low, close, 9, 26, 52)
        if ichimoku_signal_val == 1:
            bull_confluences += 1; signal_details_pine['Ichi'] = "▲"
        elif ichimoku_signal_val == -1:
            bear_confluences += 1; signal_details_pine['Ichi'] = "▼"
        else: signal_details_pine['Ichi'] = "─"
    except Exception: signal_details_pine['Ichi'] = "Err"

    confluence_value = 0
    direction = "NEUTRE"
    if bull_confluences > bear_confluences:
        direction = "HAUSSIER"; confluence_value = bull_confluences
    elif bear_confluences > bull_confluences:
        direction = "BAISSIER"; confluence_value = bear_confluences
    elif bull_confluences == bear_confluences and bull_confluences > 0:
        direction = "CONFLIT"; confluence_value = bull_confluences

    return {
        'confluence_P': confluence_value,
        'direction_P': direction,
        'bull_P': bull_confluences,
        'bear_P': bear_confluences,
        'rsi_P': signal_details_pine.get('RSI_val', "N/A"),
        'adx_P': signal_details_pine.get('ADX_val', "N/A"),
        'signals_P': signal_details_pine
    }

# --- Fonction utilitaire pour les étoiles ---
def get_stars_pine(confluence_value):
    stars_map = {6: "⭐⭐⭐⭐⭐⭐", 5: "⭐⭐⭐⭐⭐", 4: "⭐⭐⭐⭐", 3: "⭐⭐⭐", 2: "⭐⭐", 1: "⭐"}
    return stars_map.get(int(confluence_value), "WAIT")

# --- Interface utilisateur Streamlit ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres du Scan")
    min_conf = st.selectbox("Confluence minimale (0-6):",
                           options=list(range(7)), index=3,
                           format_func=lambda x: f"{x} étoile(s)")
    show_all = st.checkbox("Voir toutes les paires (ignorer filtre)")
    pair_to_debug = st.selectbox("🔍 Afficher données OHLC pour:", ["Aucune"] + PAIRS_OANDA, index=0)
    scan_btn = st.button("🚀 Lancer le Scanner (OANDA H1)", type="primary", use_container_width=True)

with col2:
    if scan_btn:
        if 'oanda' not in st.secrets or 'API_KEY' not in st.secrets.get("oanda", {}):
            st.error("⚠️ **Configuration OANDA Manquante!**")
            st.info("Ajoutez votre clé API OANDA dans les secrets Streamlit Cloud.")
            st.code("""[oanda]\nAPI_KEY = "VOTRE_CLE_API_OANDA_ICI" """, language="toml")
        else:
            st.info("🔄 Scan OANDA H1 en cours...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_list = []

            if pair_to_debug != "Aucune":
                st.subheader(f"📊 Données OHLC récentes pour {pair_to_debug} (OANDA):")
                debug_data = get_data_oanda(pair_to_debug, timeframe="H1", count=50)
                if debug_data is not None and not debug_data.empty:
                    st.dataframe(debug_data[['Open', 'High', 'Low', 'Close']].tail(10))
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
                            'Paire': pair_display_name,
                            'Direction': signals_data['direction_P'],
                            'Conf. (0-6)': signals_data['confluence_P'],
                            'Étoiles': get_stars_pine(signals_data['confluence_P']),
                            'RSI': signals_data.get('rsi_P', "N/A"),
                            'ADX': signals_data.get('adx_P', "N/A"),
                            'Bull Signals': signals_data['bull_P'],
                            'Bear Signals': signals_data['bear_P'],
                            'Détails Signaux': signals_data['signals_P']
                        })
                    else:
                        results_list.append({'Paire': pair_display_name, 'Direction': 'ERREUR CALCUL', 'Conf. (0-6)': 0, 'Étoiles': 'N/A', 'RSI': 'N/A', 'ADX': 'N/A', 'Bull Signals': 0, 'Bear Signals': 0, 'Détails Signaux': {'Info': 'Calcul signaux échoué'}})
                else:
                    results_list.append({'Paire': pair_display_name, 'Direction': 'ERREUR DONNÉES', 'Conf. (0-6)': 0, 'Étoiles': 'N/A', 'RSI': 'N/A', 'ADX': 'N/A', 'Bull Signals': 0, 'Bear Signals': 0, 'Détails Signaux': {'Info': 'Données OANDA non disponibles'}})
                
                time.sleep(0.25) # Respecter les limites API

            progress_bar.empty()
            status_text.success("✅ Scan OANDA H1 terminé!")

            if results_list:
                results_df = pd.DataFrame(results_list)
                filtered_df = results_df.copy() if show_all else results_df[results_df['Conf. (0-6)'] >= min_conf].copy()
                
                if not show_all:
                    st.success(f"🎯 {len(filtered_df)} paire(s) avec {min_conf}+ confluence (OANDA H1).")
                else:
                    st.info(f"🔍 Affichage des {len(filtered_df)} paires scannées (OANDA H1).")

                if not filtered_df.empty:
                    sorted_df = filtered_df.sort_values(by=['Conf. (0-6)', 'Paire'], ascending=[False, True])
                    display_columns = ['Paire', 'Direction', 'Conf. (0-6)', 'Étoiles', 'RSI', 'ADX', 'Bull Signals', 'Bear Signals']
                    st.dataframe(sorted_df[display_columns], use_container_width=True, hide_index=True,
                                 column_config={"Conf. (0-6)": st.column_config.NumberColumn(format="%d")})
                    
                    with st.expander("📊 Détails des signaux (OANDA H1)"):
                        for _, row in sorted_df.iterrows():
                            signal_details = row.get('Détails Signaux', {})
                            if not isinstance(signal_details, dict): signal_details = {'Info': 'Détails non disponibles'}
                            st.write(f"**{row.get('Paire', 'N/A')}** | {row.get('Étoiles', '')} ({row.get('Conf. (0-6)', 'N/A')}) | Dir: {row.get('Direction', 'N/A')}")
                            
                            detail_cols_count = max(1, len([k for k in signal_details if "_val" not in k and k != 'Info']))
                            detail_cols = st.columns(detail_cols_count)
                            
                            signal_order = ['HMA', 'RSI', 'ADX', 'HA', 'SHA', 'Ichi']
                            col_idx = 0
                            displayed_keys = set()

                            for key in signal_order:
                                if key in signal_details and col_idx < len(detail_cols):
                                    detail_cols[col_idx].metric(label=key, value=str(signal_details[key]))
                                    displayed_keys.add(key)
                                    col_idx += 1
                            
                            for key, val in signal_details.items(): # Pour ceux non listés dans signal_order
                                if key not in displayed_keys and "_val" not in key and key != 'Info' and col_idx < len(detail_cols):
                                    detail_cols[col_idx].metric(label=key, value=str(val))
                                    col_idx +=1
                            st.divider()
                else:
                    st.warning(f"❌ Aucune paire avec critères de filtrage (OANDA H1).")
            else:
                st.error("❌ Aucune paire traitée (OANDA H1).")

# --- Sections d'information ---
with st.expander("ℹ️ Comment Fonctionne ce Scanner?"):
    st.markdown("""
    Ce scanner analyse plusieurs indicateurs techniques sur l'horizon **H1** pour des signaux de confluence.
    **Indicateurs (6):** HMA(20), RSI(10, ohlc4), ADX(14), Heiken Ashi, Smoothed Heiken Ashi(10,10), Ichimoku(9,26,52).
    **Source:** API OANDA (compte de pratique).
    **Confluence:** Nombre de signaux dans la même direction. ADX >= 20 confirme la force (compte pour bull & bear).
    """)

with st.expander("🔧 Configuration API OANDA"):
    st.markdown("""
    1.  Obtenez une clé API OANDA (compte démo ou réel).
    2.  Ajoutez la clé aux Secrets Streamlit Cloud:
        ```toml
        [oanda]
        API_KEY = "VOTRE_CLE_API_OANDA_ICI"
        ```
    3.  L'application utilise l'URL de pratique par défaut.
    """)

st.caption("Scanner Forex H1 avec données OANDA • Version pour Streamlit Cloud")
