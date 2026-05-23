# ─── stock_app.py  v24.0 급등 예측 탐색기 (서버용) ─────────────
import streamlit as st
import pandas as pd
import numpy as np
import datetime, json, os, time, re, requests

# ── 섹터별 평균 PER/PBR (2026.04 기준) ──
SECTOR_PER = {
    "음식료품": 21.07, "음식점및주점업": 2.54, "섬유의복": 11.04,
    "종이목재": 31.5, "화학": 32.53, "의약품": 72.73,
    "비금속광물": 28.38, "철강금속": 27.95, "기계": 89.31,
    "전기전자": 312.33, "의료정밀": 75.45, "운수장비": 32.26,
    "유통업": 58.73, "전기가스업": 8.48, "건설업": 21.85,
    "운수창고업": 7.24, "통신업": 31.23, "금융업": 26.21,
    "보험": 10.57, "서비스업": 14.13, "제조업": 20.0,
    "소프트웨어": 46.25, "IT부품": 124.56, "반도체": 124.56,
    "IT하드웨어": 124.56, "바이오": 118.06, "게임": 29.7,
    "인터넷": 19.82, "미디어": 41.26, "엔터테인먼트": 29.7,
    "화장품": 42.18, "2차전지": 99.01, "자동차부품": 91.36,
    "기계장비": 64.3, "음식료": 10.53, "건설": 14.13,
    "교육": 7.69,
}
MARKET_AVG_PER = 15.0

def get_sector_per(sector_name):
    if not sector_name:
        return MARKET_AVG_PER
    clean = sector_name.replace("·", "").replace(",", "").replace(" ", "")
    for key in SECTOR_PER:
        clean_key = key.replace("·", "").replace(",", "").replace(" ", "")
        if clean_key in clean or clean in clean_key:
            return SECTOR_PER[key]
    return MARKET_AVG_PER

def judge_per_by_sector(per, sector_name):
    if per is None or per <= 0:
        return "N/A", "", 0
    avg = get_sector_per(sector_name)
    ratio = per / avg
    if ratio <= 0.5:
        return "저평가", f"섹터평균({avg:.1f})의 {ratio:.0%}", 2
    elif ratio <= 0.8:
        return "다소저평가", f"섹터평균({avg:.1f})의 {ratio:.0%}", 1
    elif ratio <= 1.2:
        return "적정", f"섹터평균({avg:.1f})의 {ratio:.0%}", 0
    elif ratio <= 1.5:
        return "다소고평가", f"섹터평균({avg:.1f})의 {ratio:.0%}", -1
    else:
        return "고평가", f"섹터평균({avg:.1f})의 {ratio:.0%}", -2


# ── Gemini AI ──
from google import genai
from google.genai import types

GEMINI_OK = False
gemini_model = None

def init_gemini(api_key):
    global gemini_model, GEMINI_OK
    try:
        client = genai.Client(api_key=api_key)
        gemini_model = client
        GEMINI_OK = True
    except:
        GEMINI_OK = False


@st.cache_data(ttl=300)
def gemini_judgment(name, code, price, change, score, grade, rsi, mfi, adx, per, pbr, ev_ebitda, buy_reasons, sell_reasons, support, resist, verdict):
    if not GEMINI_OK:
        return None
    try:
        buy_text = ", ".join(buy_reasons[:5]) if buy_reasons else "없음"
        sell_text = ", ".join(sell_reasons[:5]) if sell_reasons else "없음"
        ev_text = f"{ev_ebitda:.1f}" if ev_ebitda else "N/A"
        per_text = f"{per}" if per != 0 else "N/A (데이터 없음)"
        pbr_text = f"{pbr}" if pbr != 0 else "N/A (데이터 없음)"

        foreign_flow = "외국인/기관 데이터 없음"
        try:
            url = f"https://m.stock.naver.com/api/stock/{code}/integration"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            data = resp.json()
            fb = []
            ob = []
            for deal in data.get("dealTrendInfos", []):
                try:
                    f_val = int(deal.get("foreignerPureBuyQuant", "0").replace(",", "").replace("+", ""))
                    o_val = int(deal.get("organPureBuyQuant", "0").replace(",", "").replace("+", ""))
                    fb.append(f_val)
                    ob.append(o_val)
                except:
                    pass
            fb = fb[::-1]
            ob = ob[::-1]
            if fb:
                total_fb = sum(fb)
                total_ob = sum(ob)
                consec_buy = all(x > 0 for x in fb[-3:]) if len(fb) >= 3 else False
                consec_sell = all(x < 0 for x in fb[-3:]) if len(fb) >= 3 else False
                fb_text = ", ".join([f"{x:+,}" for x in fb])
                ob_text = ", ".join([f"{x:+,}" for x in ob])
                pattern = "연속매수 (야금야금 모으는 중)" if consec_buy else "연속매도 (빠지는 중)" if consec_sell else "혼조"
                foreign_flow = f"외국인 최근 5일 순매수: [{fb_text}]\n외국인 5일 합계: {total_fb:+,}주 | 패턴: {pattern}\n기관 최근 5일 순매수: [{ob_text}]\n기관 5일 합계: {total_ob:+,}주"
        except:
            pass

        data_section = f"""당신은 '전인구경제연구소' 스타일의 외국인 수급 추적 전문가이자 연봉 1조 퀀트 트레이더입니다.

[핵심 투자 철학 - 전인구 전략]
1. 외국인이 대장주를 팔 때 → 다른 종목을 사고 있다는 신호
2. 외국인이 주가 하락 시에도 야금야금 지분을 늘리는 종목 = 핵심 매수 대상
3. 모두가 한 섹터에 몰릴 때, 텅 빈 후방이 기회
4. 외국인 연속 매수 + 개인 매도 = 강력한 매수 시그널
5. 외국인 연속 매도 + 개인 매수 = 위험 시그널
6. 주가가 안 올라도 외국인이 지분 늘리면 → 배당·저PER 전략일 수 있음

[분석 대상]
종목: {name} ({code})
현재가: {price:,}원 ({change:+.2f}%)
AI 점수: {score}점 ({grade}) | 판정: {verdict}
RSI: {rsi} | MFI: {mfi} | ADX: {adx}
PER: {per_text} | PBR: {pbr_text} | EV/EBITDA: {ev_text}
매수 신호: {buy_text}
매도 신호: {sell_text}
지지선: {support:,}원 | 저항선: {resist:,}원

[외국인/기관 수급 흐름]
{foreign_flow}"""

        format_section = """
다음 형식으로 답변해주세요 (총 6줄, 이모지 포함):
1) 📊 현재 상황 한줄 요약 (기술적 지표 + 수급 종합)
2) 💡 매매 판단 (매수/매도/관망 중 택1 + 기술적 근거와 수급 근거 모두 포함)
3) 🎯 추천 전략 (진입가, 목표가, 손절가 포함)
4) 🌊 외국인 수급 판단 (전인구 전략: 외국인이 모으는 중인지, 빠지는 중인지, 대장주 매도 후 자금 이동 가능성)
5) 🔍 후방 기습 관점 (이 종목이 시장 관심 밖 저평가인지, 외국인이 남몰래 담는 종목인지 판단)
6) ⚠️ 주의할 점"""

        prompt = data_section + format_section
        response = gemini_model.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        ai_summary = response.text.strip()[:300]
    except:
        ai_summary = ""


@st.cache_data(ttl=300)
def gemini_coin_judgment(symbol, price, change_24h, rsi, mfi, adx, funding_rate, long_short_ratio, oi_change, score, grade, buy_reasons, sell_reasons, support, resist):
    if not GEMINI_OK:
        return None
    try:
        buy_text = ", ".join(buy_reasons[:5]) if buy_reasons else "없음"
        sell_text = ", ".join(sell_reasons[:5]) if sell_reasons else "없음"
        fr_text = f"{funding_rate}" if funding_rate else "N/A"
        ls_text = f"{long_short_ratio}" if long_short_ratio else "N/A"
        oi_text = f"{oi_change}" if oi_change else "N/A"
        price_str = f"{price:,.2f}"
        change_str = f"{change_24h:+.2f}"
        support_str = f"{support:,.2f}"
        resist_str = f"{resist:,.2f}"

        data_section = f"""당신은 연봉 1조의 전설적인 코인 선물 트레이더입니다.
아래 기술적 분석 + 온체인 데이터를 보고 초보 투자자도 이해할 수 있게 매매 판단을 해주세요.

코인: {symbol}
현재가: {price_str}$ ({change_str}%)
AI 점수: {score}점 ({grade})
RSI: {rsi} | MFI: {mfi} | ADX: {adx}
펀딩비: {fr_text} | 롱숏비율: {ls_text} | OI 변화: {oi_text}
매수 신호: {buy_text}
매도 신호: {sell_text}
지지선: {support_str}$ | 저항선: {resist_str}$"""

        format_section = """
다음 형식으로 답변해주세요 (총 5줄, 이모지 포함):
1) 📊 현재 상황 한줄 요약
2) 💡 포지션 판단 (롱/숏/관망 중 택1 + 이유)
3) 🎯 추천 전략 (진입가, 목표가, 손절가, 추천 레버리지 포함)
4) 📈 펀딩비·롱숏·OI 해석 (한줄)
5) ⚠️ 주의할 점"""

        prompt = data_section + format_section
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception:
        return None


@st.cache_data(ttl=600)
def gemini_news_sentiment(name, code):
    if not GEMINI_OK:
        return None
    try:
        news_url = f"https://stock.naver.com/api/domestic/detail/news?itemCode={code}&page=1&pageSize=10"
        resp = requests.get(news_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        titles = []
        if resp.status_code == 200:
            ndata = resp.json()
            for cluster in ndata.get("clusters", []):
                for art in cluster.get("items", []):
                    t = art.get("title", "").strip()
                    if t:
                        titles.append(t)
                    if len(titles) >= 10:
                        break
                if len(titles) >= 10:
                    break

        if not titles:
            headers2 = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            rss_url = f"https://news.google.com/rss/search?q={name}&hl=ko&gl=KR&ceid=KR:ko"
            rss_resp = requests.get(rss_url, headers=headers2, timeout=5)
            from bs4 import BeautifulSoup
            rss_soup = BeautifulSoup(rss_resp.text, "xml")
            for item in rss_soup.select("item"):
                t = item.select_one("title")
                if t:
                    titles.append(t.get_text(strip=True))
                    if len(titles) >= 10:
                        break

        if not titles:
            return None

        news_text = "\n".join([f"- {t}" for t in titles])

        data_section = f"""당신은 금융 뉴스 감성 분석 전문가입니다.
아래는 '{name}({code})' 관련 최신 뉴스 제목입니다:

{news_text}"""

        format_section = """
다음 형식으로 답변해주세요:
1) 📰 뉴스 감성: [긍정 / 부정 / 중립] (긍정 N개, 부정 N개, 중립 N개)
2) 📋 핵심 이슈 요약 (2줄)
3) 💡 투자 영향: 이 뉴스가 주가에 미칠 영향 (1줄)
4) 주의: 뉴스만으로 투자 결정을 내리지 마세요"""

        prompt = data_section + format_section
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception:
        return None


def get_weekly_trend(code):
    try:
        if code.isdigit() and len(code) == 6:
            from pykrx import stock as pkstock
            end = datetime.datetime.now().strftime("%Y%m%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y%m%d")
            df = pkstock.get_market_ohlcv(start, end, code)
            if df is None or len(df) < 20:
                return None
            df.index = pd.to_datetime(df.index)
            weekly = df["종가"].resample("W").last().dropna()
            if len(weekly) < 10:
                return None
            close = weekly.values
        else:
            import yfinance as yf
            tk = yf.Ticker(code)
            df = tk.history(period="6mo", interval="1wk")
            if df is None or len(df) < 10:
                return None
            close = df["Close"].values

        ema10 = pd.Series(close).ewm(span=10).mean().values
        ema30 = pd.Series(close).ewm(span=30).mean().values

        current = close[-1]
        ema10_now = ema10[-1]
        ema30_now = ema30[-1]

        if current > ema10_now > ema30_now:
            return {"trend": "상승", "emoji": "🟢", "desc": "주봉 상승추세 (가격 > EMA10 > EMA30)"}
        elif current < ema10_now < ema30_now:
            return {"trend": "하락", "emoji": "🔴", "desc": "주봉 하락추세 (가격 < EMA10 < EMA30)"}
        else:
            return {"trend": "횡보", "emoji": "🟡", "desc": "주봉 횡보 (추세 불명확)"}
    except Exception:
        return None


# ── matplotlib 한글 폰트 ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

font_path = None
for fp in [
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleGothic.ttf",
]:
    if os.path.exists(fp):
        font_path = fp
        break
if font_path:
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# ── KIS 브로커 ──
KIS_OK = False
_broker = None
try:
    from kis_api_module import KISApi
    _broker = KISApi("kis_config.yaml")
    if _broker.enabled:
        KIS_OK = True
    else:
        _broker = None
except Exception:
    pass

# ── FinanceDataReader ──
FDR_OK = False
try:
    import FinanceDataReader as fdr
    FDR_OK = True
except Exception:
    pass

# ── streamlit-autorefresh ──
AUTOREFRESH_OK = False
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except Exception:
    pass

# ── 코인 선물 (Binance) ──
COIN_FUTURES = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT",
    "BNBUSDT", "LTCUSDT", "ETCUSDT", "FILUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "SUIUSDT", "PEPEUSDT", "SHIBUSDT",
    "NEARUSDT", "ATOMUSDT", "FTMUSDT", "INJUSDT", "TIAUSDT",
    "SEIUSDT", "JUPUSDT", "WIFUSDT", "RUNEUSDT", "AAVEUSDT",
    "MKRUSDT", "RENDERUSDT", "FETUSDT", "ONDOUSDT", "ENAUSDT",
    "TRXUSDT", "TONUSDT", "ICPUSDT", "UNIUSDT", "XLMUSDT",
    "HBARUSDT", "ALGOUSDT", "VETUSDT", "GRTUSDT", "SANDUSDT",
    "MANAUSDT", "AXSUSDT", "GALAUSDT", "IMXUSDT", "LDOUSDT",
]

@st.cache_data(ttl=60)
def get_coin_klines(symbol, interval="1d", limit=200):
    try:
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "Open_time", "Open", "High", "Low", "Close", "Volume",
            "Close_time", "Quote_vol", "Trades", "Buy_vol", "Buy_quote", "Ignore"
        ])
        df["Open"] = df["Open"].astype(float)
        df["High"] = df["High"].astype(float)
        df["Low"] = df["Low"].astype(float)
        df["Close"] = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        return df
    except:
        return None

@st.cache_data(ttl=30)
def get_funding_rate(symbol):
    try:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        r = requests.get(url, params={"symbol": symbol}, timeout=5)
        data = r.json()
        return round(float(data.get("lastFundingRate", 0)) * 100, 4)
    except:
        return None

@st.cache_data(ttl=30)
def get_long_short_ratio(symbol, period="5m"):
    try:
        url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        r = requests.get(url, params={"symbol": symbol, "period": period, "limit": 1}, timeout=5)
        data = r.json()
        if data:
            return round(float(data[0].get("longShortRatio", 1)), 3)
    except:
        pass
    return None

@st.cache_data(ttl=30)
def get_open_interest(symbol):
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        r = requests.get(url, params={"symbol": symbol}, timeout=5)
        data = r.json()
        return round(float(data.get("openInterest", 0)), 2)
    except:
        return None

@st.cache_data(ttl=30)
def get_oi_history(symbol, period="5m", limit=30):
    try:
        url = "https://fapi.binance.com/futures/data/openInterestHist"
        r = requests.get(url, params={"symbol": symbol, "period": period, "limit": limit}, timeout=5)
        data = r.json()
        if data:
            return [float(d.get("sumOpenInterest", 0)) for d in data]
    except:
        pass
    return None

@st.cache_data(ttl=30)
def get_funding_history(symbol, limit=20):
    try:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=5)
        data = r.json()
        return [round(float(d.get("fundingRate", 0)) * 100, 4) for d in data]
    except:
        return []

# ── 기술 지표 함수 ──
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calc_stoch_rsi(series, period=14):
    rsi = calc_rsi(series, period)
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    return (rsi - min_rsi) / (max_rsi - min_rsi + 1e-9)

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast).mean()
    ema_s = series.ewm(span=slow).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal).mean()
    hist = macd - sig
    return macd, sig, hist

def calc_bb(series, period=20, std_n=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma + std_n * std, ma, ma - std_n * std

def calc_ichimoku(high, low, close):
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, span_a, span_b

def calc_supertrend(df, period=10, multiplier=3):
    hl2 = (df["High"] + df["Low"]) / 2
    atr = df["High"].combine(df["Close"].shift(), max) - df["Low"].combine(df["Close"].shift(), min)
    atr = atr.rolling(period).mean()
    up = hl2 - multiplier * atr
    dn = hl2 + multiplier * atr
    st_dir = pd.Series(1, index=df.index)
    final_up = up.copy()
    final_dn = dn.copy()
    for i in range(1, len(df)):
        if up.iloc[i] > final_up.iloc[i - 1] if not np.isnan(final_up.iloc[i - 1]) else True:
            final_up.iloc[i] = up.iloc[i]
        else:
            final_up.iloc[i] = final_up.iloc[i - 1]
        if dn.iloc[i] < final_dn.iloc[i - 1] if not np.isnan(final_dn.iloc[i - 1]) else True:
            final_dn.iloc[i] = dn.iloc[i]
        else:
            final_dn.iloc[i] = final_dn.iloc[i - 1]
        if st_dir.iloc[i - 1] == 1:
            if df["Close"].iloc[i] < final_up.iloc[i]:
                st_dir.iloc[i] = -1
            else:
                st_dir.iloc[i] = 1
        else:
            if df["Close"].iloc[i] > final_dn.iloc[i]:
                st_dir.iloc[i] = 1
            else:
                st_dir.iloc[i] = -1
    st_line = pd.Series(np.where(st_dir == 1, final_up, final_dn), index=df.index)
    return st_line, st_dir

def calc_obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

def calc_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / (df["Volume"].cumsum() + 1e-9)

def calc_mfi(df, period=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos = pd.Series(0.0, index=df.index)
    neg = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        if tp.iloc[i] > tp.iloc[i - 1]:
            pos.iloc[i] = mf.iloc[i]
        else:
            neg.iloc[i] = mf.iloc[i]
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum()
    mfi = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-9)))
    return mfi

def calc_adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di
    
    
# ═══════════════════════════════════════════════════════
#  코인 분석 함수
# ═══════════════════════════════════════════════════════

def analyze_coin(df, symbol, funding, ls_ratio, oi):
    if df is None or len(df) < 30:
        return None
    try:
        code = symbol
        name = symbol
        c = df["Close"]
        h = df["High"]
        l = df["Low"]
        v = df["Volume"]

        price = round(c.iloc[-1], 4)
        prev = c.iloc[-2]
        change = round((price - prev) / prev * 100, 2)
        vol_avg = v.iloc[-20:].mean()
        vol_ratio = round(v.iloc[-1] / (vol_avg + 1e-9), 2)

        rsi = calc_rsi(c)
        rsi_val = round(rsi.iloc[-1], 1) if not np.isnan(rsi.iloc[-1]) else 50

        stoch_rsi = calc_stoch_rsi(c)
        stoch_val = round(stoch_rsi.iloc[-1] * 100, 1) if not np.isnan(stoch_rsi.iloc[-1]) else 50

        macd_line, macd_sig, macd_hist = calc_macd(c)
        bb_upper, bb_mid, bb_lower = calc_bb(c)

        ema5 = c.ewm(span=5).mean()
        ema20 = c.ewm(span=20).mean()
        ema60 = c.ewm(span=60).mean()

        coin_df = pd.DataFrame({"High": h, "Low": l, "Close": c, "Volume": v})
        st_line, st_dir = calc_supertrend(coin_df)
        obv = calc_obv(c, v)
        obv_ma = obv.rolling(20).mean()
        vwap = calc_vwap(coin_df)
        mfi = calc_mfi(coin_df)
        mfi_val = round(mfi.iloc[-1], 1) if not np.isnan(mfi.iloc[-1]) else 50
        adx, plus_di, minus_di = calc_adx(coin_df)
        adx_val = round(adx.iloc[-1], 1) if not np.isnan(adx.iloc[-1]) else 0

        score = 50
        buy_reasons = []
        sell_reasons = []

        if rsi_val < 30:
            score += 7
            buy_reasons.append(f"RSI 과매도({rsi_val})")
        elif rsi_val > 70:
            score -= 7
            sell_reasons.append(f"RSI 과매수({rsi_val})")

        if stoch_val < 20:
            score += 4
            buy_reasons.append(f"StochRSI 과매도({stoch_val})")
        elif stoch_val > 80:
            score -= 4
            sell_reasons.append(f"StochRSI 과매수({stoch_val})")

        mh = macd_hist
        if len(mh) > 1 and not np.isnan(mh.iloc[-1]) and not np.isnan(mh.iloc[-2]):
            if mh.iloc[-1] > 0 and mh.iloc[-2] <= 0:
                score += 12
                buy_reasons.append("MACD 골든크로스")
            elif mh.iloc[-1] < 0 and mh.iloc[-2] >= 0:
                score -= 12
                sell_reasons.append("MACD 데드크로스")

        if not np.isnan(bb_lower.iloc[-1]) and price <= bb_lower.iloc[-1]:
            score += 5
            buy_reasons.append("볼린저 하단 터치")
        elif not np.isnan(bb_upper.iloc[-1]) and price >= bb_upper.iloc[-1]:
            score -= 5
            sell_reasons.append("볼린저 상단 돌파")

        e5, e20, e60 = ema5.iloc[-1], ema20.iloc[-1], ema60.iloc[-1]
        if e5 > e20 > e60:
            score += 5
            buy_reasons.append("EMA 정배열 (5>20>60)")
        elif e5 < e20 < e60:
            score -= 5
            sell_reasons.append("EMA 역배열 (5<20<60)")

        if len(st_dir) > 1:
            if st_dir.iloc[-1] == 1 and st_dir.iloc[-2] == -1:
                score += 10
                buy_reasons.append("슈퍼트렌드 매수 전환")
            elif st_dir.iloc[-1] == -1 and st_dir.iloc[-2] == 1:
                score -= 10
                sell_reasons.append("슈퍼트렌드 매도 전환")
            elif st_dir.iloc[-1] == 1:
                score += 2
            elif st_dir.iloc[-1] == -1:
                score -= 2

        if not np.isnan(obv_ma.iloc[-1]):
            if obv.iloc[-1] > obv_ma.iloc[-1] * 1.05:
                score += 8
                buy_reasons.append("OBV 상승 (매집 중)")
            elif obv.iloc[-1] < obv_ma.iloc[-1] * 0.95:
                score -= 8
                sell_reasons.append("OBV 하락 (매도 중)")

        vwap_val = vwap.iloc[-1]
        if not np.isnan(vwap_val):
            if price > vwap_val:
                score += 5
                buy_reasons.append("VWAP 위 (매수 영역)")
            else:
                score -= 5
                sell_reasons.append("VWAP 아래 (매도 영역)")

        if mfi_val < 20:
            score += 5
            buy_reasons.append(f"MFI 자금유입 과매도({mfi_val})")
        elif mfi_val > 80:
            score -= 5
            sell_reasons.append(f"MFI 자금유출 과매수({mfi_val})")

        if adx_val > 25:
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                score += 10
                buy_reasons.append(f"ADX 강한 상승추세({adx_val})")
            else:
                score -= 10
                sell_reasons.append(f"ADX 강한 하락추세({adx_val})")

        if vol_ratio >= 5 and change > 0:
            score += 5
            buy_reasons.append(f"거래량 폭발({vol_ratio}x) + 양봉")
        elif vol_ratio >= 3 and change > 0:
            score += 3
            buy_reasons.append(f"거래량 급증({vol_ratio}x) + 양봉")
        elif vol_ratio >= 3 and change < 0:
            score -= 5
            sell_reasons.append(f"거래량 급증({vol_ratio}x) + 음봉")

        def detect_div(prices, indicator, window=14):
            if len(prices) < window + 5 or len(indicator) < window + 5:
                return None
            p_recent = prices.iloc[-window:]
            i_recent = indicator.iloc[-window:]
            p_lows = []
            for k in range(1, len(p_recent) - 1):
                if p_recent.iloc[k] <= p_recent.iloc[k-1] and p_recent.iloc[k] <= p_recent.iloc[k+1]:
                    p_lows.append(k)
            if len(p_lows) >= 2:
                l1, l2 = p_lows[-2], p_lows[-1]
                if p_recent.iloc[l2] < p_recent.iloc[l1] and i_recent.iloc[l2] > i_recent.iloc[l1]:
                    return "bullish"
            p_highs = []
            for k in range(1, len(p_recent) - 1):
                if p_recent.iloc[k] >= p_recent.iloc[k-1] and p_recent.iloc[k] >= p_recent.iloc[k+1]:
                    p_highs.append(k)
            if len(p_highs) >= 2:
                h1, h2 = p_highs[-2], p_highs[-1]
                if p_recent.iloc[h2] > p_recent.iloc[h1] and i_recent.iloc[h2] < i_recent.iloc[h1]:
                    return "bearish"
            return None

        rsi_div = detect_div(c, rsi, window=20)
        macd_div = detect_div(c, macd_hist, window=20)

        divergence_text = ""
        if rsi_div == "bullish":
            buy_reasons.append("RSI 상승 다이버전스 (반전↑)")
            score += 6
            divergence_text = "🟢 RSI 상승 다이버전스"
        elif rsi_div == "bearish":
            sell_reasons.append("RSI 하락 다이버전스 (반전↓)")
            score -= 6
            divergence_text = "🔴 RSI 하락 다이버전스"
        if macd_div == "bullish":
            buy_reasons.append("MACD 상승 다이버전스 (반전↑)")
            score += 6
            divergence_text += " 🟢 MACD 상승 다이버전스"
        elif macd_div == "bearish":
            sell_reasons.append("MACD 하락 다이버전스 (반전↓)")
            score -= 6
            divergence_text += " 🔴 MACD 하락 다이버전스"

        vol_up_3 = all(v.iloc[-(i+1)] > v.iloc[-(i+2)] for i in range(3)) if len(v) > 4 else False
        vol_up_5 = all(v.iloc[-(i+1)] > v.iloc[-(i+2)] for i in range(5)) if len(v) > 6 else False
        if vol_up_5:
            score += 3
            buy_reasons.append("거래량 5연속 증가")
        elif vol_up_3:
            score += 2
            buy_reasons.append("거래량 3연속 증가")

        if funding is not None:
            if funding > 0.05:
                score -= 12
                sell_reasons.append(f"펀딩비 과열({funding}%) → 숏 유리")
            elif funding > 0.03:
                score -= 6
                sell_reasons.append(f"펀딩비 높음({funding}%)")
            elif funding < -0.03:
                score += 12
                buy_reasons.append(f"펀딩비 음수({funding}%) → 롱 유리")
            elif funding < -0.01:
                score += 6
                buy_reasons.append(f"펀딩비 낮음({funding}%)")

        if ls_ratio is not None:
            if ls_ratio > 2.0:
                score -= 10
                sell_reasons.append(f"롱 과밀({ls_ratio}) → 하락 주의")
            elif ls_ratio > 1.5:
                score -= 5
                sell_reasons.append(f"롱 우위({ls_ratio})")
            elif ls_ratio < 0.5:
                score += 10
                buy_reasons.append(f"숏 과밀({ls_ratio}) → 숏스퀴즈 가능")
            elif ls_ratio < 0.7:
                score += 5
                buy_reasons.append(f"숏 우위({ls_ratio})")

        has_rsi_sell = any("RSI 과매수" in s for s in sell_reasons)
        has_macd_buy = any("MACD 골든크로스" in b for b in buy_reasons)
        if has_rsi_sell and has_macd_buy:
            score -= 5

        score = max(0, min(100, score))

        if score >= 75 and len(buy_reasons) >= 3:
            position = "🟢 롱 (매수)"
        elif score <= 40 and len(sell_reasons) >= 3:
            position = "🔴 숏 (매도)"
        elif score >= 60 and len(buy_reasons) >= 2:
            position = "🟡 롱 관심"
        elif score <= 45 and len(sell_reasons) >= 2:
            position = "🟡 숏 관심"
        else:
            position = "⚪ 관망"

        grade = "A+" if score >= 80 else "A" if score >= 70 else "B" if score >= 55 else "C" if score >= 40 else "D"

        return {
            "symbol": symbol, "name": symbol.replace("USDT", ""),
            "price": price, "change": change,
            "vol_ratio": vol_ratio, "rsi": rsi_val,
            "stoch_rsi": stoch_val, "mfi": mfi_val, "adx": adx_val,
            "score": score, "grade": grade, "position": position,
            "buy": buy_reasons, "sell": sell_reasons,
            "funding": funding, "ls_ratio": ls_ratio, "oi": oi,
            "divergence": divergence_text.strip(),
            "df": df, "ema5": ema5, "ema20": ema20, "ema60": ema60,
            "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
            "macd_line": macd_line, "macd_sig": macd_sig, "macd_hist": macd_hist,
            "rsi_series": rsi, "mfi_series": mfi,
            "obv": obv, "st_line": st_line, "st_dir": st_dir,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def draw_coin_chart(r, last_n=60):
    df = r["df"].tail(last_n).reset_index(drop=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333")
    n = len(df)

    ax1 = axes[0]
    draw_candle_bars(ax1, df)
    e5 = r["ema5"].tail(last_n).reset_index(drop=True)
    e20 = r["ema20"].tail(last_n).reset_index(drop=True)
    e60 = r["ema60"].tail(last_n).reset_index(drop=True)
    bu = r["bb_upper"].tail(last_n).reset_index(drop=True)
    bl = r["bb_lower"].tail(last_n).reset_index(drop=True)
    ax1.plot(range(n), e5[:n], color="#ffeb3b", linewidth=0.8, label="EMA5")
    ax1.plot(range(n), e20[:n], color="#ff9800", linewidth=0.8, label="EMA20")
    ax1.plot(range(n), e60[:n], color="#9c27b0", linewidth=0.8, label="EMA60")
    ax1.plot(range(n), bu[:n], color="#616161", linewidth=0.5, linestyle="--")
    ax1.plot(range(n), bl[:n], color="#616161", linewidth=0.5, linestyle="--")
    ax1.set_title(f"{r['symbol']}  ${r['price']:,.4f}  {r['change']:+.2f}%", color="white", fontsize=11)
    ax1.legend(fontsize=7, loc="upper left")

    ax2 = axes[1]
    mh = r["macd_hist"].tail(last_n).reset_index(drop=True)
    colors = ["#ff1744" if v >= 0 else "#2979ff" for v in mh]
    ax2.bar(range(n), mh[:n], color=colors, width=0.7)
    ax2.set_title("MACD", color="white", fontsize=9)

    ax3 = axes[2]
    rs = r["rsi_series"].tail(last_n).reset_index(drop=True)
    ax3.plot(range(n), rs[:n], color="#ffeb3b", linewidth=0.8)
    ax3.axhline(70, color="#ff1744", linewidth=0.5, linestyle="--")
    ax3.axhline(30, color="#4caf50", linewidth=0.5, linestyle="--")
    ax3.set_title("RSI", color="white", fontsize=9)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════
#  급등 사냥 분석 함수
# ═══════════════════════════════════════════════════════

def calc_squeeze(close, period=20):
    bb_upper, bb_mid, bb_lower = calc_bb(close, period)
    bandwidth = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    if len(bandwidth) < 120:
        return False, 0
    bw_now = bandwidth.iloc[-1]
    bw_min = bandwidth.iloc[-120:].min()
    bw_max = bandwidth.iloc[-120:].max()
    if np.isnan(bw_now) or np.isnan(bw_min):
        return False, 0
    squeeze_pct = round((bw_now - bw_min) / (bw_max - bw_min + 1e-9) * 100, 1)
    is_squeeze = squeeze_pct < 15
    return is_squeeze, squeeze_pct

def detect_stealth_accumulation(close, volume, window=20):
    if len(close) < window + 5:
        return False, 0, 0
    recent_close = close.iloc[-window:]
    recent_vol = volume.iloc[-window:]
    price_change = abs((recent_close.iloc[-1] - recent_close.iloc[0]) / (recent_close.iloc[0] + 1e-9) * 100)
    vol_values = recent_vol.values
    x = np.arange(len(vol_values))
    if np.std(vol_values) < 1:
        return False, 0, 0
    slope = np.polyfit(x, vol_values, 1)[0]
    vol_trend = round(slope / (np.mean(vol_values) + 1e-9) * 100, 2)
    is_accumulating = price_change < 5 and vol_trend > 3
    return is_accumulating, round(price_change, 2), round(vol_trend, 2)

def detect_bottom_breakout(close, volume, window=60):
    if len(close) < window + 5:
        return False, 0
    base = close.iloc[-(window+1):-1]
    base_range = (base.max() - base.min()) / (base.mean() + 1e-9) * 100
    today_close = close.iloc[-1]
    yesterday_close = close.iloc[-2]
    today_vol = volume.iloc[-1]
    avg_vol = volume.iloc[-window:-1].mean()
    is_yang = today_close > yesterday_close
    vol_explosion = today_vol / (avg_vol + 1e-9)
    is_breakout = base_range < 15 and is_yang and vol_explosion >= 3
    return is_breakout, round(vol_explosion, 2)

def detect_obv_divergence(close, volume, window=20):
    if len(close) < window + 5:
        return False, 0, 0
    obv = calc_obv(close, volume)
    recent_obv = obv.iloc[-window:]
    recent_close = close.iloc[-window:]
    price_change = abs((recent_close.iloc[-1] - recent_close.iloc[0]) / (recent_close.iloc[0] + 1e-9) * 100)
    obv_change = (recent_obv.iloc[-1] - recent_obv.iloc[0])
    obv_base = abs(recent_obv.iloc[0]) + 1e-9
    obv_pct = round(obv_change / obv_base * 100, 2)
    is_diverging = price_change < 5 and obv_pct > 20
    return is_diverging, round(price_change, 2), obv_pct

def analyze_surge(df, code, name):
    if df is None or len(df) < 60:
        return None
    try:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        price = int(close.iloc[-1])
        prev_price = int(close.iloc[-2]) if len(close) > 1 else price
        change = round((price - prev_price) / (prev_price + 1e-9) * 100, 2)
        vol_mean = volume.rolling(20).mean().iloc[-1]
        vol_ratio = round(volume.iloc[-1] / (vol_mean + 1e-9), 2)
        trade_val = round(price * volume.iloc[-1] / 1e8, 1)
        trade_val_avg = round(price * vol_mean / 1e8, 1)

        score = 0
        signals = []
        details = {}

        trade_val_ratio = round(trade_val / (trade_val_avg + 1e-9), 2)
        if trade_val_ratio >= 10:
            score += 30; signals.append(f"🔥 거래대금 {trade_val_ratio}배 폭발 ({trade_val}억)")
        elif trade_val_ratio >= 5:
            score += 20; signals.append(f"🔥 거래대금 {trade_val_ratio}배 급증 ({trade_val}억)")
        elif trade_val_ratio >= 3:
            score += 12; signals.append(f"📈 거래대금 {trade_val_ratio}배 증가 ({trade_val}억)")
        details["trade_val_ratio"] = trade_val_ratio

        if vol_ratio >= 10:
            score += 20; signals.append(f"🔥 거래량 {vol_ratio}배 폭발")
        elif vol_ratio >= 5:
            score += 12; signals.append(f"📈 거래량 {vol_ratio}배 급증")
        elif vol_ratio >= 3:
            score += 8; signals.append(f"📊 거래량 {vol_ratio}배 증가")

        is_accum, price_chg, vol_trend = detect_stealth_accumulation(close, volume)
        if is_accum:
            score += 25; signals.append(f"🕵️ 세력 매집 포착 (가격 {price_chg}% 횡보, 거래량 추세 +{vol_trend}%)")
        details["accumulation"] = is_accum

        is_breakout, vol_exp = detect_bottom_breakout(close, volume)
        if is_breakout:
            score += 25; signals.append(f"💥 바닥 돌파! 첫 양봉 + 거래량 {vol_exp}배")
        details["breakout"] = is_breakout

        is_squeeze, sq_pct = calc_squeeze(close)
        if is_squeeze:
            score += 20; signals.append(f"🔋 볼린저 스퀴즈 (밴드폭 하위 {sq_pct}%) — 곧 폭발")
        details["squeeze"] = is_squeeze

        is_obv_div, obv_price_chg, obv_pct = detect_obv_divergence(close, volume)
        if is_obv_div:
            score += 20; signals.append(f"🕵️ OBV 매집 (가격 {obv_price_chg}% 횡보, OBV +{obv_pct}%)")
        details["obv_div"] = is_obv_div

        if len(close) > 2:
            prev_change = round((close.iloc[-2] - close.iloc[-3]) / (close.iloc[-3] + 1e-9) * 100, 2)
            if prev_change >= 25:
                score += 15; signals.append(f"🚀 전일 {prev_change:+.1f}% 급등 → 연속 상승 가능")
                if change > 0:
                    score += 10; signals.append(f"📈 오늘도 양봉 ({change:+.2f}%)")
            elif prev_change >= 15:
                score += 8; signals.append(f"📈 전일 {prev_change:+.1f}% 상승")

        if price <= 3000:
            score += 10; signals.append(f"💰 초저가주 ({price:,}원) — 급등 여력 큼")
        elif price <= 5000:
            score += 7; signals.append(f"💰 저가주 ({price:,}원)")
        elif price <= 10000:
            score += 4; signals.append(f"💰 만원 이하 ({price:,}원)")

        if trade_val >= 100:
            score += 10; signals.append(f"💵 거래대금 {trade_val}억 (큰 돈 유입)")
        elif trade_val >= 50:
            score += 6; signals.append(f"💵 거래대금 {trade_val}억")

        rsi = calc_rsi(close)
        rsi_val = round(rsi.iloc[-1], 1) if not np.isnan(rsi.iloc[-1]) else 50
        if rsi_val < 30 and (is_accum or is_obv_div):
            score += 10; signals.append(f"📊 RSI 과매도({rsi_val}) + 매집 신호 = 반등 임박")
        elif rsi_val > 80:
            score -= 5

        macd_line, macd_sig, macd_hist = calc_macd(close)
        if len(macd_hist) > 1 and not np.isnan(macd_hist.iloc[-1]) and not np.isnan(macd_hist.iloc[-2]):
            if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0 and vol_ratio >= 2:
                score += 15; signals.append("📊 MACD 골든크로스 + 거래량 동반")

        if len(close) >= 250:
            low_52w = close.iloc[-250:].min()
            if price <= low_52w * 1.1:
                score += 8; signals.append(f"📉 52주 바닥권 ({low_52w:,}원) → 반등 가능")

        if len(close) > 5:
            chg_5d = round((price / (close.iloc[-6] + 1e-9) - 1) * 100, 2)
        else:
            chg_5d = 0

        if chg_5d >= 30:
            score -= 25
            signals.append(f"⚠️ 5일간 +{chg_5d}% 이미 급등 → 추격매수 위험")
        elif chg_5d >= 20:
            score -= 15
            signals.append(f"⚠️ 5일간 +{chg_5d}% 상승 → 조정 가능성")
        elif chg_5d >= 10:
            score -= 5
            signals.append(f"📊 5일간 +{chg_5d}% 상승 중")

        signal_count = len(signals)
        if abs(chg_5d) <= 5 and signal_count >= 3:
            bonus = min(signal_count * 5, 25)
            score += bonus
            signals.append(f"🎯 가격 횡보({chg_5d:+.1f}%) + 신호 {signal_count}개 = 폭발 직전! (+{bonus}점)")

        if len(close) >= 60:
            low_60d = close.iloc[-60:].min()
            high_60d = close.iloc[-60:].max()
            pos_60d = (price - low_60d) / (high_60d - low_60d + 1e-9) * 100
            if pos_60d <= 20 and signal_count >= 2:
                score += 15
                signals.append(f"🔥 60일 바닥권({pos_60d:.0f}%) + 신호 집중 = 반등 가능성 높음")
            elif pos_60d <= 30 and signal_count >= 3:
                score += 10
                signals.append(f"📈 60일 하단({pos_60d:.0f}%) + 신호 다수")
            elif pos_60d >= 90:
                score -= 10
                signals.append(f"⚠️ 60일 최고점 근처({pos_60d:.0f}%) → 고점 추격 위험")

        score = max(0, min(100, score))

        if score >= 80:
            grade = "S"
            verdict = "🔥 급등 임박"
        elif score >= 60:
            grade = "A"
            verdict = "⚡ 강력 관심"
        elif score >= 40:
            grade = "B"
            verdict = "👀 관심"
        elif score >= 25:
            grade = "C"
            verdict = "🔍 모니터링"
        else:
            grade = "D"
            verdict = "⚪ 해당없음"

        recent = df.tail(20)
        support = int(recent["Low"].min())
        resist = int(recent["High"].max())
        stop_loss = int(support * 0.97)

        buy_reasons = []
        reason_set = set(buy_reasons)

        theme_bonus, matched_themes = get_combined_theme_bonus(code, name)
        if theme_bonus > 0:
            score += theme_bonus
            buy_reasons.append(f"🔥 핫테마: {', '.join(matched_themes)}")

        return {
            "code": code, "name": name, "price": price, "change": change,
            "volume": int(volume.iloc[-1]), "vol_ratio": vol_ratio,
            "trade_val": trade_val, "trade_val_avg": trade_val_avg,
            "trade_val_ratio": trade_val_ratio,
            "rsi": rsi_val, "score": score, "grade": grade, "verdict": verdict,
            "signals": signals, "details": details,
            "support": support, "resist": resist,
            "stop_loss": int(support * 0.97),
            "tp_price": resist,
            "sl_price": int(support * 0.97),
            "theme": get_theme(name),
            "df": df,
            "ema5": close.ewm(span=5).mean(),
            "ema20": close.ewm(span=20).mean(),
            "ema60": close.ewm(span=60).mean(),
            "bb_upper": calc_bb(close)[0],
            "bb_mid": calc_bb(close)[1],
            "bb_lower": calc_bb(close)[2],
            "macd_line": macd_line, "macd_sig": macd_sig, "macd_hist": macd_hist,
            "rsi_series": rsi,
            "mfi_series": calc_mfi(df),
            "obv": calc_obv(close, volume),
            "st_line": None, "st_dir": None,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════
#  코인 가상매매 / 유틸리티
# ═══════════════════════════════════════════════════════

COIN_WL_FILE = "coin_watchlist.json"
COIN_TRADE_FILE = "coin_paper_trades.json"

def load_coin_wl():
    if os.path.exists(COIN_WL_FILE):
        with open(COIN_WL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_coin_wl(wl):
    with open(COIN_WL_FILE, "w", encoding="utf-8") as f:
        json.dump(wl, f, ensure_ascii=False, indent=2)

def get_funding_trend(symbol, limit=10):
    hist = get_funding_history(symbol, limit=limit)
    if not hist or len(hist) < 3:
        return "데이터 부족", "⚪", 0
    consecutive = 1
    direction = "양" if hist[-1] > 0 else "음"
    for i in range(len(hist)-2, -1, -1):
        if (direction == "양" and hist[i] > 0) or (direction == "음" and hist[i] < 0):
            consecutive += 1
        else:
            break
    if consecutive >= 3 and direction == "양":
        return f"펀딩비 {consecutive}회 연속 양수 → 롱 과열, 숏 유리", "🔴", consecutive
    elif consecutive >= 3 and direction == "음":
        return f"펀딩비 {consecutive}회 연속 음수 → 숏 과열, 롱 유리", "🟢", consecutive
    else:
        return f"펀딩비 방향 혼재 (최근 {direction})", "🟡", consecutive

def get_oi_change(symbol, period="5m", limit=30):
    hist = get_oi_history(symbol, period=period, limit=limit)
    if not hist or len(hist) < 2:
        return None, None
    latest = hist[-1]
    prev = hist[0]
    change_pct = round((latest - prev) / (prev + 1e-9) * 100, 2)
    return latest, change_pct

def get_top_trader_ratio(symbol, period="5m"):
    try:
        url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
        r = requests.get(url, params={"symbol": symbol, "period": period, "limit": 1}, timeout=5)
        data = r.json()
        if data:
            return round(float(data[0].get("longShortRatio", 1)), 3)
    except:
        pass
    return None

def calc_liquidation_zones(price, leverage_list=None):
    if leverage_list is None:
        leverage_list = [5, 10, 20, 25, 50, 100]
    zones = []
    for lev in leverage_list:
        liq_long = round(price * (1 - 1/lev), 4)
        liq_short = round(price * (1 + 1/lev), 4)
        zones.append({"leverage": lev, "long_liq": liq_long, "short_liq": liq_short})
    return zones

def load_coin_trades():
    try:
        if os.path.exists(COIN_TRADE_FILE):
            with open(COIN_TRADE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [t for t in data if isinstance(t, dict)]
                elif isinstance(data, dict):
                    converted = []
                    for t in data.get("open", []):
                        if isinstance(t, dict):
                            t["status"] = "open"
                            converted.append(t)
                    for t in data.get("closed", []):
                        if isinstance(t, dict):
                            t["status"] = "closed"
                            converted.append(t)
                    save_coin_trades(converted)
                    return converted
        return []
    except:
        return []

def save_coin_trades(trades):
    try:
        with open(COIN_TRADE_FILE, "w", encoding="utf-8") as f:
            json.dump(trades, f, ensure_ascii=False, indent=2)
    except:
        pass

def open_paper_trade(symbol, side, entry_price, leverage, qty_usdt, tp_price=0, sl_price=0):
    trades = load_coin_trades()
    trade = {
        "id": int(time.time() * 1000),
        "symbol": symbol, "side": side,
        "entry_price": entry_price, "leverage": leverage,
        "qty_usdt": qty_usdt,
        "open_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tp_price": tp_price if tp_price else 0,
        "sl_price": sl_price if sl_price else 0,
        "status": "open",
    }
    trades.append(trade)
    save_coin_trades(trades)
    return trade

def close_paper_trade(trade_id, exit_price):
    trades = load_coin_trades()
    target = None
    for t in trades:
        if isinstance(t, dict) and t.get("id") == trade_id and t.get("status") == "open":
            target = t
            break
    if not target:
        return None
    if target["side"] == "LONG":
        pnl_pct = round((exit_price - target["entry_price"]) / target["entry_price"] * 100 * target["leverage"], 2)
    else:
        pnl_pct = round((target["entry_price"] - exit_price) / target["entry_price"] * 100 * target["leverage"], 2)
    pnl_usdt = round(target.get("qty_usdt", 0) * pnl_pct / 100, 2)
    target.update({
        "status": "closed", "exit_price": exit_price,
        "close_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pnl_pct": pnl_pct, "pnl_usdt": pnl_usdt,
    })
    save_coin_trades(trades)
    return target

def get_paper_stats():
    try:
        trades = load_coin_trades()
        closed = [t for t in trades if isinstance(t, dict) and t.get("status") == "closed"]
        opened = [t for t in trades if isinstance(t, dict) and t.get("status") == "open"]
        if not closed:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0, "open_count": len(opened)}
        wins = [t for t in closed if t.get("pnl_pct", 0) > 0]
        losses = [t for t in closed if t.get("pnl_pct", 0) <= 0]
        total_pnl = round(sum(t.get("pnl_usdt", 0) for t in closed), 2)
        avg_pnl = round(sum(t.get("pnl_pct", 0) for t in closed) / len(closed), 2)
        win_rate = round(len(wins) / len(closed) * 100, 1)
        return {
            "total": len(closed), "wins": len(wins), "losses": len(losses),
            "win_rate": win_rate, "total_pnl": total_pnl, "avg_pnl": avg_pnl,
            "open_count": len(opened),
        }
    except:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0, "open_count": 0}

def check_paper_tpsl():
    """가상매매 TP/SL 자동 체크 (서버용: send_telegram 제거)"""
    trades = load_coin_trades()
    closed_any = False
    open_trades = [t for t in trades if isinstance(t, dict) and t.get("status") == "open"]
    for t in open_trades:
        tp = t.get("tp_price", 0) or 0
        sl = t.get("sl_price", 0) or 0
        if tp <= 0 and sl <= 0:
            continue
        try:
            url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={t['symbol']}"
            resp = requests.get(url, timeout=5)
            current_price = float(resp.json()["price"])
        except:
            continue
        hit = None
        if t["side"] == "LONG":
            if tp > 0 and current_price >= tp:
                hit = "TP"
            elif sl > 0 and current_price <= sl:
                hit = "SL"
        else:
            if tp > 0 and current_price <= tp:
                hit = "TP"
            elif sl > 0 and current_price >= sl:
                hit = "SL"
        if hit:
            result = close_paper_trade(t["id"], current_price)
            if result:
                closed_any = True
    return closed_any


# ── 종목 리스트 캐시 (서버용: stock_list.json 백업) ──
@st.cache_data(ttl=3600)
def get_stocks(market="KOSDAQ"):
    # 방법 1: FinanceDataReader
    if FDR_OK:
        try:
            df = fdr.StockListing(market)
            if df is not None and len(df) > 0:
                if "Code" not in df.columns and "Symbol" in df.columns:
                    df = df.rename(columns={"Symbol": "Code"})
                if "Name" not in df.columns and "종목명" in df.columns:
                    df = df.rename(columns={"종목명": "Name"})
                return df[["Code", "Name"]].dropna().reset_index(drop=True)
        except Exception:
            pass

    # 방법 2: 로컬 JSON 파일 (서버용 백업)
    try:
        with open("stock_list.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if market in data:
            return pd.DataFrame(data[market])
    except Exception:
        pass

    return pd.DataFrame(columns=["Code", "Name"])
# ── PER/PBR 개별 조회 (네이버 API) ──
@st.cache_data(ttl=600)
def get_per_pbr(code):
    per, pbr = 0.0, 0.0
    forward_per = 0.0
    forward_eps = 0.0
    target_price = 0
    foreign_buys = []
    organ_buys = []
    high_52 = 0
    low_52 = 0
    sector = ""
    industry_per = 0.0
    industry_pbr = 0.0
    peers = []
    dividend_yield = 0.0
    dividend_amt = 0
    try:
        url = f"https://m.stock.naver.com/api/stock/{code}/integration"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        data = resp.json()
        for item in data.get("totalInfos", []):
            if item.get("code") == "per":
                val = item.get("value", "N/A").replace("배", "").replace(",", "").strip()
                if val != "N/A":
                    per = float(val)
            elif item.get("code") == "pbr":
                val = item.get("value", "N/A").replace("배", "").replace(",", "").strip()
                if val != "N/A":
                    pbr = float(val)
            elif item.get("code") == "cnsPer":
                val = item.get("value", "N/A").replace("배", "").replace(",", "").strip()
                if val != "N/A":
                    try: forward_per = float(val)
                    except: pass
            elif item.get("code") == "cnsEps":
                val = item.get("value", "N/A").replace("원", "").replace(",", "").strip()
                if val != "N/A":
                    try: forward_eps = float(val)
                    except: pass
            elif item.get("code") == "highPriceOf52Weeks":
                val = item.get("value", "0").replace(",", "").strip()
                try: high_52 = int(val)
                except: pass
            elif item.get("code") == "lowPriceOf52Weeks":
                val = item.get("value", "0").replace(",", "").strip()
                try: low_52 = int(val)
                except: pass
            elif item.get("code") == "dividendYieldRatio":
                val = item.get("value", "0").replace("%", "").replace(",", "").strip()
                try: dividend_yield = float(val)
                except: pass
            elif item.get("code") == "dividend":
                val = item.get("value", "0").replace("원", "").replace(",", "").strip()
                try: dividend_amt = int(float(val))
                except: pass
        for deal in data.get("dealTrendInfos", []):
            try:
                fb = int(deal.get("foreignerPureBuyQuant", "0").replace(",", "").replace("+", ""))
                ob = int(deal.get("organPureBuyQuant", "0").replace(",", "").replace("+", ""))
                foreign_buys.append(fb)
                organ_buys.append(ob)
            except: pass
        foreign_buys = foreign_buys[::-1]
        organ_buys = organ_buys[::-1]
        consensus = data.get("consensusInfo", {})
        if consensus:
            try:
                tp = consensus.get("priceTargetMean", "0")
                if tp: target_price = int(float(str(tp).replace(",", "")))
            except: pass
        for peer in data.get("industryCompareInfo", []):
            try:
                peers.append({
                    "name": peer.get("stockName", ""), "code": peer.get("itemCode", ""),
                    "price": peer.get("closePrice", "0"), "change": peer.get("fluctuationsRatio", "0"),
                    "marketValue": peer.get("marketValue", "0"),
                })
            except: pass
    except: pass
    try:
        url2 = f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}&cn="
        resp2 = requests.get(url2, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=5)
        text2 = resp2.text
        m = re.search(r'코스(?:피|닥)\s+([^<\n]+)', text2)
        if m: sector = m.group(1).strip()
        m2 = re.search(r'업종PER\s*<b[^>]*>\s*([\d.,]+)', text2)
        if m2: industry_per = float(m2.group(1).replace(",", ""))
        m3 = re.search(r'PBR\s*<b[^>]*>\s*([\d.,]+)', text2)
        if m3: industry_pbr = float(m3.group(1).replace(",", ""))
    except: pass

    roe = 0.0; debt_ratio = 0.0; op_margin = 0.0; revenue_growth = 0.0
    try:
        fn_url = f"https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?gicode=A{code}"
        fn_resp = requests.get(fn_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=7)
        fn_text = fn_resp.text
        roe_match = re.search(r'ROE</div></th>\s*<td class="r"[^>]*>([\-\d.,]+)', fn_text)
        if roe_match: roe = float(roe_match.group(1).replace(",", ""))
        rev_match = re.search(r'매출액</div></th>\s*<td class="r"[^>]*title="([\-\d.,]+)"', fn_text)
        op_match = re.search(r'영업이익</div></th>\s*<td class="r"[^>]*title="([\-\d.,]+)"', fn_text)
        if rev_match and op_match:
            rev_val = float(rev_match.group(1).replace(",", ""))
            op_val_raw = float(op_match.group(1).replace(",", ""))
            if rev_val > 0: op_margin = round(op_val_raw / rev_val * 100, 1)
        growth_match = re.search(r'매출액은\s*([\d.]+)%\s*증가', fn_text)
        if growth_match: revenue_growth = float(growth_match.group(1))
        else:
            decline_match = re.search(r'매출액은\s*([\d.]+)%\s*감소', fn_text)
            if decline_match: revenue_growth = -float(decline_match.group(1))
        debt_url = f"https://navercomp.wisereport.co.kr/v2/company/c1030001.aspx?cmp_cd={code}&cn="
        dresp = requests.get(debt_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=5)
        dtext = dresp.text
        debt_match = re.search(r'부채비율.*?<td class="num">([\-\d.,]+)', dtext, re.DOTALL)
        if debt_match: debt_ratio = float(debt_match.group(1).replace(",", ""))
    except: pass

    news_items = []
    try:
        news_url = f"https://stock.naver.com/api/domestic/detail/news?itemCode={code}&page=1&pageSize=5"
        nresp = requests.get(news_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if nresp.status_code == 200:
            ndata = nresp.json()
            for cluster in ndata.get("clusters", []):
                for art in cluster.get("items", []):
                    news_items.append({
                        "title": art.get("title", ""), "body": art.get("body", ""),
                        "office": art.get("officeName", ""), "datetime": art.get("datetime", ""),
                    })
    except: pass

    return per, pbr, foreign_buys, organ_buys, high_52, low_52, sector, industry_per, forward_per, target_price, industry_pbr, peers, news_items, roe, debt_ratio, op_margin, revenue_growth, dividend_yield, dividend_amt


# ── EV/EBITDA 조회 ──
from bs4 import BeautifulSoup

@st.cache_data(ttl=600)
def get_ev_ebitda(code, is_us=False):
    result = {"ev_ebitda": None, "per": None, "pbr": None, "ebitda": None}
    if is_us:
        try:
            import yfinance as yf
            ticker = yf.Ticker(code)
            info = ticker.info
            ev = info.get("enterpriseValue", 0)
            ebitda = info.get("ebitda", 0)
            if ev and ebitda and ebitda > 0:
                result["ev_ebitda"] = round(ev / ebitda, 2)
                result["ebitda"] = ebitda
            if info.get("trailingPE"): result["per"] = round(info["trailingPE"], 2)
            if info.get("priceToBook"): result["pbr"] = round(info["priceToBook"], 2)
        except: pass
    else:
        try:
            url = f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}&cn="
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            resp = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(resp.text, "html.parser")
            for row in soup.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    header = cells[0].get_text(strip=True)
                    if "EV/EBITDA" in header:
                        val = cells[1].get_text(strip=True).replace(",", "")
                        try: result["ev_ebitda"] = float(val)
                        except: pass
        except: pass
    return result


# ── 데이터 fetch ──
def fetch(code, days=200):
    code = str(code).strip()
    try:
        if FDR_OK:
            end = datetime.datetime.now()
            start = end - datetime.timedelta(days=days)
            df = fdr.DataReader(code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            if df is not None and len(df) > 20:
                return df
    except: pass
    if KIS_OK:
        try:
            df = _broker.get_daily_price(code)
            if df is not None and len(df) > 20:
                df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
                df = df.reset_index(drop=True)
                return df
        except Exception as e:
            st.warning(f"KIS 실패 [{code}]: {e}")
    return None

def fetch_minute(code):
    if not KIS_OK: return None
    try:
        url = f"{_broker.api_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        import datetime as dt
        now = dt.datetime.now().strftime("%H%M%S")
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code, "FID_INPUT_HOUR_1": now, "FID_PW_DATA_INCU_YN": "Y"}
        res = requests.get(url, headers=_broker._headers("FHKST03010200"), params=params)
        data = res.json()
        if data.get("rt_cd") == "0":
            rows = data.get("output2", [])
            if rows:
                result = []
                for r in rows:
                    result.append({
                        "Time": r.get("stck_cntg_hour", ""),
                        "Open": int(r.get("stck_oprc", 0)), "High": int(r.get("stck_hgpr", 0)),
                        "Low": int(r.get("stck_lwpr", 0)), "Close": int(r.get("stck_clpr", 0)),
                        "Volume": int(r.get("cntg_vol", 0)),
                    })
                if result: return pd.DataFrame(result[::-1])
    except: pass
    return None


# ── 페이지 설정 ──
st.set_page_config(page_title="급등 예측 탐색기 v24", layout="wide")

# ── CSS ──
st.markdown("""
<style>
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
.buy-blink{animation:blink 1s infinite;color:#ff1744;font-weight:bold;font-size:1.05em}
.sell-blink{animation:blink 1s infinite;color:#2979ff;font-weight:bold;font-size:1.05em}
.card{background:#1a1a2e;border-radius:12px;padding:16px;margin-bottom:12px;border:1px solid #333}
.badge-per-low{background:#4caf50;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.82em}
.badge-per-mid{background:#2196f3;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.82em}
.badge-per-high{background:#f44336;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.82em}
.badge-pbr-low{background:#66bb6a;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.82em}
.badge-pbr-mid{background:#42a5f5;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.82em}
.badge-pbr-high{background:#ef5350;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.82em}
.crown{font-size:1.2em;font-weight:bold}
.support-line{color:#4caf50;font-weight:bold}
.resist-line{color:#ff9800;font-weight:bold}
.stop-line{color:#f44336;font-weight:bold}
</style>
""", unsafe_allow_html=True)

# ── 파일/상수 ──
WL_FILE = "watchlist.json"
HISTORY_FILE = "scan_history.json"
PERF_FILE = "perf_history.json"
SURGE_HISTORY_FILE = "surge_history.json"
MIN_SCORE = 90

SOUND_BUY = "https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg"
SOUND_SELL = "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"

# ── 스타일 ──
STYLES = {
    "단타 (1~3일)": {
        "key": "short", "tp": 3, "sl": 1.5,
        "RSI 과매도 (반등 가능)": 35, "RSI 과매수 (조정 가능)": 72,
        "min_trade_val": 50, "자금 유입 신호": 0, "자금 유출 신호": 85,
        "adx_min": 30, "bb_weight": 15, "macd_weight": 18, "st_weight": 15,
        "vol_weight": 12, "ema_weight": 5,
    },
    "스윙 (3~15일)": {
        "key": "swing", "tp": 10, "sl": 5,
        "RSI 과매도 (반등 가능)": 40, "RSI 과매수 (조정 가능)": 70,
        "min_trade_val": 20, "자금 유입 신호": 0, "자금 유출 신호": 80,
        "adx_min": 25, "bb_weight": 10, "macd_weight": 15, "st_weight": 12,
        "vol_weight": 8, "ema_weight": 10,
    },
    "중장기 (15일+)": {
        "key": "long", "tp": 20, "sl": 10,
        "RSI 과매도 (반등 가능)": 45, "RSI 과매수 (조정 가능)": 75,
        "min_trade_val": 10, "자금 유입 신호": 0, "자금 유출 신호": 75,
        "adx_min": 20, "bb_weight": 5, "macd_weight": 10, "st_weight": 8,
        "vol_weight": 5, "ema_weight": 15,
    },
}

KEY_TO_STYLE = {v["key"]: k for k, v in STYLES.items()}

INDICATOR_WEIGHTS = {
    "macd_cross": 18, "divergence": 17, "vol_explosion": 15, "supertrend": 14,
    "bb_touch": 11, "ema_align": 10, "obv": 9, "adx": 9, "mfi": 8, "rsi": 8,
    "vwap": 5, "ichimoku": 5, "vol_consec": 4,
}

THEME_KW = {
    "2차전지": ["배터리", "리튬", "양극", "음극", "전해", "에코프로", "엘앤에프"],
    "반도체": ["반도체", "칩", "웨이퍼", "팹", "HBM", "SK하이닉스"],
    "AI": ["AI", "인공지능", "딥러닝", "GPT", "엔비디아"],
    "바이오": ["바이오", "제약", "셀트리온", "신약", "임상"],
    "자동차": ["자동차", "EV", "전기차", "현대차", "기아"],
    "로봇": ["로봇", "두산로보", "레인보우"],
    "조선": ["조선", "HD한국조선", "한화오션"],
    "방산": ["방산", "한화에어로", "LIG넥스원"],
    "원전": ["원전", "원자력", "두산에너빌"],
}

EASY = {
    "RSI 과매도 (반등 가능)": "📊 RSI 낮음 → 많이 떨어져서 반등할 수 있어요",
    "RSI 과매수 (조정 가능)": "📊 RSI 높음 → 많이 올라서 쉬어갈 수 있어요",
    "MACD 매수 신호": "📈 MACD 골든크로스 → 상승 흐름으로 바뀌는 신호",
    "MACD 매도 신호": "📉 MACD 데드크로스 → 하락 흐름으로 바뀌는 신호",
    "볼린저 하단 터치 (반등 기대)": "📊 볼린저밴드 하단 → 가격이 바닥 근처예요",
    "볼린저 상단 터치 (조정 기대)": "📊 볼린저밴드 상단 → 가격이 천장 근처예요",
    "추세 구름 돌파 (상승 신호)": "☁️ 일목균형 매수 → 구름대 위로 올라왔어요",
    "추세 구름 이탈 (하락 신호)": "☁️ 일목균형 매도 → 구름대 아래로 내려갔어요",
    "슈퍼트렌드 매수 전환 (상승 시작)": "🔄 슈퍼트렌드 매수 전환 → 추세가 상승으로 바뀜",
    "슈퍼트렌드 매도 전환 (하락 시작)": "🔄 슈퍼트렌드 매도 전환 → 추세가 하락으로 바뀜",
    "거래량 동반 상승 (돈이 들어옴)": "💰 OBV 상승 → 큰손들이 조용히 사들이는 중",
    "거래량 동반 하락 (돈이 빠짐)": "💰 OBV 하락 → 큰손들이 조용히 팔고 있는 중",
    "자금 유입 신호": "💵 MFI 자금유입 → 돈이 이 종목으로 들어오고 있어요",
    "자금 유출 신호": "💵 MFI 자금유출 → 돈이 이 종목에서 빠지고 있어요",
    "강한 상승 힘 감지": "💪 ADX 강한 상승 → 확실한 상승 추세예요",
    "강한 하락 힘 감지": "💪 ADX 강한 하락 → 확실한 하락 추세예요",
    "평균 매수가 위 (매수세 우위)": "🏦 VWAP 위 → 기관·큰손 평균 매수가보다 위에 있어요",
    "평균 매수가 아래 (매도세 우위)": "🏦 VWAP 아래 → 기관·큰손 평균 매수가보다 아래예요",
    "vol3_buy": "📊 거래량 3일 연속 증가 → 관심이 계속 늘고 있어요",
    "vol5_buy": "🔥 거래량 5일 연속 증가 → 강한 관심 집중!",
    "ema_buy": "📈 이동평균 정배열 → 단기·중기·장기 모두 상승 방향",
    "ema_sell": "📉 이동평균 역배열 → 단기·중기·장기 모두 하락 방향",
}


# ── 워치리스트 ──
def load_wl():
    if os.path.exists(WL_FILE):
        with open(WL_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return []

def save_wl(wl):
    with open(WL_FILE, "w", encoding="utf-8") as f: json.dump(wl, f, ensure_ascii=False, indent=2)

def add_to_wl(code, name):
    wl = load_wl()
    if not any(w["code"] == code for w in wl):
        wl.append({"code": code, "name": name}); save_wl(wl)

def remove_from_wl(code):
    wl = load_wl(); wl = [w for w in wl if w["code"] != code]; save_wl(wl)

# ── 스캔 히스토리 ──
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return []

def save_history(hist):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f: json.dump(hist, f, ensure_ascii=False, indent=2)

def add_to_history_direct(style_key, records):
    hist = load_history()
    entry = {"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "style": style_key, "stocks": records}
    hist.insert(0, entry)
    if len(hist) > 50: hist = hist[:50]
    save_history(hist)

def clear_history():
    save_history([])

# ── 성과 추적 ──
def save_perf_snapshot(all_results):
    snapshot = {"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "stocks": []}
    seen = set()
    for skey, rlist in all_results.items():
        for r in rlist:
            if r["code"] not in seen:
                seen.add(r["code"])
                snapshot["stocks"].append({
                    "code": r["code"], "name": r["name"], "score": r["score"], "grade": r["grade"],
                    "verdict": r["verdict"], "price": r["price"], "tp_price": r["tp_price"], "sl_price": r["sl_price"],
                    "buy_reasons": r.get("buy_reasons", []), "sell_reasons": r.get("sell_reasons", []),
                })
    if os.path.exists(PERF_FILE):
        with open(PERF_FILE, "r", encoding="utf-8") as f: perf = json.load(f)
    else: perf = []
    perf.insert(0, snapshot)
    if len(perf) > 30: perf = perf[:30]
    with open(PERF_FILE, "w", encoding="utf-8") as f: json.dump(perf, f, ensure_ascii=False, indent=2)

def load_perf_snapshot():
    if os.path.exists(PERF_FILE):
        with open(PERF_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return []

def load_surge_history():
    if os.path.exists(SURGE_HISTORY_FILE):
        with open(SURGE_HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return []

def save_surge_history(hist):
    with open(SURGE_HISTORY_FILE, "w", encoding="utf-8") as f: json.dump(hist, f, ensure_ascii=False, indent=2)

def add_surge_record(results, country):
    hist = load_surge_history()
    record = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "country": country,
        "count": len(results),
        "stocks": [
            {"code": r["code"], "name": r["name"], "score": r["score"], "grade": r["grade"],
             "price": r["price"], "change": r["change"], "vol_ratio": r["vol_ratio"],
             "trade_val": r["trade_val"], "signals": r["signals"][:5], "market": r.get("market", "")}
            for r in results[:30]
        ],
    }
    hist.insert(0, record)
    if len(hist) > 30: hist = hist[:30]
    save_surge_history(hist)

def generate_perf_report():
    perf = load_perf_snapshot()
    if not perf: return None
    latest = perf[0]
    report_date = latest["date"]
    results = []
    for s in latest["stocks"]:
        code = s["code"]
        try:
            df = fetch(code, days=5)
            if df is None or len(df) < 2: continue
            current_price = int(df["Close"].iloc[-1])
            entry_price = s["price"]
            pnl = round((current_price - entry_price) / entry_price * 100, 2)
            hit_tp = current_price >= s["tp_price"]
            hit_sl = current_price <= s["sl_price"]
            status = "🎯 목표달성" if hit_tp else "🛑 손절" if hit_sl else "📊 보유중"
            results.append({
                "code": code, "name": s["name"], "score": s["score"], "grade": s["grade"],
                "entry_price": entry_price, "current_price": current_price,
                "pnl": pnl, "status": status, "tp_price": s["tp_price"], "sl_price": s["sl_price"],
            })
        except: continue
    if not results: return None
    wins = len([r for r in results if r["pnl"] > 0])
    losses = len([r for r in results if r["pnl"] <= 0])
    avg_pnl = round(sum(r["pnl"] for r in results) / len(results), 2)
    best = max(results, key=lambda x: x["pnl"])
    worst = min(results, key=lambda x: x["pnl"])
    return {
        "date": report_date, "results": sorted(results, key=lambda x: x["pnl"], reverse=True),
        "total": len(results), "wins": wins, "losses": losses,
        "win_rate": round(wins / len(results) * 100, 1) if results else 0,
        "avg_pnl": avg_pnl, "best": best, "worst": worst,
    }

def generate_indicator_report():
    perf = load_perf_snapshot()
    if not perf: return None
    indicator_stats = {}
    for snapshot in perf:
        for s in snapshot["stocks"]:
            code = s["code"]; entry_price = s["price"]
            buy_reasons = s.get("buy_reasons", [])
            if not buy_reasons: continue
            try:
                df = fetch(code, days=10)
                if df is None or len(df) < 2: continue
                current_price = int(df["Close"].iloc[-1])
                pnl = round((current_price - entry_price) / (entry_price + 1e-9) * 100, 2)
                win = pnl > 0
                for reason in buy_reasons:
                    if reason not in indicator_stats:
                        indicator_stats[reason] = {"total": 0, "wins": 0, "total_pnl": 0}
                    indicator_stats[reason]["total"] += 1
                    if win: indicator_stats[reason]["wins"] += 1
                    indicator_stats[reason]["total_pnl"] += pnl
            except: continue
    if not indicator_stats: return None
    results = []
    for reason, stats in indicator_stats.items():
        if stats["total"] >= 2:
            results.append({
                "indicator": reason, "total": stats["total"], "wins": stats["wins"],
                "win_rate": round(stats["wins"] / stats["total"] * 100, 1),
                "avg_pnl": round(stats["total_pnl"] / stats["total"], 2),
            })
    results.sort(key=lambda x: x["win_rate"], reverse=True)
    return results


# ── 백테스트 ──
BACKTEST_FILE = "backtest_results.json"

def save_backtest(results):
    with open(BACKTEST_FILE, "w", encoding="utf-8") as f: json.dump(results, f, ensure_ascii=False, indent=2)

def load_backtest():
    if os.path.exists(BACKTEST_FILE):
        with open(BACKTEST_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return []

def run_backtest(days_back=120, hold_days=5, max_stocks=200, _progress_callback=None, cfg=None):
    results = []
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    try:
        stocks_df = fdr.StockListing("KOSDAQ")
        if stocks_df is None or len(stocks_df) == 0: return []
        codes = stocks_df["Code"].tolist()[:max_stocks]
        names = dict(zip(stocks_df["Code"], stocks_df["Name"]))
    except: return []
    test_dates = []
    current = start_date
    while current < end_date - datetime.timedelta(days=hold_days + 5):
        test_dates.append(current); current += datetime.timedelta(days=10)
    total_tasks = len(test_dates) * len(codes); done = 0
    for test_date in test_dates:
        for code in codes:
            done += 1
            if _progress_callback and done % 50 == 0: _progress_callback(done / total_tasks)
            try:
                fetch_start = (test_date - datetime.timedelta(days=250)).strftime("%Y-%m-%d")
                fetch_end = (test_date + datetime.timedelta(days=hold_days + 3)).strftime("%Y-%m-%d")
                df = fdr.DataReader(code, fetch_start, fetch_end)
                if df is None or len(df) < 60: continue
                df_analysis = df[df.index <= test_date.strftime("%Y-%m-%d")]
                if len(df_analysis) < 60: continue
                name = names.get(code, code)
                r = analyze(df_analysis, code, name, cfg)
                if r["score"] < MIN_SCORE: continue
                if len(r.get("buy_reasons", [])) < 3: continue
                df_after = df[df.index > test_date.strftime("%Y-%m-%d")]
                if len(df_after) < hold_days: continue
                entry_price = r["price"]
                exit_price = int(df_after["Close"].iloc[hold_days - 1])
                pnl = round((exit_price - entry_price) / (entry_price + 1e-9) * 100, 2)
                results.append({
                    "date": test_date.strftime("%Y-%m-%d"), "code": code, "name": name,
                    "score": r["score"], "grade": r["grade"],
                    "buy_reasons": r.get("buy_reasons", []), "sell_reasons": r.get("sell_reasons", []),
                    "entry_price": entry_price, "exit_price": exit_price, "pnl": pnl, "win": pnl > 0,
                })
                if len(results) % 50 == 0: save_backtest(results)
            except: continue
    if results: save_backtest(results)
    return results

def analyze_backtest(results):
    if not results: return None
    indicator_stats = {}
    for r in results:
        for reason in r.get("buy_reasons", []):
            if reason not in indicator_stats:
                indicator_stats[reason] = {"total": 0, "wins": 0, "total_pnl": 0}
            indicator_stats[reason]["total"] += 1
            if r["win"]: indicator_stats[reason]["wins"] += 1
            indicator_stats[reason]["total_pnl"] += r["pnl"]
    report = []
    for reason, stats in indicator_stats.items():
        if stats["total"] >= 3:
            report.append({
                "indicator": reason, "total": stats["total"], "wins": stats["wins"],
                "win_rate": round(stats["wins"] / stats["total"] * 100, 1),
                "avg_pnl": round(stats["total_pnl"] / stats["total"], 2),
            })
    report.sort(key=lambda x: x["win_rate"], reverse=True)
    return report

def analyze_combo(results, min_combo=2):
    if not results: return None
    from itertools import combinations
    combo_stats = {}
    for r in results:
        reasons = sorted(r.get("buy_reasons", []))
        if len(reasons) < min_combo: continue
        for combo in combinations(reasons, min_combo):
            key = " + ".join(combo)
            if key not in combo_stats:
                combo_stats[key] = {"total": 0, "wins": 0, "total_pnl": 0}
            combo_stats[key]["total"] += 1
            if r["win"]: combo_stats[key]["wins"] += 1
            combo_stats[key]["total_pnl"] += r["pnl"]
    report = []
    for combo, stats in combo_stats.items():
        if stats["total"] >= 3:
            report.append({
                "combo": combo, "total": stats["total"], "wins": stats["wins"],
                "win_rate": round(stats["wins"] / stats["total"] * 100, 1),
                "avg_pnl": round(stats["total_pnl"] / stats["total"], 2),
            })
    report.sort(key=lambda x: x["win_rate"], reverse=True)
    return report


# ── 핫 테마 ──
@st.cache_data(ttl=1800)
def get_hot_themes(top_n=10):
    hot_themes = []
    try:
        url = "https://finance.naver.com/sise/theme.naver"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("table.type_1 tr")
        count = 0
        for row in rows:
            if count >= top_n: break
            cols = row.select("td")
            if len(cols) < 6: continue
            theme_link = cols[0].select_one("a")
            rate_span = cols[1].select_one("span")
            if not theme_link or not rate_span: continue
            theme_name = theme_link.text.strip()
            theme_url = "https://finance.naver.com" + theme_link["href"]
            change_rate = rate_span.text.strip()
            theme_codes = []
            try:
                r2 = requests.get(theme_url, headers=headers, timeout=10)
                soup2 = BeautifulSoup(r2.text, "html.parser")
                stock_links = soup2.select("a[href*='code=']")
                for sl in stock_links:
                    href = sl["href"]
                    if "code=" in href:
                        code = href.split("code=")[-1][:6]
                        if code.isdigit() and len(code) == 6: theme_codes.append(code)
                theme_codes = list(set(theme_codes))
            except: pass
            hot_themes.append({"name": theme_name, "change_rate": change_rate, "codes": theme_codes})
            count += 1
    except: pass
    return hot_themes

def get_hot_theme_bonus(code, hot_themes=None):
    if not hot_themes: return 0, []
    matched_themes = []
    for theme in hot_themes:
        if code in theme.get("codes", []): matched_themes.append(theme["name"])
    if len(matched_themes) >= 2: return 10, matched_themes
    elif len(matched_themes) == 1: return 6, matched_themes
    return 0, matched_themes

MANUAL_THEME_FILE = "manual_hot_themes.json"

def load_manual_themes():
    if os.path.exists(MANUAL_THEME_FILE):
        with open(MANUAL_THEME_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return {"enabled": True, "auto": True, "manual_keywords": []}

def save_manual_themes(data):
    with open(MANUAL_THEME_FILE, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

def get_combined_theme_bonus(code, name):
    settings = load_manual_themes()
    if not settings.get("enabled", True): return 0, []
    matched = []
    if settings.get("auto", True):
        hot_themes = get_hot_themes()
        bonus, auto_matched = get_hot_theme_bonus(code, hot_themes)
        matched.extend(auto_matched)
    for kw in settings.get("manual_keywords", []):
        if kw in name: matched.append(f"수동:{kw}")
    matched = list(set(matched))
    if len(matched) >= 2: return 10, matched
    elif len(matched) == 1: return 6, matched
    return 0, matched


# ── 뉴스 / 테마 / 시장국면 / 지지저항 ──
@st.cache_data(ttl=300)
def get_stock_news(code, name):
    news_list = []
    try:
        search_query = name.replace(" ", "+")
        url = f"https://search.naver.com/search.naver?where=news&query={search_query}+주식&sort=1"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=5)
        text = resp.text
        titles = re.findall(r'class="news_tit"[^>]*title="([^"]*)"', text)
        links = re.findall(r'class="news_tit"[^>]*href="([^"]*)"', text)
        sources = re.findall(r'class="info press"[^>]*>([^<]*)<', text)
        for i in range(min(5, len(titles))):
            news_list.append({
                "title": titles[i] if i < len(titles) else "",
                "link": links[i] if i < len(links) else "",
                "source": sources[i] if i < len(sources) else "",
            })
    except: pass
    return news_list

def get_theme(name):
    for theme, keywords in THEME_KW.items():
        for kw in keywords:
            if kw in name: return theme
    return ""

def market_phase():
    if not FDR_OK: return "확인불가", "⚪"
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=60)
        df = fdr.DataReader("KS11", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is None or len(df) < 20: return "확인불가", "⚪"
        c = df["Close"].iloc[-1]
        ma20 = df["Close"].rolling(20).mean().iloc[-1]
        if c > ma20 * 1.02: return "상승장", "🟢"
        elif c < ma20 * 0.98: return "하락장", "🔴"
        else: return "횡보장", "🟡"
    except: return "확인불가", "⚪"

def calc_support_resist(df, period=20):
    recent = df.tail(period)
    support = int(recent["Low"].min())
    resist = int(recent["High"].max())
    bb_upper, bb_mid, bb_lower = calc_bb(df["Close"])
    bb_low = bb_lower.iloc[-1]; bb_high = bb_upper.iloc[-1]
    if not np.isnan(bb_low): support = max(support, int(bb_low))
    if not np.isnan(bb_high): resist = min(resist, int(bb_high))
    if support >= resist:
        support = int(recent["Low"].min()); resist = int(recent["High"].max())
    stop_loss = int(support * 0.97)
    return support, resist, stop_loss


# ── 세션 초기화 ──
for key, val in [
    ("all_scan_results", {}), ("cached_data", {}), ("prev_signals", {}),
    ("hist_prev_signals", {}), ("sound_counter", 0), ("scan_done", False), ("scan_running", False),
]:
    if key not in st.session_state: st.session_state[key] = val

# ─── 사이드바 ─────────────
with st.sidebar:
    st.markdown("## 🔥 급등 예측 탐색기")
    st.caption("v24.0 PRO | AI 매매 + 감시 투자")
    st.divider()

    # ── Gemini API 키 입력 (서버용) ──
    st.markdown("**🔑 Gemini API 키**")
    _saved_key = ""
    try:
        if os.path.exists("gemini_key.txt"):
            with open("gemini_key.txt", "r") as f:
                _saved_key = f.read().strip()
    except: pass
    _user_key = st.text_input("API 키 입력", value=_saved_key, type="password", placeholder="AIzaSy...", key="gemini_key_input", label_visibility="collapsed")
    if _user_key:
        st.session_state["gemini_api_key"] = _user_key
        init_gemini(_user_key)
        try:
            with open("gemini_key.txt", "w") as f:
                f.write(_user_key)
        except: pass
    elif _saved_key:
        init_gemini(_saved_key)
    st.caption("[무료 API 키 발급받기](https://aistudio.google.com/apikey)")
    st.sidebar.markdown(f"**🧠 Gemini AI:** {'🟢 연결됨' if GEMINI_OK else '⚪ 미연결'}")
    st.divider()

    # ── 이용 가이드 ──
    with st.sidebar.expander("📖 이용 가이드", expanded=False):
        st.markdown("""
**🚀 급등 예측 탐색기 사용법**

**📡 스캔/검색**
- 종목명 입력 → 개별 분석
- 전체 스캔 → AI가 600개 종목 자동 분석
- 점수 85점 이상만 결과에 표시

**👀 내 종목 감시**
- 관심 종목을 등록하고 실시간 모니터링
- 목표가/손절가 알림 설정 가능

**🪙 코인 선물**
- 바이낸스 코인 선물 50개 자동 분석
- 롱/숏 신호 + 가상매매 지원

**🎯 투자 브리핑**
- AI에게 자유롭게 질문
- 종목명 포함 시 실시간 데이터 자동 수집
- 실시간 뉴스/공시 기반 답변

**📊 성과 리포트**
- 스캔 결과의 수익률 추적

**💡 팁**
- Gemini API 키 입력 필수!
- [무료 API 키 발급](https://aistudio.google.com/apikey)
- 시장(KOSPI/KOSDAQ) 전환은 사이드바에서
        """)


    menu = st.radio("메뉴", [
        "📡 스캔/검색",
        "👀 내 종목 감시",
        "📊 성과 리포트",
        "📜 스캔 기록",
        "🔥 섹터 동반 상승",
        "🪙 코인 선물",
        "💰 매매",
        "🚀 급등 사냥",
        "🌊 외국인 수급 추적",
        "🎯 투자 브리핑",
    ], label_visibility="collapsed")


    st.divider()
    st.markdown("**시장**")
    market = st.selectbox("시장", ["KOSDAQ", "KOSPI", "NASDAQ", "S&P500", "NYSE"], index=0, label_visibility="collapsed")
    sound_on = st.checkbox("🔊 사운드 알림", value=False)

    # ── 핫 테마 설정 ──
    with st.expander("🔥 핫 테마 설정", expanded=False):
        theme_settings = load_manual_themes()
        theme_enabled = st.checkbox("테마 보너스 활성화", value=theme_settings.get("enabled", True), key="theme_enabled")
        theme_auto = st.checkbox("자동 감지 (네이버 실시간)", value=theme_settings.get("auto", True), key="theme_auto")
        if theme_auto:
            try:
                hot = get_hot_themes(top_n=5)
                if hot:
                    st.markdown("**📈 오늘의 핫 테마 TOP 5**")
                    for h in hot: st.caption(f"🔥 {h['name']} ({h['change_rate']})")
            except: pass
        st.markdown("**✏️ 수동 키워드 추가**")
        st.caption("종목명에 포함된 키워드로 매칭")
        current_keywords = theme_settings.get("manual_keywords", [])
        new_kw = st.text_input("키워드 입력", placeholder="예: 로봇, AI, 전력", key="new_theme_kw")
        if st.button("➕ 추가", key="add_theme_kw"):
            if new_kw and new_kw not in current_keywords:
                current_keywords.append(new_kw)
                theme_settings["manual_keywords"] = current_keywords
                theme_settings["enabled"] = theme_enabled
                theme_settings["auto"] = theme_auto
                save_manual_themes(theme_settings); st.rerun()
        if current_keywords:
            st.markdown("**현재 수동 키워드:**")
            for i, kw in enumerate(current_keywords):
                col_kw, col_del = st.columns([3, 1])
                with col_kw: st.caption(f"🏷️ {kw}")
                with col_del:
                    if st.button("❌", key=f"del_kw_{i}"):
                        current_keywords.pop(i)
                        theme_settings["manual_keywords"] = current_keywords
                        save_manual_themes(theme_settings); st.rerun()
        if theme_enabled != theme_settings.get("enabled") or theme_auto != theme_settings.get("auto"):
            theme_settings["enabled"] = theme_enabled
            theme_settings["auto"] = theme_auto
            save_manual_themes(theme_settings)

    # ── 커스텀 알림 조건 ──
    with st.expander("🔔 커스텀 알림 조건", expanded=False):
        preset = st.selectbox("📋 프리셋 선택", ["직접 설정", "💎 저평가 우량주", "🚀 급등 후보", "📉 바닥 반등 종목"], key="alert_preset")
        if "prev_preset" not in st.session_state: st.session_state["prev_preset"] = "직접 설정"
        if preset != st.session_state["prev_preset"]:
            st.session_state["prev_preset"] = preset
            if preset == "💎 저평가 우량주":
                st.session_state.update({"alert_per": 15.0, "alert_pbr": 1.5, "alert_score": 60, "alert_rsi_low": 0, "alert_vol": 0.0, "alert_pattern": []})
            elif preset == "🚀 급등 후보":
                st.session_state.update({"alert_per": 0.0, "alert_pbr": 0.0, "alert_score": 70, "alert_rsi_low": 0, "alert_vol": 2.0, "alert_pattern": []})
            elif preset == "📉 바닥 반등 종목":
                st.session_state.update({"alert_per": 0.0, "alert_pbr": 0.0, "alert_score": 50, "alert_rsi_low": 35, "alert_vol": 0.0, "alert_pattern": ["W자 바닥", "N자 반등"]})
            else:
                st.session_state.update({"alert_per": 0.0, "alert_pbr": 0.0, "alert_score": 0, "alert_rsi_low": 0, "alert_vol": 0.0, "alert_pattern": []})
            st.rerun()
        alert_per = st.number_input("PER 이하", min_value=0.0, max_value=100.0, step=1.0, key="alert_per", help="0이면 비활성")
        alert_pbr = st.number_input("PBR 이하", min_value=0.0, max_value=50.0, step=0.5, key="alert_pbr", help="0이면 비활성")
        alert_score = st.number_input("점수 이상", min_value=0, max_value=100, step=5, key="alert_score", help="0이면 비활성")
        alert_rsi_low = st.number_input("RSI 이하 (과매도)", min_value=0, max_value=100, step=5, key="alert_rsi_low", help="0이면 비활성")
        alert_vol = st.number_input("거래량 배율 이상", min_value=0.0, max_value=50.0, step=0.5, key="alert_vol", help="0이면 비활성")
        alert_pattern = st.multiselect("패턴 필터", ["W자 바닥", "N자 반등", "골든크로스"], key="alert_pattern")

    # ── 공포/탐욕 지수 ──
    try:
        fg_res = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5).json()
        fg_val = int(fg_res["data"][0]["value"])
        fg_label = fg_res["data"][0]["value_classification"]
        if fg_val <= 25: fg_emoji, fg_color, fg_kr = "😱", "#f44336", "극도의 공포"
        elif fg_val <= 45: fg_emoji, fg_color, fg_kr = "😰", "#ff9800", "공포"
        elif fg_val <= 55: fg_emoji, fg_color, fg_kr = "😐", "#ffeb3b", "중립"
        elif fg_val <= 75: fg_emoji, fg_color, fg_kr = "😊", "#8bc34a", "탐욕"
        else: fg_emoji, fg_color, fg_kr = "🤑", "#4caf50", "극도의 탐욕"
        st.sidebar.markdown(
            f'<div style="background:#1a1a2e;padding:10px;border-radius:8px;border:1px solid {fg_color};text-align:center">'
            f'{fg_emoji} <b>크립토 공포/탐욕 지수</b><br>'
            f'<span style="font-size:1.8em;color:{fg_color}">{fg_val}</span><br>'
            f'<span style="color:{fg_color}">{fg_kr} ({fg_label})</span></div>', unsafe_allow_html=True)
    except: pass

    # ── 미국 경제 지표 ──
    try:
        eco_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        cpi_val = ""; unemp_val = ""; rate_val = ""
        import re as _re
        try:
            cpi_resp = requests.get("https://tradingeconomics.com/united-states/inflation-cpi", headers=eco_headers, timeout=5)
            cpi_m = _re.search(r'Inflation Rate in the United States (?:increased|decreased|remained unchanged at) (?:to )?([\d.]+)', cpi_resp.text)
            if cpi_m: cpi_val = cpi_m.group(1)
        except: pass
        try:
            unemp_resp = requests.get("https://tradingeconomics.com/united-states/unemployment-rate", headers=eco_headers, timeout=5)
            unemp_m = _re.search(r'Unemployment Rate in the United States (?:increased|decreased|remained unchanged at) (?:to )?([\d.]+)', unemp_resp.text)
            if unemp_m: unemp_val = unemp_m.group(1)
        except: pass
        try:
            rate_resp = requests.get("https://tradingeconomics.com/united-states/interest-rate", headers=eco_headers, timeout=5)
            rate_m = _re.search(r'last recorded at ([\d.]+) percent', rate_resp.text)
            if rate_m: rate_val = rate_m.group(1)
        except: pass

        if cpi_val or unemp_val or rate_val:
            cpi_f = float(cpi_val) if cpi_val else 0
            if cpi_f >= 4: cpi_color, cpi_icon, cpi_label = "#f44336", "🔴", "인플레 과열"
            elif cpi_f >= 3: cpi_color, cpi_icon, cpi_label = "#ff9800", "🟡", "인플레 주의"
            elif cpi_f >= 2: cpi_color, cpi_icon, cpi_label = "#4caf50", "🟢", "안정적"
            else: cpi_color, cpi_icon, cpi_label = "#2196f3", "🔵", "디플레 우려"
            unemp_f = float(unemp_val) if unemp_val else 0
            if unemp_f >= 6: unemp_color, unemp_icon, unemp_label = "#f44336", "🔴", "고용 위기"
            elif unemp_f >= 5: unemp_color, unemp_icon, unemp_label = "#ff9800", "🟡", "주의"
            elif unemp_f >= 4: unemp_color, unemp_icon, unemp_label = "#ff9800", "🟡", "보통"
            else: unemp_color, unemp_icon, unemp_label = "#4caf50", "🟢", "양호"
            rate_f = float(rate_val) if rate_val else 0
            if rate_f >= 5: rate_color, rate_icon, rate_label = "#f44336", "🔴", "긴축"
            elif rate_f >= 4: rate_color, rate_icon, rate_label = "#ff9800", "🟡", "중립~긴축"
            elif rate_f >= 2: rate_color, rate_icon, rate_label = "#ff9800", "🟡", "중립"
            else: rate_color, rate_icon, rate_label = "#4caf50", "🟢", "완화적"
            if cpi_f >= 3.5 and unemp_f < 5: outlook, outlook_color = "📈 금리 인상 가능성 ↑", "#f44336"
            elif cpi_f < 2.5 and unemp_f >= 4.5: outlook, outlook_color = "📉 금리 인하 가능성 ↑", "#4caf50"
            else: outlook, outlook_color = "➡️ 금리 동결 전망", "#ff9800"
            st.sidebar.markdown(
                f'<div style="background:#1a1a2e;padding:10px;border-radius:8px;border:1px solid #444;margin-top:10px">'
                f'🇺🇸 <b>미국 경제 지표</b><br>'
                f'<table style="width:100%;font-size:0.85em;margin-top:5px">'
                f'<tr><td>{cpi_icon} CPI</td><td style="text-align:right;color:{cpi_color};font-weight:bold">{cpi_val}%</td><td style="text-align:right;color:{cpi_color};font-size:0.8em">{cpi_label}</td></tr>'
                f'<tr><td>{unemp_icon} 실업률</td><td style="text-align:right;color:{unemp_color};font-weight:bold">{unemp_val}%</td><td style="text-align:right;color:{unemp_color};font-size:0.8em">{unemp_label}</td></tr>'
                f'<tr><td>💵 기준금리</td><td style="text-align:right;color:{rate_color};font-weight:bold">{rate_val}%</td><td style="text-align:right;color:{rate_color};font-size:0.8em">{rate_label}</td></tr>'
                f'</table>'
                f'<div style="margin-top:8px;padding:5px;background:#111;border-radius:5px;text-align:center;color:{outlook_color};font-weight:bold">{outlook}</div>'
                f'</div>', unsafe_allow_html=True)
    except: pass

    # ── 한국 경제지표 + VIX + 미국 10년물 국채 + 매크로 점수 ──
    try:
        import re as _re2
        bok_rate = ""; krw_val = ""; vix_val = ""; bond_val = ""; short_val = ""
        try:
            _te_h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            bok_page = requests.get("https://tradingeconomics.com/south-korea/interest-rate", headers=_te_h, timeout=5).text
            bok_match = _re2.search(r'last recorded at ([\d.]+) percent', bok_page)
            if bok_match: bok_rate = bok_match.group(1)
        except: pass
        try:
            _er_resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5).json()
            if _er_resp.get("result") == "success":
                _krw_rate = _er_resp["rates"].get("KRW")
                if _krw_rate: krw_val = str(round(_krw_rate, 1))
        except: pass
        try:
            import yfinance as _yf
            _vix_tk = _yf.Ticker("^VIX"); _vix_hist = _vix_tk.history(period="1d")
            if not _vix_hist.empty: vix_val = str(round(_vix_hist["Close"].iloc[-1], 2))
        except: pass
        try:
            import yfinance as _yf2
            _tnx_tk = _yf2.Ticker("^TNX"); _tnx_hist = _tnx_tk.history(period="1d")
            if not _tnx_hist.empty: bond_val = str(round(_tnx_hist["Close"].iloc[-1], 2))
        except: pass

        bok_f = float(bok_rate) if bok_rate else 0
        if bok_f <= 2.0: bok_icon, bok_color, bok_label = "🟢", "#4caf50", "저금리 (호재)"
        elif bok_f <= 3.0: bok_icon, bok_color, bok_label = "🟡", "#ff9800", "보통"
        else: bok_icon, bok_color, bok_label = "🔴", "#f44336", "고금리 (부담)"
        krw_f = float(krw_val) if krw_val else 0
        if krw_f <= 1250: krw_icon, krw_color, krw_label = "🟢", "#4caf50", "원화 강세"
        elif krw_f <= 1350: krw_icon, krw_color, krw_label = "🟡", "#ff9800", "보통"
        elif krw_f <= 1450: krw_icon, krw_color, krw_label = "🟠", "#ff5722", "원화 약세 주의"
        else: krw_icon, krw_color, krw_label = "🔴", "#f44336", "원화 급락"
        vix_f = float(vix_val) if vix_val else 0
        if vix_f <= 15: vix_icon, vix_color, vix_label = "😎", "#4caf50", "안정"
        elif vix_f <= 20: vix_icon, vix_color, vix_label = "😐", "#ff9800", "보통"
        elif vix_f <= 30: vix_icon, vix_color, vix_label = "😰", "#ff5722", "불안"
        else: vix_icon, vix_color, vix_label = "😱", "#f44336", "공포"
        bond_f = float(bond_val) if bond_val else 0
        if bond_f <= 3.5: bond_icon, bond_color, bond_label = "🟢", "#4caf50", "저금리 (주식 호재)"
        elif bond_f <= 4.0: bond_icon, bond_color, bond_label = "🟡", "#ff9800", "보통"
        elif bond_f <= 4.5: bond_icon, bond_color, bond_label = "🟠", "#ff5722", "부담"
        else: bond_icon, bond_color, bond_label = "🔴", "#f44336", "고금리 (주식 악재)"

        # 매크로 보너스
        macro_bonus = 0; macro_reasons = []
        try:
            _cpi_f = float(cpi_val) if cpi_val else 0; _unemp_f = float(unemp_val) if unemp_val else 0
            if _cpi_f < 2.5 and _unemp_f >= 4.5: macro_bonus += 3; macro_reasons.append("CPI 안정+금리 인하 기대")
            elif _cpi_f >= 4.0 and _unemp_f < 4.0: macro_bonus -= 3; macro_reasons.append("CPI 과열+긴축 지속")
        except: pass
        if bok_f > 0:
            if bok_f <= 2.0: macro_bonus += 2; macro_reasons.append("한국 저금리")
            elif bok_f >= 3.5: macro_bonus -= 2; macro_reasons.append("한국 고금리")
        if vix_f > 0:
            if vix_f <= 15: macro_bonus += 2; macro_reasons.append("VIX 안정")
            elif vix_f >= 30: macro_bonus -= 3; macro_reasons.append("VIX 공포")
            elif vix_f >= 25: macro_bonus -= 2; macro_reasons.append("VIX 불안")
        if bond_f > 0:
            if bond_f <= 3.5: macro_bonus += 2; macro_reasons.append("미국채 저금리")
            elif bond_f >= 4.5: macro_bonus -= 2; macro_reasons.append("미국채 고금리")
        if krw_f > 0:
            if krw_f <= 1200: macro_bonus += 1; macro_reasons.append("원화 강세")
            elif krw_f >= 1400: macro_bonus -= 1; macro_reasons.append("원화 약세")
        macro_bonus = max(-5, min(5, macro_bonus))
        st.session_state["macro_bonus"] = macro_bonus
        st.session_state["macro_reasons"] = macro_reasons

        kr_rows = ""
        if bok_rate: kr_rows += f'<tr><td>{bok_icon} 기준금리</td><td style="text-align:right;color:{bok_color};font-weight:bold">{bok_rate}%</td><td style="text-align:right;color:{bok_color};font-size:0.8em">{bok_label}</td></tr>'
        if krw_val: kr_rows += f'<tr><td>{krw_icon} 원/달러</td><td style="text-align:right;color:{krw_color};font-weight:bold">{krw_f:,.0f}원</td><td style="text-align:right;color:{krw_color};font-size:0.8em">{krw_label}</td></tr>'
        if kr_rows:
            st.sidebar.markdown(f'<div style="background:#1a1a2e;padding:10px;border-radius:8px;border:1px solid #444;margin-top:10px">🇰🇷 <b>한국 경제 지표</b><br><table style="width:100%;font-size:0.85em;margin-top:5px">{kr_rows}</table></div>', unsafe_allow_html=True)
        gl_rows = ""
        if vix_val: gl_rows += f'<tr><td>{vix_icon} VIX</td><td style="text-align:right;color:{vix_color};font-weight:bold">{vix_val}</td><td style="text-align:right;color:{vix_color};font-size:0.8em">{vix_label}</td></tr>'
        if bond_val: gl_rows += f'<tr><td>{bond_icon} 미국 10Y</td><td style="text-align:right;color:{bond_color};font-weight:bold">{bond_val}%</td><td style="text-align:right;color:{bond_color};font-size:0.8em">{bond_label}</td></tr>'
        if gl_rows:
            st.sidebar.markdown(f'<div style="background:#1a1a2e;padding:10px;border-radius:8px;border:1px solid #444;margin-top:10px">🌐 <b>글로벌 리스크 지표</b><br><table style="width:100%;font-size:0.85em;margin-top:5px">{gl_rows}</table></div>', unsafe_allow_html=True)
        if macro_bonus > 0: m_color, m_icon, m_label = "#4caf50", "🟢", "우호적"
        elif macro_bonus < 0: m_color, m_icon, m_label = "#f44336", "🔴", "비우호적"
        else: m_color, m_icon, m_label = "#ff9800", "🟡", "중립"
        macro_detail = ", ".join(macro_reasons) if macro_reasons else "특이사항 없음"
        st.sidebar.markdown(
            f'<div style="background:#1a1a2e;padding:10px;border-radius:8px;border:1px solid {m_color};margin-top:10px;text-align:center">'
            f'📊 <b>매크로 점수</b><br><span style="font-size:1.5em;color:{m_color};font-weight:bold">{macro_bonus:+d}점</span><br>'
            f'{m_icon} {m_label}<br><span style="font-size:0.8em;color:#888">{macro_detail}</span></div>', unsafe_allow_html=True)

        # Gemini AI 경제 해석
        if GEMINI_OK and "eco_ai_cache" not in st.session_state:
            try:
                _nums = []
                if cpi_val: _nums.append(f"미국CPI {cpi_val}%")
                if unemp_val: _nums.append(f"실업률 {unemp_val}%")
                if rate_val: _nums.append(f"미국금리 {rate_val}%")
                if bok_rate: _nums.append(f"한국금리 {bok_rate}%")
                if krw_val: _nums.append(f"환율 {krw_f:,.0f}원")
                if vix_val: _nums.append(f"VIX {vix_val}")
                if bond_val: _nums.append(f"10년물 {bond_val}%")
                if _nums:
                    _prompt = ("초보 투자자를 위한 경제 해설가로서 아래 지표를 3줄로 설명하세요.\n"
                        f"지표: {', '.join(_nums)}\n1줄: 경제 상황 초등학생 수준 요약 (이모지)\n2줄: 주식 투자 영향 (이모지)\n3줄: 지금 사도 될까? 의견 (이모지)\n한국어, 각 줄 줄바꿈")
                    _resp = gemini_model.generate_content(_prompt)
                    _txt = _resp.text.strip().replace("\n", "<br>")
                    st.session_state["eco_ai_cache"] = _txt
            except: pass
        if "eco_ai_cache" in st.session_state:
            st.sidebar.markdown(
                f'<div style="background:#1a2e1a;padding:10px;border-radius:8px;border:1px solid #4caf50;margin-top:10px">'
                f'🤖 <b>AI 경제 해석</b><br><br><span style="font-size:0.85em">{st.session_state["eco_ai_cache"]}</span></div>', unsafe_allow_html=True)
    except: pass

    if menu == "👀 내 종목 감시":
        st.divider()
        st.markdown("**📊 감시 설정**")
        wl_style_name = st.selectbox("분석 스타일", list(STYLES.keys()), key="wl_style_side")
        st.button("🔄 새로고침", key="wl_refresh_side", use_container_width=True)
        st.button("🗑️ 전체삭제", key="wl_clear_side", use_container_width=True)
        st.button("🗑️ 선택삭제", key="wl_del_side", use_container_width=True)

    st.divider()
    phase, phase_icon = market_phase()
    st.markdown(f"**{phase_icon} {phase}**")
    st.divider()
    
    # ── 방문자 수 ──
    VISIT_FILE = "visit_count.json"
    try:
        if os.path.exists(VISIT_FILE):
            with open(VISIT_FILE, "r") as f:
                visit_data = json.load(f)
        else:
            visit_data = {"total": 0, "today": 0, "date": ""}
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        if visit_data.get("date") != today_str:
            visit_data["today"] = 0
            visit_data["date"] = today_str
        if "visited_this_session" not in st.session_state:
            visit_data["total"] += 1
            visit_data["today"] += 1
            st.session_state["visited_this_session"] = True
            with open(VISIT_FILE, "w") as f:
                json.dump(visit_data, f)
        st.markdown(
            f'<div style="background:#1a1a2e;padding:8px;border-radius:8px;text-align:center;margin-bottom:10px">'
            f'👁️ 오늘 <b>{visit_data["today"]}</b>명 | 누적 <b>{visit_data["total"]}</b>명</div>',
            unsafe_allow_html=True)
    except:
        pass

    wl = load_wl()
    st.markdown(f"**관심종목: {len(wl)}개**")
    if wl:
        for w in wl: st.caption(f"• {w['name']} ({w['code']})")
    hist = load_history()
    st.caption(f"스캔기록: {len(hist)}개")
    st.divider()
    st.markdown(f"**KIS:** {'🟢' if KIS_OK else '🔴'}")
    st.markdown(f"**FDR:** {'🟢' if FDR_OK else '🔴'}")
    st.markdown(f"**자동갱신:** {'🟢' if AUTOREFRESH_OK else '⚪'}")
    st.divider()
    st.caption("거래대금 필터: 단타≥50억 | 스윙≥20억 | 중장기≥10억")
    st.caption("PER/PBR: 네이버 실시간 조회")
    st.caption("목표가/손절가: 지지선·저항선 기반")
# ─── 분석 함수 ───────────────────────────────────────
def analyze(df, code, name, cfg, market="KOSDAQ"):
    code = str(code).strip()
    close = df["Close"]; high = df["High"]; low = df["Low"]; volume = df["Volume"]
    price = int(close.iloc[-1])
    prev_price = int(close.iloc[-2]) if len(close) > 1 else price
    change = round((price - prev_price) / (prev_price + 1e-9) * 100, 2)
    vol_mean = volume.rolling(20).mean().iloc[-1]
    vol_ratio = round(volume.iloc[-1] / (vol_mean + 1e-9), 2)
    trade_val = round(price * volume.iloc[-1] / 1e8, 1)
    vol_up_3 = all(volume.iloc[-(i+1)] > volume.iloc[-(i+2)] for i in range(3)) if len(volume) > 4 else False
    vol_up_5 = all(volume.iloc[-(i+1)] > volume.iloc[-(i+2)] for i in range(5)) if len(volume) > 6 else False
    chg5 = round((price / (close.iloc[-6] + 1e-9) - 1) * 100, 1) if len(close) > 6 else 0
    chg10 = round((price / (close.iloc[-11] + 1e-9) - 1) * 100, 1) if len(close) > 11 else 0
    chg20 = round((price / (close.iloc[-21] + 1e-9) - 1) * 100, 1) if len(close) > 21 else 0
    ema5 = close.ewm(span=5).mean(); ema20 = close.ewm(span=20).mean(); ema60 = close.ewm(span=60).mean()
    e5 = ema5.iloc[-1]; e20 = ema20.iloc[-1]; e60 = ema60.iloc[-1]
    rsi = calc_rsi(close); rsi_val = round(rsi.iloc[-1], 1) if not np.isnan(rsi.iloc[-1]) else 50
    stoch_rsi = calc_stoch_rsi(close)
    macd_line, macd_sig, macd_hist = calc_macd(close)
    bb_upper, bb_mid, bb_lower = calc_bb(close)
    tenkan, kijun, span_a, span_b = calc_ichimoku(high, low, close)
    st_line, st_dir = calc_supertrend(df)
    obv = calc_obv(close, volume); vwap = calc_vwap(df)
    mfi = calc_mfi(df); mfi_val = round(mfi.iloc[-1], 1) if not np.isnan(mfi.iloc[-1]) else 50
    adx, plus_di, minus_di = calc_adx(df)
    adx_val = round(adx.iloc[-1], 1) if not np.isnan(adx.iloc[-1]) else 0

    if st.session_state.get("scan_running", False):
        per, pbr = 0.0, 0.0; foreign_buys, organ_buys = [], []; high_52, low_52 = 0, 0
        sector = ""; industry_per = 0.0; forward_per = 0.0; target_price = 0
        industry_pbr = 0.0; peers = []; news_items = []; roe = 0.0; debt_ratio = 0.0
        op_margin = 0.0; revenue_growth = 0.0; dividend_yield = 0.0; dividend_amt = 0
    else:
        per, pbr, foreign_buys, organ_buys, high_52, low_52, sector, industry_per, forward_per, target_price, industry_pbr, peers, news_items, roe, debt_ratio, op_margin, revenue_growth, dividend_yield, dividend_amt = get_per_pbr(code)
    theme = get_theme(name)
    support, resist, stop_loss = calc_support_resist(df)
    tp_price = max(resist, int(price * (1 + cfg["tp"] / 100)))
    sl_price = min(stop_loss, int(price * (1 - cfg["sl"] / 100)))
    buy_reasons = []; sell_reasons = []

    _phase, _ = market_phase()
    if _phase == "상승장": _trend_mult = 1.0; _osc_mult = 1.0; _oversold_bonus = 1.2
    elif _phase == "하락장": _trend_mult = 1.0; _osc_mult = 0.7; _oversold_bonus = 0.5
    else: _trend_mult = 0.7; _osc_mult = 1.0; _oversold_bonus = 1.0

    score = 50
    if rsi_val < cfg["RSI 과매도 (반등 가능)"]:
        buy_reasons.append("RSI 과매도 (반등 가능)"); score += int(12 * _oversold_bonus)
    elif rsi_val > cfg["RSI 과매수 (조정 가능)"]:
        sell_reasons.append("RSI 과매수 (조정 가능)"); score -= 12

    mh = macd_hist; _macd_w = cfg.get("macd_weight", 15)
    if len(mh) > 1 and not np.isnan(mh.iloc[-1]) and not np.isnan(mh.iloc[-2]):
        if mh.iloc[-1] > 0 and mh.iloc[-2] <= 0: buy_reasons.append("MACD 매수 신호"); score += int(_macd_w * _trend_mult)
        elif mh.iloc[-1] < 0 and mh.iloc[-2] >= 0: sell_reasons.append("MACD 매도 신호"); score -= int(_macd_w * _trend_mult)

    _bb_w = cfg.get("bb_weight", 10)
    if not np.isnan(bb_lower.iloc[-1]) and price <= bb_lower.iloc[-1]: buy_reasons.append("볼린저 하단 터치 (반등 기대)"); score += _bb_w
    elif not np.isnan(bb_upper.iloc[-1]) and price >= bb_upper.iloc[-1]: sell_reasons.append("볼린저 상단 터치 (조정 기대)"); score -= _bb_w

    def detect_divergence(prices, indicator, window=14):
        if len(prices) < window + 5 or len(indicator) < window + 5: return None
        p_recent = prices.iloc[-window:]; i_recent = indicator.iloc[-window:]
        p_lows = [k for k in range(1, len(p_recent)-1) if p_recent.iloc[k] <= p_recent.iloc[k-1] and p_recent.iloc[k] <= p_recent.iloc[k+1]]
        if len(p_lows) >= 2:
            l1, l2 = p_lows[-2], p_lows[-1]
            if p_recent.iloc[l2] < p_recent.iloc[l1] and i_recent.iloc[l2] > i_recent.iloc[l1]: return "bullish"
        p_highs = [k for k in range(1, len(p_recent)-1) if p_recent.iloc[k] >= p_recent.iloc[k-1] and p_recent.iloc[k] >= p_recent.iloc[k+1]]
        if len(p_highs) >= 2:
            h1, h2 = p_highs[-2], p_highs[-1]
            if p_recent.iloc[h2] > p_recent.iloc[h1] and i_recent.iloc[h2] < i_recent.iloc[h1]: return "bearish"
        return None

    rsi_div = detect_divergence(close, rsi, window=20); macd_div = detect_divergence(close, macd_hist, window=20)
    divergence_text = ""
    if rsi_div == "bullish": buy_reasons.append("RSI 상승 다이버전스 (반전↑)"); score += 15; divergence_text = "🟢 RSI 상승 다이버전스"
    elif rsi_div == "bearish": sell_reasons.append("RSI 하락 다이버전스 (반전↓)"); score -= 15; divergence_text = "🔴 RSI 하락 다이버전스"
    if macd_div == "bullish": buy_reasons.append("MACD 상승 다이버전스 (반전↑)"); score += 15; divergence_text += " 🟢 MACD 상승 다이버전스"
    elif macd_div == "bearish": sell_reasons.append("MACD 하락 다이버전스 (반전↓)"); score -= 15; divergence_text += " 🔴 MACD 하락 다이버전스"

    vol_spike = ""
    if vol_ratio >= 5 and change > 0: buy_reasons.append(f"거래량 폭발 ({vol_ratio}배) + 양봉 → 세력 매집 가능"); score += 5; vol_spike = f"🔥 거래량 {vol_ratio}배 폭발"
    elif vol_ratio >= 3 and change > 0: buy_reasons.append(f"거래량 급증 ({vol_ratio}배) + 양봉"); score += 3; vol_spike = f"⚡ 거래량 {vol_ratio}배 급증"
    elif vol_ratio >= 3 and change < 0: sell_reasons.append(f"거래량 급증 ({vol_ratio}배) + 음봉 → 매도세 주의"); score -= 3; vol_spike = f"⚠️ 거래량 {vol_ratio}배 (음봉)"

    ma5 = close.rolling(5).mean().iloc[-1]; ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else None
    ma120 = close.rolling(120).mean().iloc[-1] if len(close) >= 120 else None
    ma_align = ""; _ema_w = cfg.get("ema_weight", 4)
    if ma120 and ma60:
        if ma5 > ma20 > ma60 > ma120: ma_align = "🟢 완전 정배열"; buy_reasons.append("5>20>60>120 완전 정배열 (강한 상승추세)"); score += int((_ema_w + 3) * _trend_mult)
        elif ma5 > ma20 > ma60: ma_align = "🟢 정배열 (60일선까지)"; buy_reasons.append("5>20>60 정배열 (상승추세)"); score += int(_ema_w * _trend_mult)
        elif ma5 < ma20 < ma60 < ma120: ma_align = "🔴 완전 역배열"; sell_reasons.append("5<20<60<120 완전 역배열 (강한 하락추세)"); score -= int((_ema_w + 3) * _trend_mult)
        elif ma5 < ma20 < ma60: ma_align = "🔴 역배열 (60일선까지)"; sell_reasons.append("5<20<60 역배열 (하락추세)"); score -= int(_ema_w * _trend_mult)
        elif ma5 > ma20 and ma20 < ma60: ma_align = "🟡 정배열 초입"
    elif ma60:
        if ma5 > ma20 > ma60: ma_align = "🟢 정배열"; buy_reasons.append("5>20>60 정배열"); score += int(_ema_w * _trend_mult)
        elif ma5 < ma20 < ma60: ma_align = "🔴 역배열"; sell_reasons.append("5<20<60 역배열"); score -= int(_ema_w * _trend_mult)

    sa = span_a.iloc[-1] if not np.isnan(span_a.iloc[-1]) else 0
    sb = span_b.iloc[-1] if not np.isnan(span_b.iloc[-1]) else 0
    if price > max(sa, sb) and sa > sb: buy_reasons.append("추세 구름 돌파 (상승 신호)"); score += 8
    elif price < min(sa, sb) and sa < sb: sell_reasons.append("추세 구름 이탈 (하락 신호)"); score -= 8

    _st_w = cfg.get("st_weight", 6)
    if st_dir.iloc[-1] == 1 and (len(st_dir) > 1 and st_dir.iloc[-2] == -1): buy_reasons.append("슈퍼트렌드 매수 전환 (상승 시작)"); score += int(_st_w * _trend_mult)
    elif st_dir.iloc[-1] == -1 and (len(st_dir) > 1 and st_dir.iloc[-2] == 1): sell_reasons.append("슈퍼트렌드 매도 전환 (하락 시작)"); score -= int(_st_w * _trend_mult)

    # W자 바닥 / N자 반등 / 역N자 패턴
    try:
        _closes = df["Close"].values; _len = len(_closes)
        if _len >= 60:
            _recent = _closes[-60:]; _mid = len(_recent) // 2
            _left = _recent[:_mid]; _right = _recent[_mid:]
            _left_min_idx = np.argmin(_left); _left_min = _left[_left_min_idx]
            _right_min_idx = np.argmin(_right); _right_min = _right[_right_min_idx]
            _mid_max = np.max(_recent[_left_min_idx:_mid + _right_min_idx + 1]) if _mid + _right_min_idx > _left_min_idx else 0
            _cur = _closes[-1]
            if _left_min > 0 and _right_min > 0 and _mid_max > 0:
                _bottom_diff = abs(_left_min - _right_min) / _left_min
                _bounce = (_mid_max - min(_left_min, _right_min)) / min(_left_min, _right_min)
                if _bottom_diff < 0.05 and _bounce > 0.03 and _cur >= _mid_max * 0.98:
                    score += 3; buy_reasons.append("W자 바닥 패턴 (쌍바닥 돌파, 강한 매수 신호)")
            _r30 = _closes[-30:]
            if len(_r30) >= 30:
                _p1 = np.max(_r30[:10]); _p2 = np.min(_r30[5:20]); _p3 = np.max(_r30[10:25]); _p4 = np.min(_r30[20:]); _p5 = _closes[-1]
                if _p1 > 0 and _p2 > 0 and _p3 > 0 and _p4 > 0:
                    _drop1 = (_p1 - _p2) / _p1; _bounce1 = (_p3 - _p2) / _p2; _pullback = (_p3 - _p4) / _p3; _recovery = (_p5 - _p4) / _p4 if _p4 > 0 else 0
                    if _drop1 > 0.05 and _bounce1 > 0.03 and _pullback > 0.02 and _pullback < _drop1 * 0.7 and _recovery > 0.02 and _p5 > _p4:
                        score += 2; buy_reasons.append("N자 반등 패턴 (눌림 후 재상승, 추세 전환)")
                _h1 = np.max(_r30[:10]); _l1 = np.min(_r30[10:20]); _h2 = np.max(_r30[15:25]); _cur30 = _closes[-1]
                if _h1 > 0 and _l1 > 0 and _h2 > 0:
                    _fall1 = (_h1 - _l1) / _h1; _bounce2 = (_h2 - _l1) / _l1; _fall2 = (_h2 - _cur30) / _h2
                    if _fall1 > 0.05 and _bounce2 > 0.02 and _h2 < _h1 * 0.97 and _fall2 > 0.03:
                        score -= 8; sell_reasons.append("역N자 패턴 (반등 후 재하락, 하락 추세)")
                        sell_reasons.insert(0, "⚠️ 역N자 경고: 점수 높아도 추세 하락 중!")
    except: pass

    obv_ma = obv.rolling(20).mean()
    if not np.isnan(obv_ma.iloc[-1]):
        if obv.iloc[-1] > obv_ma.iloc[-1] * 1.05: buy_reasons.append("거래량 동반 상승 (돈이 들어옴)"); score += 8
        elif obv.iloc[-1] < obv_ma.iloc[-1] * 0.95: sell_reasons.append("거래량 동반 하락 (돈이 빠짐)"); score -= 8

    _mfi_sell = cfg.get("자금 유출 신호", 80)
    if mfi_val > _mfi_sell: sell_reasons.append("자금 유출 신호"); score -= int(5 * _osc_mult)

    _adx_min = cfg.get("adx_min", 25)
    if adx_val > _adx_min:
        if plus_di.iloc[-1] > minus_di.iloc[-1]: buy_reasons.append("강한 상승 힘 감지"); score += 10
        else: sell_reasons.append("강한 하락 힘 감지"); score -= 10

    vwap_val = vwap.iloc[-1]
    if not np.isnan(vwap_val):
        if price > vwap_val: buy_reasons.append("평균 매수가 위 (매수세 우위)"); score += 6
        else: sell_reasons.append("평균 매수가 아래 (매도세 우위)"); score -= 6

    _vol_w = cfg.get("vol_weight", 3)
    if vol_up_5: buy_reasons.append("vol5_buy"); score += _vol_w + 1

    if "RSI 과매수 (조정 가능)" in sell_reasons and "MACD 매수 신호" in buy_reasons: score -= 5
    score = max(0, min(100, score))

    if score >= 80: grade = "A+"
    elif score >= 70: grade = "A"
    elif score >= 60: grade = "B+"
    elif score >= 50: grade = "B"
    elif score >= 40: grade = "C"
    else: grade = "D"

    buy_count = len(buy_reasons); sell_count = len(sell_reasons)
    if buy_count >= 4: verdict = "적극 매수"
    elif buy_count >= 2 and buy_count > sell_count: verdict = "매수 관심"
    elif sell_count >= 4: verdict = "적극 매도"
    elif sell_count >= 2 and sell_count > buy_count: verdict = "매도 관심"
    else: verdict = "중립 관망"

    # AI 요약
    easy_parts = []
    if chg5 > 3: easy_parts.append(f"최근 5일간 {chg5}% 올랐어요 📈")
    elif chg5 < -3: easy_parts.append(f"최근 5일간 {abs(chg5)}% 빠졌어요 📉")
    else: easy_parts.append("최근 5일간 큰 변동 없이 횡보 중이에요")

    if high_52 > 0 and low_52 > 0 and high_52 != low_52:
        pos_52 = round((price - low_52) / (high_52 - low_52) * 100, 1)
        if pos_52 <= 20: trend_pos = "바닥권 초입"; trend_pos_icon = "🟢"
        elif pos_52 <= 40: trend_pos = "하단부"; trend_pos_icon = "🟡"
        elif pos_52 <= 60: trend_pos = "중간대"; trend_pos_icon = "🟠"
        elif pos_52 <= 80: trend_pos = "상단부"; trend_pos_icon = "🟠"
        else: trend_pos = "고점권 주의"; trend_pos_icon = "🔴"
    else: pos_52 = 0; trend_pos = ""; trend_pos_icon = ""

    if trend_pos:
        if "바닥권" in trend_pos or "하단" in trend_pos: easy_parts.append(f"52주 중 {trend_pos_icon} {trend_pos}이라 저점 매수 기회일 수 있어요")
        elif "고점" in trend_pos or "상단" in trend_pos: easy_parts.append(f"52주 중 {trend_pos_icon} {trend_pos}이라 추격 매수는 위험해요")

    if vol_up_5: easy_parts.append("거래량이 5일 연속 늘고 있어요 — 시장의 관심이 집중되는 중!")
    elif vol_up_3: easy_parts.append("거래량이 3일 연속 늘고 있어요")
    elif vol_ratio > 2: easy_parts.append(f"오늘 거래량이 평소의 {vol_ratio}배 — 뭔가 움직임이 있어요")

    if e5 > e20 > e60: easy_parts.append("이동평균선이 정배열 — 상승 추세가 이어지고 있어요")
    elif e5 < e20 < e60: easy_parts.append("이동평균선이 역배열 — 하락 추세가 이어지고 있어요")

    if per > 0:
        if industry_per > 0: _avg = industry_per
        else: _avg = get_sector_per(sector if sector else theme)
        _ratio = per / _avg
        if _ratio <= 0.5: easy_parts.append(f"PER {per:.1f}배 — 업종평균({_avg:.1f})보다 훨씬 싸요! 💰")
        elif _ratio <= 0.8: easy_parts.append(f"PER {per:.1f}배 — 업종평균({_avg:.1f})보다 싼 편이에요 💰")
        elif _ratio <= 1.2: easy_parts.append(f"PER {per:.1f}배 — 업종평균({_avg:.1f})과 비슷한 수준이에요")
        elif _ratio <= 1.5: easy_parts.append(f"PER {per:.1f}배 — 업종평균({_avg:.1f})보다 다소 비싼 편이에요 ⚠️")
        else: easy_parts.append(f"PER {per:.1f}배 — 업종평균({_avg:.1f})보다 많이 비싸요 ⚠️")

    # EV/EBITDA
    ev_ebitda_val = None; ev_comment = ""
    if not st.session_state.get("scan_running", False):
        ev_data = get_ev_ebitda(code, is_us=(market in ["NASDAQ", "NYSE", "S&P500", "AMEX"]))
        ev_ebitda_val = ev_data.get("ev_ebitda")
        if ev_ebitda_val and ev_ebitda_val > 0:
            if ev_ebitda_val < 5: score -= 3; sell_reasons.append(f"EV/EBITDA {ev_ebitda_val:.1f} 극도로 저평가 (가치함정 주의)")
            elif ev_ebitda_val >= 6 and ev_ebitda_val <= 8: score += 8; buy_reasons.append(f"EV/EBITDA {ev_ebitda_val:.1f} 저평가")
            elif ev_ebitda_val > 8 and ev_ebitda_val < 12: score += 2
            elif ev_ebitda_val > 25: score -= 3; sell_reasons.append(f"EV/EBITDA {ev_ebitda_val:.1f} 고평가")
        score = max(0, min(100, score))

    if verdict in ["적극 매수", "매수 관심"]:
        easy_parts.append(f"👉 매수 신호 {buy_count}개 감지! 진입가 {support:,}원 부근, 목표가 {tp_price:,}원, 손절가 {sl_price:,}원")
    elif verdict in ["적극 매도", "매도 관심"]:
        easy_parts.append(f"👉 매도 신호 {sell_count}개. {support:,}원 깨지면 빠져나오세요")
    else:
        easy_parts.append(f"👉 아직 방향이 안 잡혔어요. {support:,}원 지지 확인 후 진입 고려")

    ai_summary = "🤖 " + " | ".join(easy_parts)

    return {
        "code": code, "name": name, "price": price, "change": change,
        "volume": int(volume.iloc[-1]), "vol_ratio": vol_ratio, "trade_val": trade_val,
        "rsi": rsi_val, "mfi": mfi_val, "adx": adx_val,
        "score": score, "grade": grade, "verdict": verdict,
        "buy_reasons": buy_reasons, "sell_reasons": sell_reasons,
        "ai_summary": ai_summary, "theme": sector if sector else theme,
        "tp_price": tp_price, "sl_price": sl_price,
        "support": support, "resist": resist, "stop_loss": stop_loss,
        "per": per, "pbr": pbr, "industry_per": industry_per,
        "forward_per": forward_per, "target_price": target_price,
        "industry_pbr": industry_pbr, "peers": peers, "news_items": news_items,
        "roe": roe, "debt_ratio": debt_ratio, "op_margin": op_margin,
        "revenue_growth": revenue_growth, "dividend_yield": dividend_yield,
        "dividend_amt": dividend_amt, "ev_ebitda": ev_ebitda_val,
        "is_us": market in ["NASDAQ", "NYSE", "S&P500", "AMEX"],
        "high_52": high_52, "low_52": low_52, "pos_52": pos_52,
        "trend_pos": trend_pos, "trend_pos_icon": trend_pos_icon,
        "foreign_buys": foreign_buys, "organ_buys": organ_buys,
        "chg5": chg5, "chg10": chg10, "chg20": chg20,
        "df": df, "ema5": ema5, "ema20": ema20, "ema60": ema60,
        "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
        "macd_line": macd_line, "macd_sig": macd_sig, "macd_hist": macd_hist,
        "rsi_series": rsi, "mfi_series": mfi,
        "obv": obv, "st_line": st_line, "st_dir": st_dir,
        "divergence": divergence_text.strip(), "vol_spike": vol_spike, "ma_align": ma_align,
    }

# ── 매도 알림 ──
def check_sell(r, cfg):
    alerts = []
    if r["price"] >= r["resist"]: alerts.append(f"🎯 {r['name']} 저항선 {r['resist']:,}원 도달! 익절 고려")
    if r["price"] <= r["stop_loss"]: alerts.append(f"🛑 {r['name']} 손절선 {r['stop_loss']:,}원 이탈! 손절 고려")
    if r["price"] <= r["support"]: alerts.append(f"⚠️ {r['name']} 지지선 {r['support']:,}원 근접! 주의")
    return alerts

# ── 캔들 그리기 ──
def draw_candle_bars(ax, df):
    for i in range(len(df)):
        o, h, l, c = df["Open"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i]
        color = "#ff1744" if c >= o else "#2979ff"
        ax.plot([i, i], [l, h], color=color, linewidth=0.7)
        ax.plot([i, i], [min(o, c), max(o, c)], color=color, linewidth=2.5)

# ── 차트 ──
def draw_chart(r, cfg, last_n=60):
    df = r["df"].tail(last_n).reset_index(drop=True)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117"); ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values(): spine.set_color("#333")
    n = len(df)
    ax1 = axes[0]; draw_candle_bars(ax1, df)
    e5 = r["ema5"].tail(last_n).reset_index(drop=True); e20 = r["ema20"].tail(last_n).reset_index(drop=True); e60 = r["ema60"].tail(last_n).reset_index(drop=True)
    bu = r["bb_upper"].tail(last_n).reset_index(drop=True); bl = r["bb_lower"].tail(last_n).reset_index(drop=True)
    ax1.plot(range(n), e5[:n], color="#ffeb3b", linewidth=0.8, label="EMA5")
    ax1.plot(range(n), e20[:n], color="#ff9800", linewidth=0.8, label="EMA20")
    ax1.plot(range(n), e60[:n], color="#9c27b0", linewidth=0.8, label="EMA60")
    ax1.plot(range(n), bu[:n], color="#616161", linewidth=0.5, linestyle="--")
    ax1.plot(range(n), bl[:n], color="#616161", linewidth=0.5, linestyle="--")
    if r.get("st_line") is not None and r.get("st_dir") is not None:
        stl = r["st_line"].tail(last_n).reset_index(drop=True); std = r["st_dir"].tail(last_n).reset_index(drop=True)
        price_min = df["Low"].min(); price_max = df["High"].max(); price_margin = (price_max - price_min) * 0.1
        for i in range(1, n):
            y0 = stl.iloc[i-1]; y1 = stl.iloc[i]
            if pd.isna(y0) or pd.isna(y1) or y0 < price_min - price_margin or y1 < price_min - price_margin or y0 > price_max + price_margin or y1 > price_max + price_margin: continue
            color = "#4caf50" if std.iloc[i] == 1 else "#f44336"
            ax1.plot([i-1, i], [y0, y1], color=color, linewidth=1.5, alpha=0.7)

    # 세력매집/매도경고 신호
    vol = df["Volume"].values; close = df["Close"].values; low_arr = df["Low"].values; opn = df["Open"].values; high_arr = df["High"].values
    vol_ma20 = pd.Series(vol).rolling(20).mean().values
    obv_s = r["obv"].tail(last_n).reset_index(drop=True); obv_ma10 = obv_s.rolling(10).mean()
    for i in range(20, n):
        body = abs(close[i] - opn[i]); candle_range = high_arr[i] - low_arr[i]
        if candle_range == 0: continue
        body_ratio = body / candle_range; vol_r = vol[i] / vol_ma20[i] if vol_ma20[i] > 0 else 0
        lower_wick = min(opn[i], close[i]) - low_arr[i]; wick_ratio = lower_wick / candle_range
        sc = 0
        if vol_r >= 3.0: sc += 3
        elif vol_r >= 2.0: sc += 2
        if body_ratio <= 0.25: sc += 3
        elif body_ratio <= 0.35: sc += 2
        if wick_ratio >= 0.5: sc += 3
        elif wick_ratio >= 0.35: sc += 2
        if not pd.isna(obv_ma10.iloc[i]) and obv_s.iloc[i] >= obv_ma10.iloc[i]: sc += 2
        if sc >= 7:
            marker_y = low_arr[i] - (high_arr[i] - low_arr[i]) * 0.8
            ax1.plot(i, marker_y, "^", color="#00e676", markersize=12, alpha=0.95, zorder=10)
    for i in range(20, n):
        body = abs(close[i] - opn[i]); candle_range = high_arr[i] - low_arr[i]
        if candle_range == 0: continue
        body_ratio = body / candle_range; vol_r = vol[i] / vol_ma20[i] if vol_ma20[i] > 0 else 0
        upper_wick = high_arr[i] - max(opn[i], close[i]); wick_ratio = upper_wick / candle_range
        sc = 0
        if vol_r >= 3.0: sc += 3
        elif vol_r >= 2.0: sc += 2
        if body_ratio <= 0.25: sc += 3
        elif body_ratio <= 0.35: sc += 2
        if wick_ratio >= 0.5: sc += 3
        elif wick_ratio >= 0.35: sc += 2
        if not pd.isna(obv_ma10.iloc[i]) and obv_s.iloc[i] < obv_ma10.iloc[i]: sc += 2
        if sc >= 7:
            marker_y = high_arr[i] + (high_arr[i] - low_arr[i]) * 0.8
            ax1.plot(i, marker_y, "v", color="#ff1744", markersize=12, alpha=0.95, zorder=10)

    ax1.axhline(r["support"], color="#4caf50", linewidth=1, linestyle="--", label=f"지지 {r['support']:,}")
    ax1.axhline(r["resist"], color="#ff9800", linewidth=1, linestyle="--", label=f"저항 {r['resist']:,}")
    ax1.axhline(r["stop_loss"], color="#f44336", linewidth=1, linestyle="--", label=f"손절 {r['stop_loss']:,}")
    ax1.set_title(f"{r['name']} ({r['code']})  {r['price']:,}원  {r['change']:+.2f}%", color="white", fontsize=11)
    ax1.legend(fontsize=7, loc="upper left")

    ax2 = axes[1]; mh = r["macd_hist"].tail(last_n).reset_index(drop=True); ml = r["macd_line"].tail(last_n).reset_index(drop=True); ms = r["macd_sig"].tail(last_n).reset_index(drop=True)
    colors = ["#ff1744" if v >= 0 else "#2979ff" for v in mh]
    ax2.bar(range(n), mh[:n], color=colors, width=0.7); ax2.plot(range(n), ml[:n], color="#ffeb3b", linewidth=0.8); ax2.plot(range(n), ms[:n], color="#ff9800", linewidth=0.8)
    ax2.set_title("MACD", color="white", fontsize=9)

    ax3 = axes[2]; rs = r["rsi_series"].tail(last_n).reset_index(drop=True); mf = r["mfi_series"].tail(last_n).reset_index(drop=True)
    ax3.plot(range(n), rs[:n], color="#ffeb3b", linewidth=0.8, label="RSI"); ax3.plot(range(n), mf[:n], color="#26c6da", linewidth=0.8, label="MFI")
    ax3.axhline(70, color="#ff1744", linewidth=0.5, linestyle="--"); ax3.axhline(30, color="#4caf50", linewidth=0.5, linestyle="--")
    ax3.set_title("RSI / MFI", color="white", fontsize=9); ax3.legend(fontsize=7)

    ax4 = axes[3]; ob = r["obv"].tail(last_n).reset_index(drop=True)
    ax4.plot(range(n), ob[:n], color="#ab47bc", linewidth=0.8); ax4.set_title("OBV", color="white", fontsize=9)
    plt.tight_layout(); return fig


# ── 호가창 분석 ──
def analyze_orderbook(code):
    if not KIS_OK or _broker is None: return None
    try:
        ob = _broker.get_orderbook(code)
        if not ob: return None
        asks = ob["asks"]; bids = ob["bids"]; total_ask = ob["total_ask_vol"]; total_bid = ob["total_bid_vol"]
        if total_ask == 0: return None
        bid_ask_ratio = round(total_bid / total_ask, 2)
        ask_vols = [a["volume"] for a in asks if a["volume"] > 0]
        avg_ask = sum(ask_vols) / len(ask_vols) if ask_vols else 0
        sell_walls = [a for a in asks if a["volume"] >= avg_ask * 3]
        bid_vols = [b["volume"] for b in bids if b["volume"] > 0]
        avg_bid = sum(bid_vols) / len(bid_vols) if bid_vols else 0
        buy_walls = [b for b in bids if b["volume"] >= avg_bid * 3]
        if bid_ask_ratio >= 2.0: signal = "🔥 매수세 압도 (급등 가능)"; signal_color = "#ff1744"
        elif bid_ask_ratio >= 1.3: signal = "👀 매수세 우위"; signal_color = "#ff9800"
        elif bid_ask_ratio <= 0.5: signal = "🧊 매도세 압도 (하락 주의)"; signal_color = "#2979ff"
        elif bid_ask_ratio <= 0.7: signal = "⚠️ 매도세 우위"; signal_color = "#42a5f5"
        else: signal = "😐 균형 상태"; signal_color = "#888"
        return {"asks": asks, "bids": bids, "total_ask": total_ask, "total_bid": total_bid, "ratio": bid_ask_ratio, "sell_walls": sell_walls, "buy_walls": buy_walls, "signal": signal, "signal_color": signal_color}
    except: return None

# ── 섹터 동반 상승 ──
def detect_sector_surge(all_results):
    theme_map = {}
    for skey, rlist in all_results.items():
        for r in rlist:
            th = r.get("theme", "")
            if not th: continue
            if th not in theme_map: theme_map[th] = []
            theme_map[th].append({"name": r["name"], "code": r["code"], "score": r["score"], "change": r["change"], "style": skey})
    surges = []
    for theme, stocks in theme_map.items():
        unique_codes = list(set([s["code"] for s in stocks]))
        if len(unique_codes) >= 2:
            avg_score = round(sum(s["score"] for s in stocks) / len(stocks), 1)
            avg_change = round(sum(s["change"] for s in stocks) / len(stocks), 2)
            surges.append({"theme": theme, "count": len(unique_codes), "avg_score": avg_score, "avg_change": avg_change, "stocks": stocks})
    surges.sort(key=lambda x: x["count"], reverse=True)
    return surges

# ── 전체 스캔 ──
def run_full_scan(stocks):
    stocks = stocks.reset_index(drop=True); total = len(stocks)
    if total == 0: return {}, {}
    cached_data = {}; all_results = {}

    st.subheader("📡 데이터 수집 중..."); dl_bar = st.progress(0.0); dl_text = st.empty()
    total = min(total, 1200)
    for i in range(total):
        row = stocks.iloc[i]; code = str(row["Code"]).strip(); name = str(row["Name"]).strip()
        dl_bar.progress(min((i+1)/total, 1.0)); dl_text.text(f"다운로드: {name} ({i+1}/{total})")
        df = fetch(code)
        if df is not None: cached_data[code] = {"name": name, "df": df}
    dl_bar.empty(); dl_text.empty()
    st.info(f"📊 데이터 수집 완료: {len(cached_data)}개 종목 / 전체 {total}개")

    phase, phase_icon = market_phase()
    if phase == "하락장":
        adjusted_min_score = 85
        st.warning(f"{phase_icon} 하락장 감지 — 최소 점수 85점")
    elif phase == "상승장":
        adjusted_min_score = 85
        st.success(f"{phase_icon} 상승장 감지 — 최소 점수 85점")
    else:
        adjusted_min_score = 85
        st.info(f"{phase_icon} 횡보장 — 기본 점수 85점")

    st.subheader("🔎 3개 스타일 동시 분석 중..."); an_bar = st.progress(0.0); an_text = st.empty()
    total_work = len(cached_data) * len(STYLES); done = 0
    for style_name, cfg in STYLES.items():
        short_key = cfg["key"]; results = []
        for code, cdata in cached_data.items():
            done += 1; an_bar.progress(min(done/(total_work+1), 1.0)); an_text.text(f"분석: {cdata['name']} - {style_name}")
            try:
                r = analyze(cdata["df"], code, cdata["name"], cfg)
                if r["trade_val"] < cfg["min_trade_val"]: continue
                if r["score"] >= adjusted_min_score: r["crown"] = ""; results.append(r)
            except: pass
        all_results[short_key] = results
    an_bar.empty(); an_text.empty()

    # EV/EBITDA 추가 조회
    st.subheader("💎 상위 종목 EV/EBITDA 조회 중..."); ev_bar = st.progress(0.0); ev_count = 0
    for skey, rlist in all_results.items():
        top_list = sorted(rlist, key=lambda x: x["score"], reverse=True)[:15]
        for r in top_list:
            ev_count += 1; ev_bar.progress(min(ev_count/(len(all_results)*15), 1.0))
            try:
                is_us = market in ["NASDAQ", "NYSE", "S&P500", "AMEX"]
                ev_data = get_ev_ebitda(r["code"], is_us=is_us); ev_val = ev_data.get("ev_ebitda"); r["ev_ebitda"] = ev_val
                if ev_val and ev_val > 0:
                    if ev_val < 5: r["score"] += 8; r["buy_reasons"].append(f"EV/EBITDA {ev_val:.1f} 극도로 저평가")
                    elif ev_val < 8: r["score"] += 5; r["buy_reasons"].append(f"EV/EBITDA {ev_val:.1f} 저평가")
                    elif ev_val > 25: r["score"] -= 3; r["sell_reasons"].append(f"EV/EBITDA {ev_val:.1f} 고평가")
                    r["score"] = max(0, min(100, r["score"]))
            except: pass
    ev_bar.empty()

    _macro = st.session_state.get("macro_bonus", 0); _macro_reasons = st.session_state.get("macro_reasons", [])
    if _macro != 0:
        for skey, rlist in all_results.items():
            for r in rlist:
                r["score"] = max(0, min(100, r["score"] + _macro))
                if _macro > 0: r["buy_reasons"].append(f"매크로 우호 ({', '.join(_macro_reasons)})")
                else: r["sell_reasons"].append(f"매크로 악재 ({', '.join(_macro_reasons)})")

    for skey, rlist in all_results.items():
        theme_count = {}; filtered_list = []; rlist.sort(key=lambda x: x["score"], reverse=True)
        for r in rlist:
            th = r.get("theme", "")
            if th: theme_count[th] = theme_count.get(th, 0) + 1
            if th and theme_count[th] > 3: continue
            filtered_list.append(r)
        all_results[skey] = filtered_list

    code_styles = {}
    for skey, rlist in all_results.items():
        for r in rlist:
            if r["code"] not in code_styles: code_styles[r["code"]] = set()
            code_styles[r["code"]].add(skey)
    for skey, rlist in all_results.items():
        for r in rlist:
            cnt = len(code_styles.get(r["code"], set()))
            if cnt >= 3: r["crown"] = "🏆 3관왕"
            elif cnt >= 2: r["crown"] = "⭐ 2관왕"
            else: r["crown"] = ""

    return all_results, cached_data

# ── 카드 UI ──
def show_card(r, prefix, cfg, is_new_buy=False, is_new_sell=False, show_wl_btn=True):
    score = r["score"]
    if score >= 80: score_color = "#4caf50"; score_label = "상승 잠재력"
    elif score >= 60: score_color = "#ff9800"; score_label = "상승 잠재력"
    else: score_color = "#f44336"; score_label = "하락 신호"
    verdict_emoji = {"적극 매수": "🔥", "매수 관심": "👀", "적극 매도": "🧊", "매도 관심": "⚠️", "중립 관망": "😐"}.get(r["verdict"], "")
    crown = r.get("crown", ""); is_us_stock = r.get("is_us", False)
    currency = "$" if is_us_stock else ""; unit = "" if is_us_stock else "원"
    chg_color = "#ff1744" if r["change"] >= 0 else "#2979ff"

    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div><span style="font-size:1.4em;font-weight:bold">{crown} {r["name"]} ({r["code"]}) | {r.get("theme","")}</span><br>'
        f'<span style="color:{chg_color};font-size:1.3em;font-weight:bold">{currency}{r["price"]:,}{unit} ({r["change"]:+.2f}%)</span> '
        f'<span style="font-size:0.9em;color:#aaa">거래대금 {r["trade_val"]:,.0f}{"만$" if is_us_stock else "억"} | 거래량비 {r["vol_ratio"]}배</span></div>'
        f'<div style="text-align:right"><span style="font-size:2.2em;font-weight:bold;color:{score_color}">{score}점</span><br>'
        f'<span style="font-size:1.1em;color:{score_color}">{r["grade"]} {verdict_emoji} {r["verdict"]}</span></div></div>', unsafe_allow_html=True)

    # PER/PBR 배지
    per_html = ""; fper_html = ""; tp_html = ""; pbr_html = ""; ev_html = ""
    if r["per"] > 0:
        _avg = r["industry_per"] if r.get("industry_per", 0) > 0 else get_sector_per(r.get("theme", ""))
        _ratio = r["per"] / _avg
        if _ratio <= 0.5: _label = "저평가"
        elif _ratio <= 0.8: _label = "다소저평가"
        elif _ratio <= 1.2: _label = "적정"
        elif _ratio <= 1.5: _label = "다소고평가"
        else: _label = "고평가"
        _detail = f"업종PER({_avg:.1f})의 {_ratio:.0%}"
        if _label in ["저평가", "다소저평가"]: per_html = f'<span class="badge-per-low">PER {r["per"]:.1f} {_label} ({_detail})</span>'
        elif _label == "적정": per_html = f'<span class="badge-per-mid">PER {r["per"]:.1f} {_label} ({_detail})</span>'
        else: per_html = f'<span class="badge-per-high">PER {r["per"]:.1f} {_label} ({_detail})</span>'
    if r.get("forward_per", 0) > 0:
        fper_html = f'<span style="background:#1565c0;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">예상PER {r["forward_per"]:.1f}</span>'
    if r.get("target_price", 0) > 0:
        _upside = round((r["target_price"] / r["price"] - 1) * 100, 1)
        if _upside > 0: tp_html = f'<span style="background:#2e7d32;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">목표가 {r["target_price"]:,}원 (▲{_upside}%)</span>'
        else: tp_html = f'<span style="background:#c62828;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">목표가 {r["target_price"]:,}원 (▼{_upside}%)</span>'
    if r["pbr"] > 0:
        if r.get("industry_pbr", 0) > 0:
            _pavg = r["industry_pbr"]; _pratio = r["pbr"] / _pavg
            if _pratio <= 0.8: pbr_html = f'<span class="badge-pbr-low">PBR {r["pbr"]:.2f} 저평가</span>'
            elif _pratio <= 1.2: pbr_html = f'<span class="badge-pbr-mid">PBR {r["pbr"]:.2f} 적정</span>'
            else: pbr_html = f'<span class="badge-pbr-high">PBR {r["pbr"]:.2f} 고평가</span>'
        else:
            if r["pbr"] <= 1: pbr_html = f'<span class="badge-pbr-low">PBR {r["pbr"]:.2f} 저평가</span>'
            elif r["pbr"] <= 3: pbr_html = f'<span class="badge-pbr-mid">PBR {r["pbr"]:.2f} 보통</span>'
            else: pbr_html = f'<span class="badge-pbr-high">PBR {r["pbr"]:.2f} 고평가</span>'
    if r.get("ev_ebitda") and r["ev_ebitda"] > 0:
        if r["ev_ebitda"] < 8: ev_html = f'<span class="badge-pbr-low">EV/EBITDA {r["ev_ebitda"]:.1f} 저평가</span>'
        elif r["ev_ebitda"] <= 15: ev_html = f'<span class="badge-pbr-mid">EV/EBITDA {r["ev_ebitda"]:.1f} 보통</span>'
        else: ev_html = f'<span class="badge-pbr-high">EV/EBITDA {r["ev_ebitda"]:.1f} 고평가</span>'
    if per_html or fper_html or tp_html or pbr_html or ev_html:
        st.markdown(f"{per_html} {fper_html} {tp_html} {pbr_html} {ev_html}", unsafe_allow_html=True)

    col_left, col_right = st.columns([6, 4])
    with col_left:
        # 밸류에이션 코멘트
        val_comments = []
        if r["per"] > 0 and r.get("forward_per", 0) > 0:
            if r["forward_per"] < r["per"] * 0.5: val_comments.append(f"📊 현재 PER {r['per']:.1f}배, 예상PER {r['forward_per']:.1f}배 → 미래 기준으로 싸요! 💰")
            elif r["forward_per"] > r["per"] * 1.5: val_comments.append(f"📊 현재 PER {r['per']:.1f}배, 예상PER {r['forward_per']:.1f}배 → 이익 감소 전망 ⚠️")
        if r.get("target_price", 0) > 0:
            _up = round((r["target_price"] / r["price"] - 1) * 100, 1)
            if _up > 20: val_comments.append(f"🎯 증권사 목표가 {r['target_price']:,}원 — {_up}% 상승 여력! 📈")
            elif _up > 0: val_comments.append(f"🎯 증권사 목표가 {r['target_price']:,}원 — {_up}% 상승 여력 📈")
            else: val_comments.append(f"🎯 증권사 목표가 {r['target_price']:,}원 — 현재가가 이미 넘었어요 ⚠️")
        if val_comments:
            st.markdown(f'<div style="background:#1a1a2e;border-left:4px solid #00b4d8;padding:10px 15px;border-radius:5px;margin:5px 0;font-size:0.9em"><b>💡 밸류에이션</b><br>{"<br>".join(val_comments)}</div>', unsafe_allow_html=True)

        # 추세/수급 뱃지
        extra_badges = ""
        if r.get("trend_pos"):
            tp_colors = {"바닥권 초입": "#4caf50", "하단부": "#8bc34a", "중간대": "#ff9800", "상단부": "#ff5722", "고점권 주의": "#f44336"}
            tp_col = tp_colors.get(r["trend_pos"], "#888")
            extra_badges += f'<span style="background:{tp_col};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">{r["trend_pos_icon"]} {r["trend_pos"]}</span> '
        if r.get("foreign_buys"):
            consec = 0
            for fb in r["foreign_buys"]:
                if fb > 0: consec += 1
                else: break
            if consec >= 3: extra_badges += f'<span style="background:#e91e63;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">🔥 외국인 {consec}일 연속 순매수</span> '
        if r.get("divergence"):
            div_col = "#4caf50" if "상승" in r["divergence"] else "#f44336"
            extra_badges += f'<span style="background:{div_col};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">{r["divergence"]}</span> '
        if r.get("vol_spike"): extra_badges += f'<span style="background:#ff6f00;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">{r["vol_spike"]}</span> '
        if r.get("ma_align"):
            ma_col = "#4caf50" if "정배열" in r["ma_align"] else "#f44336" if "역배열" in r["ma_align"] else "#ff9800"
            extra_badges += f'<span style="background:{ma_col};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">{r["ma_align"]}</span> '
        if extra_badges: st.markdown(extra_badges, unsafe_allow_html=True)

        buy_count = len(r["buy_reasons"]); sell_count = len(r["sell_reasons"])
        if buy_count > 0:
            new_tag = "🆕 " if is_new_buy else ""
            st.markdown(f'<div class="buy-blink" style="background:#ff1744;color:#fff;padding:10px 14px;border-radius:8px;margin:8px 0;font-size:1.15em">{new_tag}🔴 매수 신호 {buy_count}개 감지!</div>', unsafe_allow_html=True)
        if sell_count > 0:
            new_tag = "🆕 " if is_new_sell else ""
            st.markdown(f'<div class="sell-blink" style="background:#2979ff;color:#fff;padding:10px 14px;border-radius:8px;margin:8px 0;font-size:1.15em">{new_tag}🔵 매도 신호 {sell_count}개 감지!</div>', unsafe_allow_html=True)

        wt = get_weekly_trend(r["code"])
        if wt:
            if wt["trend"] == "상승" and r["score"] >= 80: mtf_msg = f'{wt["emoji"]} {wt["desc"]} + 일봉 매수 신호 → <b>🔥 강력 매수 구간</b>'; mtf_bg = "#1a2e1a"; mtf_border = "#4caf50"
            elif wt["trend"] == "하락" and r["score"] >= 80: mtf_msg = f'{wt["emoji"]} {wt["desc"]} + 일봉 매수 신호 → <b>⚠️ 역추세 매매 (주의)</b>'; mtf_bg = "#2e1a1a"; mtf_border = "#f44336"
            elif wt["trend"] == "하락": mtf_msg = f'{wt["emoji"]} {wt["desc"]} → <b>매수 비추천</b>'; mtf_bg = "#2e1a1a"; mtf_border = "#f44336"
            else: mtf_msg = f'{wt["emoji"]} {wt["desc"]}'; mtf_bg = "#2a2a3e"; mtf_border = "#ff9800"
            st.markdown(f'<div style="background:{mtf_bg};padding:10px 14px;border-radius:8px;margin:6px 0;border-left:4px solid {mtf_border}">📅 <b>멀티타임프레임:</b> {mtf_msg}</div>', unsafe_allow_html=True)

        st.markdown(f'<div style="background:#2a2a3e;padding:10px 14px;border-radius:8px;margin:6px 0;border-left:4px solid #ff9800">🤖 <b>AI 판단:</b> {r["ai_summary"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:#1a2e1a;padding:8px 14px;border-radius:8px;margin:4px 0">🟢 <b>추천 진입가(지지선):</b> <span class="support-line">{currency}{r["support"]:,}{unit}</span> &nbsp;&nbsp; 🟡 <b>목표가(저항선):</b> <span class="resist-line">{currency}{r["tp_price"]:,}{unit}</span> (+{round((r["tp_price"]/(r["price"]+1e-9)-1)*100,1)}%)</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:#2e1a1a;padding:8px 14px;border-radius:8px;margin:4px 0">🔴 <b>손절가:</b> <span class="stop-line">{currency}{r["sl_price"]:,}{unit}</span> ({round((r["sl_price"]/(r["price"]+1e-9)-1)*100,1)}%) &nbsp;&nbsp; 📏 <b>손익비:</b> {min(round((r["tp_price"]-r["price"])/(r["price"]-r["sl_price"]+1e-9),1), 99.9)}:1</div>', unsafe_allow_html=True)
        bc, sc = st.columns(2)
        with bc:
            for br in r["buy_reasons"]: st.markdown(f'🔴 {EASY.get(br, br)}')
        with sc:
            for sr in r["sell_reasons"]: st.markdown(f'🔵 {EASY.get(sr, sr)}')

    with col_right:
        if r.get("peers") and len(r["peers"]) > 0:
            peer_rows = ""
            for p in r["peers"][:5]:
                p_chg_val = float(p.get("change", "0")) if p.get("change") else 0
                p_chg_color = "#ff4444" if p_chg_val > 0 else "#4488ff" if p_chg_val < 0 else "#aaa"
                peer_rows += f'<tr><td style="padding:4px 8px">{p.get("name","")}</td><td style="padding:4px 8px;text-align:right">{p.get("price","0")}원</td><td style="padding:4px 8px;text-align:right;color:{p_chg_color}">{p_chg_val:+.2f}%</td></tr>'
            st.markdown(f'<div style="background:#1a1a2e;border-left:4px solid #ff9800;padding:10px 15px;border-radius:5px;margin:8px 0"><b>🏭 동종업계 비교</b><table style="width:100%;margin-top:8px;border-collapse:collapse;font-size:0.85em"><tr style="border-bottom:1px solid #333"><th style="padding:4px 8px;text-align:left">종목</th><th style="padding:4px 8px;text-align:right">현재가</th><th style="padding:4px 8px;text-align:right">등락률</th></tr>{peer_rows}</table></div>', unsafe_allow_html=True)

        if r.get("news_items") and len(r["news_items"]) > 0:
            pos_words = ["상승", "급등", "호실적", "최고", "성장", "흑자", "수혜", "기대", "호재", "강세"]
            neg_words = ["하락", "급락", "적자", "손실", "위기", "우려", "매도", "약세", "폭락", "부진"]
            pos_count = 0; neg_count = 0; news_html = ""
            for n_item in r["news_items"]:
                title = n_item.get("title", ""); text = title + " " + n_item.get("body", "")
                is_pos = any(w in text for w in pos_words); is_neg = any(w in text for w in neg_words)
                if is_pos and not is_neg: icon = "🟢"; pos_count += 1
                elif is_neg and not is_pos: icon = "🔴"; neg_count += 1
                else: icon = "⚪"
                news_html += f'<div style="padding:3px 0;border-bottom:1px solid #222">{icon} {title[:60]}</div>'
            total_n = len(r["news_items"]); neutral_count = total_n - pos_count - neg_count
            if pos_count > neg_count: sentiment = "긍정적 📈"; sent_color = "#4caf50"
            elif neg_count > pos_count: sentiment = "부정적 📉"; sent_color = "#f44336"
            else: sentiment = "중립 ➡️"; sent_color = "#ff9800"
            st.markdown(f'<div style="background:#1a1a2e;border-left:4px solid {sent_color};padding:10px 15px;border-radius:5px;margin:8px 0"><b>📰 뉴스 감성</b> — <span style="color:{sent_color};font-weight:bold">{sentiment}</span> <span style="font-size:0.85em;color:#aaa">(긍정 {pos_count} / 부정 {neg_count} / 중립 {neutral_count})</span>{news_html}</div>', unsafe_allow_html=True)

        # 재무 건전성
        roe_val = r.get("roe", 0); debt_val = r.get("debt_ratio", 0); op_val = r.get("op_margin", 0); rev_val = r.get("revenue_growth", 0)
        if roe_val or debt_val or op_val or rev_val:
            fin_items = ""
            if roe_val:
                if roe_val >= 15: roe_color, roe_label = "#4caf50", "우량"
                elif roe_val >= 10: roe_color, roe_label = "#ff9800", "양호"
                elif roe_val > 0: roe_color, roe_label = "#aaa", "보통"
                else: roe_color, roe_label = "#f44336", "적자"
                fin_items += f'<tr><td style="padding:4px 8px">ROE</td><td style="padding:4px 8px;text-align:right;color:{roe_color};font-weight:bold">{roe_val:.1f}%</td><td style="padding:4px 8px;text-align:right;color:{roe_color}">{roe_label}</td></tr>'
            if debt_val:
                if debt_val <= 100: d_color, d_label = "#4caf50", "안전"
                elif debt_val <= 200: d_color, d_label = "#ff9800", "보통"
                else: d_color, d_label = "#f44336", "위험"
                fin_items += f'<tr><td style="padding:4px 8px">부채비율</td><td style="padding:4px 8px;text-align:right;color:{d_color};font-weight:bold">{debt_val:.1f}%</td><td style="padding:4px 8px;text-align:right;color:{d_color}">{d_label}</td></tr>'
            if op_val:
                if op_val >= 15: o_color, o_label = "#4caf50", "고수익"
                elif op_val >= 5: o_color, o_label = "#ff9800", "양호"
                elif op_val > 0: o_color, o_label = "#aaa", "저수익"
                else: o_color, o_label = "#f44336", "적자"
                fin_items += f'<tr><td style="padding:4px 8px">영업이익률</td><td style="padding:4px 8px;text-align:right;color:{o_color};font-weight:bold">{op_val:.1f}%</td><td style="padding:4px 8px;text-align:right;color:{o_color}">{o_label}</td></tr>'
            st.markdown(f'<div style="background:#1a1a2e;border-left:4px solid #2196f3;padding:10px 15px;border-radius:5px;margin:8px 0"><b>📊 재무 건전성</b><table style="width:100%;margin-top:8px;border-collapse:collapse;font-size:0.85em"><tr style="border-bottom:1px solid #333"><th style="padding:4px 8px;text-align:left">항목</th><th style="padding:4px 8px;text-align:right">수치</th><th style="padding:4px 8px;text-align:right">평가</th></tr>{fin_items}</table></div>', unsafe_allow_html=True)

        # 수급 흐름 차트
        fb = r.get("foreign_buys", []); ob = r.get("organ_buys", [])
        if fb and ob and len(fb) >= 3:
            try:
                import plotly.graph_objects as go
                days_label = [f"D-{len(fb)-1-i}" for i in range(len(fb))]; days_label[-1] = "오늘"
                fig_supply = go.Figure()
                fig_supply.add_trace(go.Bar(x=days_label, y=fb, name="외국인", marker_color=["#ff4444" if v < 0 else "#4488ff" for v in fb]))
                fig_supply.add_trace(go.Bar(x=days_label, y=ob, name="기관", marker_color=["#ff8844" if v < 0 else "#44cc88" for v in ob]))
                fig_supply.update_layout(title=dict(text="📈 외국인·기관 순매수 추이", font=dict(size=14)), barmode="group", height=250, margin=dict(l=10,r=10,t=40,b=30), paper_bgcolor="#0e1117", plot_bgcolor="#1a1a2e", font=dict(color="#ccc",size=11), legend=dict(orientation="h",y=1.15,x=0.5,xanchor="center"))
                st.plotly_chart(fig_supply, use_container_width=True)
            except: pass

        div_yield = r.get("dividend_yield", 0); div_amt = r.get("dividend_amt", 0)
        if div_yield > 0:
            if div_yield >= 5: dy_color, dy_label = "#4caf50", "고배당"
            elif div_yield >= 2: dy_color, dy_label = "#ff9800", "양호"
            else: dy_color, dy_label = "#aaa", "보통"
            st.markdown(f'<div style="background:#1a1a2e;border-left:4px solid #9c27b0;padding:10px 15px;border-radius:5px;margin:8px 0"><b>💰 배당</b> <span style="font-size:1.2em;color:{dy_color};font-weight:bold">{div_yield:.2f}%</span> <span style="color:{dy_color}">({dy_label})</span> <span style="color:#aaa;margin-left:15px">주당 {div_amt:,}원</span></div>', unsafe_allow_html=True)

    # Gemini AI 심층 분석
    if GEMINI_OK:
        gm_state_key = f"gm_result_{r['code']}"
        if st.button("🧠 Gemini AI 심층 분석", key=f"{prefix}_gemini_{r['code']}"):
            with st.spinner("🧠 Gemini AI가 분석 중..."):
                gm = gemini_judgment(r["name"], r["code"], r["price"], r["change"], r["score"], r["grade"], r["rsi"], r["mfi"], r["adx"], r.get("per",0), r.get("pbr",0), r.get("ev_ebitda",None), r.get("buy_reasons",[]), r.get("sell_reasons",[]), r.get("support",0), r.get("resist",0), r.get("verdict",""))
            if gm: st.session_state[gm_state_key] = gm
            else: st.warning("Gemini AI 응답을 가져올 수 없습니다.")
        if gm_state_key in st.session_state:
            st.markdown(f'<div style="background:#1a2e1a;padding:14px;border-radius:10px;margin:8px 0;border-left:4px solid #4caf50">🧠 <b>Gemini AI 심층 분석</b><br><br>{st.session_state[gm_state_key]}</div>', unsafe_allow_html=True)
    if GEMINI_OK:
        news_state_key = f"news_result_{r['code']}"
        if st.button("📰 뉴스 감성 분석", key=f"{prefix}_gemini_news_{r['code']}"):
            with st.spinner("📰 뉴스 수집 및 감성 분석 중..."):
                news_result = gemini_news_sentiment(r["name"], r["code"])
            if news_result: st.session_state[news_state_key] = news_result
            else: st.warning("뉴스를 가져올 수 없습니다.")
        if news_state_key in st.session_state:
            st.markdown(f'<div style="background:#2e2a1a;padding:14px;border-radius:10px;margin:8px 0;border-left:4px solid #ffc107">📰 <b>뉴스 감성 분석</b><br><br>{st.session_state[news_state_key]}</div>', unsafe_allow_html=True)

    # 버튼
    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
    with btn_c1:
        if show_wl_btn:
            wl = load_wl(); is_in_wl = any(w["code"] == r["code"] for w in wl)
            if is_in_wl:
                if st.button("관심 삭제", key=f"{prefix}_rm_{r['code']}"): remove_from_wl(r["code"]); st.rerun()
            else:
                if st.button("⭐ 관심 추가", key=f"{prefix}_add_{r['code']}"): add_to_wl(r["code"], r["name"]); st.rerun()
    with btn_c2:
        if st.button("차트 보기", key=f"{prefix}_chart_{r['code']}"):
            fig = draw_chart(r, cfg); st.pyplot(fig); plt.close(fig)
    with btn_c3:
        if KIS_OK:
            if st.button("⏱️ 분봉", key=f"{prefix}_min_{r['code']}"):
                df_min = fetch_minute(r["code"])
                if df_min is not None and len(df_min) > 0: st.dataframe(df_min)
                else: st.warning("분봉 데이터 없음")
    with btn_c4:
        if st.button("📰 뉴스", key=f"{prefix}_news_{r['code']}"):
            news = get_stock_news(r["code"], r["name"])
            if news:
                for n_item in news:
                    st.markdown(f'<div style="background:#1a1a2e;padding:8px 12px;border-radius:8px;margin:3px 0;border-left:3px solid #ff9800">📰 <a href="{n_item["link"]}" target="_blank" style="color:#fff;text-decoration:none">{n_item["title"]}</a> <span style="color:#888;font-size:0.8em">— {n_item["source"]}</span></div>', unsafe_allow_html=True)
            else: st.info("관련 뉴스를 찾을 수 없습니다")

    alerts = check_sell(r, cfg)
    for a in alerts:
        st.markdown(f'<div style="background:#4a1a1a;padding:8px 14px;border-radius:8px;margin:4px 0;border-left:4px solid #f44336">{a}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─── 메인 ────────────────────────────────────────────

if menu == "📡 스캔/검색":
    st.header("📡 스캔/검색")
    with st.expander("ℹ️ 사용법", expanded=False):
        st.markdown("""
- **개별 검색**: 종목명 입력 → 기술 분석 + AI 심층분석
- **전체 스캔**: 1200개 종목 자동 분석 → 85점 이상만 표시
- 결과 카드에서 📌 관심종목 등록 가능
        """)    
    stocks = get_stocks(market)
    if st.session_state["scan_done"] and not st.session_state.get("scan_running", False):
        all_results = st.session_state["all_scan_results"]
        total_found = sum(len(v) for v in all_results.values())
        st.session_state["scan_results_cache"] = all_results
        st.success(f"✅ 자동검색 완료: {total_found}개 종목 발견")
        if GEMINI_OK:
            if st.button("📰 AI 마켓 브리핑 생성", key="market_briefing_btn"):
                with st.spinner("🧠 AI가 오늘의 시장을 분석 중..."):
                    _top_stocks = []
                    for _skey, _rlist in all_results.items():
                        for _r in sorted(_rlist, key=lambda x: x["score"], reverse=True)[:5]:
                            _top_stocks.append(f"{_r['name']}({_r['code']}) {_r['score']}점 {_r['verdict']}")
                    _briefing_prompt = ("당신은 한국 주식시장 전문 애널리스트입니다.\n\n📊 스캔 결과 상위 종목:\n" + "\n".join(_top_stocks[:10]) + "\n\n아래 형식:\n1️⃣ 시장 한줄 요약\n2️⃣ 주목 종목 TOP 3\n3️⃣ 투자 전략\n4️⃣ 주의할 점\n한국어, 이모지 활용")
                    try:
                        _br_resp = gemini_model.generate_content(_briefing_prompt)
                        st.markdown(f'<div style="background:linear-gradient(135deg,#1a2e1a,#1a1a2e);padding:20px;border-radius:12px;border:2px solid #4caf50;margin:15px 0">📰 <b>AI 마켓 브리핑</b><br><br>{_br_resp.text.strip().replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
                    except: st.warning("마켓 브리핑 생성 실패")

        # 크라운
        crown_3 = []; crown_2 = []; seen_3 = set(); seen_2 = set()
        for skey, rlist in all_results.items():
            for r in rlist:
                if r.get("crown") == "🏆 3관왕" and r["code"] not in seen_3: crown_3.append(r); seen_3.add(r["code"])
                elif r.get("crown") == "⭐ 2관왕" and r["code"] not in seen_2 and r["code"] not in seen_3: crown_2.append(r); seen_2.add(r["code"])
        if crown_3:
            st.subheader("🏆 3관왕")
            for r in sorted(crown_3, key=lambda x: x["score"], reverse=True): st.markdown(f"🏆 **{r['name']}** ({r['code']}) — {r['score']}점 {r['grade']}")
        if crown_2:
            st.subheader("⭐ 2관왕")
            for r in sorted(crown_2, key=lambda x: x["score"], reverse=True): st.markdown(f"⭐ **{r['name']}** ({r['code']}) — {r['score']}점 {r['grade']}")

        surges = detect_sector_surge(all_results)
        if surges:
            st.markdown("---"); st.subheader("🔥 섹터 동반 상승 감지")
            for sg in surges:
                icon = "🚀" if sg["count"] >= 3 else "⚡"; color = "#ff1744" if sg["count"] >= 3 else "#ff9800"
                st.markdown(f'<div style="background:{color};color:#fff;padding:12px 16px;border-radius:10px;margin:6px 0">{icon} <b>{sg["theme"]}</b> — {sg["count"]}종목 | 평균 {sg["avg_score"]}점</div>', unsafe_allow_html=True)
        st.divider()

        tabs = st.tabs(list(STYLES.keys()))
        for i, (style_name, cfg) in enumerate(STYLES.items()):
            skey = cfg["key"]; rlist = all_results.get(skey, [])
            with tabs[i]:
                st.subheader(f"{style_name} — {len(rlist)}개")
                for r in sorted(rlist, key=lambda x: x["score"], reverse=True):
                    _a_per = st.session_state.get("alert_per", 0); _a_pbr = st.session_state.get("alert_pbr", 0)
                    _a_score = st.session_state.get("alert_score", 0); _a_vol = st.session_state.get("alert_vol", 0)
                    _a_pattern = st.session_state.get("alert_pattern", [])
                    _any_alert = (_a_per > 0 or _a_pbr > 0 or _a_score > 0 or _a_vol > 0 or len(_a_pattern) > 0)
                    if _any_alert:
                        _alert_match = True
                        if _a_per > 0:
                            try:
                                if float(r.get("per", 999)) > _a_per: _alert_match = False
                            except: _alert_match = False
                        if _a_pbr > 0:
                            try:
                                if float(r.get("pbr", 999)) > _a_pbr: _alert_match = False
                            except: _alert_match = False
                        if _a_score > 0 and r["score"] < _a_score: _alert_match = False
                        if _a_vol > 0:
                            try:
                                if float(r.get("vol_ratio", 0)) < _a_vol: _alert_match = False
                            except: pass
                        if _a_pattern:
                            _reasons_all = " ".join(r.get("buy_reasons", []))
                            if not any(_pat in _reasons_all for _pat in _a_pattern): _alert_match = False
                        if not _alert_match: continue
                        _alert_icon = "🔔 "
                    else: _alert_icon = ""
                    with st.expander(f"{_alert_icon}{r.get('crown','')} {r['name']} ({r['code']}) — {r['score']}점 {r['grade']} | {r['verdict']}", expanded=False):
                        show_card(r, f"scan_{skey}_{r['code']}", cfg)
        st.divider()

    st.subheader("🔍 종목 검색")
    query = st.text_input("종목명 또는 코드 입력")
    if query:
        query = query.strip()
        exact_match = stocks[stocks["Code"].str.upper() == query.upper()]
        matched = exact_match if not exact_match.empty else stocks[stocks["Name"].str.contains(query, case=False, na=False) | stocks["Code"].str.upper().str.contains(query.upper(), na=False)]
        if matched.empty: st.warning("검색 결과가 없습니다.")
        else:
            style_name = st.selectbox("분석 스타일", list(STYLES.keys()), key="search_style"); cfg = STYLES[style_name]
            for _, row in matched.head(10).iterrows():
                code = str(row["Code"]).strip(); name = str(row["Name"]).strip()
                with st.expander(f"📋 {name} ({code})", expanded=True):
                    df = fetch(code)
                    if df is not None: r = analyze(df, code, name, cfg, market=market); r["crown"] = ""; show_card(r, f"search_{code}", cfg)
                    else: st.error(f"{name} 데이터를 불러올 수 없습니다.")
    st.divider()

    if st.button("🚀 전체 스캔 시작 (3개 스타일 동시)", type="primary", use_container_width=True):
        if stocks.empty: st.error("종목 리스트를 불러올 수 없습니다.")
        else:
            st.session_state["scan_running"] = True
            all_results, cached_data = run_full_scan(stocks)
            st.session_state["all_scan_results"] = all_results; st.session_state["cached_data"] = cached_data
            st.session_state["scan_done"] = True; st.session_state["scan_running"] = False
            save_perf_snapshot(all_results)
            for skey, rlist in all_results.items():
                records = [{"code": r["code"], "name": r["name"], "score": r["score"], "verdict": r["verdict"], "crown": r.get("crown","")} for r in rlist]
                if records: add_to_history_direct(skey, records)
            surges = detect_sector_surge(all_results)
            if surges:
                SECTOR_FILE = "sector_history.json"
                if os.path.exists(SECTOR_FILE):
                    with open(SECTOR_FILE, "r", encoding="utf-8") as f: sector_hist = json.load(f)
                else: sector_hist = []
                sector_hist.append({"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "surges": surges})
                with open(SECTOR_FILE, "w", encoding="utf-8") as f: json.dump(sector_hist, f, ensure_ascii=False, indent=2)
            st.rerun()

elif menu == "👀 내 종목 감시":
    st.header("👀 내 종목 감시")
    with st.expander("ℹ️ 사용법", expanded=False):
        st.markdown("""
- 스캔/검색에서 ⭐ 버튼으로 관심종목 등록
- 등록된 종목의 실시간 가격/신호 모니터링
- 자동 갱신 켜면 60초마다 업데이트
        """)

    wl = load_wl()
    if not wl: st.info("감시 종목이 없습니다. 스캔/검색에서 ⭐ 버튼으로 추가하세요.")
    else:
        style_name = st.session_state.get("wl_style_side", list(STYLES.keys())[0]); cfg = STYLES[style_name]
        if st.session_state.get("wl_clear_side"): save_wl([]); st.rerun()
        if AUTOREFRESH_OK:
            if st.checkbox("⏱️ 자동 갱신 (60초)", value=False, key="wl_auto"): st_autorefresh(interval=60000, key="wl_refresh")
        for item in wl:
            code = str(item["code"]).strip(); name = item["name"]; df = fetch(code)
            if df is not None:
                r = analyze(df, code, name, cfg); r["crown"] = ""
                chg_c = "#ff1744" if r["change"] >= 0 else "#2979ff"
                sc = "#4caf50" if r["score"] >= 80 else "#ff9800" if r["score"] >= 60 else "#f44336"
                st.markdown(f'<div style="background:#1a1a2e;padding:10px 14px;border-radius:10px;margin:4px 0;display:flex;justify-content:space-between"><div><b>{name}</b> ({code}) <span style="color:{chg_c};font-weight:bold">{r["price"]:,}원 ({r["change"]:+.2f}%)</span></div><div><span style="font-size:1.4em;font-weight:bold;color:{sc}">{r["score"]}점</span> {r["grade"]}</div></div>', unsafe_allow_html=True)
                with st.expander("상세 분석 보기", expanded=False):
                    prev = st.session_state["prev_signals"].get(code, {"buy": [], "sell": []})
                    new_buy = [b for b in r["buy_reasons"] if b not in prev.get("buy", [])]
                    new_sell = [s for s in r["sell_reasons"] if s not in prev.get("sell", [])]
                    st.session_state["prev_signals"][code] = {"buy": r["buy_reasons"], "sell": r["sell_reasons"]}
                    is_new_buy = len(new_buy) > 0; is_new_sell = len(new_sell) > 0
                    if sound_on and (is_new_buy or is_new_sell):
                        if is_new_buy: st.markdown(f'<audio autoplay><source src="{SOUND_BUY}"></audio>', unsafe_allow_html=True)
                        if is_new_sell: st.markdown(f'<audio autoplay><source src="{SOUND_SELL}"></audio>', unsafe_allow_html=True)
                    show_card(r, f"wl_{code}", cfg, is_new_buy=is_new_buy, is_new_sell=is_new_sell)
            else: st.error(f"{name} 데이터 불러오기 실패")

elif menu == "📊 성과 리포트":
    st.header("📊 성과 리포트")
    with st.expander("ℹ️ 사용법", expanded=False):
        st.markdown("""
- 전체 스캔 후 자동 저장된 추천 종목의 수익률 추적
- 과거 추천이 실제로 올랐는지 확인 가능
        """)

    perf_data = load_perf_snapshot()
    if not perf_data: st.info("아직 스캔 기록이 없습니다.")
    else:
        if st.button("📊 성과 분석 실행", type="primary", use_container_width=True):
            with st.spinner("전일 추천 종목 현재가 조회 중..."): report = generate_perf_report()
            if report:
                win_color = "#4caf50" if report["win_rate"] >= 50 else "#f44336"
                st.markdown(f'<div style="background:#1a1a2e;padding:20px;border-radius:12px;border:2px solid {win_color}"><div style="display:flex;justify-content:space-around;text-align:center"><div><span style="font-size:2em;color:{win_color}">{report["win_rate"]}%</span><br>승률</div><div><span style="font-size:2em">{report["avg_pnl"]:+.2f}%</span><br>평균수익</div><div>{report["wins"]}승/{report["losses"]}패</div></div></div>', unsafe_allow_html=True)
                for r in report["results"]:
                    pnl_c = "#4caf50" if r["pnl"] >= 0 else "#f44336"
                    st.markdown(f'{r["status"]} **{r["name"]}** — {r["entry_price"]:,}원 → {r["current_price"]:,}원 <span style="color:{pnl_c}">{r["pnl"]:+.2f}%</span>', unsafe_allow_html=True)
                st.divider(); st.subheader("🔬 지표별 승률 분석")
                with st.spinner("지표별 성과 계산 중..."): ind_report = generate_indicator_report()
                if ind_report:
                    for ind in ind_report:
                        wr = ind["win_rate"]; wr_color = "#4caf50" if wr >= 60 else "#ff9800" if wr >= 40 else "#f44336"
                        st.markdown(f'<div style="background:#1a1a2e;padding:10px 14px;border-radius:10px;margin:4px 0"><b>{ind["indicator"]}</b> — {ind["total"]}건 | <span style="color:{wr_color}">{wr}%</span> 승률 | {ind["avg_pnl"]:+.2f}% 평균</div>', unsafe_allow_html=True)
            else: st.warning("성과 데이터를 생성할 수 없습니다.")

elif menu == "📜 스캔 기록":
    st.header("📜 스캔 기록")
    with st.expander("ℹ️ 사용법", expanded=False):
        st.markdown("""
- 과거 전체 스캔 결과 기록 확인
- 날짜별로 어떤 종목이 발견됐는지 조회
        """)

    hist = load_history()
    if not hist: st.info("스캔 기록이 없습니다.")
    else:
        if st.button("🗑️ 기록 전체 삭제"): clear_history(); st.rerun()
        tabs = st.tabs(["단타", "스윙", "중장기"])
        for ti, skey in enumerate(["short", "swing", "long"]):
            with tabs[ti]:
                filtered = [(i, h) for i, h in enumerate(hist) if isinstance(h, dict) and h.get("style") == skey]
                if not filtered: st.info("해당 스타일 기록 없음"); continue
                cfg = STYLES[KEY_TO_STYLE[skey]]
                for orig_idx, entry in filtered:
                    with st.expander(f"📅 {entry['date']} — {len(entry['stocks'])}종목", expanded=False):
                        for stock_info in entry["stocks"]:
                            code = str(stock_info["code"]).strip(); name = stock_info["name"]
                            st.markdown(f"**{name}** ({code}) — 당시 {stock_info.get('score','?')}점")
                            df = fetch(code)
                            if df is not None:
                                r = analyze(df, code, name, cfg); r["crown"] = ""
                                show_card(r, f"hist_{skey}_{orig_idx}_{code}", cfg)

elif menu == "🔥 섹터 동반 상승":
    st.header("🔥 섹터 동반 상승 기록")
    with st.expander("ℹ️ 사용법", expanded=False):
        st.markdown("""
- 같은 섹터/테마에서 여러 종목이 동시에 강세일 때 감지
- 섹터 전체가 움직이는 큰 흐름 포착
        """)

    SECTOR_FILE = "sector_history.json"
    if os.path.exists(SECTOR_FILE):
        with open(SECTOR_FILE, "r", encoding="utf-8") as f: sector_hist = json.load(f)
    else: sector_hist = []
    if not sector_hist: st.info("아직 섹터 동반 상승 기록이 없어요.")
    else:
        for ri, record in enumerate(reversed(sector_hist)):
            with st.expander(f"📅 {record.get('time','')} — {len(record.get('surges',[]))}개 섹터", expanded=(ri==0)):
                for sg in record.get("surges", []):
                    icon = "🚀" if sg["count"] >= 3 else "⚡"; color = "#ff1744" if sg["count"] >= 3 else "#ff9800"
                    st.markdown(f'<div style="background:{color};color:#fff;padding:12px 16px;border-radius:10px;margin:6px 0">{icon} <b>{sg["theme"]}</b> — {sg["count"]}종목 | 평균 {sg["avg_score"]}점</div>', unsafe_allow_html=True)
                    for s in sg.get("stocks", []): st.write(f"• {s['name']} ({s['code']}) — {s['score']}점")

elif menu == "🪙 코인 선물":
    st.header("🪙 코인 선물 분석")
    with st.expander("ℹ️ 사용법", expanded=False):
        st.markdown("""
- 바이낸스 코인 선물 50개 자동 분석
- 롱/숏 신호 + 펀딩비/롱숏비율/OI 데이터
- 가상매매로 실전 연습 가능 (실제 돈 X)
        """)

    check_paper_tpsl()
    coin_tabs = st.tabs(["📊 전체 스캔", "🔍 개별 분석", "⭐ 관심종목", "💰 가상매매"])
    with coin_tabs[0]:
        st.subheader("📊 코인 선물 전체 스캔")
        timeframe = st.selectbox("분석 기간", ["1h","4h","1d"], index=2, key="coin_tf")
        min_score_coin = st.slider("최소 점수", 0, 100, 50, key="coin_min_score")
        if st.button("🔍 코인 전체 스캔 시작", key="coin_scan_btn", use_container_width=True):
            results = []; prog = st.progress(0); total = len(COIN_FUTURES)
            for i, symbol in enumerate(COIN_FUTURES):
                prog.progress((i+1)/total)
                try:
                    df = get_coin_klines(symbol, interval=timeframe); funding = get_funding_rate(symbol)
                    ls_ratio = get_long_short_ratio(symbol); oi = get_open_interest(symbol)
                    r = analyze_coin(df, symbol, funding, ls_ratio, oi)
                    if r and r["score"] >= min_score_coin: results.append(r)
                except: pass
                time.sleep(0.1)
            prog.empty()
            st.session_state["coin_scan_results"] = [r for r in results if r is not None]
            st.success(f"✅ {len(st.session_state['coin_scan_results'])}개 종목 감지")
        if "coin_scan_results" in st.session_state:
            for r in sorted(st.session_state["coin_scan_results"], key=lambda x: x.get("score",0), reverse=True):
                if not isinstance(r, dict): continue
                with st.expander(f"{r['position']} {r['symbol']} | {r['score']}점 | ${r['price']:,.4f} ({r['change']:+.2f}%)", expanded=False):
                    if r.get("buy"): st.success("매수: " + " | ".join(r["buy"][:5]))
                    if r.get("sell"): st.error("매도: " + " | ".join(r["sell"][:5]))
                    try: fig = draw_coin_chart(r); st.pyplot(fig); plt.close(fig)
                    except: pass

    with coin_tabs[1]:
        st.subheader("🔍 코인 개별 분석")
        coin_input = st.text_input("코인 심볼 (예: BTCUSDT)", value="BTCUSDT", key="coin_single")
        if st.button("🔍 분석", key="coin_analyze"):
            symbol = coin_input.upper().strip()
            if not symbol.endswith("USDT"): symbol += "USDT"
            with st.spinner(f"{symbol} 분석 중..."):
                df = get_coin_klines(symbol); funding = get_funding_rate(symbol)
                ls_ratio = get_long_short_ratio(symbol); oi = get_open_interest(symbol)
                r = analyze_coin(df, symbol, funding, ls_ratio, oi)
            if r:
                st.session_state["coin_detail"] = r
                score_color = "#ff4444" if r["score"] >= 80 else "#ff8800" if r["score"] >= 60 else "#4488ff"
                st.markdown(f'<div style="background:#1a1a2e;padding:20px;border-radius:15px"><h2 style="color:white">{symbol} ${r["price"]:,.4f} ({r["change"]:+.2f}%)</h2><p style="color:{score_color};font-size:24px">점수: {r["score"]} | {r["position"]}</p></div>', unsafe_allow_html=True)
                if r.get("buy"): st.success("매수: " + " | ".join(r["buy"][:5]))
                if r.get("sell"): st.error("매도: " + " | ".join(r["sell"][:5]))
                if GEMINI_OK:
                    if st.button("🧠 Gemini AI 코인 분석", key=f"gemini_coin_{symbol}"):
                        with st.spinner("🧠 분석 중..."):
                            gm_coin = gemini_coin_judgment(symbol, r["price"], r["change"], r.get("rsi",0), r.get("mfi",0), r.get("adx",0), funding, ls_ratio, None, r["score"], r["grade"], r.get("buy",[]), r.get("sell",[]), 0, 0)
                        if gm_coin: st.markdown(f'<div style="background:#1a2e1a;padding:14px;border-radius:10px;border-left:4px solid #4caf50">🧠 {gm_coin}</div>', unsafe_allow_html=True)
                try: fig = draw_coin_chart(r); st.pyplot(fig); plt.close(fig)
                except: pass
                # 가상매매 진입
                st.markdown("---"); st.markdown("**⚡ 바로 가상매매 진입**")
                pt_c1, pt_c2 = st.columns(2)
                with pt_c1: pt_lev = st.selectbox("레버리지", [5,10,20,25,50,100], index=1, key=f"pt_lev_{symbol}")
                with pt_c2: pt_qty = st.number_input("투자금 (USDT)", min_value=10.0, value=100.0, step=10.0, key=f"pt_qty_{symbol}")
                btn_l, btn_s = st.columns(2)
                with btn_l:
                    if st.button("🟢 롱 진입", key=f"pt_long_{symbol}", use_container_width=True, type="primary"):
                        open_paper_trade(symbol, "LONG", r["price"], pt_lev, pt_qty); st.success(f"🟢 롱 진입! ${r['price']:,.4f}"); time.sleep(1); st.rerun()
                with btn_s:
                    if st.button("🔴 숏 진입", key=f"pt_short_{symbol}", use_container_width=True):
                        open_paper_trade(symbol, "SHORT", r["price"], pt_lev, pt_qty); st.success(f"🔴 숏 진입! ${r['price']:,.4f}"); time.sleep(1); st.rerun()
            else: st.error("분석 데이터 부족")

    with coin_tabs[2]:
        st.subheader("⭐ 코인 관심종목"); cwl = load_coin_wl()
        new_coin = st.text_input("종목 추가 (예: BTCUSDT)", key="cwl_add_input")
        if st.button("➕ 추가", key="cwl_add_btn"):
            sym = new_coin.upper().strip()
            if sym and not sym.endswith("USDT"): sym += "USDT"
            if sym and sym not in cwl: cwl.append(sym); save_coin_wl(cwl); st.rerun()
        if cwl and st.button("🔄 관심종목 분석", key="cwl_scan_btn", use_container_width=True):
            cwl_results = []
            for sym in cwl:
                try:
                    df = get_coin_klines(sym); r = analyze_coin(df, sym, get_funding_rate(sym), get_long_short_ratio(sym), get_open_interest(sym))
                    if r: cwl_results.append(r)
                except: continue
            st.session_state["cwl_results"] = cwl_results
        if "cwl_results" in st.session_state:
            for r in st.session_state["cwl_results"]:
                with st.expander(f"{r['position']} {r['symbol']} | {r['score']}점 | ${r['price']:,.4f}", expanded=False):
                    if r.get("buy"): st.success("매수: " + " | ".join(r["buy"][:3]))
                    if r.get("sell"): st.error("매도: " + " | ".join(r["sell"][:3]))
                    if st.button(f"❌ {r['symbol']} 삭제", key=f"cwl_del_{r['symbol']}"):
                        cwl = load_coin_wl();
                        if r["symbol"] in cwl: cwl.remove(r["symbol"]); save_coin_wl(cwl); st.rerun()

    with coin_tabs[3]:
        st.subheader("💰 가상매매"); trades = load_coin_trades()
        open_trades = [t for t in trades if isinstance(t, dict) and t.get("status") == "open"]
        stats = get_paper_stats()
        if stats is None: stats = {"total": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0}
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("총 거래", stats.get("total",0)); sc2.metric("승률", f"{stats.get('win_rate',0):.1f}%")
        sc3.metric("총 수익", f"${stats.get('total_pnl',0):,.2f}"); sc4.metric("평균 수익", f"${stats.get('avg_pnl',0):,.2f}")
        st.markdown("### 📂 진행 중인 포지션")
        if not open_trades: st.info("진행 중인 가상매매가 없습니다.")
        for t in open_trades:
            try: cur_price = float(requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={t['symbol']}", timeout=5).json()["price"])
            except: cur_price = t["entry_price"]
            if t["side"] == "LONG": pnl_pct = round((cur_price - t["entry_price"]) / t["entry_price"] * 100 * t["leverage"], 2)
            else: pnl_pct = round((t["entry_price"] - cur_price) / t["entry_price"] * 100 * t["leverage"], 2)
            pnl_color = "#00ff88" if pnl_pct >= 0 else "#ff4444"
            side_icon = "🟢" if t["side"] == "LONG" else "🔴"
            st.markdown(f'<div style="background:#1e1e2e;padding:15px;border-radius:10px;margin:10px 0;border-left:4px solid {pnl_color}"><b>{side_icon} {t["symbol"]} {t["side"]} {t["leverage"]}x</b> | 진입: ${t["entry_price"]:,.4f} → 현재: ${cur_price:,.4f} | <span style="color:{pnl_color};font-size:20px;font-weight:bold">{pnl_pct:+.2f}%</span></div>', unsafe_allow_html=True)
            if st.button(f"💰 {t['symbol']} 종료", key=f"close_{t.get('id',t['symbol'])}"):
                close_paper_trade(t["id"], cur_price); st.success(f"종료! {pnl_pct:+.2f}%"); time.sleep(1); st.rerun()

elif menu == "🚀 급등 사냥":
    st.markdown("## 🚀 급등 사냥")
    with st.expander("ℹ️ 사용법", expanded=False):
        st.markdown("""
- 종목명 입력 → 과거 급등 패턴 분석
- 거래량 폭증 + 가격 급등 구간 자동 감지
        """)

    if not FDR_OK: st.error("FinanceDataReader 미설치")
    else:
        surge_query = st.text_input("종목명 또는 코드", key="surge_single_query")
        if surge_query:
            found = None
            for mkt in ["KOSPI", "KOSDAQ"]:
                stocks_df = get_stocks(mkt)
                if stocks_df.empty: continue
                match = stocks_df[stocks_df["Name"].str.contains(surge_query, case=False, na=False) | stocks_df["Code"].str.contains(surge_query, case=False, na=False)]
                if not match.empty: found = match.iloc[0]; break
            if found is not None:
                code = str(found["Code"]).strip(); name = str(found["Name"]).strip()
                df = fetch(code, days=300)
                if df is not None and len(df) >= 60:
                    r = analyze_surge(df, code, name)
                    if r and r["score"] >= 10:
                        st.markdown(f"**{r['grade']} {r['name']}** ({r['code']}) — {r['score']}점 {r['verdict']}")
                        for sig in r["signals"]: st.markdown(f"- {sig}")
                    else: st.info("급등 신호 미감지")
                else: st.error("데이터 부족")
            else: st.warning("종목을 찾을 수 없습니다.")
        st.divider()
        if st.button("🚀 급등 사냥 시작!", use_container_width=True, type="primary"):
            all_stocks = pd.DataFrame()
            for mkt in ["KOSPI", "KOSDAQ"]:
                tmp = get_stocks(mkt)
                if not tmp.empty: tmp["market"] = mkt; all_stocks = pd.concat([all_stocks, tmp], ignore_index=True)
            if all_stocks.empty: st.warning("종목 리스트 불러오기 실패")
            else:
                total = len(all_stocks); progress = st.progress(0); status = st.empty(); results = []
                for idx, row in all_stocks.iterrows():
                    code = str(row["Code"]).strip(); name = str(row["Name"]).strip()
                    if idx % 50 == 0: progress.progress((idx+1)/total); status.text(f"분석 중... {idx+1}/{total} | {name}")
                    try:
                        df = fetch(code, days=300)
                        if df is None or len(df) < 60: continue
                        vol_mean = df["Volume"].rolling(20).mean().iloc[-1]
                        if df["Volume"].iloc[-1] / (vol_mean + 1e-9) < 1.5: continue
                        r = analyze_surge(df, code, name)
                        if r and r["score"] >= 40: r["market"] = row["market"]; results.append(r)
                    except: continue
                progress.empty(); status.empty()
                if results:
                    results.sort(key=lambda x: x["score"], reverse=True)
                    add_surge_record(results, "한국"); st.success(f"🎯 {len(results)}개 발견!")
                    for i, r in enumerate(results[:20]):
                        st.markdown(f"**{i+1}. {r['grade']} {r['name']}** ({r['code']}) — {r['score']}점 | {r['price']:,}원 ({r['change']:+.2f}%)")
                        with st.expander(f"신호 ({len(r['signals'])}개)", expanded=(i<3)):
                            for sig in r["signals"]: st.markdown(f"- {sig}")
                else: st.info("급등 후보 없음")

elif menu == "🌊 외국인 수급 추적":
    st.markdown("## 🌊 외국인 수급 추적")
    if st.button("🔍 외국인 수급 스캔 시작", use_container_width=True, type="primary"):
        with st.spinner("수급 데이터 수집 중..."):
            try:
                url = "https://www.truefriend.com/main/research/research/Sell.jsp"
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15); resp.encoding = "utf-8"
                soup = BeautifulSoup(resp.text, "html.parser"); tables = soup.find_all("table")
                def parse_table(table):
                    results = []
                    for row in table.find_all("tr"):
                        cols = row.find_all("td")
                        if len(cols) >= 4:
                            results.append({"sell_name": cols[0].get_text(strip=True), "sell_amount": cols[1].get_text(strip=True), "buy_name": cols[2].get_text(strip=True), "buy_amount": cols[3].get_text(strip=True)})
                    return results
                all_data = [parse_table(t) for t in tables if len(parse_table(t)) >= 3]
                if len(all_data) >= 2:
                    inst_data = all_data[0]; foreign_data = all_data[1]
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### 📈 외국인 순매수")
                        for item in foreign_data: st.markdown(f'🟢 **{item["buy_name"]}** — {item["buy_amount"]}백만')
                    with c2:
                        st.markdown("### 📉 외국인 순매도")
                        for item in foreign_data: st.markdown(f'🔴 **{item["sell_name"]}** — {item["sell_amount"]}백만')
                    overlap_buy = [item["buy_name"] for item in foreign_data for inst in inst_data if item["buy_name"] == inst["buy_name"]]
                    if overlap_buy: st.success(f"🎯 외국인+기관 동시 순매수: {', '.join(set(overlap_buy))}")
                    st.success("✅ 스캔 완료!")
                else: st.warning("데이터 파싱 실패")
            except Exception as e: st.error(f"스캔 오류: {e}")

elif menu == "💰 매매":
    st.header("💰 매매")
    if not KIS_OK: st.error("KIS API 미연결")
    else: st.info("KIS API 매매 기능은 로컬 전용입니다.")

elif menu == "🎯 투자 브리핑":
    st.header("🎯 AI 투자 브리핑")
    with st.expander("ℹ️ 사용법", expanded=False):
        st.markdown("""
- 종목명 포함해서 질문하면 실시간 데이터 자동 수집
- 실시간 뉴스/공시 기반 답변 (Google 검색 연동)
- 예: "삼성전자 지금 사도 될까?", "오늘 시장 어때?"
        """)
    st.caption("실시간 뉴스·공시 기반 AI 투자 상담")

    if not GEMINI_OK:
        st.warning("🔑 사이드바에서 Gemini API 키를 입력하면 투자 브리핑을 이용할 수 있습니다.")
    else:
        BRIEFING_PERSONA = """당신은 '세이브티커(SaveTicker)' 소속 실시간 모니터링 요원이자 패턴 매칭 전문가입니다.

[정체성]
- 한국 주식 시장의 실시간 공시(DART), 뉴스, 수급 흐름을 24시간 감시하는 역할
- 차트 패턴(상승삼각형, 역머리어깨형, 상승깃발형, 박스권 등)을 절대 기준으로 정밀 분석
- 시장의 주관적 예측은 배제하고, [확정 호재] + [외인·기관 수급] + [기술적 패턴] 일치 시에만 신호 발생

[소통 원칙]
- 두괄식 구조, 볼드체, 기호 활용 → 가독성과 스캔 가능성 최우선
- 딱딱한 금융 리포트가 아닌, 위트 있고 든든한 투자 파트너 어조
- 조건 충족 시 "사라! 🔥", "기다려! ⏳", "도망쳐! 🏃" 등 직관적 신호
- 확신 없으면 절대 추천하지 않음 → "아직 아니야 ⚪" 로 명확히 표현
- 항상 마지막에 "⚠️ 투자 판단의 책임은 본인에게 있습니다" 명시

[분석 프레임워크]
1. 📢 호재 체크: 공시/뉴스에서 확정된 호재가 있는가?
2. 🌊 수급 체크: 외국인·기관이 순매수 중인가?
3. 📊 패턴 체크: 차트에서 상승 패턴이 확인되는가?
4. 💰 밸류 체크: PER/PBR/EV/EBITDA 기준 저평가인가?
→ 3개 이상 일치 = "사라! 🔥"
→ 2개 일치 = "관심 종목 등록 👀"
→ 1개 이하 = "아직 아니야 ⚪"

[답변 형식]
항상 아래 형식을 지켜주세요:
━━━━━━━━━━━━━━━━━━━━
🎯 **[종목명/주제]** — [한줄 판정]
━━━━━━━━━━━━━━━━━━━━
📢 **호재:** (있으면 구체적으로 / 없으면 "특이사항 없음")
🌊 **수급:** (외국인/기관 동향)
📊 **패턴:** (차트 패턴 분석)
💰 **밸류:** (PER/PBR/EV 등)
━━━━━━━━━━━━━━━━━━━━
✅ **결론:** [사라!/기다려!/도망쳐!/아직 아니야]
📌 **전략:** [구체적 진입가/목표가/손절가]
━━━━━━━━━━━━━━━━━━━━
⚠️ 본 분석은 정보 제공 목적이며, 투자 판단의 책임은 본인에게 있습니다.

한국어로 답변하고, 이모지를 적극 활용하세요."""

        if "briefing_history" not in st.session_state:
            st.session_state["briefing_history"] = []

        # 이전 대화 표시
        for msg in st.session_state["briefing_history"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="background:#2a2a4e;padding:12px;border-radius:10px;margin:8px 0;border-left:4px solid #ff9800">'
                    f'🧑 <b>질문:</b> {msg["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="background:#1a2e1a;padding:12px;border-radius:10px;margin:8px 0;border-left:4px solid #4caf50">'
                    f'🎯 {msg["text"]}</div>', unsafe_allow_html=True)

        # 자유 질문
        user_input = st.text_area("질문을 입력하세요", placeholder="예: 삼성전자 지금 사도 될까? / 오늘 시장 어때? / 2차전지 전망은?", key="briefing_input", height=100)

        col_send, col_clear = st.columns([1, 1])
        with col_send:
            send_btn = st.button("📤 브리핑 요청", key="briefing_send", use_container_width=True, type="primary")
        with col_clear:
            if st.button("🗑️ 대화 초기화", key="briefing_clear", use_container_width=True):
                st.session_state["briefing_history"] = []
                st.rerun()

        if send_btn and user_input:
            st.session_state["briefing_history"].append({"role": "user", "text": user_input})
            with st.spinner("🎯 브리핑 생성 중..."):
                try:
                    # 종목명 감지 → 실제 데이터 수집
                    stock_data = ""
                    try:
                        stocks_df = get_stocks(market)
                        for _word in user_input.replace(",", " ").split():
                            _match = stocks_df[stocks_df["Name"].str.contains(_word, case=False, na=False)]
                            if not _match.empty:
                                _row = _match.iloc[0]
                                _code = str(_row["Code"]).strip()
                                _name = str(_row["Name"]).strip()
                                _df = fetch(_code)
                                if _df is not None:
                                    _cfg = list(STYLES.values())[0]
                                    _r = analyze(_df, _code, _name, _cfg, market=market)
                                    stock_data = (
                                        f"\n[{_name}({_code}) 실시간 데이터]\n"
                                        f"현재가: {_r['price']:,}원 ({_r['change']:+.2f}%)\n"
                                        f"AI점수: {_r['score']}점 ({_r['grade']}) | 판정: {_r['verdict']}\n"
                                        f"RSI: {_r['rsi']} | MFI: {_r['mfi']} | ADX: {_r['adx']}\n"
                                        f"PER: {_r.get('per',0)} | PBR: {_r.get('pbr',0)}\n"
                                        f"지지선: {_r['support']:,}원 | 저항선: {_r['resist']:,}원\n"
                                        f"매수신호: {', '.join(_r.get('buy_reasons',[])[:5])}\n"
                                        f"매도신호: {', '.join(_r.get('sell_reasons',[])[:5])}\n"
                                        f"거래량비: {_r['vol_ratio']}배\n"
                                    )
                                break
                    except:
                        pass

                    context = ""
                    for msg in st.session_state["briefing_history"][-10:]:
                        if msg["role"] == "user": context += f"사용자: {msg['text']}\n"
                        else: context += f"AI: {msg['text']}\n"

                    full_prompt = BRIEFING_PERSONA + "\n\n이전 대화:\n" + context + "\n" + stock_data + "\n질문: " + user_input
                    response = gemini_model.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            tools=[types.Tool(google_search=types.GoogleSearch())]
                        )
                    )
                    ai_text = response.text.strip()
                    st.session_state["briefing_history"].append({"role": "ai", "text": ai_text})
                except Exception as e:
                    st.session_state["briefing_history"].append({"role": "ai", "text": f"오류 발생: {str(e)[:100]}"})
            st.rerun()

        # 추천 질문
        st.divider()
        st.markdown("**💡 추천 질문:**")
        suggest_cols = st.columns(3)
        suggestions = [
            "오늘 시장 어때?",
            "이번 주 급등 후보 종목은?",
            "외국인이 지금 뭘 사고 있어?",
            "2차전지 섹터 전망은?",
            "비트코인 지금 들어가도 돼?",
            "초보자 추천 투자 전략",
        ]
        for i, sug in enumerate(suggestions):
            with suggest_cols[i % 3]:
                if st.button(sug, key=f"suggest_{i}"):
                    st.session_state["briefing_history"].append({"role": "user", "text": sug})
                    with st.spinner("🎯 브리핑 생성 중..."):
                        try:
                            response = gemini_model.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=BRIEFING_PERSONA + "\n\n질문: " + sug,
                                config=types.GenerateContentConfig(
                                    tools=[types.Tool(google_search=types.GoogleSearch())]
                                )
                            )
                            st.session_state["briefing_history"].append({"role": "ai", "text": response.text.strip()})
                        except Exception as e:
                            st.session_state["briefing_history"].append({"role": "ai", "text": f"오류: {str(e)[:100]}"})
                    st.rerun()

# ── 푸터 ──
st.divider()
st.caption(f"🔥 급등 예측 탐색기 v24.0 PRO (서버용) | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | ⚠️ 투자 판단의 책임은 본인에게 있습니다.")
