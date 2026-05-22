# ─── stock_app.py  v24.0 급등 예측 탐색기 ─────────────
import streamlit as st
import pandas as pd
import numpy as np
import datetime, json, os, time, re, requests

# ── 섹터별 평균 PER/PBR (2026.04 기준) ──
SECTOR_PER = {
    # 코스피
    "음식료품": 21.07, "음식점및주점업": 2.54, "섬유의복": 11.04,
    "종이목재": 31.5, "화학": 32.53, "의약품": 72.73,
    "비금속광물": 28.38, "철강금속": 27.95, "기계": 89.31,
    "전기전자": 312.33, "의료정밀": 75.45, "운수장비": 32.26,
    "유통업": 58.73, "전기가스업": 8.48, "건설업": 21.85,
    "운수창고업": 7.24, "통신업": 31.23, "금융업": 26.21,
    "보험": 10.57, "서비스업": 14.13, "제조업": 20.0,
    # 코스닥
    "소프트웨어": 46.25, "IT부품": 124.56, "반도체": 124.56,
    "IT하드웨어": 124.56, "바이오": 118.06, "게임": 29.7,
    "인터넷": 19.82, "미디어": 41.26, "엔터테인먼트": 29.7,
    "화장품": 42.18, "2차전지": 99.01, "자동차부품": 91.36,
    "기계장비": 64.3, "음식료": 10.53, "건설": 14.13,
    "교육": 7.69,
}
MARKET_AVG_PER = 15.0  # 전체 시장 평균 (섹터 못 찾을 때 기본값)

def get_sector_per(sector_name):
    """종목의 섹터명으로 해당 섹터 평균 PER 반환"""
    if not sector_name:
        return MARKET_AVG_PER
    clean = sector_name.replace("·", "").replace(",", "").replace(" ", "")
    for key in SECTOR_PER:
        clean_key = key.replace("·", "").replace(",", "").replace(" ", "")
        if clean_key in clean or clean in clean_key:
            return SECTOR_PER[key]
    return MARKET_AVG_PER


def judge_per_by_sector(per, sector_name):
    """섹터 평균 대비 PER 고평가/저평가 판단"""
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
GEMINI_OK = False
gemini_model = None

        
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

        # ── 외국인/기관 수급 데이터 가져오기 ──
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
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini 오류: {e}")
        return None


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
    """주봉 기준 추세 판단: 상승/하락/횡보 — 일봉을 주봉으로 변환"""
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
    except Exception as e:
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
    """바이낸스 선물 캔들 데이터"""
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
    """현재 펀딩비"""
    try:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        r = requests.get(url, params={"symbol": symbol}, timeout=5)
        data = r.json()
        return round(float(data.get("lastFundingRate", 0)) * 100, 4)
    except:
        return None

@st.cache_data(ttl=30)
def get_long_short_ratio(symbol, period="5m"):
    """롱숏 비율"""
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
    """미결제약정"""
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        r = requests.get(url, params={"symbol": symbol}, timeout=5)
        data = r.json()
        return round(float(data.get("openInterest", 0)), 2)
    except:
        return None

@st.cache_data(ttl=30)
def get_oi_history(symbol, period="5m", limit=30):
    """미결제약정 변화"""
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
    """최근 펀딩비 히스토리"""
    try:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=5)
        data = r.json()
        return [round(float(d.get("fundingRate", 0)) * 100, 4) for d in data]
    except:
        return []

def analyze_coin(df, symbol, funding, ls_ratio, oi):
    """코인 선물 분석 (고급 지표 포함)"""
    if df is None or len(df) < 30:
        print(f"⚠️ 데이터 부족: {symbol}, len={len(df) if df is not None else 'None'}")
        return None
    try:
        print(f"🔍 1단계 시작: {symbol}")
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
        print(f"🔍 2단계 완료: price={price}, change={change}")

        # ── 기술지표 계산 ──
        rsi = calc_rsi(c)
        rsi_val = round(rsi.iloc[-1], 1) if not np.isnan(rsi.iloc[-1]) else 50

        stoch_rsi = calc_stoch_rsi(c)
        stoch_val = round(stoch_rsi.iloc[-1] * 100, 1) if not np.isnan(stoch_rsi.iloc[-1]) else 50

        macd_line, macd_sig, macd_hist = calc_macd(c)
        bb_upper, bb_mid, bb_lower = calc_bb(c)

        ema5 = c.ewm(span=5).mean()
        ema20 = c.ewm(span=20).mean()
        ema60 = c.ewm(span=60).mean()
        e5 = df['Close'].ewm(span=5).mean()
        e20 = df['Close'].ewm(span=20).mean()
        e60 = df['Close'].ewm(span=60).mean()
        # SuperTrend
        coin_df = pd.DataFrame({"High": h, "Low": l, "Close": c, "Volume": v})
        st_line, st_dir = calc_supertrend(coin_df)

        # OBV
        obv = calc_obv(c, v)
        obv_ma = obv.rolling(20).mean()

        # VWAP
        vwap = calc_vwap(coin_df)

        # MFI
        mfi = calc_mfi(coin_df)
        mfi_val = round(mfi.iloc[-1], 1) if not np.isnan(mfi.iloc[-1]) else 50

        # ADX
        adx, plus_di, minus_di = calc_adx(coin_df)
        adx_val = round(adx.iloc[-1], 1) if not np.isnan(adx.iloc[-1]) else 0

        score = 50
        buy_reasons = []
        sell_reasons = []


        # ── 1. RSI ──
        if rsi_val < 30:
            score += 7               
            buy_reasons.append(f"RSI 과매도({rsi_val})")
        elif rsi_val > 70:
            score -= 7
            sell_reasons.append(f"RSI 과매수({rsi_val})")

        # ── 2. Stochastic RSI ──
        if stoch_val < 20:
            score += 4
            buy_reasons.append(f"StochRSI 과매도({stoch_val})")
        elif stoch_val > 80:
            score -= 4
            sell_reasons.append(f"StochRSI 과매수({stoch_val})")

        # ── 3. MACD ── 
        mh = macd_hist
        if len(mh) > 1 and not np.isnan(mh.iloc[-1]) and not np.isnan(mh.iloc[-2]):
            if mh.iloc[-1] > 0 and mh.iloc[-2] <= 0:
                score += 12          
                buy_reasons.append("MACD 골든크로스")
            elif mh.iloc[-1] < 0 and mh.iloc[-2] >= 0:
                score -= 12
                sell_reasons.append("MACD 데드크로스")

        # ── 4. 볼린저밴드 ──
        if not np.isnan(bb_lower.iloc[-1]) and price <= bb_lower.iloc[-1]:
            score += 5              
            buy_reasons.append("볼린저 하단 터치")
        elif not np.isnan(bb_upper.iloc[-1]) and price >= bb_upper.iloc[-1]:
            score -= 5
            sell_reasons.append("볼린저 상단 돌파")

        # ── 5. EMA 배열 ──
        e5, e20, e60 = ema5.iloc[-1], ema20.iloc[-1], ema60.iloc[-1]
        if e5 > e20 > e60:
            score += 5
            buy_reasons.append("EMA 정배열 (5>20>60)")
        elif e5 < e20 < e60:
            score -= 5
            sell_reasons.append("EMA 역배열 (5<20<60)")

        # ── 6. SuperTrend ── 
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

        # ── 7. OBV ──
        if not np.isnan(obv_ma.iloc[-1]):
            if obv.iloc[-1] > obv_ma.iloc[-1] * 1.05:
                score += 8
                buy_reasons.append("OBV 상승 (매집 중)")
            elif obv.iloc[-1] < obv_ma.iloc[-1] * 0.95:
                score -= 8
                sell_reasons.append("OBV 하락 (매도 중)")

        # ── 8. VWAP ──
        vwap_val = vwap.iloc[-1]
        if not np.isnan(vwap_val):
            if price > vwap_val:
                score += 5
                buy_reasons.append("VWAP 위 (매수 영역)")
            else:
                score -= 5
                sell_reasons.append("VWAP 아래 (매도 영역)")

        per = 0
        forward_per = 0
        target_price = 0

        # ── 8-1. PER 업종 비교 ──
        if per > 0:
            _per_avg = industry_per if industry_per > 0 else get_sector_per(sector or theme)
            _per_ratio = per / _per_avg
            if _per_ratio <= 0.5:
                score += 8; buy_reasons.append(f"PER {per:.1f} 저평가 (업종PER {_per_avg:.1f}의 {_per_ratio:.0%})")
            elif _per_ratio <= 0.8:
                score += 4; buy_reasons.append(f"PER {per:.1f} 다소저평가 (업종PER {_per_avg:.1f}의 {_per_ratio:.0%})")
            elif _per_ratio > 1.5:
                score -= 8; sell_reasons.append(f"PER {per:.1f} 고평가 (업종PER {_per_avg:.1f}의 {_per_ratio:.0%})")
            elif _per_ratio > 1.2:
                score -= 4; sell_reasons.append(f"PER {per:.1f} 다소고평가 (업종PER {_per_avg:.1f}의 {_per_ratio:.0%})")

        # ── 8-2. 예상PER (미래 실적) ──
        if forward_per > 0 and per > 0:
            if forward_per < per * 0.5:
                score += 8; buy_reasons.append(f"예상PER {forward_per:.1f} — 이익 대폭 증가 전망")
            elif forward_per < per * 0.8:
                score += 4; buy_reasons.append(f"예상PER {forward_per:.1f} — 이익 증가 전망")
            elif forward_per > per * 1.5:
                score -= 4; sell_reasons.append(f"예상PER {forward_per:.1f} — 이익 감소 전망")

        # ── 8-3. 컨센서스 목표가 ──
        if target_price > 0 and price > 0:
            _upside = (target_price / price - 1) * 100
            if _upside > 30:
                score += 8; buy_reasons.append(f"목표가 {target_price:,}원 (▲{_upside:.1f}%) 상승여력 큼")
            elif _upside > 10:
                score += 4; buy_reasons.append(f"목표가 {target_price:,}원 (▲{_upside:.1f}%) 상승여력")
            elif _upside < -10:
                score -= 6; sell_reasons.append(f"목표가 {target_price:,}원 (▼{_upside:.1f}%) 현재가 과열")

        # ── 9. MFI ──
        if mfi_val < 20:
            score += 5
            buy_reasons.append(f"MFI 자금유입 과매도({mfi_val})")
        elif mfi_val > 80:
            score -= 5
            sell_reasons.append(f"MFI 자금유출 과매수({mfi_val})")

        # ── 10. ADX ──
        if adx_val > 25:
            if plus_di.iloc[-1] > minus_di.iloc[-1]:
                score += 10
                buy_reasons.append(f"ADX 강한 상승추세({adx_val})")
            else:
                score -= 10
                sell_reasons.append(f"ADX 강한 하락추세({adx_val})")

        # ── 11. 거래량 ──
        if vol_ratio >= 5 and change > 0:
            score += 5
            buy_reasons.append(f"거래량 폭발({vol_ratio}x) + 양봉")
        elif vol_ratio >= 3 and change > 0:
            score += 3
            buy_reasons.append(f"거래량 급증({vol_ratio}x) + 양봉")
        elif vol_ratio >= 3 and change < 0:
            score -= 5
            sell_reasons.append(f"거래량 급증({vol_ratio}x) + 음봉")

        # ── 12. 다이버전스 ──
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


        # ── 13. 거래량 연속 증가 ──
        vol_up_3 = all(v.iloc[-(i+1)] > v.iloc[-(i+2)] for i in range(3)) if len(v) > 4 else False
        vol_up_5 = all(v.iloc[-(i+1)] > v.iloc[-(i+2)] for i in range(5)) if len(v) > 6 else False
        if vol_up_5:
            score += 3
            buy_reasons.append("거래량 5연속 증가")
        elif vol_up_3:
            score += 2
            buy_reasons.append("거래량 3연속 증가")

        # ══════ 선물 전용 지표 ══════

        # ── 14. 펀딩비 ──
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

        # ── 15. 롱숏 비율 ──
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

        # ── 충돌 보정 ──
        has_rsi_sell = any("RSI 과매수" in s for s in sell_reasons)
        has_macd_buy = any("MACD 골든크로스" in b for b in buy_reasons)
        if has_rsi_sell and has_macd_buy:
            score -= 5

        score = max(0, min(100, score))

        # ── 포지션 판단 ──
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
        print(f"❌ {symbol} 에러: {e}")
        import traceback
        traceback.print_exc()
        return None


def draw_coin_chart(r, last_n=60):
    """코인 차트"""
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
    """볼린저밴드 스퀴즈 감지 — 밴드폭이 최근 120일 중 최소에 가까우면 스퀴즈"""
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
    is_squeeze = squeeze_pct < 15  # 하위 15% 이면 스퀴즈
    return is_squeeze, squeeze_pct

def detect_stealth_accumulation(close, volume, window=20):
    """세력 매집 감지 — 거래량은 증가하는데 가격은 횡보"""
    if len(close) < window + 5:
        return False, 0, 0
    recent_close = close.iloc[-window:]
    recent_vol = volume.iloc[-window:]
    price_change = abs((recent_close.iloc[-1] - recent_close.iloc[0]) / (recent_close.iloc[0] + 1e-9) * 100)
    # 거래량 추세 (선형 회귀 기울기)
    vol_values = recent_vol.values
    x = np.arange(len(vol_values))
    if np.std(vol_values) < 1:
        return False, 0, 0
    slope = np.polyfit(x, vol_values, 1)[0]
    vol_trend = round(slope / (np.mean(vol_values) + 1e-9) * 100, 2)
    # 가격 변동 5% 이하인데 거래량 추세 상승이면 매집
    is_accumulating = price_change < 5 and vol_trend > 3
    return is_accumulating, round(price_change, 2), round(vol_trend, 2)

def detect_bottom_breakout(close, volume, window=60):
    """바닥 횡보 후 첫 양봉 + 거래량 폭발 감지"""
    if len(close) < window + 5:
        return False, 0
    base = close.iloc[-(window+1):-1]  # 최근 60일 (오늘 제외)
    base_range = (base.max() - base.min()) / (base.mean() + 1e-9) * 100
    today_close = close.iloc[-1]
    yesterday_close = close.iloc[-2]
    today_vol = volume.iloc[-1]
    avg_vol = volume.iloc[-window:-1].mean()
    is_yang = today_close > yesterday_close
    vol_explosion = today_vol / (avg_vol + 1e-9)
    # 바닥 횡보(변동폭 15% 이내) + 오늘 양봉 + 거래량 3배 이상
    is_breakout = base_range < 15 and is_yang and vol_explosion >= 3
    return is_breakout, round(vol_explosion, 2)

def detect_obv_divergence(close, volume, window=20):
    """OBV 상승 + 가격 횡보 감지 (조용한 매집)"""
    if len(close) < window + 5:
        return False, 0, 0
    obv = calc_obv(close, volume)
    recent_obv = obv.iloc[-window:]
    recent_close = close.iloc[-window:]
    price_change = abs((recent_close.iloc[-1] - recent_close.iloc[0]) / (recent_close.iloc[0] + 1e-9) * 100)
    obv_change = (recent_obv.iloc[-1] - recent_obv.iloc[0])
    # OBV 변화를 퍼센트로
    obv_base = abs(recent_obv.iloc[0]) + 1e-9
    obv_pct = round(obv_change / obv_base * 100, 2)
    # 가격은 5% 이내 횡보인데 OBV가 20% 이상 증가
    is_diverging = price_change < 5 and obv_pct > 20
    return is_diverging, round(price_change, 2), obv_pct

def analyze_surge(df, code, name):
    """급등 사냥 전용 분석 — 수익률 극대화 초점"""
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

        # ── 1. 거래대금 폭발 (핵심 지표) ──
        trade_val_ratio = round(trade_val / (trade_val_avg + 1e-9), 2)
        if trade_val_ratio >= 10:
            score += 30; signals.append(f"🔥 거래대금 {trade_val_ratio}배 폭발 ({trade_val}억)")
        elif trade_val_ratio >= 5:
            score += 20; signals.append(f"🔥 거래대금 {trade_val_ratio}배 급증 ({trade_val}억)")
        elif trade_val_ratio >= 3:
            score += 12; signals.append(f"📈 거래대금 {trade_val_ratio}배 증가 ({trade_val}억)")
        details["trade_val_ratio"] = trade_val_ratio

        # ── 2. 거래량 폭발 ──
        if vol_ratio >= 10:
            score += 20; signals.append(f"🔥 거래량 {vol_ratio}배 폭발")
        elif vol_ratio >= 5:
            score += 12; signals.append(f"📈 거래량 {vol_ratio}배 급증")
        elif vol_ratio >= 3:
            score += 8; signals.append(f"📊 거래량 {vol_ratio}배 증가")

        # ── 3. 세력 매집 감지 ──
        is_accum, price_chg, vol_trend = detect_stealth_accumulation(close, volume)
        if is_accum:
            score += 25; signals.append(f"🕵️ 세력 매집 포착 (가격 {price_chg}% 횡보, 거래량 추세 +{vol_trend}%)")
        details["accumulation"] = is_accum

        # ── 4. 바닥 돌파 ──
        is_breakout, vol_exp = detect_bottom_breakout(close, volume)
        if is_breakout:
            score += 25; signals.append(f"💥 바닥 돌파! 첫 양봉 + 거래량 {vol_exp}배")
        details["breakout"] = is_breakout

        # ── 5. 볼린저 스퀴즈 ──
        is_squeeze, sq_pct = calc_squeeze(close)
        if is_squeeze:
            score += 20; signals.append(f"🔋 볼린저 스퀴즈 (밴드폭 하위 {sq_pct}%) — 곧 폭발")
        details["squeeze"] = is_squeeze

        # ── 6. OBV 다이버전스 (조용한 매집) ──
        is_obv_div, obv_price_chg, obv_pct = detect_obv_divergence(close, volume)
        if is_obv_div:
            score += 20; signals.append(f"🕵️ OBV 매집 (가격 {obv_price_chg}% 횡보, OBV +{obv_pct}%)")
        details["obv_div"] = is_obv_div

        # ── 7. 전일 상한가 후 연속 상승 가능성 ──
        if len(close) > 2:
            prev_change = round((close.iloc[-2] - close.iloc[-3]) / (close.iloc[-3] + 1e-9) * 100, 2)
            if prev_change >= 25:  # 전일 상한가급
                score += 15; signals.append(f"🚀 전일 {prev_change:+.1f}% 급등 → 연속 상승 가능")
                if change > 0:
                    score += 10; signals.append(f"📈 오늘도 양봉 ({change:+.2f}%)")
            elif prev_change >= 15:
                score += 8; signals.append(f"📈 전일 {prev_change:+.1f}% 상승")

        # ── 8. 저가주 보너스 ──
        if price <= 3000:
            score += 10; signals.append(f"💰 초저가주 ({price:,}원) — 급등 여력 큼")
        elif price <= 5000:
            score += 7; signals.append(f"💰 저가주 ({price:,}원)")
        elif price <= 10000:
            score += 4; signals.append(f"💰 만원 이하 ({price:,}원)")

        # ── 9. 거래대금 절대값 ──
        if trade_val >= 100:
            score += 10; signals.append(f"💵 거래대금 {trade_val}억 (큰 돈 유입)")
        elif trade_val >= 50:
            score += 6; signals.append(f"💵 거래대금 {trade_val}억")

        # ── 10. RSI 보조 ──
        rsi = calc_rsi(close)
        rsi_val = round(rsi.iloc[-1], 1) if not np.isnan(rsi.iloc[-1]) else 50
        if rsi_val < 30 and (is_accum or is_obv_div):
            score += 10; signals.append(f"📊 RSI 과매도({rsi_val}) + 매집 신호 = 반등 임박")
        elif rsi_val > 80:
            score -= 5  # 이미 과열이면 약간 감점

        # ── 11. MACD 골든크로스 + 거래량 동시 ──
        macd_line, macd_sig, macd_hist = calc_macd(close)
        if len(macd_hist) > 1 and not np.isnan(macd_hist.iloc[-1]) and not np.isnan(macd_hist.iloc[-2]):
            if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0 and vol_ratio >= 2:
                score += 15; signals.append("📊 MACD 골든크로스 + 거래량 동반")

        # ── 12. 52주 신저가 근처에서 반등 ──
        if len(close) >= 250:
            low_52w = close.iloc[-250:].min()
            if price <= low_52w * 1.1:  # 52주 신저가 대비 10% 이내
                score += 8; signals.append(f"📉 52주 바닥권 ({low_52w:,}원) → 반등 가능")

        # ── 13. 이미 급등 vs 아직 안 오른 종목 판별 ──
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

        # ── 14. 바닥권에서 신호 집중 보너스 ──
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


        # 점수 보정
        score = max(0, min(100, score))

        # 등급
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

        # 지지/저항
        recent = df.tail(20)
        support = int(recent["Low"].min())
        resist = int(recent["High"].max())
        stop_loss = int(support * 0.97)

        # ── 조합 보너스 (2차 백테스트 검증) ──
        reason_set = set(buy_reasons)

        # EV/EBITDA 6점대 + 강한 상승 힘 → 83~100%
        if any("EV/EBITDA 6" in r for r in reason_set) and any("강한 상승 힘" in r for r in reason_set):
            score += 10

        # EV/EBITDA 7점대 + 추세 구름 돌파 → 83%
        if any("EV/EBITDA 7" in r for r in reason_set) and any("추세 구름 돌파" in r for r in reason_set):
            score += 8

        # MACD 매수 + 정배열 → 80%
        if any("MACD 골든크로스" in r for r in reason_set) and any("정배열" in r for r in reason_set):
            score += 8

        # EV/EBITDA 6점대 + 거래량 동반 → 83%
        if any("EV/EBITDA 6" in r for r in reason_set) and any("거래량 동반" in r for r in reason_set):
            score += 8

        # RSI 과매도 + RSI 다이버전스 → 83.3%
        if any("RSI 과매도" in r for r in reason_set) and any("RSI 상승 다이버전스" in r for r in reason_set):
            score += 8

        # MACD 매수 + 슈퍼트렌드 매수 전환 → 80%
        if any("MACD 골든크로스" in r for r in reason_set) and any("슈퍼트렌드 매수 전환" in r for r in reason_set):
            score += 8

        # 정배열 + 슈퍼트렌드 매수 전환 → 80%
        if any("정배열" in r for r in reason_set) and any("슈퍼트렌드 매수 전환" in r for r in reason_set):
            score += 6

        # ── 감점 패널티 (승률 낮은 지표) ──
        if any("RSI 상승 다이버전스" in r for r in reason_set):
            strong = any(x in r for r in reason_set for x in ["MACD 골든크로스", "슈퍼트렌드 매수 전환", "EV/EBITDA", "거래량 동반"])
            if not strong:
                score -= 5

        if any("MACD 상승 다이버전스" in r for r in reason_set):
            strong2 = any(x in r for r in reason_set for x in ["정배열", "슈퍼트렌드 매수 전환", "EV/EBITDA", "거래량 동반"])
            if not strong2:
                score -= 5

        # ── 핫 테마 보너스 ──
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
        print(f"❌ {symbol} 에러: {e}")          
        import traceback                        
        traceback.print_exc()                    
        return None         


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
    """펀딩비 추세 분석 (연속 양/음 감지)"""
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
    """미결제약정 변화율"""
    hist = get_oi_history(symbol, period=period, limit=limit)
    if not hist or len(hist) < 2:
        return None, None
    latest = hist[-1]
    prev = hist[0]
    change_pct = round((latest - prev) / (prev + 1e-9) * 100, 2)
    return latest, change_pct

def get_top_trader_ratio(symbol, period="5m"):
    """탑 트레이더(고래) 롱숏 비율"""
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
    """레버리지별 예상 청산가 계산"""
    if leverage_list is None:
        leverage_list = [5, 10, 20, 25, 50, 100]
    zones = []
    for lev in leverage_list:
        liq_long = round(price * (1 - 1/lev), 4)
        liq_short = round(price * (1 + 1/lev), 4)
        zones.append({
            "leverage": lev,
            "long_liq": liq_long,
            "short_liq": liq_short,
        })
    return zones

def load_coin_trades():
    """코인 가상매매 내역 로드"""
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
    """코인 가상매매 내역 저장"""
    try:
        with open(COIN_TRADE_FILE, "w", encoding="utf-8") as f:
            json.dump(trades, f, ensure_ascii=False, indent=2)
    except:
        pass


def open_paper_trade(symbol, side, entry_price, leverage, qty_usdt, tp_price=0, sl_price=0):
    """가상매매 진입"""
    trades = load_coin_trades()
    trade = {
        "id": int(time.time() * 1000),
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "leverage": leverage,
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
    """가상매매 종료"""
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
        "status": "closed",
        "exit_price": exit_price,
        "close_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pnl_pct": pnl_pct,
        "pnl_usdt": pnl_usdt,
    })
    save_coin_trades(trades)
    return target

def get_paper_stats():
    """가상매매 통계"""
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
            "total": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "open_count": len(opened),
        }
    except:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0, "open_count": 0}


def check_paper_tpsl():
    """가상매매 TP/SL 자동 체크"""
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
                emoji = "🎯" if hit == "TP" else "🛑"
                closed_any = True
    return closed_any


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
        "min_trade_val": 50,
        "자금 유입 신호": 0, "자금 유출 신호": 85,
        "adx_min": 30,
        "bb_weight": 15, "macd_weight": 18, "st_weight": 15,
        "vol_weight": 12, "ema_weight": 5,
    },
    "스윙 (3~15일)": {
        "key": "swing", "tp": 10, "sl": 5,
        "RSI 과매도 (반등 가능)": 40, "RSI 과매수 (조정 가능)": 70,
        "min_trade_val": 20,
        "자금 유입 신호": 0, "자금 유출 신호": 80,
        "adx_min": 25,
        "bb_weight": 10, "macd_weight": 15, "st_weight": 12,
        "vol_weight": 8, "ema_weight": 10,
    },
    "중장기 (15일+)": {
        "key": "long", "tp": 20, "sl": 10,
        "RSI 과매도 (반등 가능)": 45, "RSI 과매수 (조정 가능)": 75,
        "min_trade_val": 10,
        "자금 유입 신호": 0, "자금 유출 신호": 75,
        "adx_min": 20,
        "bb_weight": 5, "macd_weight": 10, "st_weight": 8,
        "vol_weight": 5, "ema_weight": 15,
    },
}

KEY_TO_STYLE = {v["key"]: k for k, v in STYLES.items()}

# ── 백테스트 기반 가중치 (지표별 실전 수익 기여도) ──
INDICATOR_WEIGHTS = {
    # 높은 기여도 (백테스트 승률 60%+ 기여 지표)
    "macd_cross": 18,      # MACD 골든/데드크로스
    "divergence": 17,       # RSI/MACD 다이버전스
    "vol_explosion": 15,    # 거래량 폭발 + 양봉
    "supertrend": 14,       # 슈퍼트렌드 전환
    
    # 중간 기여도
    "bb_touch": 11,         # 볼린저밴드 터치
    "ema_align": 10,        # EMA 정/역배열
    "obv": 9,               # OBV 매집/매도
    "adx": 9,               # ADX 추세 강도
    "mfi": 8,               # MFI 자금 흐름
    "rsi": 8,               # RSI 과매수/과매도
    
    # 낮은 기여도 (보조 확인용)
    "vwap": 5,              # VWAP 위치
    "ichimoku": 5,          # 일목균형
    "vol_consec": 4,        # 거래량 연속 증가
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
    	"추세 구름 돌파 (상승 신호)": "☁️ 일목균형 매수 → 구름대 위로 올라왔어요 (상승 신호)",
    	"추세 구름 이탈 (하락 신호)": "☁️ 일목균형 매도 → 구름대 아래로 내려갔어요 (하락 신호)",
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
        with open(WL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_wl(wl):
    with open(WL_FILE, "w", encoding="utf-8") as f:
        json.dump(wl, f, ensure_ascii=False, indent=2)

def add_to_wl(code, name):
    wl = load_wl()
    if not any(w["code"] == code for w in wl):
        wl.append({"code": code, "name": name})
        save_wl(wl)

def remove_from_wl(code):
    wl = load_wl()
    wl = [w for w in wl if w["code"] != code]
    save_wl(wl)

# ── 스캔 히스토리 ──
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(hist):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

def add_to_history_direct(style_key, records):
    hist = load_history()
    entry = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "style": style_key,
        "stocks": records,
    }
    hist.insert(0, entry)
    if len(hist) > 50:
        hist = hist[:50]
    save_history(hist)

def clear_history():
    save_history([])

# ── 성과 추적 ──
def save_perf_snapshot(all_results):
    """스캔 결과를 성과 추적용으로 저장"""
    snapshot = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "stocks": []
    }
    seen = set()
    for skey, rlist in all_results.items():
        for r in rlist:
            if r["code"] not in seen:
                seen.add(r["code"])
                snapshot["stocks"].append({
                    "code": r["code"], "name": r["name"],
                    "score": r["score"], "grade": r["grade"],
                    "verdict": r["verdict"], "price": r["price"],
                    "tp_price": r["tp_price"], "sl_price": r["sl_price"],
                    "buy_reasons": r.get("buy_reasons", []),
                    "sell_reasons": r.get("sell_reasons", []),
                })

    if os.path.exists(PERF_FILE):
        with open(PERF_FILE, "r", encoding="utf-8") as f:
            perf = json.load(f)
    else:
        perf = []
    perf.insert(0, snapshot)
    if len(perf) > 30:
        perf = perf[:30]
    with open(PERF_FILE, "w", encoding="utf-8") as f:
        json.dump(perf, f, ensure_ascii=False, indent=2)

def load_perf_snapshot():
    if os.path.exists(PERF_FILE):
        with open(PERF_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def load_surge_history():
    if os.path.exists(SURGE_HISTORY_FILE):
        with open(SURGE_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_surge_history(hist):
    with open(SURGE_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

def add_surge_record(results, country):
    hist = load_surge_history()
    record = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "country": country,
        "count": len(results),
        "stocks": [
            {
                "code": r["code"], "name": r["name"], "score": r["score"],
                "grade": r["grade"], "price": r["price"], "change": r["change"],
                "vol_ratio": r["vol_ratio"], "trade_val": r["trade_val"],
                "signals": r["signals"][:5], "market": r.get("market", ""),
            }
            for r in results[:30]
        ],
    }
    hist.insert(0, record)
    if len(hist) > 30:
        hist = hist[:30]
    save_surge_history(hist)


def generate_perf_report():
    """전일 추천 종목의 현재 성과를 계산"""
    perf = load_perf_snapshot()
    if not perf:
        return None
    latest = perf[0]
    report_date = latest["date"]
    results = []
    for s in latest["stocks"]:
        code = s["code"]
        try:
            df = fetch(code, days=5)
            if df is None or len(df) < 2:
                continue
            current_price = int(df["Close"].iloc[-1])
            entry_price = s["price"]
            pnl = round((current_price - entry_price) / entry_price * 100, 2)
            hit_tp = current_price >= s["tp_price"]
            hit_sl = current_price <= s["sl_price"]
            status = "🎯 목표달성" if hit_tp else "🛑 손절" if hit_sl else "📊 보유중"
            results.append({
                "code": code, "name": s["name"],
                "score": s["score"], "grade": s["grade"],
                "entry_price": entry_price, "current_price": current_price,
                "pnl": pnl, "status": status,
                "tp_price": s["tp_price"], "sl_price": s["sl_price"],
            })
        except Exception:
            continue
    if not results:
        return None
    wins = len([r for r in results if r["pnl"] > 0])
    losses = len([r for r in results if r["pnl"] <= 0])
    avg_pnl = round(sum(r["pnl"] for r in results) / len(results), 2)
    best = max(results, key=lambda x: x["pnl"])
    worst = min(results, key=lambda x: x["pnl"])
    return {
        "date": report_date,
        "results": sorted(results, key=lambda x: x["pnl"], reverse=True),
        "total": len(results), "wins": wins, "losses": losses,
        "win_rate": round(wins / len(results) * 100, 1) if results else 0,
        "avg_pnl": avg_pnl, "best": best, "worst": worst,
    }

def generate_indicator_report():
    """지표별 승률 분석"""
    perf = load_perf_snapshot()
    if not perf:
        return None

    indicator_stats = {}

    for snapshot in perf:
        for s in snapshot["stocks"]:
            code = s["code"]
            entry_price = s["price"]
            buy_reasons = s.get("buy_reasons", [])
            if not buy_reasons:
                continue
            try:
                df = fetch(code, days=10)
                if df is None or len(df) < 2:
                    continue
                current_price = int(df["Close"].iloc[-1])
                pnl = round((current_price - entry_price) / (entry_price + 1e-9) * 100, 2)
                win = pnl > 0

                for reason in buy_reasons:
                    if reason not in indicator_stats:
                        indicator_stats[reason] = {"total": 0, "wins": 0, "total_pnl": 0}
                    indicator_stats[reason]["total"] += 1
                    if win:
                        indicator_stats[reason]["wins"] += 1
                    indicator_stats[reason]["total_pnl"] += pnl
            except:
                continue

    if not indicator_stats:
        return None

    results = []
    for reason, stats in indicator_stats.items():
        if stats["total"] >= 2:
            results.append({
                "indicator": reason,
                "total": stats["total"],
                "wins": stats["wins"],
                "win_rate": round(stats["wins"] / stats["total"] * 100, 1),
                "avg_pnl": round(stats["total_pnl"] / stats["total"], 2),
            })

    results.sort(key=lambda x: x["win_rate"], reverse=True)
    return results



# ──────────────────────────────────────────
# 백테스트 시스템
# ──────────────────────────────────────────
BACKTEST_FILE = "backtest_results.json"

def save_backtest(results):
    with open(BACKTEST_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def load_backtest():
    if os.path.exists(BACKTEST_FILE):
        with open(BACKTEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def run_backtest(days_back=120, hold_days=5, max_stocks=200, _progress_callback=None, cfg=None):
    """과거 데이터로 백테스트 실행"""
    results = []
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)

    # 코스닥 종목 리스트
    try:
        stocks_df = fdr.StockListing("KOSDAQ")
        if stocks_df is None or len(stocks_df) == 0:
            return []
        codes = stocks_df["Code"].tolist()[:max_stocks]
        names = dict(zip(stocks_df["Code"], stocks_df["Name"]))
    except:
        return []

    # 10일 간격으로 과거 시점 생성
    test_dates = []
    current = start_date
    while current < end_date - datetime.timedelta(days=hold_days + 5):
        test_dates.append(current)
        current += datetime.timedelta(days=10)

    total_tasks = len(test_dates) * len(codes)
    done = 0

    for test_date in test_dates:
        for code in codes:
            done += 1
            if _progress_callback and done % 50 == 0:
                _progress_callback(done / total_tasks)
            try:
                # 해당 시점 전후 데이터 가져오기
                fetch_start = (test_date - datetime.timedelta(days=250)).strftime("%Y-%m-%d")
                fetch_end = (test_date + datetime.timedelta(days=hold_days + 3)).strftime("%Y-%m-%d")
                df = fdr.DataReader(code, fetch_start, fetch_end)

                if df is None or len(df) < 60:
                    print(f"❌ {code} 데이터 부족: {len(df) if df is not None else 'None'}")
                    continue

                # test_date 시점까지만 잘라서 분석
                df_analysis = df[df.index <= test_date.strftime("%Y-%m-%d")]
                if len(df_analysis) < 60:
                    continue

                name = names.get(code, code)
                r = analyze(df_analysis, code, name, cfg)
                print(f"✅ {name}({code}) 점수: {r['score']}")

                if r["score"] < MIN_SCORE:
                    continue
                if len(r.get("buy_reasons", [])) < 3:
                    continue

                # 매수 후 hold_days일 뒤 가격 확인
                df_after = df[df.index > test_date.strftime("%Y-%m-%d")]
                if len(df_after) < hold_days:
                    continue

                entry_price = r["price"]
                exit_price = int(df_after["Close"].iloc[hold_days - 1])
                pnl = round((exit_price - entry_price) / (entry_price + 1e-9) * 100, 2)

                results.append({
                    "date": test_date.strftime("%Y-%m-%d"),
                    "code": code,
                    "name": name,
                    "score": r["score"],
                    "grade": r["grade"],
                    "buy_reasons": r.get("buy_reasons", []),
                    "sell_reasons": r.get("sell_reasons", []),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "win": pnl > 0,
                })
                # 50건마다 중간 저장
                if len(results) % 50 == 0:
                    save_backtest(results)
                    print(f"💾 중간 저장: {len(results)}건")
            except Exception as e:
                print(f"🔥 에러: {code} - {e}")
                continue

    # 결과 저장
    print(f"📊 백테스트 완료: {len(results)}건 발견 (총 {done}건 분석)")
    if results:
        save_backtest(results)

    return results



def analyze_backtest(results):
    """백테스트 결과에서 지표별 승률 분석"""
    if not results:
        return None

    indicator_stats = {}
    for r in results:
        for reason in r.get("buy_reasons", []):
            if reason not in indicator_stats:
                indicator_stats[reason] = {"total": 0, "wins": 0, "total_pnl": 0}
            indicator_stats[reason]["total"] += 1
            if r["win"]:
                indicator_stats[reason]["wins"] += 1
            indicator_stats[reason]["total_pnl"] += r["pnl"]

    report = []
    for reason, stats in indicator_stats.items():
        if stats["total"] >= 3:
            report.append({
                "indicator": reason,
                "total": stats["total"],
                "wins": stats["wins"],
                "win_rate": round(stats["wins"] / stats["total"] * 100, 1),
                "avg_pnl": round(stats["total_pnl"] / stats["total"], 2),
            })

    report.sort(key=lambda x: x["win_rate"], reverse=True)
    return report


def analyze_combo(results, min_combo=2):
    """지표 조합별 승률 분석"""
    if not results:
        return None

    from itertools import combinations
    combo_stats = {}

    for r in results:
        reasons = sorted(r.get("buy_reasons", []))
        if len(reasons) < min_combo:
            continue
        # 2개 조합
        for combo in combinations(reasons, min_combo):
            key = " + ".join(combo)
            if key not in combo_stats:
                combo_stats[key] = {"total": 0, "wins": 0, "total_pnl": 0}
            combo_stats[key]["total"] += 1
            if r["win"]:
                combo_stats[key]["wins"] += 1
            combo_stats[key]["total_pnl"] += r["pnl"]

    report = []
    for combo, stats in combo_stats.items():
        if stats["total"] >= 3:
            report.append({
                "combo": combo,
                "total": stats["total"],
                "wins": stats["wins"],
                "win_rate": round(stats["wins"] / stats["total"] * 100, 1),
                "avg_pnl": round(stats["total_pnl"] / stats["total"], 2),
            })

    report.sort(key=lambda x: x["win_rate"], reverse=True)
    return report

# ── 핫 테마 자동 감지 (네이버 증권) ──
@st.cache_data(ttl=1800)
def get_hot_themes(top_n=10):
    """네이버 증권에서 오늘의 핫 테마 + 소속 종목 가져오기"""
    hot_themes = []
    try:
        url = "https://finance.naver.com/sise/theme.naver"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        
        rows = soup.select("table.type_1 tr")
        count = 0
        for row in rows:
            if count >= top_n:
                break
            cols = row.select("td")
            if len(cols) < 6:
                continue
            theme_link = cols[0].select_one("a")
            rate_span = cols[1].select_one("span")
            if not theme_link or not rate_span:
                continue
            
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
                        if code.isdigit() and len(code) == 6:
                            theme_codes.append(code)
                theme_codes = list(set(theme_codes))
            except:
                pass
            
            hot_themes.append({
                "name": theme_name,
                "change_rate": change_rate,
                "codes": theme_codes,
            })
            count += 1
    except:
        pass
    return hot_themes


def get_hot_theme_bonus(code, hot_themes=None):
    """종목이 핫 테마에 속하면 보너스 점수 반환"""
    if not hot_themes:
        return 0, []
    
    matched_themes = []
    for theme in hot_themes:
        if code in theme.get("codes", []):
            matched_themes.append(theme["name"])
    
    if len(matched_themes) >= 2:
        return 10, matched_themes
    elif len(matched_themes) == 1:
        return 6, matched_themes
    return 0, matched_themes


# ── 수동 핫 테마 설정 ──
MANUAL_THEME_FILE = "manual_hot_themes.json"

def load_manual_themes():
    if os.path.exists(MANUAL_THEME_FILE):
        with open(MANUAL_THEME_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"enabled": True, "auto": True, "manual_keywords": []}

def save_manual_themes(data):
    with open(MANUAL_THEME_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_combined_theme_bonus(code, name):
    """자동 + 수동 테마 보너스 통합"""
    settings = load_manual_themes()
    if not settings.get("enabled", True):
        return 0, []
    
    matched = []
    
    # 자동 감지
    if settings.get("auto", True):
        hot_themes = get_hot_themes()
        bonus, auto_matched = get_hot_theme_bonus(code, hot_themes)
        matched.extend(auto_matched)
    
    # 수동 키워드 매칭
    for kw in settings.get("manual_keywords", []):
        if kw in name:
            matched.append(f"수동:{kw}")
    
    matched = list(set(matched))
    if len(matched) >= 2:
        return 10, matched
    elif len(matched) == 1:
        return 6, matched
    return 0, matched


# ── 뉴스 조회 ──
@st.cache_data(ttl=300)
def get_stock_news(code, name):
    """네이버 뉴스에서 종목 관련 뉴스 가져오기"""
    news_list = []
    try:
        search_query = name.replace(" ", "+")
        url = f"https://search.naver.com/search.naver?where=news&query={search_query}+주식&sort=1"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=5)
        text = resp.text
        # 간단한 파싱
        titles = re.findall(r'class="news_tit"[^>]*title="([^"]*)"', text)
        links = re.findall(r'class="news_tit"[^>]*href="([^"]*)"', text)
        sources = re.findall(r'class="info press"[^>]*>([^<]*)<', text)
        for i in range(min(5, len(titles))):
            news_list.append({
                "title": titles[i] if i < len(titles) else "",
                "link": links[i] if i < len(links) else "",
                "source": sources[i] if i < len(sources) else "",
            })
    except Exception:
        pass
    return news_list


# ── 테마 ──
def get_theme(name):
    for theme, keywords in THEME_KW.items():
        for kw in keywords:
            if kw in name:
                return theme
    return ""

# ── 시장 국면 ──
def market_phase():
    if not FDR_OK:
        return "확인불가", "⚪"
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=60)
        df = fdr.DataReader("KS11", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if df is None or len(df) < 20:
            return "확인불가", "⚪"
        c = df["Close"].iloc[-1]
        ma20 = df["Close"].rolling(20).mean().iloc[-1]
        if c > ma20 * 1.02:
            return "상승장", "🟢"
        elif c < ma20 * 0.98:
            return "하락장", "🔴"
        else:
            return "횡보장", "🟡"
    except Exception:
        return "확인불가", "⚪"

# ── 지지선/저항선 계산 ──
def calc_support_resist(df, period=20):
    recent = df.tail(period)
    support = int(recent["Low"].min())
    resist = int(recent["High"].max())
    bb_upper, bb_mid, bb_lower = calc_bb(df["Close"])
    bb_low = bb_lower.iloc[-1]
    bb_high = bb_upper.iloc[-1]
    if not np.isnan(bb_low):
        support = max(support, int(bb_low))
    if not np.isnan(bb_high):
        resist = min(resist, int(bb_high))
    if support >= resist:
        support = int(recent["Low"].min())
        resist = int(recent["High"].max())
    stop_loss = int(support * 0.97)
    return support, resist, stop_loss

# ── 기술 지표 ──
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

# ── 종목 리스트 캐시 ──
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

    # 방법 2: 로컬 JSON 파일 (해외 서버용 백업)
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
                    try:
                        forward_per = float(val)
                    except:
                        pass
            elif item.get("code") == "cnsEps":
                val = item.get("value", "N/A").replace("원", "").replace(",", "").strip()
                if val != "N/A":
                    try:
                        forward_eps = float(val)
                    except:
                        pass
            elif item.get("code") == "highPriceOf52Weeks":
                val = item.get("value", "0").replace(",", "").strip()
                try:
                    high_52 = int(val)
                except Exception:
                    pass
            elif item.get("code") == "lowPriceOf52Weeks":
                val = item.get("value", "0").replace(",", "").strip()
                try:
                    low_52 = int(val)
                except Exception:
                    pass
            elif item.get("code") == "dividendYieldRatio":
                val = item.get("value", "0").replace("%", "").replace(",", "").strip()
                try:
                    dividend_yield = float(val)
                except:
                    pass
            elif item.get("code") == "dividend":
                val = item.get("value", "0").replace("원", "").replace(",", "").strip()
                try:
                    dividend_amt = int(float(val))
                except:
                    pass
        for deal in data.get("dealTrendInfos", []):
            try:
                fb = int(deal.get("foreignerPureBuyQuant", "0").replace(",", "").replace("+", ""))
                ob = int(deal.get("organPureBuyQuant", "0").replace(",", "").replace("+", ""))
                foreign_buys.append(fb)
                organ_buys.append(ob)
            except Exception:
                pass
        foreign_buys = foreign_buys[::-1]
        organ_buys = organ_buys[::-1]

        # 컨센서스 목표가
        consensus = data.get("consensusInfo", {})
        if consensus:
            try:
                tp = consensus.get("priceTargetMean", "0")
                if tp:
                    target_price = int(float(str(tp).replace(",", "")))
            except:
                pass
        # 동종업계 비교
        for peer in data.get("industryCompareInfo", []):
            try:
                peers.append({
                    "name": peer.get("stockName", ""),
                    "code": peer.get("itemCode", ""),
                    "price": peer.get("closePrice", "0"),
                    "change": peer.get("fluctuationsRatio", "0"),
                    "marketValue": peer.get("marketValue", "0"),
                })
            except:
                pass
    except Exception:
        pass
    # 업종 정보 + 업종PER/PBR 가져오기
    try:
        url2 = f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}&cn="
        resp2 = requests.get(url2, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=5)
        text2 = resp2.text
        m = re.search(r'코스(?:피|닥)\s+([^<\n]+)', text2)
        if m:
            sector = m.group(1).strip()
        m2 = re.search(r'업종PER\s*<b[^>]*>\s*([\d.,]+)', text2)
        if m2:
            industry_per = float(m2.group(1).replace(",", ""))
        m3 = re.search(r'PBR\s*<b[^>]*>\s*([\d.,]+)', text2)
        if m3:
            industry_pbr = float(m3.group(1).replace(",", ""))
    except Exception:
        pass

    # ── 재무 건전성 지표 (FnGuide) ──
    roe = 0.0
    debt_ratio = 0.0
    op_margin = 0.0
    revenue_growth = 0.0
    try:
        fn_url = f"https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?gicode=A{code}"
        fn_resp = requests.get(fn_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=7)
        fn_text = fn_resp.text
        # ROE (업종비교 테이블)
        roe_match = re.search(r'ROE</div></th>\s*<td class="r"[^>]*>([\-\d.,]+)', fn_text)
        if roe_match:
            roe = float(roe_match.group(1).replace(",", ""))
        # 매출액, 영업이익 추출 → 영업이익률 계산
        rev_match = re.search(r'매출액</div></th>\s*<td class="r"[^>]*title="([\-\d.,]+)"', fn_text)
        op_match = re.search(r'영업이익</div></th>\s*<td class="r"[^>]*title="([\-\d.,]+)"', fn_text)
        if rev_match and op_match:
            rev_val = float(rev_match.group(1).replace(",", ""))
            op_val_raw = float(op_match.group(1).replace(",", ""))
            if rev_val > 0:
                op_margin = round(op_val_raw / rev_val * 100, 1)
        # 매출액 성장률 (Business Summary에서)
        growth_match = re.search(r'매출액은\s*([\d.]+)%\s*증가', fn_text)
        if growth_match:
            revenue_growth = float(growth_match.group(1))
        else:
            decline_match = re.search(r'매출액은\s*([\d.]+)%\s*감소', fn_text)
            if decline_match:
                revenue_growth = -float(decline_match.group(1))
        # 부채비율 (navercomp에서 가져오기)
        debt_url = f"https://navercomp.wisereport.co.kr/v2/company/c1030001.aspx?cmp_cd={code}&cn="
        dresp = requests.get(debt_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=5)
        dtext = dresp.text
        debt_match = re.search(r'부채비율.*?<td class="num">([\-\d.,]+)', dtext, re.DOTALL)
        if debt_match:
            debt_ratio = float(debt_match.group(1).replace(",", ""))
    except Exception:
        pass


    # ── 뉴스 가져오기 ──
    news_items = []
    try:
        news_url = f"https://stock.naver.com/api/domestic/detail/news?itemCode={code}&page=1&pageSize=5"
        nresp = requests.get(news_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if nresp.status_code == 200:
            ndata = nresp.json()
            for cluster in ndata.get("clusters", []):
                for art in cluster.get("items", []):
                    news_items.append({
                        "title": art.get("title", ""),
                        "body": art.get("body", ""),
                        "office": art.get("officeName", ""),
                        "datetime": art.get("datetime", ""),
                    })
    except:
        pass

    return per, pbr, foreign_buys, organ_buys, high_52, low_52, sector, industry_per, forward_per, target_price, industry_pbr, peers, news_items, roe, debt_ratio, op_margin, revenue_growth, dividend_yield, dividend_amt


# ── EV/EBITDA 조회 (네이버 컴프) ──
from bs4 import BeautifulSoup

@st.cache_data(ttl=600)
def get_ev_ebitda(code, is_us=False):
    """EV/EBITDA 가져오기 — 한국: 네이버 컴프 / 미국: 야후 파이낸스"""
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
            if info.get("trailingPE"):
                result["per"] = round(info["trailingPE"], 2)
            if info.get("priceToBook"):
                result["pbr"] = round(info["priceToBook"], 2)
        except:
            pass
    else:
        try:
            url = f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}&cn="
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}
            resp = requests.get(url, headers=headers, timeout=5)
            resp = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(resp.text, "html.parser")
            for row in soup.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    header = cells[0].get_text(strip=True)
                    if "EV/EBITDA" in header:
                        val = cells[1].get_text(strip=True).replace(",", "")
                        try:
                            result["ev_ebitda"] = float(val)
                        except:
                            pass
        except:
            pass
    
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
    except Exception as e:
        pass
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
    if not KIS_OK:
        return None
    try:
        url = f"{_broker.api_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        import datetime as dt
        now = dt.datetime.now().strftime("%H%M%S")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_HOUR_1": now,
            "FID_PW_DATA_INCU_YN": "Y"
        }
        res = requests.get(url, headers=_broker._headers("FHKST03010200"), params=params)
        data = res.json()
        if data.get("rt_cd") == "0":
            rows = data.get("output2", [])
            if rows:
                result = []
                for r in rows:
                    result.append({
                        "Time": r.get("stck_cntg_hour", ""),
                        "Open": int(r.get("stck_oprc", 0)),
                        "High": int(r.get("stck_hgpr", 0)),
                        "Low": int(r.get("stck_lwpr", 0)),
                        "Close": int(r.get("stck_clpr", 0)),
                        "Volume": int(r.get("cntg_vol", 0)),
                    })
                if result:
                    return pd.DataFrame(result[::-1])
    except Exception:
        pass
    return None

# ─── 분석 함수 ───────────────────────────────────────
def analyze(df, code, name, cfg, market="KOSDAQ"):
    code = str(code).strip()
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

    vol_up_3 = all(volume.iloc[-(i+1)] > volume.iloc[-(i+2)] for i in range(3)) if len(volume) > 4 else False
    vol_up_5 = all(volume.iloc[-(i+1)] > volume.iloc[-(i+2)] for i in range(5)) if len(volume) > 6 else False

    chg5 = round((price / (close.iloc[-6] + 1e-9) - 1) * 100, 1) if len(close) > 6 else 0
    chg10 = round((price / (close.iloc[-11] + 1e-9) - 1) * 100, 1) if len(close) > 11 else 0
    chg20 = round((price / (close.iloc[-21] + 1e-9) - 1) * 100, 1) if len(close) > 21 else 0

    ema5 = close.ewm(span=5).mean()
    ema20 = close.ewm(span=20).mean()
    ema60 = close.ewm(span=60).mean()
    e5 = ema5.iloc[-1]
    e20 = ema20.iloc[-1]
    e60 = ema60.iloc[-1]

    rsi = calc_rsi(close)
    rsi_val = round(rsi.iloc[-1], 1) if not np.isnan(rsi.iloc[-1]) else 50
    stoch_rsi = calc_stoch_rsi(close)
    macd_line, macd_sig, macd_hist = calc_macd(close)
    bb_upper, bb_mid, bb_lower = calc_bb(close)
    tenkan, kijun, span_a, span_b = calc_ichimoku(high, low, close)
    st_line, st_dir = calc_supertrend(df)
    obv = calc_obv(close, volume)
    vwap = calc_vwap(df)
    mfi = calc_mfi(df)
    mfi_val = round(mfi.iloc[-1], 1) if not np.isnan(mfi.iloc[-1]) else 50
    adx, plus_di, minus_di = calc_adx(df)
    adx_val = round(adx.iloc[-1], 1) if not np.isnan(adx.iloc[-1]) else 0

    if st.session_state.get("scan_running", False):
        per, pbr = 0.0, 0.0
        foreign_buys, organ_buys = [], []
        high_52, low_52 = 0, 0
        sector = ""
        industry_per = 0.0
        forward_per = 0.0
        target_price = 0
        industry_pbr = 0.0
        peers = []
        news_items = []
        roe = 0.0
        debt_ratio = 0.0
        op_margin = 0.0
        revenue_growth = 0.0
        dividend_yield = 0.0
        dividend_amt = 0
    else:
        per, pbr, foreign_buys, organ_buys, high_52, low_52, sector, industry_per, forward_per, target_price, industry_pbr, peers, news_items, roe, debt_ratio, op_margin, revenue_growth, dividend_yield, dividend_amt = get_per_pbr(code)
    theme = get_theme(name)

    # 지지선/저항선 기반 목표가·손절가
    support, resist, stop_loss = calc_support_resist(df)
    tp_price = max(resist, int(price * (1 + cfg["tp"] / 100)))
    sl_price = min(stop_loss, int(price * (1 - cfg["sl"] / 100)))

    buy_reasons = []
    sell_reasons = []

    # ── 시장 국면별 가중치 ──
    _phase, _ = market_phase()
    if _phase == "상승장":
        _trend_mult = 1.0
        _osc_mult = 1.0
        _oversold_bonus = 1.2
    elif _phase == "하락장":
        _trend_mult = 1.0
        _osc_mult = 0.7
        _oversold_bonus = 0.5
    else:
        _trend_mult = 0.7
        _osc_mult = 1.0
        _oversold_bonus = 1.0
    print(f"📊 시장 국면: {_phase}, 추세={_trend_mult}, 오실레이터={_osc_mult}, 과매도보너스={_oversold_bonus}")

    score = 50


    # RSI
    if rsi_val < cfg["RSI 과매도 (반등 가능)"]:
        buy_reasons.append("RSI 과매도 (반등 가능)")
        score += int(12 * _oversold_bonus)
    elif rsi_val > cfg["RSI 과매수 (조정 가능)"]:
        sell_reasons.append("RSI 과매수 (조정 가능)")
        score -= 12

    # MACD
    mh = macd_hist
    _macd_w = cfg.get("macd_weight", 15)
    if len(mh) > 1 and not np.isnan(mh.iloc[-1]) and not np.isnan(mh.iloc[-2]):
        if mh.iloc[-1] > 0 and mh.iloc[-2] <= 0:
            buy_reasons.append("MACD 매수 신호")
            score += int(_macd_w * _trend_mult)
        elif mh.iloc[-1] < 0 and mh.iloc[-2] >= 0:
            sell_reasons.append("MACD 매도 신호")
            score -= int(_macd_w * _trend_mult)


    # 볼린저밴드
    _bb_w = cfg.get("bb_weight", 10)
    if not np.isnan(bb_lower.iloc[-1]) and price <= bb_lower.iloc[-1]:
        buy_reasons.append("볼린저 하단 터치 (반등 기대)")
        score += _bb_w
    elif not np.isnan(bb_upper.iloc[-1]) and price >= bb_upper.iloc[-1]:
        sell_reasons.append("볼린저 상단 터치 (조정 기대)")
        score -= _bb_w

    # 다이버전스 감지
    def detect_divergence(prices, indicator, window=14):
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

    rsi_div = detect_divergence(close, rsi, window=20)
    macd_div = detect_divergence(close, macd_hist, window=20)

    divergence_text = ""
    if rsi_div == "bullish":
        buy_reasons.append("RSI 상승 다이버전스 (반전↑)")
        score += 15
        divergence_text = "🟢 RSI 상승 다이버전스"
    elif rsi_div == "bearish":
        sell_reasons.append("RSI 하락 다이버전스 (반전↓)")
        score -= 15
        divergence_text = "🔴 RSI 하락 다이버전스"
    if macd_div == "bullish":
        buy_reasons.append("MACD 상승 다이버전스 (반전↑)")
        score += 15
        divergence_text += " 🟢 MACD 상승 다이버전스"
    elif macd_div == "bearish":
        sell_reasons.append("MACD 하락 다이버전스 (반전↓)")
        score -= 15
        divergence_text += " 🔴 MACD 하락 다이버전스"

    # 거래량 이상 감지
    vol_mean_20 = volume.iloc[-21:-1].mean() if len(volume) > 21 else volume.mean()
    vol_today = volume.iloc[-1]
    vol_spike = ""
    if vol_ratio >= 5 and change > 0:
        buy_reasons.append(f"거래량 폭발 ({vol_ratio}배) + 양봉 → 세력 매집 가능")
        score += 5
        vol_spike = f"🔥 거래량 {vol_ratio}배 폭발"
    elif vol_ratio >= 3 and change > 0:
        buy_reasons.append(f"거래량 급증 ({vol_ratio}배) + 양봉")
        score += 3
        vol_spike = f"⚡ 거래량 {vol_ratio}배 급증"
    elif vol_ratio >= 3 and change < 0:
        sell_reasons.append(f"거래량 급증 ({vol_ratio}배) + 음봉 → 매도세 주의")
        score -= 3
        vol_spike = f"⚠️ 거래량 {vol_ratio}배 (음봉)"


    # 이동평균선 배열 감지
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else None
    ma120 = close.rolling(120).mean().iloc[-1] if len(close) >= 120 else None

    ma_align = ""
    _ema_w = cfg.get("ema_weight", 4)
    if ma120 and ma60:
        if ma5 > ma20 > ma60 > ma120:
            ma_align = "🟢 완전 정배열"
            buy_reasons.append("5>20>60>120 완전 정배열 (강한 상승추세)")
            score += int((_ema_w + 3) * _trend_mult)
        elif ma5 > ma20 > ma60:
            ma_align = "🟢 정배열 (60일선까지)"
            buy_reasons.append("5>20>60 정배열 (상승추세)")
            score += int(_ema_w * _trend_mult)
        elif ma5 < ma20 < ma60 < ma120:
            ma_align = "🔴 완전 역배열"
            sell_reasons.append("5<20<60<120 완전 역배열 (강한 하락추세)")
            score -= int((_ema_w + 3) * _trend_mult)
        elif ma5 < ma20 < ma60:
            ma_align = "🔴 역배열 (60일선까지)"
            sell_reasons.append("5<20<60 역배열 (하락추세)")
            score -= int(_ema_w * _trend_mult)
        elif ma5 > ma20 and ma20 < ma60:
            ma_align = "🟡 정배열 초입"
    elif ma60:
        if ma5 > ma20 > ma60:
            ma_align = "🟢 정배열"
            buy_reasons.append("5>20>60 정배열")
            score += int(_ema_w * _trend_mult)
        elif ma5 < ma20 < ma60:
            ma_align = "🔴 역배열"
            sell_reasons.append("5<20<60 역배열")
            score -= int(_ema_w * _trend_mult)



    # 일목균형
    sa = span_a.iloc[-1] if not np.isnan(span_a.iloc[-1]) else 0
    sb = span_b.iloc[-1] if not np.isnan(span_b.iloc[-1]) else 0
    if price > max(sa, sb) and sa > sb:
        buy_reasons.append(	"추세 구름 돌파 (상승 신호)")
        score += 8
    elif price < min(sa, sb) and sa < sb:
        sell_reasons.append(	"추세 구름 이탈 (하락 신호)")
        score -= 8

    # 슈퍼트렌드
    _st_w = cfg.get("st_weight", 6)
    if st_dir.iloc[-1] == 1 and (len(st_dir) > 1 and st_dir.iloc[-2] == -1):
        buy_reasons.append("슈퍼트렌드 매수 전환 (상승 시작)")
        score += int(_st_w * _trend_mult)
    elif st_dir.iloc[-1] == -1 and (len(st_dir) > 1 and st_dir.iloc[-2] == 1):
        sell_reasons.append("슈퍼트렌드 매도 전환 (하락 시작)")
        score -= int(_st_w * _trend_mult)


    # ── W자 바닥 / N자 반등 패턴 감지 ──
    try:
        _closes = df["Close"].values
        _len = len(_closes)
        if _len >= 60:
            # W자 바닥 (쌍바닥): 두 번 바닥 찍고 반등
            _recent = _closes[-60:]
            _mid = len(_recent) // 2
            _left = _recent[:_mid]
            _right = _recent[_mid:]
            _left_min_idx = np.argmin(_left)
            _left_min = _left[_left_min_idx]
            _right_min_idx = np.argmin(_right)
            _right_min = _right[_right_min_idx]
            _mid_max = np.max(_recent[_left_min_idx:_mid + _right_min_idx + 1]) if _mid + _right_min_idx > _left_min_idx else 0
            _cur = _closes[-1]

            # W자 조건: 두 바닥이 비슷하고, 중간 반등이 있고, 현재가가 중간 고점 돌파
            if _left_min > 0 and _right_min > 0 and _mid_max > 0:
                _bottom_diff = abs(_left_min - _right_min) / _left_min
                _bounce = (_mid_max - min(_left_min, _right_min)) / min(_left_min, _right_min)
                if _bottom_diff < 0.05 and _bounce > 0.03 and _cur >= _mid_max * 0.98:
                    score += 3            
                    buy_reasons.append("W자 바닥 패턴 (쌍바닥 돌파, 강한 매수 신호)")

            # N자 반등: 하락 → 반등 → 눌림 → 재상승
            _r30 = _closes[-30:]
            if len(_r30) >= 30:
                _p1 = np.max(_r30[:10])   # 초기 고점
                _p2 = np.min(_r30[5:20])  # 중간 저점
                _p3 = np.max(_r30[10:25]) # 반등 고점
                _p4 = np.min(_r30[20:])   # 눌림 저점
                _p5 = _closes[-1]         # 현재가

                if _p1 > 0 and _p2 > 0 and _p3 > 0 and _p4 > 0:
                    _drop1 = (_p1 - _p2) / _p1          # 첫 하락폭
                    _bounce1 = (_p3 - _p2) / _p2         # 반등폭
                    _pullback = (_p3 - _p4) / _p3        # 눌림폭
                    _recovery = (_p5 - _p4) / _p4 if _p4 > 0 else 0  # 재상승

                    # N자 조건: 하락 후 반등, 얕은 눌림, 재상승 중
                    if (_drop1 > 0.05 and _bounce1 > 0.03 and
                        _pullback > 0.02 and _pullback < _drop1 * 0.7 and
                        _recovery > 0.02 and _p5 > _p4):
                        score += 2      
                        buy_reasons.append("N자 반등 패턴 (눌림 후 재상승, 추세 전환)")

            # 역N자 (하락 패턴): 상승 → 고점 → 반락 → 반등 → 재하락
            if len(_r30) >= 30:
                _h1 = np.max(_r30[:10])
                _l1 = np.min(_r30[10:20])
                _h2 = np.max(_r30[15:25])
                _cur30 = _closes[-1]

                if _h1 > 0 and _l1 > 0 and _h2 > 0:
                    _fall1 = (_h1 - _l1) / _h1
                    _bounce2 = (_h2 - _l1) / _l1
                    _fall2 = (_h2 - _cur30) / _h2

                    if (_fall1 > 0.05 and _bounce2 > 0.02 and
                        _h2 < _h1 * 0.97 and _fall2 > 0.03):
                        score -= 8
                        sell_reasons.append("역N자 패턴 (반등 후 재하락, 하락 추세)")
                        sell_reasons.insert(0, "⚠️ 역N자 경고: 점수 높아도 추세 하락 중!")

    except:
        pass


    # OBV
    obv_ma = obv.rolling(20).mean()
    if not np.isnan(obv_ma.iloc[-1]):
        if obv.iloc[-1] > obv_ma.iloc[-1] * 1.05:
            buy_reasons.append(	"거래량 동반 상승 (돈이 들어옴)")
            score += 8
        elif obv.iloc[-1] < obv_ma.iloc[-1] * 0.95:
            sell_reasons.append("거래량 동반 하락 (돈이 빠짐)")
            score -= 8

    # MFI
    _mfi_buy = cfg.get("자금 유입 신호", 25)
    _mfi_sell = cfg.get("자금 유출 신호", 80)
    if mfi_val < _mfi_buy:
        pass  # 자금 유입 신호 비활성화 (백테스트 승률 6.7%)
    elif mfi_val > _mfi_sell:
        sell_reasons.append("자금 유출 신호")
        score -= int(5 * _osc_mult)


    # ADX
    _adx_min = cfg.get("adx_min", 25)
    if adx_val > _adx_min:
        if plus_di.iloc[-1] > minus_di.iloc[-1]:
            buy_reasons.append("강한 상승 힘 감지")
            score += 10
        else:
            sell_reasons.append("강한 하락 힘 감지")
            score -= 10

    # VWAP
    vwap_val = vwap.iloc[-1]
    if not np.isnan(vwap_val):
        if price > vwap_val:
            buy_reasons.append("평균 매수가 위 (매수세 우위)")
            score += 6
        else:
            sell_reasons.append("평균 매수가 아래 (매도세 우위)")
            score -= 6

    # 거래량 연속 증가
    _vol_w = cfg.get("vol_weight", 3)
    if vol_up_5:
        buy_reasons.append("vol5_buy")
        score += _vol_w + 1
    elif vol_up_3:
        pass  # vol3_buy 비활성화 (백테스트 승률 36.8%, 수익 -1.07%)


    # 충돌 보정
    if "RSI 과매수 (조정 가능)" in sell_reasons and "MACD 매수 신호" in buy_reasons:
        score -= 5

    score = max(0, min(100, score))

    if score >= 80:
        grade = "A+"
    elif score >= 70:
        grade = "A"
    elif score >= 60:
        grade = "B+"
    elif score >= 50:
        grade = "B"
    elif score >= 40:
        grade = "C"
    else:
        grade = "D"

    buy_count = len(buy_reasons)
    sell_count = len(sell_reasons)

    if buy_count >= 4:
        verdict = "적극 매수"
    elif buy_count >= 2 and buy_count > sell_count:
        verdict = "매수 관심"
    elif sell_count >= 4:
        verdict = "적극 매도"
    elif sell_count >= 2 and sell_count > buy_count:
        verdict = "매도 관심"
    else:
        verdict = "중립 관망"

    # AI 요약
    momentum = ""
    if chg5 > 0:
        momentum += f"5일 +{chg5}% "
    else:
        momentum += f"5일 {chg5}% "
    if chg10 > 0:
        momentum += f"10일 +{chg10}%"
    else:
        momentum += f"10일 {chg10}%"

    vol_comment = ""
    if vol_up_5:
        vol_comment = "거래량 5일 연속 증가 중! "
    elif vol_up_3:
        vol_comment = "거래량 3일 연속 증가. "
    elif vol_ratio > 2:
        vol_comment = f"거래량 {vol_ratio}배 폭증. "

    trend_comment = ""
    if e5 > e20 > e60:
        trend_comment = "EMA 정배열로 상승 추세. "
    elif e5 < e20 < e60:
        trend_comment = "EMA 역배열로 하락 추세. "

    per_comment = ""
    if per > 0:
        if industry_per > 0:
            avg = industry_per
        else:
            sector_name = sector if sector else theme
            avg = get_sector_per(sector_name)
        ratio = per / avg
        if ratio <= 0.5:
            per_label, per_detail = "저평가", f"업종PER({avg:.1f})의 {ratio:.0%}"
        elif ratio <= 0.8:
            per_label, per_detail = "다소저평가", f"업종PER({avg:.1f})의 {ratio:.0%}"
        elif ratio <= 1.2:
            per_label, per_detail = "적정", f"업종PER({avg:.1f})의 {ratio:.0%}"
        elif ratio <= 1.5:
            per_label, per_detail = "다소고평가", f"업종PER({avg:.1f})의 {ratio:.0%}"
        else:
            per_label, per_detail = "고평가", f"업종PER({avg:.1f})의 {ratio:.0%}"
        per_comment = f"PER {per:.1f} {per_label} — {per_detail}. "

    # EV/EBITDA 분석
    ev_ebitda_val = None
    ev_comment = ""
    if not st.session_state.get("scan_running", False):
        ev_data = get_ev_ebitda(code, is_us=(market in ["NASDAQ", "NYSE", "S&P500", "AMEX"]))
        ev_ebitda_val = ev_data.get("ev_ebitda")
        if ev_ebitda_val and ev_ebitda_val > 0:
            if ev_ebitda_val < 5:
                score -= 3                # ← +8에서 -3으로 (싼 데는 이유가 있다)
                sell_reasons.append(f"EV/EBITDA {ev_ebitda_val:.1f} 극도로 저평가 (가치함정 주의)")
                ev_comment = f"EV/EBITDA {ev_ebitda_val:.1f} 가치함정 주의. "
            elif ev_ebitda_val >= 6 and ev_ebitda_val <= 8:
                score += 8                # ← 5에서 8로 상향 (6~8이 승률 최고)
                buy_reasons.append(f"EV/EBITDA {ev_ebitda_val:.1f} 저평가")
                ev_comment = f"EV/EBITDA {ev_ebitda_val:.1f} 저평가. "
            elif ev_ebitda_val > 8 and ev_ebitda_val < 12:
                score += 2
                ev_comment = f"EV/EBITDA {ev_ebitda_val:.1f} 적정. "
            elif ev_ebitda_val > 25:
                score -= 3
                sell_reasons.append(f"EV/EBITDA {ev_ebitda_val:.1f} 고평가")
                ev_comment = f"EV/EBITDA {ev_ebitda_val:.1f} 고평가 주의. "
        score = max(0, min(100, score))

    support_comment = f"지지선 {support:,}원, 저항선 {resist:,}원. "

    if verdict in ["적극 매수", "매수 관심"]:
        action = f"매수 신호 {buy_count}개 감지. 추천 진입가 {support:,}원 부근, 목표가 {resist:,}원, 손절가 {stop_loss:,}원."
    elif verdict in ["적극 매도", "매도 관심"]:
        action = f"매도 신호 {sell_count}개. 지지선 {support:,}원 이탈 시 손절."
    else:
        action = f"뚜렷한 방향성 없음. 지지선 {support:,}원 확인 후 진입 고려."

    # 추세 위치
    if high_52 > 0 and low_52 > 0 and high_52 != low_52:
        pos_52 = round((price - low_52) / (high_52 - low_52) * 100, 1)
        if pos_52 <= 20:
            trend_pos = "바닥권 초입"
            trend_pos_icon = "🟢"
        elif pos_52 <= 40:
            trend_pos = "하단부"
            trend_pos_icon = "🟡"
        elif pos_52 <= 60:
            trend_pos = "중간대"
            trend_pos_icon = "🟠"
        elif pos_52 <= 80:
            trend_pos = "상단부"
            trend_pos_icon = "🟠"
        else:
            trend_pos = "고점권 주의"
            trend_pos_icon = "🔴"
    else:
        pos_52 = 0
        trend_pos = ""
        trend_pos_icon = ""

    # 외국인/기관 수급
    foreign_comment = ""
    organ_comment = ""
    if foreign_buys:
        consec_fb = 0
        for fb in foreign_buys:
            if fb > 0:
                consec_fb += 1
            else:
                break
        if consec_fb >= 3:
            foreign_comment = f"외국인 {consec_fb}일 연속 순매수! "
        elif consec_fb >= 1:
            foreign_comment = f"외국인 순매수 중. "
        else:
            neg_count = 0
            for fb in foreign_buys:
                if fb < 0:
                    neg_count += 1
                else:
                    break
            if neg_count >= 3:
                foreign_comment = f"외국인 {neg_count}일 연속 순매도. "
    if organ_buys:
        consec_ob = 0
        for ob in organ_buys:
            if ob > 0:
                consec_ob += 1
            else:
                break
        if consec_ob >= 3:
            organ_comment = f"기관 {consec_ob}일 연속 순매수! "
        elif consec_ob >= 1:
            organ_comment = f"기관 순매수 중. "

    # 52주 대비
    week52_comment = ""
    if high_52 > 0 and low_52 > 0:
        from_low = round((price / low_52 - 1) * 100, 1)
        from_high = round((price / high_52 - 1) * 100, 1)
        week52_comment = f"52주 최저 대비 +{from_low}%, 최고 대비 {from_high}%. "


    trend_pos_text = f"{trend_pos_icon} {trend_pos}. " if trend_pos else ""
    # 초보자용 AI 요약
    easy_parts = []

    if chg5 > 3:
        easy_parts.append(f"최근 5일간 {chg5}% 올랐어요 📈")
    elif chg5 < -3:
        easy_parts.append(f"최근 5일간 {abs(chg5)}% 빠졌어요 📉")
    else:
        easy_parts.append("최근 5일간 큰 변동 없이 횡보 중이에요")

    if trend_pos:
        if "바닥권" in trend_pos or "하단" in trend_pos:
            easy_parts.append(f"52주 중 {trend_pos_icon} {trend_pos}이라 저점 매수 기회일 수 있어요")
        elif "고점" in trend_pos or "상단" in trend_pos:
            easy_parts.append(f"52주 중 {trend_pos_icon} {trend_pos}이라 추격 매수는 위험해요")

    if vol_up_5:
        easy_parts.append("거래량이 5일 연속 늘고 있어요 — 시장의 관심이 집중되는 중!")
    elif vol_up_3:
        easy_parts.append("거래량이 3일 연속 늘고 있어요")
    elif vol_ratio > 2:
        easy_parts.append(f"오늘 거래량이 평소의 {vol_ratio}배 — 뭔가 움직임이 있어요")

    if e5 > e20 > e60:
        easy_parts.append("이동평균선이 정배열 — 상승 추세가 이어지고 있어요")
    elif e5 < e20 < e60:
        easy_parts.append("이동평균선이 역배열 — 하락 추세가 이어지고 있어요")

    if foreign_comment:
        easy_parts.append(foreign_comment.strip())
    if organ_comment:
        easy_parts.append(organ_comment.strip())

    if per > 0:
        if industry_per > 0:
            _avg = industry_per
        else:
            _avg = get_sector_per(sector if sector else theme)
        _ratio = per / _avg
        if _ratio <= 0.5:
            easy_parts.append(f"PER {per:.1f}배 — 업종평균({_avg:.1f})보다 훨씬 싸요! 💰")
        elif _ratio <= 0.8:
            easy_parts.append(f"PER {per:.1f}배 — 업종평균({_avg:.1f})보다 싼 편이에요 💰")
        elif _ratio <= 1.2:
            easy_parts.append(f"PER {per:.1f}배 — 업종평균({_avg:.1f})과 비슷한 수준이에요")
        elif _ratio <= 1.5:
            easy_parts.append(f"PER {per:.1f}배 — 업종평균({_avg:.1f})보다 다소 비싼 편이에요 ⚠️")
        else:
            easy_parts.append(f"PER {per:.1f}배 — 업종평균({_avg:.1f})보다 많이 비싸요 ⚠️")


    if pbr > 0:
        if pbr <= 1:
            easy_parts.append(f"PBR {pbr:.2f}배 — 회사 재산보다 주가가 싸요")
        elif pbr > 3:
            easy_parts.append(f"PBR {pbr:.2f}배 — 회사 재산 대비 주가가 비싼 편이에요")

    if ev_ebitda_val and ev_ebitda_val > 0:
        if ev_ebitda_val < 5:
            easy_parts.append(f"EV/EBITDA {ev_ebitda_val:.1f}배 — 영업이익 대비 기업 가치가 매우 싸요 💎")
        elif ev_ebitda_val < 8:
            easy_parts.append(f"EV/EBITDA {ev_ebitda_val:.1f}배 — 영업이익 대비 적당히 싼 편이에요")
        elif ev_ebitda_val > 25:
            easy_parts.append(f"EV/EBITDA {ev_ebitda_val:.1f}배 — 영업이익 대비 비싼 편이에요 ⚠️")

    if high_52 > 0 and low_52 > 0:
        from_low = round((price / low_52 - 1) * 100, 1)
        from_high = round((price / high_52 - 1) * 100, 1)
        if from_high > -10:
            easy_parts.append("52주 최고가에 거의 근접 — 신고가 도전 중이에요")
        elif from_low < 20:
            easy_parts.append("52주 최저가 근처 — 바닥에서 반등할 수 있어요")

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
        "ai_summary": ai_summary,
        "theme": sector if sector else theme,
        "tp_price": tp_price, "sl_price": sl_price,
        "support": support, "resist": resist, "stop_loss": stop_loss,
        "per": per, "pbr": pbr,
        "industry_per": industry_per,
        "forward_per": forward_per,
        "target_price": target_price,
        "industry_pbr": industry_pbr,
        "peers": peers,
        "news_items": news_items,
        "roe": roe,
        "debt_ratio": debt_ratio,
        "op_margin": op_margin,
        "revenue_growth": revenue_growth,
        "dividend_yield": dividend_yield,
        "dividend_amt": dividend_amt,
        "ev_ebitda": ev_ebitda_val,
        "is_us": market in ["NASDAQ", "NYSE", "S&P500", "AMEX"],
        "high_52": high_52,
        "low_52": low_52,
        "pos_52": pos_52,
        "trend_pos": trend_pos,
        "trend_pos_icon": trend_pos_icon,
        "foreign_buys": foreign_buys,
        "organ_buys": organ_buys,
        "chg5": chg5, "chg10": chg10, "chg20": chg20,
        "df": df,
        "ema5": ema5, "ema20": ema20, "ema60": ema60,
        "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
        "macd_line": macd_line, "macd_sig": macd_sig, "macd_hist": macd_hist,
        "rsi_series": rsi, "mfi_series": mfi,
        "obv": obv, "st_line": st_line, "st_dir": st_dir,
        "divergence": divergence_text.strip(),
        "vol_spike": vol_spike,
        "ma_align": ma_align,
    }

# ── 매도 알림 ──
def check_sell(r, cfg):
    alerts = []
    if r["price"] >= r["resist"]:
        alerts.append(f"🎯 {r['name']} 저항선 {r['resist']:,}원 도달! 익절 고려")
    if r["price"] <= r["stop_loss"]:
        alerts.append(f"🛑 {r['name']} 손절선 {r['stop_loss']:,}원 이탈! 손절 고려")
    if r["price"] <= r["support"]:
        alerts.append(f"⚠️ {r['name']} 지지선 {r['support']:,}원 근접! 주의")
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

    # ── SuperTrend 선 ──
    if r.get("st_line") is not None and r.get("st_dir") is not None:
        stl = r["st_line"].tail(last_n).reset_index(drop=True)
        std = r["st_dir"].tail(last_n).reset_index(drop=True)
        price_min = df["Low"].min()
        price_max = df["High"].max()
        price_margin = (price_max - price_min) * 0.1
        for i in range(1, n):
            y0 = stl.iloc[i - 1]
            y1 = stl.iloc[i]
            if (pd.isna(y0) or pd.isna(y1)
                    or y0 < price_min - price_margin
                    or y1 < price_min - price_margin
                    or y0 > price_max + price_margin
                    or y1 > price_max + price_margin):
                continue
            color = "#4caf50" if std.iloc[i] == 1 else "#f44336"
            ax1.plot([i - 1, i], [y0, y1],
                     color=color, linewidth=1.5, alpha=0.7)


    # ── 세력매집 신호 (초록 큰 점 ● 캔들 아래) ──
    vol = df["Volume"].values
    close = df["Close"].values
    low = df["Low"].values
    opn = df["Open"].values
    high = df["High"].values
    vol_ma20 = pd.Series(vol).rolling(20).mean().values
    obv_s = r["obv"].tail(last_n).reset_index(drop=True)
    obv_ma10 = obv_s.rolling(10).mean()

    accumulation_pts = []
    for i in range(20, n):
        body = abs(close[i] - opn[i])
        candle_range = high[i] - low[i]
        if candle_range == 0:
            continue
        body_ratio = body / candle_range
        vol_ratio = vol[i] / vol_ma20[i] if vol_ma20[i] > 0 else 0
        lower_wick = min(opn[i], close[i]) - low[i]
        wick_ratio = lower_wick / candle_range

        score = 0

        # 거래량 급증
        if vol_ratio >= 3.0:
            score += 3
        elif vol_ratio >= 2.0:
            score += 2
        # 작은 몸통
        if body_ratio <= 0.25:
            score += 3
        elif body_ratio <= 0.35:
            score += 2
        # 긴 아래꼬리
        if wick_ratio >= 0.5:
            score += 3
        elif wick_ratio >= 0.35:
            score += 2
        # OBV 상승
        if (not pd.isna(obv_ma10.iloc[i])
                and obv_s.iloc[i] >= obv_ma10.iloc[i]):
            score += 2

        # 7점 이상이면 세력매집 신호
        if score >= 7:
            accumulation_pts.append(i)

    for i in accumulation_pts:
        marker_y = low[i] - (high[i] - low[i]) * 0.8
        ax1.plot(i, marker_y, "^", color="#00e676",
                 markersize=12, alpha=0.95, zorder=10)

    # ── 매도 경고 신호 (빨간 큰 점 ▼ 캔들 위) ──
    distribution_pts = []
    for i in range(20, n):
        body = abs(close[i] - opn[i])
        candle_range = high[i] - low[i]
        if candle_range == 0:
            continue
        body_ratio = body / candle_range
        vol_ratio = vol[i] / vol_ma20[i] if vol_ma20[i] > 0 else 0
        upper_wick = high[i] - max(opn[i], close[i])
        wick_ratio = upper_wick / candle_range

        score = 0
        # 거래량 급증
        if vol_ratio >= 3.0:
            score += 3
        elif vol_ratio >= 2.0:
            score += 2
        # 작은 몸통
        if body_ratio <= 0.25:
            score += 3
        elif body_ratio <= 0.35:
            score += 2
        # 긴 윗꼬리
        if wick_ratio >= 0.5:
            score += 3
        elif wick_ratio >= 0.35:
            score += 2
        # OBV 하락
        if (not pd.isna(obv_ma10.iloc[i])
                and obv_s.iloc[i] < obv_ma10.iloc[i]):
            score += 2

        # 7점 이상이면 매도 경고
        if score >= 7:
            distribution_pts.append(i)

    for i in distribution_pts:
        marker_y = high[i] + (high[i] - low[i]) * 0.8
        ax1.plot(i, marker_y, "v", color="#ff1744",
                 markersize=12, alpha=0.95, zorder=10)


    # 지지선/저항선 표시
    ax1.axhline(r["support"], color="#4caf50", linewidth=1, linestyle="--", label=f"지지 {r['support']:,}")
    ax1.axhline(r["resist"], color="#ff9800", linewidth=1, linestyle="--", label=f"저항 {r['resist']:,}")
    ax1.axhline(r["stop_loss"], color="#f44336", linewidth=1, linestyle="--", label=f"손절 {r['stop_loss']:,}")
    ax1.set_title(f"{r['name']} ({r['code']})  {r['price']:,}원  {r['change']:+.2f}%", color="white", fontsize=11)
    ax1.legend(fontsize=7, loc="upper left")

    ax2 = axes[1]
    mh = r["macd_hist"].tail(last_n).reset_index(drop=True)
    ml = r["macd_line"].tail(last_n).reset_index(drop=True)
    ms = r["macd_sig"].tail(last_n).reset_index(drop=True)
    colors = ["#ff1744" if v >= 0 else "#2979ff" for v in mh]
    ax2.bar(range(n), mh[:n], color=colors, width=0.7)
    ax2.plot(range(n), ml[:n], color="#ffeb3b", linewidth=0.8)
    ax2.plot(range(n), ms[:n], color="#ff9800", linewidth=0.8)
    ax2.set_title("MACD", color="white", fontsize=9)

    ax3 = axes[2]
    rs = r["rsi_series"].tail(last_n).reset_index(drop=True)
    mf = r["mfi_series"].tail(last_n).reset_index(drop=True)
    ax3.plot(range(n), rs[:n], color="#ffeb3b", linewidth=0.8, label="RSI")
    ax3.plot(range(n), mf[:n], color="#26c6da", linewidth=0.8, label="MFI")
    ax3.axhline(70, color="#ff1744", linewidth=0.5, linestyle="--")
    ax3.axhline(30, color="#4caf50", linewidth=0.5, linestyle="--")
    ax3.set_title("RSI / MFI", color="white", fontsize=9)
    ax3.legend(fontsize=7)

    ax4 = axes[3]
    ob = r["obv"].tail(last_n).reset_index(drop=True)
    ax4.plot(range(n), ob[:n], color="#ab47bc", linewidth=0.8)
    ax4.set_title("OBV", color="white", fontsize=9)

    plt.tight_layout()
    return fig

# ── 카드 UI (2열 레이아웃) ──
def show_card(r, prefix, cfg, is_new_buy=False, is_new_sell=False, show_wl_btn=True):
    score = r["score"]
    if score >= 80:
        score_color = "#4caf50"
        score_label = "상승 잠재력"
    elif score >= 60:
        score_color = "#ff9800"
        score_label = "상승 잠재력"
    else:
        score_color = "#f44336"
        score_label = "하락 신호"

    verdict_emoji = {"적극 매수": "🔥", "매수 관심": "👀", "적극 매도": "🧊", "매도 관심": "⚠️", "중립 관망": "😐"}.get(r["verdict"], "")
    crown = r.get("crown", "")
    theme_badge = f" | {r['theme']}" if r["theme"] else ""
    chg_color = "#ff1744" if r["change"] >= 0 else "#2979ff"
    is_us_stock = r.get("is_us", False)
    currency = "$" if is_us_stock else ""
    unit = "" if is_us_stock else "원"

    st.markdown(f'<div class="card">', unsafe_allow_html=True)

    # ── 헤더: 종목명 + 점수 (전체 너비) ──
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div>'
        f'<span style="font-size:1.4em;font-weight:bold">{crown} {r["name"]} ({r["code"]}) | {r.get("theme","")}</span><br>'
        f'<span style="color:{chg_color};font-size:1.3em;font-weight:bold">{currency}{r["price"]:,}{unit} ({r["change"]:+.2f}%)</span> '
        f'<span style="font-size:0.9em;color:#aaa">거래대금 {r["trade_val"]:,.0f}{"만$" if is_us_stock else "억"} | 거래량비 {r["vol_ratio"]}배</span>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<span style="font-size:2.2em;font-weight:bold;color:{score_color}">{score}점</span><br>'
        f'<span style="font-size:1.1em;color:{score_color}">{r["grade"]} {verdict_emoji} {r["verdict"]}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── PER/PBR 배지 (전체 너비) ──
    per_html = ""
    if r["per"] > 0:
        if r.get("industry_per", 0) > 0:
            _avg = r["industry_per"]
        else:
            _sector = r.get("theme", "")
            _avg = get_sector_per(_sector)
        _ratio = r["per"] / _avg
        if _ratio <= 0.5:
            _label = "저평가"
        elif _ratio <= 0.8:
            _label = "다소저평가"
        elif _ratio <= 1.2:
            _label = "적정"
        elif _ratio <= 1.5:
            _label = "다소고평가"
        else:
            _label = "고평가"
        _detail = f"업종PER({_avg:.1f})의 {_ratio:.0%}"
        if _label in ["저평가", "다소저평가"]:
            per_html = f'<span class="badge-per-low">PER {r["per"]:.1f} {_label} ({_detail})</span>'
        elif _label == "적정":
            per_html = f'<span class="badge-per-mid">PER {r["per"]:.1f} {_label} ({_detail})</span>'
        else:
            per_html = f'<span class="badge-per-high">PER {r["per"]:.1f} {_label} ({_detail})</span>'

    fper_html = ""
    if r.get("forward_per", 0) > 0:
        fper_html = f'<span style="background:#1565c0;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">예상PER {r["forward_per"]:.1f}</span>'

    tp_html = ""
    if r.get("target_price", 0) > 0:
        _upside = round((r["target_price"] / r["price"] - 1) * 100, 1)
        if _upside > 0:
            tp_html = f'<span style="background:#2e7d32;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">목표가 {r["target_price"]:,}원 (▲{_upside}%)</span>'
        else:
            tp_html = f'<span style="background:#c62828;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">목표가 {r["target_price"]:,}원 (▼{_upside}%)</span>'

    pbr_html = ""
    if r["pbr"] > 0:
        if r.get("industry_pbr", 0) > 0:
            _pavg = r["industry_pbr"]
            _pratio = r["pbr"] / _pavg
            if _pratio <= 0.5:
                _plabel = "저평가"
            elif _pratio <= 0.8:
                _plabel = "다소저평가"
            elif _pratio <= 1.2:
                _plabel = "적정"
            elif _pratio <= 1.5:
                _plabel = "다소고평가"
            else:
                _plabel = "고평가"
            _pdetail = f"업종PBR({_pavg:.2f})의 {_pratio:.0%}"
            if _plabel in ["저평가", "다소저평가"]:
                pbr_html = f'<span class="badge-pbr-low">PBR {r["pbr"]:.2f} {_plabel} ({_pdetail})</span>'
            elif _plabel == "적정":
                pbr_html = f'<span class="badge-pbr-mid">PBR {r["pbr"]:.2f} {_plabel} ({_pdetail})</span>'
            else:
                pbr_html = f'<span class="badge-pbr-high">PBR {r["pbr"]:.2f} {_plabel} ({_pdetail})</span>'
        else:
            if r["pbr"] <= 1:
                pbr_html = f'<span class="badge-pbr-low">PBR {r["pbr"]:.2f} 저평가</span>'
            elif r["pbr"] <= 3:
                pbr_html = f'<span class="badge-pbr-mid">PBR {r["pbr"]:.2f} 보통</span>'
            else:
                pbr_html = f'<span class="badge-pbr-high">PBR {r["pbr"]:.2f} 고평가</span>'

    ev_html = ""
    if r.get("ev_ebitda") and r["ev_ebitda"] > 0:
        if r["ev_ebitda"] < 8:
            ev_html = f'<span class="badge-pbr-low">EV/EBITDA {r["ev_ebitda"]:.1f} 저평가</span>'
        elif r["ev_ebitda"] <= 15:
            ev_html = f'<span class="badge-pbr-mid">EV/EBITDA {r["ev_ebitda"]:.1f} 보통</span>'
        else:
            ev_html = f'<span class="badge-pbr-high">EV/EBITDA {r["ev_ebitda"]:.1f} 고평가</span>'

    if per_html or fper_html or tp_html or pbr_html or ev_html:
        st.markdown(f"{per_html} {fper_html} {tp_html} {pbr_html} {ev_html}", unsafe_allow_html=True)

    # ═══ 2열 레이아웃 시작 ═══
    col_left, col_right = st.columns([6, 4])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  왼쪽: 차트 + 신호 + AI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with col_left:
        # 💡 밸류에이션 코멘트
        val_comments = []
        if r["per"] > 0:
            if r.get("forward_per", 0) > 0 and r["per"] > r["forward_per"] * 2:
                val_comments.append(f"📊 현재 PER {r['per']:.1f}배로 비싸 보이지만, 예상PER {r['forward_per']:.1f}배로 이익이 크게 늘어날 전망이에요 → 미래 기준으로는 싸요! 💰")
            elif r.get("forward_per", 0) > 0 and r["forward_per"] > r["per"] * 1.5:
                val_comments.append(f"📊 현재 PER {r['per']:.1f}배인데, 예상PER {r['forward_per']:.1f}배로 이익이 줄어들 전망이에요 → 주의가 필요해요 ⚠️")
            elif r.get("forward_per", 0) > 0:
                val_comments.append(f"📊 현재 PER {r['per']:.1f}배, 예상PER {r['forward_per']:.1f}배로 비슷한 수준이에요")
        if r.get("target_price", 0) > 0:
            _upside = round((r["target_price"] / r["price"] - 1) * 100, 1)
            if _upside > 20:
                val_comments.append(f"🎯 증권사 목표가 {r['target_price']:,}원 — 현재가보다 {_upside}% 상승 여력! 기대되는 종목이에요 📈")
            elif _upside > 0:
                val_comments.append(f"🎯 증권사 목표가 {r['target_price']:,}원 — 현재가보다 {_upside}% 상승 여력이 있어요 📈")
            else:
                val_comments.append(f"🎯 증권사 목표가 {r['target_price']:,}원 — 현재가가 목표가를 이미 넘었어요. 차익실현 고려 ⚠️")
        if r["pbr"] > 0:
            if r.get("industry_pbr", 0) > 0:
                _pratio = r["pbr"] / r["industry_pbr"]
                if _pratio <= 0.8:
                    val_comments.append(f"🏢 PBR {r['pbr']:.2f}배로 업종평균({r['industry_pbr']:.2f})보다 싸요 — 회사 재산 대비 저평가 💰")
                elif _pratio <= 1.2:
                    val_comments.append(f"🏢 PBR {r['pbr']:.2f}배로 업종평균({r['industry_pbr']:.2f})과 비슷해요 — 적정 수준")
                else:
                    val_comments.append(f"🏢 PBR {r['pbr']:.2f}배로 업종평균({r['industry_pbr']:.2f})보다 비싸요 ⚠️")
        if r.get("ev_ebitda") and r["ev_ebitda"] > 0:
            if r["ev_ebitda"] < 8:
                val_comments.append(f"💎 EV/EBITDA {r['ev_ebitda']:.1f}배 — 영업이익 대비 기업 가치가 싸요!")
            elif r["ev_ebitda"] > 20:
                val_comments.append(f"⚠️ EV/EBITDA {r['ev_ebitda']:.1f}배 — 영업이익 대비 기업 가치가 비싸요")
        if val_comments:
            val_text = "<br>".join(val_comments)
            st.markdown(
                f'<div style="background:#1a1a2e;border-left:4px solid #00b4d8;padding:10px 15px;border-radius:5px;margin:5px 0;font-size:0.9em">'
                f'<b>💡 밸류에이션 한눈에 보기</b><br>{val_text}</div>',
                unsafe_allow_html=True,
            )

        # 추세 위치 + 52주 + 수급 뱃지
        extra_badges = ""
        if r.get("trend_pos"):
            tp_colors = {"바닥권 초입": "#4caf50", "하단부": "#8bc34a", "중간대": "#ff9800", "상단부": "#ff5722", "고점권 주의": "#f44336"}
            tp_col = tp_colors.get(r["trend_pos"], "#888")
            extra_badges += f'<span style="background:{tp_col};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">{r["trend_pos_icon"]} {r["trend_pos"]}</span> '
        if r.get("pos_52", 0) > 0:
            extra_badges += f'<span style="background:#333;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">52주 위치 {r["pos_52"]}%</span> '
        if r.get("foreign_buys"):
            consec = 0
            for fb in r["foreign_buys"]:
                if fb > 0:
                    consec += 1
                else:
                    break
            if consec >= 3:
                extra_badges += f'<span style="background:#e91e63;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">🔥 외국인 {consec}일 연속 순매수</span> '
        if r.get("organ_buys"):
            consec = 0
            for ob in r["organ_buys"]:
                if ob > 0:
                    consec += 1
                else:
                    break
            if consec >= 3:
                extra_badges += f'<span style="background:#9c27b0;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">🔥 기관 {consec}일 연속 순매수</span> '
        if r.get("divergence"):
            div_col = "#4caf50" if "상승" in r["divergence"] else "#f44336"
            extra_badges += f'<span style="background:{div_col};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">{r["divergence"]}</span> '
        if r.get("vol_spike"):
            extra_badges += f'<span style="background:#ff6f00;color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">{r["vol_spike"]}</span> '
        if r.get("ma_align"):
            ma_col = "#4caf50" if "정배열" in r["ma_align"] else "#f44336" if "역배열" in r["ma_align"] else "#ff9800"
            extra_badges += f'<span style="background:{ma_col};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.85em">{r["ma_align"]}</span> '
        if extra_badges:
            st.markdown(extra_badges, unsafe_allow_html=True)

        # 매수/매도 신호 바
        buy_count = len(r["buy_reasons"])
        sell_count = len(r["sell_reasons"])
        if buy_count > 0:
            new_tag = "🆕 " if is_new_buy else ""
            st.markdown(
                f'<div class="buy-blink" style="background:#ff1744;color:#fff;padding:10px 14px;border-radius:8px;margin:8px 0;font-size:1.15em">'
                f'{new_tag}🔴 매수 신호 {buy_count}개 감지!</div>',
                unsafe_allow_html=True,
            )
        if sell_count > 0:
            new_tag = "🆕 " if is_new_sell else ""
            st.markdown(
                f'<div class="sell-blink" style="background:#2979ff;color:#fff;padding:10px 14px;border-radius:8px;margin:8px 0;font-size:1.15em">'
                f'{new_tag}🔵 매도 신호 {sell_count}개 감지!</div>',
                unsafe_allow_html=True,
            )

        # 멀티타임프레임
        wt = get_weekly_trend(r["code"])
        if wt:
            if wt["trend"] == "상승" and r["score"] >= 80:
                mtf_msg = f'{wt["emoji"]} {wt["desc"]} + 일봉 매수 신호 → <b>🔥 강력 매수 구간</b>'
                mtf_bg = "#1a2e1a"
                mtf_border = "#4caf50"
            elif wt["trend"] == "하락" and r["score"] >= 80:
                mtf_msg = f'{wt["emoji"]} {wt["desc"]} + 일봉 매수 신호 → <b>⚠️ 역추세 매매 (주의)</b>'
                mtf_bg = "#2e1a1a"
                mtf_border = "#f44336"
            elif wt["trend"] == "하락":
                mtf_msg = f'{wt["emoji"]} {wt["desc"]} → <b>매수 비추천</b>'
                mtf_bg = "#2e1a1a"
                mtf_border = "#f44336"
            else:
                mtf_msg = f'{wt["emoji"]} {wt["desc"]}'
                mtf_bg = "#2a2a3e"
                mtf_border = "#ff9800"
            st.markdown(
                f'<div style="background:{mtf_bg};padding:10px 14px;border-radius:8px;margin:6px 0;border-left:4px solid {mtf_border}">'
                f'📅 <b>멀티타임프레임:</b> {mtf_msg}</div>',
                unsafe_allow_html=True,
            )

        # AI 요약
        st.markdown(
            f'<div style="background:#2a2a3e;padding:10px 14px;border-radius:8px;margin:6px 0;border-left:4px solid #ff9800">'
            f'🤖 <b>AI 판단:</b> {r["ai_summary"]}</div>',
            unsafe_allow_html=True,
        )

        # 지지선/저항선/손절가
        st.markdown(
            f'<div style="background:#1a2e1a;padding:8px 14px;border-radius:8px;margin:4px 0">'
            f'🟢 <b>추천 진입가(지지선):</b> <span class="support-line">{currency}{r["support"]:,}{unit}</span> &nbsp;&nbsp; '
            f'🟡 <b>목표가(저항선):</b> <span class="resist-line">{currency}{r["tp_price"]:,}{unit}</span> '
            f'(+{round((r["tp_price"]/(r["price"]+1e-9)-1)*100,1)}%)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#2e1a1a;padding:8px 14px;border-radius:8px;margin:4px 0">'
            f'🔴 <b>손절가(지지선 이탈):</b> <span class="stop-line">{currency}{r["sl_price"]:,}{unit}</span> '
            f'({round((r["sl_price"]/(r["price"]+1e-9)-1)*100,1)}%) &nbsp;&nbsp; '
            f'📏 <b>손익비:</b> {min(round((r["tp_price"]-r["price"])/(r["price"]-r["sl_price"]+1e-9),1), 99.9)}:1</div>',
            unsafe_allow_html=True,
        )

        # 매수/매도 사유 목록
        bc, sc = st.columns(2)
        with bc:
            for br in r["buy_reasons"]:
                st.markdown(f'🔴 {EASY.get(br, br)}')
        with sc:
            for sr in r["sell_reasons"]:
                st.markdown(f'🔵 {EASY.get(sr, sr)}')
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  오른쪽: 데이터 패널
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with col_right:
        # ── 동종업계 비교 ──
        if r.get("peers") and len(r["peers"]) > 0:
            peer_rows = ""
            for p in r["peers"][:5]:
                p_name = p.get("name", "")
                p_price = p.get("price", "0")
                p_chg = p.get("change", "0")
                p_chg_val = float(p_chg) if p_chg else 0
                p_chg_color = "#ff4444" if p_chg_val > 0 else "#4488ff" if p_chg_val < 0 else "#aaa"
                p_chg_sign = "+" if p_chg_val > 0 else ""
                p_mktcap = p.get("marketValue", "0")
                try:
                    p_mktcap_num = int(str(p_mktcap).replace(",", ""))
                    if p_mktcap_num > 10000:
                        p_mktcap_str = f"{p_mktcap_num / 10000:.1f}조"
                    else:
                        p_mktcap_str = f"{p_mktcap_num:,}억"
                except:
                    p_mktcap_str = str(p_mktcap)
                peer_rows += f'<tr><td style="padding:4px 8px">{p_name}</td><td style="padding:4px 8px;text-align:right">{p_price}원</td><td style="padding:4px 8px;text-align:right;color:{p_chg_color}">{p_chg_sign}{p_chg_val:.2f}%</td><td style="padding:4px 8px;text-align:right">{p_mktcap_str}</td></tr>'
            st.markdown(
                f'<div style="background:#1a1a2e;border-left:4px solid #ff9800;padding:10px 15px;border-radius:5px;margin:8px 0">'
                f'<b>🏭 동종업계 비교</b>'
                f'<table style="width:100%;margin-top:8px;border-collapse:collapse;font-size:0.85em">'
                f'<tr style="border-bottom:1px solid #333"><th style="padding:4px 8px;text-align:left">종목</th><th style="padding:4px 8px;text-align:right">현재가</th><th style="padding:4px 8px;text-align:right">등락률</th><th style="padding:4px 8px;text-align:right">시가총액</th></tr>'
                f'{peer_rows}'
                f'</table></div>',
                unsafe_allow_html=True,
            )

        # ── 뉴스 감성 분석 ──
        if r.get("news_items") and len(r["news_items"]) > 0:
            pos_words = ["상승", "급등", "호실적", "최고", "성장", "흑자", "수혜", "기대", "호재", "강세", "돌파", "신고가", "매수", "목표가 상향", "실적 개선", "수주", "계약"]
            neg_words = ["하락", "급락", "적자", "손실", "위기", "우려", "매도", "약세", "폭락", "파업", "리스크", "부진", "감소", "악재", "목표가 하향", "공매도", "조정"]
            pos_count = 0
            neg_count = 0
            news_html = ""
            for n in r["news_items"]:
                title = n.get("title", "")
                body = n.get("body", "")
                text = title + " " + body
                is_pos = any(w in text for w in pos_words)
                is_neg = any(w in text for w in neg_words)
                if is_pos and not is_neg:
                    icon = "🟢"
                    pos_count += 1
                elif is_neg and not is_pos:
                    icon = "🔴"
                    neg_count += 1
                elif is_pos and is_neg:
                    icon = "🟡"
                else:
                    icon = "⚪"
                dt = n.get("datetime", "")
                if len(dt) >= 12:
                    time_str = f"{dt[4:6]}/{dt[6:8]} {dt[8:10]}:{dt[10:12]}"
                else:
                    time_str = dt
                news_html += f'<div style="padding:3px 0;border-bottom:1px solid #222">{icon} <span style="color:#aaa;font-size:0.8em">{time_str} | {n.get("office","")}</span> {title}</div>'
            total = len(r["news_items"])
            neutral_count = total - pos_count - neg_count
            if pos_count > neg_count:
                sentiment = "긍정적 📈"
                sent_color = "#4caf50"
            elif neg_count > pos_count:
                sentiment = "부정적 📉"
                sent_color = "#f44336"
            else:
                sentiment = "중립 ➡️"
                sent_color = "#ff9800"
            st.markdown(
                f'<div style="background:#1a1a2e;border-left:4px solid {sent_color};padding:10px 15px;border-radius:5px;margin:8px 0">'
                f'<b>📰 뉴스 감성 분석</b> — <span style="color:{sent_color};font-weight:bold">{sentiment}</span> '
                f'<span style="font-size:0.85em;color:#aaa">(긍정 {pos_count} / 부정 {neg_count} / 중립 {neutral_count})</span>'
                f'{news_html}</div>',
                unsafe_allow_html=True,
            )

        # ── 재무 건전성 ──
        roe_val = r.get("roe", 0)
        debt_val = r.get("debt_ratio", 0)
        op_val = r.get("op_margin", 0)
        rev_val = r.get("revenue_growth", 0)
        if roe_val or debt_val or op_val or rev_val:
            fin_items = ""
            if roe_val:
                if roe_val >= 15:
                    roe_color, roe_label = "#4caf50", "우량"
                elif roe_val >= 10:
                    roe_color, roe_label = "#ff9800", "양호"
                elif roe_val > 0:
                    roe_color, roe_label = "#aaa", "보통"
                else:
                    roe_color, roe_label = "#f44336", "적자"
                fin_items += f'<tr><td style="padding:4px 8px">ROE</td><td style="padding:4px 8px;text-align:right;color:{roe_color};font-weight:bold">{roe_val:.1f}%</td><td style="padding:4px 8px;text-align:right;color:{roe_color}">{roe_label}</td></tr>'
            if debt_val:
                if debt_val <= 100:
                    d_color, d_label = "#4caf50", "안전"
                elif debt_val <= 200:
                    d_color, d_label = "#ff9800", "보통"
                else:
                    d_color, d_label = "#f44336", "위험"
                fin_items += f'<tr><td style="padding:4px 8px">부채비율</td><td style="padding:4px 8px;text-align:right;color:{d_color};font-weight:bold">{debt_val:.1f}%</td><td style="padding:4px 8px;text-align:right;color:{d_color}">{d_label}</td></tr>'
            if op_val:
                if op_val >= 15:
                    o_color, o_label = "#4caf50", "고수익"
                elif op_val >= 5:
                    o_color, o_label = "#ff9800", "양호"
                elif op_val > 0:
                    o_color, o_label = "#aaa", "저수익"
                else:
                    o_color, o_label = "#f44336", "적자"
                fin_items += f'<tr><td style="padding:4px 8px">영업이익률</td><td style="padding:4px 8px;text-align:right;color:{o_color};font-weight:bold">{op_val:.1f}%</td><td style="padding:4px 8px;text-align:right;color:{o_color}">{o_label}</td></tr>'
            if rev_val:
                if rev_val >= 20:
                    rv_color, rv_label = "#4caf50", "고성장"
                elif rev_val >= 5:
                    rv_color, rv_label = "#ff9800", "성장"
                elif rev_val > 0:
                    rv_color, rv_label = "#aaa", "정체"
                else:
                    rv_color, rv_label = "#f44336", "역성장"
                fin_items += f'<tr><td style="padding:4px 8px">매출 성장률</td><td style="padding:4px 8px;text-align:right;color:{rv_color};font-weight:bold">{rev_val:.1f}%</td><td style="padding:4px 8px;text-align:right;color:{rv_color}">{rv_label}</td></tr>'
            st.markdown(
                f'<div style="background:#1a1a2e;border-left:4px solid #2196f3;padding:10px 15px;border-radius:5px;margin:8px 0">'
                f'<b>📊 재무 건전성</b>'
                f'<table style="width:100%;margin-top:8px;border-collapse:collapse;font-size:0.85em">'
                f'<tr style="border-bottom:1px solid #333"><th style="padding:4px 8px;text-align:left">항목</th><th style="padding:4px 8px;text-align:right">수치</th><th style="padding:4px 8px;text-align:right">평가</th></tr>'
                f'{fin_items}'
                f'</table></div>',
                unsafe_allow_html=True,
            )

        # ── 수급 흐름 차트 ──
        fb = r.get("foreign_buys", [])
        ob = r.get("organ_buys", [])
        if fb and ob and len(fb) >= 3:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            days_label = [f"D-{len(fb)-1-i}" for i in range(len(fb))]
            days_label[-1] = "오늘"
            fig_supply = make_subplots(specs=[[{"secondary_y": True}]])
            fig_supply.add_trace(
                go.Bar(x=days_label, y=fb, name="외국인",
                       marker_color=["#ff4444" if v < 0 else "#4488ff" for v in fb]),
                secondary_y=False,
            )
            fig_supply.add_trace(
                go.Bar(x=days_label, y=ob, name="기관",
                       marker_color=["#ff8844" if v < 0 else "#44cc88" for v in ob]),
                secondary_y=False,
            )
            fig_supply.update_layout(
                title=dict(text="📈 외국인·기관 순매수 추이", font=dict(size=14)),
                barmode="group",
                height=250,
                margin=dict(l=10, r=10, t=40, b=30),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1a1a2e",
                font=dict(color="#ccc", size=11),
                legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
                yaxis=dict(gridcolor="#333", title="순매수(주)"),
            )
            st.plotly_chart(fig_supply, use_container_width=True)

        # ── 배당 정보 ──
        div_yield = r.get("dividend_yield", 0)
        div_amt = r.get("dividend_amt", 0)
        if div_yield > 0:
            if div_yield >= 5:
                dy_color, dy_label = "#4caf50", "고배당"
            elif div_yield >= 2:
                dy_color, dy_label = "#ff9800", "양호"
            elif div_yield >= 1:
                dy_color, dy_label = "#aaa", "보통"
            else:
                dy_color, dy_label = "#666", "저배당"
            st.markdown(
                f'<div style="background:#1a1a2e;border-left:4px solid #9c27b0;padding:10px 15px;border-radius:5px;margin:8px 0">'
                f'<b>💰 배당 정보</b><br>'
                f'<span style="font-size:1.2em;color:{dy_color};font-weight:bold">{div_yield:.2f}%</span> '
                f'<span style="color:{dy_color}">({dy_label})</span>'
                f'<span style="color:#aaa;margin-left:15px">주당 배당금 {div_amt:,}원</span></div>',
                unsafe_allow_html=True,
            )
    # ═══ 2열 레이아웃 끝, 다시 전체 너비 ═══

    # ── Gemini AI 심층 분석 ──
    if GEMINI_OK:
        gm_state_key = f"gm_result_{r['code']}"
        if st.button("🧠 Gemini AI 심층 분석", key=f"{prefix}_gemini_{r['code']}"):
            with st.spinner("🧠 Gemini AI가 분석 중..."):
                gm = gemini_judgment(
                    r["name"], r["code"], r["price"], r["change"],
                    r["score"], r["grade"], r["rsi"], r["mfi"], r["adx"],
                    r.get("per", 0), r.get("pbr", 0), r.get("ev_ebitda", None),
                    r.get("buy_reasons", []), r.get("sell_reasons", []),
                    r.get("support", 0), r.get("resist", 0), r.get("verdict", "")
                )
            if gm:
                st.session_state[gm_state_key] = gm
            else:
                st.warning("Gemini AI 응답을 가져올 수 없습니다.")
        if gm_state_key in st.session_state:
            st.markdown(
                f'<div style="background:#1a2e1a;padding:14px;border-radius:10px;margin:8px 0;border-left:4px solid #4caf50">'
                f'🧠 <b>Gemini AI 심층 분석</b><br><br>'
                f'{st.session_state[gm_state_key]}</div>',
                unsafe_allow_html=True,
            )

    # ── Gemini 뉴스 감성 분석 ──
    if GEMINI_OK:
        news_state_key = f"news_result_{r['code']}"
        if st.button("📰 뉴스 감성 분석", key=f"{prefix}_gemini_news_{r['code']}"):
            with st.spinner("📰 뉴스 수집 및 감성 분석 중..."):
                news_result = gemini_news_sentiment(r["name"], r["code"])
            if news_result:
                st.session_state[news_state_key] = news_result
            else:
                st.warning("뉴스를 가져올 수 없습니다.")
        if news_state_key in st.session_state:
            st.markdown(
                f'<div style="background:#2e2a1a;padding:14px;border-radius:10px;margin:8px 0;border-left:4px solid #ffc107">'
                f'📰 <b>뉴스 감성 분석</b><br><br>'
                f'{st.session_state[news_state_key]}</div>',
                unsafe_allow_html=True,
            )

    # 버튼
    btn_c1, btn_c2, btn_c3, btn_c4, btn_c5, btn_c6 = st.columns(6)
    with btn_c1:
        if show_wl_btn:
            wl = load_wl()
            is_in_wl = any(w["code"] == r["code"] for w in wl)
            if is_in_wl:
                if st.button("관심 삭제", key=f"{prefix}_rm_{r['code']}"):
                    remove_from_wl(r["code"])
                    st.rerun()
            else:
                if st.button("⭐ 관심 추가", key=f"{prefix}_add_{r['code']}"):
                    add_to_wl(r["code"], r["name"])
                    st.rerun()
    with btn_c2:
        if st.button("차트 보기", key=f"{prefix}_chart_{r['code']}"):
            fig = draw_chart(r, cfg)
            st.pyplot(fig)
            plt.close(fig)
    with btn_c3:
        if KIS_OK:
            if st.button("⏱️ 분봉", key=f"{prefix}_min_{r['code']}"):
                df_min = fetch_minute(r["code"])
                if df_min is not None and len(df_min) > 0:
                    fig = draw_minute_chart(df_min)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("분봉 데이터 없음")
    with btn_c4:
        if st.button("📈 백테스트", key=f"{prefix}_bt_{r['code']}"):
            with st.spinner("백테스팅 중..."):
                bt = run_backtest(r["code"], r["name"], cfg)
            if bt:
                if bt["win_rate"] >= 60:
                    bt_color = "#4caf50"
                elif bt["win_rate"] >= 40:
                    bt_color = "#ff9800"
                else:
                    bt_color = "#f44336"
                st.markdown(
                    f'<div style="background:#1a1a2e;padding:14px;border-radius:10px;border:1px solid {bt_color}">'
                    f'<b>📈 백테스트 결과 (최근 120일)</b><br>'
                    f'총 거래: {bt["total_trades"]}회 | '
                    f'<span style="color:#4caf50">승: {bt["wins"]}회</span> | '
                    f'<span style="color:#f44336">패: {bt["losses"]}회</span><br>'
                    f'<span style="font-size:1.3em;color:{bt_color}">승률 {bt["win_rate"]}%</span> | '
                    f'총 수익률 {bt["total_pnl"]:+.2f}% | 최대낙폭(MDD) -{bt.get("mdd", 0):.2f}%<br>'
                    f'평균 수익 {bt["avg_win"]:+.2f}% | 평균 손실 {bt["avg_loss"]:+.2f}%'
                    f'</div>', unsafe_allow_html=True
                )
                with st.expander("거래 내역"):
                    for t in bt["trades"]:
                        pnl_color = "#4caf50" if t["pnl"] > 0 else "#f44336"
                        st.markdown(f'<span style="color:{pnl_color}">{t["entry_date"]} → {t["exit_date"]} | {t["entry_price"]:,}원 → {t["exit_price"]:,}원 | {t["pnl"]:+.2f}% ({t["reason"]})</span>', unsafe_allow_html=True)
            else:
                st.info("백테스트 데이터 부족 (신호 발생 이력 없음)")
    with btn_c5:
        if KIS_OK:
            if st.button("📋 호가/체결", key=f"{prefix}_ob_{r['code']}"):
                with st.spinner("조회 중..."):
                    ob_result = analyze_orderbook(r["code"])
                    try:
                        price_info = _broker.get_price(r["code"])
                    except Exception:
                        price_info = None
                    st.write(f"DEBUG price_info: {price_info}")

                if price_info and "체결강도" in price_info:
                    ts = price_info["체결강도"]
                    if ts >= 150:
                        ts_color = "#f44336"
                        ts_label = "🔥 매우 강한 매수세"
                    elif ts >= 120:
                        ts_color = "#ff5722"
                        ts_label = "💪 강한 매수세"
                    elif ts >= 100:
                        ts_color = "#4caf50"
                        ts_label = "📈 매수 우위"
                    elif ts >= 80:
                        ts_color = "#ff9800"
                        ts_label = "⚖️ 보합"
                    else:
                        ts_color = "#2196f3"
                        ts_label = "📉 매도 우위"
                    st.markdown(
                        f'<div style="background:#1a1a2e;padding:10px;border-radius:10px;border:2px solid {ts_color};margin:6px 0">'
                        f'<b>⚡ 체결강도 <span style="font-size:1.5em;color:{ts_color}">{ts:.1f}%</span></b> '
                        f'<span style="color:{ts_color}">{ts_label}</span>'
                        f'</div>', unsafe_allow_html=True)
                if ob_result:
                    st.markdown(
                        f'<div style="background:#1a1a2e;padding:10px;border-radius:10px;border:1px solid {ob_result["signal_color"]};margin:6px 0">'
                        f'<b>📋 호가</b> <span style="color:{ob_result["signal_color"]}">{ob_result["signal"]}</span><br>'
                        f'매수잔량 {ob_result["total_bid"]:,} | 매도잔량 {ob_result["total_ask"]:,} | 비율 {ob_result["ratio"]:.1f}%'
                        f'</div>', unsafe_allow_html=True)
                    if ob_result.get("sell_walls"):
                        for w in ob_result["sell_walls"]:
                            st.warning(f"매도벽: {w['price']:,}원 — {w['volume']:,}주")
                    if ob_result.get("buy_walls"):
                        for w in ob_result["buy_walls"]:
                            st.success(f"매수벽: {w['price']:,}원 — {w['volume']:,}주")
                    with st.expander("호가 상세"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("매도호가")
                            for a in ob_result["asks"][::-1]:
                                st.write(f"{a['price']:,}원 — {a['volume']:,}주")
                        with c2:
                            st.caption("매수호가")
                            for b in ob_result["bids"]:
                                st.write(f"{b['price']:,}원 — {b['volume']:,}주")
                if not ob_result and (not price_info or "체결강도" not in price_info):
                    st.info("호가/체결강도 조회 불가 (장중에만 가능)")
        else:
            st.caption("KIS 미연결")

    with btn_c6:
        if st.button("📰 뉴스", key=f"{prefix}_news_{r['code']}"):
            news = get_stock_news(r["code"], r["name"])
            if news:
                for n in news:
                    st.markdown(
                        f'<div style="background:#1a1a2e;padding:8px 12px;border-radius:8px;margin:3px 0;border-left:3px solid #ff9800">'
                        f'📰 <a href="{n["link"]}" target="_blank" style="color:#fff;text-decoration:none">{n["title"]}</a> '
                        f'<span style="color:#888;font-size:0.8em">— {n["source"]}</span>'
                        f'</div>', unsafe_allow_html=True
                    )
            else:
                st.info("관련 뉴스를 찾을 수 없습니다")

    # 매도 알림
    alerts = check_sell(r, cfg)
    for a in alerts:
        st.markdown(
            f'<div style="background:#4a1a1a;padding:8px 14px;border-radius:8px;margin:4px 0;border-left:4px solid #f44336">'
            f'{a}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ─── 호가창 분석 ───
def analyze_orderbook(code):
    if not KIS_OK or _broker is None:
        return None
    try:
        ob = _broker.get_orderbook(code)
        if not ob:
            return None
        asks = ob["asks"]
        bids = ob["bids"]
        total_ask = ob["total_ask_vol"]
        total_bid = ob["total_bid_vol"]
        if total_ask == 0:
            return None
        bid_ask_ratio = round(total_bid / total_ask, 2)
        # 매도벽 감지 (매도 호가 중 평균의 3배 이상)
        ask_vols = [a["volume"] for a in asks if a["volume"] > 0]
        avg_ask = sum(ask_vols) / len(ask_vols) if ask_vols else 0
        sell_walls = [a for a in asks if a["volume"] >= avg_ask * 3]
        # 매수벽 감지
        bid_vols = [b["volume"] for b in bids if b["volume"] > 0]
        avg_bid = sum(bid_vols) / len(bid_vols) if bid_vols else 0
        buy_walls = [b for b in bids if b["volume"] >= avg_bid * 3]
        # 판단
        signal = ""
        signal_color = "#888"
        if bid_ask_ratio >= 2.0:
            signal = "🔥 매수세 압도 (급등 가능)"
            signal_color = "#ff1744"
        elif bid_ask_ratio >= 1.3:
            signal = "👀 매수세 우위"
            signal_color = "#ff9800"
        elif bid_ask_ratio <= 0.5:
            signal = "🧊 매도세 압도 (하락 주의)"
            signal_color = "#2979ff"
        elif bid_ask_ratio <= 0.7:
            signal = "⚠️ 매도세 우위"
            signal_color = "#42a5f5"
        else:
            signal = "😐 균형 상태"
            signal_color = "#888"
        return {
            "asks": asks,
            "bids": bids,
            "total_ask": total_ask,
            "total_bid": total_bid,
            "ratio": bid_ask_ratio,
            "sell_walls": sell_walls,
            "buy_walls": buy_walls,
            "signal": signal,
            "signal_color": signal_color,
        }
    except Exception:
        return None

# ─── 섹터 동반 상승 감지 ───
def detect_sector_surge(all_results):
    theme_map = {}
    for skey, rlist in all_results.items():
        for r in rlist:
            th = r.get("theme", "")
            if not th:
                continue
            if th not in theme_map:
                theme_map[th] = []
            theme_map[th].append({
                "name": r["name"],
                "code": r["code"],
                "score": r["score"],
                "change": r["change"],
                "style": skey,
            })
    surges = []
    for theme, stocks in theme_map.items():
        unique_codes = list(set([s["code"] for s in stocks]))
        if len(unique_codes) >= 3:
            avg_score = round(sum(s["score"] for s in stocks) / len(stocks), 1)
            avg_change = round(sum(s["change"] for s in stocks) / len(stocks), 2)
            surges.append({
                "theme": theme,
                "count": len(unique_codes),
                "avg_score": avg_score,
                "avg_change": avg_change,
                "stocks": stocks,
            })
        elif len(unique_codes) >= 2:
            avg_score = round(sum(s["score"] for s in stocks) / len(stocks), 1)
            avg_change = round(sum(s["change"] for s in stocks) / len(stocks), 2)
            surges.append({
                "theme": theme,
                "count": len(unique_codes),
                "avg_score": avg_score,
                "avg_change": avg_change,
                "stocks": stocks,
            })
    surges.sort(key=lambda x: x["count"], reverse=True)
    return surges

# ─── 전체 스캔 ───────────────────────────────────────
def run_full_scan(stocks):
    stocks = stocks.reset_index(drop=True)
    total = len(stocks)
    if total == 0:
        return {}, {}
    cached_data = {}
    all_results = {}

    st.subheader("📡 데이터 수집 중...")
    dl_bar = st.progress(0.0)
    dl_text = st.empty()
    for i in range(total):
        row = stocks.iloc[i]
        code = str(row["Code"]).strip()
        name = str(row["Name"]).strip()
        dl_bar.progress(min((i + 1) / total, 1.0))
        dl_text.text(f"다운로드: {name} ({i + 1}/{total})")
        df = fetch(code)
        if df is not None:
            cached_data[code] = {"name": name, "df": df}
    dl_bar.empty()
    dl_text.empty()
    
    # ── 시장 국면별 최소 점수 조정 ──
    phase, phase_icon = market_phase()
    if phase == "하락장":
        adjusted_min_score = 90
        st.warning(f"{phase_icon} 하락장 감지 — 최소 점수 90점으로 상향 (엄격 모드)")
    elif phase == "상승장":
        adjusted_min_score = 70
        st.success(f"{phase_icon} 상승장 감지 — 최소 점수 70점으로 하향 (적극 모드)")
    else:
        adjusted_min_score = MIN_SCORE
        st.info(f"{phase_icon} 횡보장 — 기본 점수 {MIN_SCORE}점 유지")


    st.subheader("🔎 3개 스타일 동시 분석 중...")
    an_bar = st.progress(0.0)
    an_text = st.empty()
    total_work = len(cached_data) * len(STYLES)
    done = 0
    for style_name, cfg in STYLES.items():
        short_key = cfg["key"]
        results = []
        for code, cdata in cached_data.items():
            done += 1
            an_bar.progress(min(done / (total_work + 1), 1.0))
            an_text.text(f"분석: {cdata['name']} - {style_name}")
            try:
                r = analyze(cdata["df"], code, cdata["name"], cfg)
                if r["trade_val"] < cfg["min_trade_val"]:
                    continue
                if r["score"] >= adjusted_min_score:
                    r["crown"] = ""
                    results.append(r)
            except Exception:
                pass
        all_results[short_key] = results
    an_bar.empty()
    an_text.empty()

    # 상위 종목 EV/EBITDA 추가 조회
    st.subheader("💎 상위 종목 EV/EBITDA 조회 중...")
    ev_bar = st.progress(0.0)
    ev_count = 0
    for skey, rlist in all_results.items():
        top_list = sorted(rlist, key=lambda x: x["score"], reverse=True)[:15]
        for r in top_list:
            ev_count += 1
            ev_bar.progress(min(ev_count / (len(all_results) * 15), 1.0))
            try:
                is_us = market in ["NASDAQ", "NYSE", "S&P500", "AMEX"]
                ev_data = get_ev_ebitda(r["code"], is_us=is_us)
                ev_val = ev_data.get("ev_ebitda")
                r["ev_ebitda"] = ev_val
                if ev_val and ev_val > 0:
                    if ev_val < 5:
                        r["score"] += 8
                        r["buy_reasons"].append(f"EV/EBITDA {ev_val:.1f} 극도로 저평가")
                    elif ev_val < 8:
                        r["score"] += 5
                        r["buy_reasons"].append(f"EV/EBITDA {ev_val:.1f} 저평가")
                    elif ev_val > 25:
                        r["score"] -= 3
                        r["sell_reasons"].append(f"EV/EBITDA {ev_val:.1f} 고평가")
                    r["score"] = max(0, min(100, r["score"]))
            except:
                pass
    ev_bar.empty()

    # ── 매크로 보너스 반영 ──
    _macro = st.session_state.get("macro_bonus", 0)
    _macro_reasons = st.session_state.get("macro_reasons", [])
    if _macro != 0:
        for skey, rlist in all_results.items():
            for r in rlist:
                r["score"] = max(0, min(100, r["score"] + _macro))
                if _macro > 0:
                    r["buy_reasons"].append(f"매크로 우호 ({', '.join(_macro_reasons)})")
                else:
                    r["sell_reasons"].append(f"매크로 악재 ({', '.join(_macro_reasons)})")

    # return all_results, cached_data    ← ★ 이 줄 삭제!

    # ── 섹터 집중 방지 (같은 테마 최대 3종목) ──
    for skey, rlist in all_results.items():
        theme_count = {}
        filtered_list = []
        rlist.sort(key=lambda x: x["score"], reverse=True)
        for r in rlist:
            th = r.get("theme", "")
            if th:
                theme_count[th] = theme_count.get(th, 0) + 1
                if theme_count[th] > 3:
                    continue
            filtered_list.append(r)
        all_results[skey] = filtered_list

    # 크라운 (중복 코드 제거 — 하나만 남김)
    code_styles = {}
    for skey, rlist in all_results.items():
        for r in rlist:
            c = r["code"]
            if c not in code_styles:
                code_styles[c] = set()
            code_styles[c].add(skey)
    for skey, rlist in all_results.items():
        for r in rlist:
            c = r["code"]
            cnt = len(code_styles.get(c, set()))
            if cnt >= 3:
                r["crown"] = "🏆 3관왕"
            elif cnt >= 2:
                r["crown"] = "⭐ 2관왕"
            else:
                r["crown"] = ""

    return all_results, cached_data   

# ─── 세션 초기화 ─────────────────────────────────────
for key, val in [
    ("all_scan_results", {}),
    ("cached_data", {}),
    ("prev_signals", {}),
    ("hist_prev_signals", {}),
    ("sound_counter", 0),
    ("scan_done", False),
    ("scan_running", False),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─── 사이드바 (급등 예측 탐색기 스타일) ─────────────
with st.sidebar:
    st.markdown("## 🔥 급등 예측 탐색기")
    st.caption("v24.0 PRO | AI 매매 + 감시 투자")
    st.sidebar.markdown(f"**🧠 Gemini AI:** {'🟢 연결됨' if GEMINI_OK else '⚪ 미연결'}")
    st.divider()

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
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**🔑 Gemini API 키**")
    _user_key = st.text_input("API 키 입력", type="password", placeholder="AIzaSy...", key="gemini_key_input", label_visibility="collapsed")
    if _user_key:
        st.session_state["gemini_api_key"] = _user_key
    st.caption("[무료 API 키 발급받기](https://aistudio.google.com/apikey)")
    st.divider()
    st.markdown("**시장**")
    market = st.selectbox("시장", ["KOSDAQ", "KOSPI", "NASDAQ", "S&P500", "NYSE"], index=0, label_visibility="collapsed")
    sound_on = st.checkbox("🔊 사운드 알림", value=False)

    # ── 핫 테마 설정 ──
    with st.expander("🔥 핫 테마 설정", expanded=False):
        theme_settings = load_manual_themes()
        
        theme_enabled = st.checkbox("테마 보너스 활성화", value=theme_settings.get("enabled", True), key="theme_enabled")
        theme_auto = st.checkbox("자동 감지 (네이버 실시간)", value=theme_settings.get("auto", True), key="theme_auto")
        
        # 현재 자동 감지된 핫 테마 표시
        if theme_auto:
            try:
                hot = get_hot_themes(top_n=5)
                if hot:
                    st.markdown("**📈 오늘의 핫 테마 TOP 5**")
                    for h in hot:
                        st.caption(f"🔥 {h['name']} ({h['change_rate']})")
            except:
                pass
        
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
                save_manual_themes(theme_settings)
                st.rerun()
        
        if current_keywords:
            st.markdown("**현재 수동 키워드:**")
            for i, kw in enumerate(current_keywords):
                col_kw, col_del = st.columns([3, 1])
                with col_kw:
                    st.caption(f"🏷️ {kw}")
                with col_del:
                    if st.button("❌", key=f"del_kw_{i}"):
                        current_keywords.pop(i)
                        theme_settings["manual_keywords"] = current_keywords
                        save_manual_themes(theme_settings)
                        st.rerun()
        
        # 설정 저장
        if theme_enabled != theme_settings.get("enabled") or theme_auto != theme_settings.get("auto"):
            theme_settings["enabled"] = theme_enabled
            theme_settings["auto"] = theme_auto
            save_manual_themes(theme_settings)

    # ── 커스텀 알림 조건 ──
    with st.expander("🔔 커스텀 알림 조건", expanded=False):
        preset = st.selectbox("📋 프리셋 선택", [
            "직접 설정",
            "💎 저평가 우량주",
            "🚀 급등 후보",
            "📉 바닥 반등 종목"
        ], key="alert_preset")

        # 프리셋 변경 감지 → 값 자동 세팅
        if "prev_preset" not in st.session_state:
            st.session_state["prev_preset"] = "직접 설정"

        if preset != st.session_state["prev_preset"]:
            st.session_state["prev_preset"] = preset
            if preset == "💎 저평가 우량주":
                st.session_state["alert_per"] = 15.0
                st.session_state["alert_pbr"] = 1.5
                st.session_state["alert_score"] = 60
                st.session_state["alert_rsi_low"] = 0
                st.session_state["alert_vol"] = 0.0
                st.session_state["alert_pattern"] = []
            elif preset == "🚀 급등 후보":
                st.session_state["alert_per"] = 0.0
                st.session_state["alert_pbr"] = 0.0
                st.session_state["alert_score"] = 70
                st.session_state["alert_rsi_low"] = 0
                st.session_state["alert_vol"] = 2.0
                st.session_state["alert_pattern"] = []
            elif preset == "📉 바닥 반등 종목":
                st.session_state["alert_per"] = 0.0
                st.session_state["alert_pbr"] = 0.0
                st.session_state["alert_score"] = 50
                st.session_state["alert_rsi_low"] = 35
                st.session_state["alert_vol"] = 0.0
                st.session_state["alert_pattern"] = ["W자 바닥", "N자 반등"]
            else:  # 직접 설정
                st.session_state["alert_per"] = 0.0
                st.session_state["alert_pbr"] = 0.0
                st.session_state["alert_score"] = 0
                st.session_state["alert_rsi_low"] = 0
                st.session_state["alert_vol"] = 0.0
                st.session_state["alert_pattern"] = []
            st.rerun()

        alert_per = st.number_input("PER 이하", min_value=0.0, max_value=100.0, step=1.0, key="alert_per", help="0이면 비활성")
        alert_pbr = st.number_input("PBR 이하", min_value=0.0, max_value=50.0, step=0.5, key="alert_pbr", help="0이면 비활성")
        alert_score = st.number_input("점수 이상", min_value=0, max_value=100, step=5, key="alert_score", help="0이면 비활성")
        alert_rsi_low = st.number_input("RSI 이하 (과매도)", min_value=0, max_value=100, step=5, key="alert_rsi_low", help="0이면 비활성")
        alert_vol = st.number_input("거래량 배율 이상", min_value=0.0, max_value=50.0, step=0.5, key="alert_vol", help="0이면 비활성")
        alert_pattern = st.multiselect("패턴 필터", ["W자 바닥", "N자 반등", "골든크로스"], key="alert_pattern", help="선택한 패턴이 감지된 종목만")

    # ── 공포/탐욕 지수 ──
    try:
        fg_res = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5).json()
        fg_val = int(fg_res["data"][0]["value"])
        fg_label = fg_res["data"][0]["value_classification"]
        if fg_val <= 25:
            fg_emoji, fg_color = "😱", "#f44336"
            fg_kr = "극도의 공포"
        elif fg_val <= 45:
            fg_emoji, fg_color = "😰", "#ff9800"
            fg_kr = "공포"
        elif fg_val <= 55:
            fg_emoji, fg_color = "😐", "#ffeb3b"
            fg_kr = "중립"
        elif fg_val <= 75:
            fg_emoji, fg_color = "😊", "#8bc34a"
            fg_kr = "탐욕"
        else:
            fg_emoji, fg_color = "🤑", "#4caf50"
            fg_kr = "극도의 탐욕"
        st.sidebar.markdown(
            f'<div style="background:#1a1a2e;padding:10px;border-radius:8px;border:1px solid {fg_color};text-align:center">'
            f'{fg_emoji} <b>크립토 공포/탐욕 지수</b><br>'
            f'<span style="font-size:1.8em;color:{fg_color}">{fg_val}</span><br>'
            f'<span style="color:{fg_color}">{fg_kr} ({fg_label})</span></div>',
            unsafe_allow_html=True
        )
    except Exception:
        pass

    # ── 미국 경제 지표 ──
    try:
        eco_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        # CPI (소비자물가지수)
        cpi_resp = requests.get("https://tradingeconomics.com/united-states/inflation-cpi", headers=eco_headers, timeout=5)
        cpi_val = ""
        import re as _re
        cpi_m = _re.search(r'Inflation Rate in the United States (?:increased|decreased|remained unchanged at) (?:to )?([\d.]+)', cpi_resp.text)
        if cpi_m:
            cpi_val = cpi_m.group(1)

        # 실업률
        unemp_resp = requests.get("https://tradingeconomics.com/united-states/unemployment-rate", headers=eco_headers, timeout=5)
        unemp_val = ""
        unemp_m = _re.search(r'Unemployment Rate in the United States (?:increased|decreased|remained unchanged at) (?:to )?([\d.]+)', unemp_resp.text)
        if unemp_m:
            unemp_val = unemp_m.group(1)

        # 기준금리
        rate_resp = requests.get("https://tradingeconomics.com/united-states/interest-rate", headers=eco_headers, timeout=5)
        rate_val = ""
        rate_m = _re.search(r'last recorded at ([\d.]+) percent', rate_resp.text)
        if rate_m:
            rate_val = rate_m.group(1)

        if cpi_val or unemp_val or rate_val:
            # CPI 색상
            cpi_f = float(cpi_val) if cpi_val else 0
            if cpi_f >= 4:
                cpi_color, cpi_icon = "#f44336", "🔴"
                cpi_label = "인플레 과열"
            elif cpi_f >= 3:
                cpi_color, cpi_icon = "#ff9800", "🟡"
                cpi_label = "인플레 주의"
            elif cpi_f >= 2:
                cpi_color, cpi_icon = "#4caf50", "🟢"
                cpi_label = "안정적"
            else:
                cpi_color, cpi_icon = "#2196f3", "🔵"
                cpi_label = "디플레 우려"

            # 실업률 색상
            unemp_f = float(unemp_val) if unemp_val else 0
            if unemp_f >= 6:
                unemp_color, unemp_icon = "#f44336", "🔴"
                unemp_label = "고용 위기"
            elif unemp_f >= 5:
                unemp_color, unemp_icon = "#ff9800", "🟡"
                unemp_label = "주의"
            elif unemp_f >= 4:
                unemp_color, unemp_icon = "#ff9800", "🟡"
                unemp_label = "보통"
            else:
                unemp_color, unemp_icon = "#4caf50", "🟢"
                unemp_label = "양호"

            # 금리 색상
            rate_f = float(rate_val) if rate_val else 0
            if rate_f >= 5:
                rate_color, rate_icon = "#f44336", "🔴"
                rate_label = "긴축"
            elif rate_f >= 4:
                rate_color, rate_icon = "#ff9800", "🟡"
                rate_label = "중립~긴축"
            elif rate_f >= 2:
                rate_color, rate_icon = "#ff9800", "🟡"
                rate_label = "중립"
            else:
                rate_color, rate_icon = "#4caf50", "🟢"
                rate_label = "완화적"

            # 금리 전망 판단
            if cpi_f >= 3.5 and unemp_f < 5:
                outlook = "📈 금리 인상 가능성 ↑"
                outlook_color = "#f44336"
            elif cpi_f < 2.5 and unemp_f >= 4.5:
                outlook = "📉 금리 인하 가능성 ↑"
                outlook_color = "#4caf50"
            else:
                outlook = "➡️ 금리 동결 전망"
                outlook_color = "#ff9800"

            st.sidebar.markdown(
                f'<div style="background:#1a1a2e;padding:10px;border-radius:8px;border:1px solid #444;margin-top:10px">'
                f'🇺🇸 <b>미국 경제 지표</b><br>'
                f'<table style="width:100%;font-size:0.85em;margin-top:5px">'
                f'<tr><td>{cpi_icon} CPI</td><td style="text-align:right;color:{cpi_color};font-weight:bold">{cpi_val}%</td><td style="text-align:right;color:{cpi_color};font-size:0.8em">{cpi_label}</td></tr>'
                f'<tr><td>{unemp_icon} 실업률</td><td style="text-align:right;color:{unemp_color};font-weight:bold">{unemp_val}%</td><td style="text-align:right;color:{unemp_color};font-size:0.8em">{unemp_label}</td></tr>'
                f'<tr><td>💵 기준금리</td><td style="text-align:right;color:{rate_color};font-weight:bold">{rate_val}%</td><td style="text-align:right;color:{rate_color};font-size:0.8em">{rate_label}</td></tr>'
                f'</table>'
                f'<div style="margin-top:8px;padding:5px;background:#111;border-radius:5px;text-align:center;color:{outlook_color};font-weight:bold">{outlook}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    except Exception:
        pass


    # ── 한국 경제지표 + VIX + 미국 10년물 국채 ──
    try:
        import re as _re2

        # 한국 기준금리 (tradingeconomics)
        bok_rate = ""
        try:
            _te_h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            bok_page = requests.get("https://tradingeconomics.com/south-korea/interest-rate", headers=_te_h, timeout=5).text
            bok_match = _re2.search(r'last recorded at ([\d.]+) percent', bok_page)
            if bok_match:
                bok_rate = bok_match.group(1)
        except:
            pass

        # 원/달러 환율 (exchangerate-api, 키 필요 없음)
        krw_val = ""
        try:
            _er_resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5).json()
            if _er_resp.get("result") == "success":
                _krw_rate = _er_resp["rates"].get("KRW")
                if _krw_rate:
                    krw_val = str(round(_krw_rate, 1))
        except:
            pass

        # VIX (yfinance)
        vix_val = ""
        try:
            import yfinance as _yf
            _vix_tk = _yf.Ticker("^VIX")
            _vix_hist = _vix_tk.history(period="1d")
            if not _vix_hist.empty:
                vix_val = str(round(_vix_hist["Close"].iloc[-1], 2))
        except:
            pass


        # 미국 10년물 국채 (yfinance)
        bond_val = ""
        try:
            import yfinance as _yf2
            _tnx_tk = _yf2.Ticker("^TNX")
            _tnx_hist = _tnx_tk.history(period="1d")
            if not _tnx_hist.empty:
                bond_val = str(round(_tnx_hist["Close"].iloc[-1], 2))
        except:
            pass

        # 코스피 공매도 비중 (KRX)
        short_val = ""
        try:
            import datetime as _dt
            _today = _dt.datetime.now().strftime("%Y%m%d")
            _short_resp = requests.post(
                "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd",
                data={
                    "bld": "dbms/MDC/STAT/srt/MDCSTAT1001",
                    "locale": "ko_KR",
                    "trdDd": _today,
                    "share": "1",
                    "money": "1",
                    "csvxls_is498": "false",
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5
            )
            _short_data = _short_resp.json()
            if _short_data.get("output"):
                _ratios = []
                for _item in _short_data["output"]:
                    try:
                        _ratios.append(float(_item.get("SHRT_SELL_RATIO", "0")))
                    except:
                        pass
                if _ratios:
                    short_val = f"{sum(_ratios)/len(_ratios):.2f}"
        except:
            pass

        # ── 색상/라벨 결정 ──
        bok_f = float(bok_rate) if bok_rate else 0
        if bok_f <= 2.0:
            bok_icon, bok_color, bok_label = "🟢", "#4caf50", "저금리 (호재)"
        elif bok_f <= 3.0:
            bok_icon, bok_color, bok_label = "🟡", "#ff9800", "보통"
        else:
            bok_icon, bok_color, bok_label = "🔴", "#f44336", "고금리 (부담)"

        krw_f = float(krw_val) if krw_val else 0
        if krw_f <= 1250:
            krw_icon, krw_color, krw_label = "🟢", "#4caf50", "원화 강세"
        elif krw_f <= 1350:
            krw_icon, krw_color, krw_label = "🟡", "#ff9800", "보통"
        elif krw_f <= 1450:
            krw_icon, krw_color, krw_label = "🟠", "#ff5722", "원화 약세 주의"
        else:
            krw_icon, krw_color, krw_label = "🔴", "#f44336", "원화 급락"

        vix_f = float(vix_val) if vix_val else 0
        if vix_f <= 15:
            vix_icon, vix_color, vix_label = "😎", "#4caf50", "안정"
        elif vix_f <= 20:
            vix_icon, vix_color, vix_label = "😐", "#ff9800", "보통"
        elif vix_f <= 30:
            vix_icon, vix_color, vix_label = "😰", "#ff5722", "불안"
        else:
            vix_icon, vix_color, vix_label = "😱", "#f44336", "공포"

        bond_f = float(bond_val) if bond_val else 0
        if bond_f <= 3.5:
            bond_icon, bond_color, bond_label = "🟢", "#4caf50", "저금리 (주식 호재)"
        elif bond_f <= 4.0:
            bond_icon, bond_color, bond_label = "🟡", "#ff9800", "보통"
        elif bond_f <= 4.5:
            bond_icon, bond_color, bond_label = "🟠", "#ff5722", "부담"
        else:
            bond_icon, bond_color, bond_label = "🔴", "#f44336", "고금리 (주식 악재)"

        short_f = float(short_val) if short_val else 0
        if short_f <= 5:
            short_icon, short_color, short_label = "🟢", "#4caf50", "낮음 (긍정)"
        elif short_f <= 10:
            short_icon, short_color, short_label = "🟡", "#ff9800", "보통"
        else:
            short_icon, short_color, short_label = "🔴", "#f44336", "높음 (부정)"

        # ── 매크로 보너스 점수 ──
        macro_bonus = 0
        macro_reasons = []
        try:
            _cpi_f = float(cpi_val) if cpi_val else 0
            _unemp_f = float(unemp_val) if unemp_val else 0
            if _cpi_f < 2.5 and _unemp_f >= 4.5:
                macro_bonus += 3
                macro_reasons.append("CPI 안정+금리 인하 기대")
            elif _cpi_f >= 4.0 and _unemp_f < 4.0:
                macro_bonus -= 3
                macro_reasons.append("CPI 과열+긴축 지속")
        except:
            pass
        if bok_f > 0:
            if bok_f <= 2.0:
                macro_bonus += 2; macro_reasons.append("한국 저금리")
            elif bok_f >= 3.5:
                macro_bonus -= 2; macro_reasons.append("한국 고금리")
        if vix_f > 0:
            if vix_f <= 15:
                macro_bonus += 2; macro_reasons.append("VIX 안정")
            elif vix_f >= 30:
                macro_bonus -= 3; macro_reasons.append("VIX 공포")
            elif vix_f >= 25:
                macro_bonus -= 2; macro_reasons.append("VIX 불안")
        if bond_f > 0:
            if bond_f <= 3.5:
                macro_bonus += 2; macro_reasons.append("미국채 저금리")
            elif bond_f >= 4.5:
                macro_bonus -= 2; macro_reasons.append("미국채 고금리")
        if krw_f > 0:
            if krw_f <= 1200:
                macro_bonus += 1; macro_reasons.append("원화 강세")
            elif krw_f >= 1400:
                macro_bonus -= 1; macro_reasons.append("원화 약세")

        macro_bonus = max(-5, min(5, macro_bonus))
        st.session_state["macro_bonus"] = macro_bonus
        st.session_state["macro_reasons"] = macro_reasons

        # ── 사이드바: 한국 경제지표 ──
        kr_rows = ""
        if bok_rate:
            kr_rows += f'<tr><td>{bok_icon} 기준금리</td><td style="text-align:right;color:{bok_color};font-weight:bold">{bok_rate}%</td><td style="text-align:right;color:{bok_color};font-size:0.8em">{bok_label}</td></tr>'
        if krw_val:
            kr_rows += f'<tr><td>{krw_icon} 원/달러</td><td style="text-align:right;color:{krw_color};font-weight:bold">{krw_f:,.0f}원</td><td style="text-align:right;color:{krw_color};font-size:0.8em">{krw_label}</td></tr>'
        if short_val:
            kr_rows += f'<tr><td>{short_icon} 공매도</td><td style="text-align:right;color:{short_color};font-weight:bold">{short_val}%</td><td style="text-align:right;color:{short_color};font-size:0.8em">{short_label}</td></tr>'
        if kr_rows:
            st.sidebar.markdown(
                f'<div style="background:#1a1a2e;padding:10px;border-radius:8px;border:1px solid #444;margin-top:10px">'
                f'🇰🇷 <b>한국 경제 지표</b><br>'
                f'<table style="width:100%;font-size:0.85em;margin-top:5px">{kr_rows}</table>'
                f'</div>', unsafe_allow_html=True)

        # ── 사이드바: 글로벌 리스크 ──
        gl_rows = ""
        if vix_val:
            gl_rows += f'<tr><td>{vix_icon} VIX</td><td style="text-align:right;color:{vix_color};font-weight:bold">{vix_val}</td><td style="text-align:right;color:{vix_color};font-size:0.8em">{vix_label}</td></tr>'
        if bond_val:
            gl_rows += f'<tr><td>{bond_icon} 미국 10Y</td><td style="text-align:right;color:{bond_color};font-weight:bold">{bond_val}%</td><td style="text-align:right;color:{bond_color};font-size:0.8em">{bond_label}</td></tr>'
        if gl_rows:
            st.sidebar.markdown(
                f'<div style="background:#1a1a2e;padding:10px;border-radius:8px;border:1px solid #444;margin-top:10px">'
                f'🌐 <b>글로벌 리스크 지표</b><br>'
                f'<table style="width:100%;font-size:0.85em;margin-top:5px">{gl_rows}</table>'
                f'</div>', unsafe_allow_html=True)

        # ── 매크로 점수 종합 ──
        if macro_bonus > 0:
            m_color, m_icon, m_label = "#4caf50", "🟢", "우호적"
        elif macro_bonus < 0:
            m_color, m_icon, m_label = "#f44336", "🔴", "비우호적"
        else:
            m_color, m_icon, m_label = "#ff9800", "🟡", "중립"
        macro_detail = ", ".join(macro_reasons) if macro_reasons else "특이사항 없음"
        st.sidebar.markdown(
            f'<div style="background:#1a1a2e;padding:10px;border-radius:8px;border:1px solid {m_color};margin-top:10px;text-align:center">'
            f'📊 <b>매크로 점수</b><br>'
            f'<span style="font-size:1.5em;color:{m_color};font-weight:bold">{macro_bonus:+d}점</span><br>'
            f'{m_icon} {m_label}<br>'
            f'<span style="font-size:0.8em;color:#888">{macro_detail}</span>'
            f'</div>', unsafe_allow_html=True)

        # ── Gemini AI 경제 해석 (캐싱) ──
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
                    _prompt = (
                        "초보 투자자를 위한 경제 해설가로서 아래 지표를 3줄로 설명하세요.\n"
                        f"지표: {', '.join(_nums)}\n"
                        "1줄: 경제 상황 초등학생 수준 요약 (이모지)\n"
                        "2줄: 주식 투자 영향 (이모지)\n"
                        "3줄: 지금 사도 될까? 의견 (이모지)\n"
                        "한국어, 각 줄 줄바꿈"
                    )
                    _resp = gemini_model.generate_content(_prompt)
                    _txt = _resp.text.strip().replace("\n", "<br>")
                    st.session_state["eco_ai_cache"] = _txt
                    st.sidebar.markdown(
                        f'<div style="background:#1a2e1a;padding:10px;border-radius:8px;border:1px solid #4caf50;margin-top:10px">'
                        f'🤖 <b>AI 경제 해석</b><br><br>'
                        f'<span style="font-size:0.85em">{_txt}</span>'
                        f'</div>', unsafe_allow_html=True)
            except:
                pass
        elif "eco_ai_cache" in st.session_state:
            st.sidebar.markdown(
                f'<div style="background:#1a2e1a;padding:10px;border-radius:8px;border:1px solid #4caf50;margin-top:10px">'
                f'🤖 <b>AI 경제 해석</b><br><br>'
                f'<span style="font-size:0.85em">{st.session_state["eco_ai_cache"]}</span>'
                f'</div>', unsafe_allow_html=True)

    except Exception:
        pass


    # 내 종목 감시 컨트롤 (사이드바 고정)
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
    wl = load_wl()
    st.markdown(f"**관심종목: {len(wl)}개**")
    if wl:
        for w in wl:
            st.caption(f"• {w['name']} ({w['code']})")

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

# ─── 메인 ────────────────────────────────────────────

# ======================== 스캔/검색 ========================
if menu == "📡 스캔/검색":
    stocks = get_stocks(market)

    # 이전 스캔 결과가 있으면 맨 위에 표시
    if st.session_state["scan_done"] and not st.session_state.get("scan_running", False):
        all_results = st.session_state["all_scan_results"]
        total_found = sum(len(v) for v in all_results.values())
        st.session_state["scan_results_cache"] = all_results
        st.success(f"✅ 자동검색 완료: {total_found}개 종목 발견 (결과 유지됨)")

        # ── AI 마켓 브리핑 ──
        if GEMINI_OK:
            if st.button("📰 AI 마켓 브리핑 생성", key="market_briefing_btn"):
                with st.spinner("🧠 AI가 오늘의 시장을 분석 중..."):
                    # 상위 종목 정리
                    _top_stocks = []
                    for _skey, _rlist in all_results.items():
                        for _r in sorted(_rlist, key=lambda x: x["score"], reverse=True)[:5]:
                            _top_stocks.append(f"{_r['name']}({_r['code']}) {_r['score']}점 {_r['verdict']} 등락{_r['change']:+.2f}%")

                    # 경제지표 정리
                    _eco_info = []
                    _m_bonus = st.session_state.get("macro_bonus", 0)
                    _m_reasons = st.session_state.get("macro_reasons", [])
                    _eco_info.append(f"매크로 점수: {_m_bonus:+d}점 ({', '.join(_m_reasons) if _m_reasons else '중립'})")

                    # 크라운 종목
                    _crowns = []
                    for _skey, _rlist in all_results.items():
                        for _r in _rlist:
                            if _r.get("crown") and _r["crown"] not in ["", " "]:
                                _crowns.append(f"{_r['crown']} {_r['name']} {_r['score']}점")

                    _briefing_prompt = (
                        "당신은 한국 주식시장 전문 애널리스트입니다. 초보 투자자를 위해 오늘의 마켓 브리핑을 작성하세요.\n\n"
                        f"📊 스캔 결과 상위 종목:\n" + "\n".join(_top_stocks[:10]) + "\n\n"
                        f"🏆 크라운 종목 (다중 스타일 적합):\n" + ("\n".join(_crowns[:5]) if _crowns else "없음") + "\n\n"
                        f"📈 경제 환경: {', '.join(_eco_info)}\n\n"
                        "아래 형식으로 작성하세요:\n"
                        "1️⃣ 오늘의 시장 한줄 요약 (이모지 포함)\n"
                        "2️⃣ 주목할 종목 TOP 3와 이유 (각 1줄씩, 쉬운 말로)\n"
                        "3️⃣ 오늘의 투자 전략 (초보자용, 2~3줄)\n"
                        "4️⃣ 주의할 점 (1~2줄)\n"
                        "한국어, 이모지 적극 활용, 전문용어 사용 시 괄호로 쉬운 설명 추가"
                    )
                    try:
                        _br_resp = gemini_model.generate_content(_briefing_prompt)
                        _br_txt = _br_resp.text.strip().replace("\n", "<br>")
                        st.markdown(
                            f'<div style="background:linear-gradient(135deg,#1a2e1a,#1a1a2e);padding:20px;border-radius:12px;'
                            f'border:2px solid #4caf50;margin:15px 0">'
                            f'📰 <b style="font-size:1.2em">AI 마켓 브리핑</b><br><br>'
                            f'<span style="font-size:0.95em;line-height:1.8">{_br_txt}</span>'
                            f'</div>', unsafe_allow_html=True)
                    except Exception as _br_err:
                        st.warning(f"마켓 브리핑 생성 실패: {str(_br_err)[:100]}")


        # 크라운
        crown_3 = []
        crown_2 = []
        seen_3 = set()
        seen_2 = set()
        for skey, rlist in all_results.items():
            for r in rlist:
                if r.get("crown") == "🏆 3관왕" and r["code"] not in seen_3 and r["score"] >= MIN_SCORE:
                    crown_3.append(r)
                    seen_3.add(r["code"])
                elif r.get("crown") == "⭐ 2관왕" and r["code"] not in seen_2 and r["code"] not in seen_3 and r["score"] >= MIN_SCORE:
                    crown_2.append(r)
                    seen_2.add(r["code"])
        crown_3.sort(key=lambda x: x["score"], reverse=True)
        crown_2.sort(key=lambda x: x["score"], reverse=True)

        if crown_3:
            st.subheader("🏆 3관왕 (단타+스윙+중장기 모두 적합)")
            for r in crown_3:
                st.markdown(f"🏆 **{r['name']}** ({r['code']}) — {r['score']}점 {r['grade']} | 진입 {r['support']:,}원 → 목표 {r['tp_price']:,}원")
        if crown_2:
            st.subheader("⭐ 2관왕 (2개 스타일 적합)")
            for r in crown_2:
                st.markdown(f"⭐ **{r['name']}** ({r['code']}) — {r['score']}점 {r['grade']} | 진입 {r['support']:,}원 → 목표 {r['tp_price']:,}원")

        # 섹터 동반 상승 (간략 표시)
        surges = detect_sector_surge(all_results)
        if surges:
            st.markdown("---")
            st.subheader("🔥 섹터 동반 상승 감지")
            st.caption("자세한 내용은 사이드바 '🔥 섹터 동반 상승' 메뉴에서 확인하세요")
            for sg in surges:
                if sg["count"] >= 3:
                    icon = "🚀"
                    color = "#ff1744"
                    label = "강력"
                else:
                    icon = "⚡"
                    color = "#ff9800"
                    label = "주의"
                st.markdown(
                    f'<div style="background:{color};color:#fff;padding:12px 16px;border-radius:10px;margin:6px 0;font-size:1.1em">'
                    f'{icon} <b>[{label}] {sg["theme"]}</b> — {sg["count"]}종목 동시 신호 | '
                    f'평균 점수 {sg["avg_score"]}점 | 평균 등락률 {sg["avg_change"]:+.2f}%'
                    f'</div>', unsafe_allow_html=True
                )

        st.divider()

        tabs = st.tabs(list(STYLES.keys()))
        for i, (style_name, cfg) in enumerate(STYLES.items()):
            skey = cfg["key"]
            rlist = all_results.get(skey, [])
            with tabs[i]:
                st.subheader(f"{style_name} — {len(rlist)}개")
                rlist_sorted = sorted(rlist, key=lambda x: x["score"], reverse=True)
                for r in rlist_sorted:
                    # ── 커스텀 알림 체크 ──
                    _a_per = st.session_state.get("alert_per", 0)
                    _a_pbr = st.session_state.get("alert_pbr", 0)
                    _a_score = st.session_state.get("alert_score", 0)
                    _a_rsi = st.session_state.get("alert_rsi_low", 0)
                    _a_vol = st.session_state.get("alert_vol", 0)
                    _a_pattern = st.session_state.get("alert_pattern", [])
                    _any_alert = (_a_per > 0 or _a_pbr > 0 or _a_score > 0 or _a_rsi > 0 or _a_vol > 0 or len(_a_pattern) > 0)

                    if _any_alert:
                        _alert_match = True
                        if _a_per > 0:
                            try:
                                if float(r.get("per", 999)) > _a_per:
                                    _alert_match = False
                            except:
                                _alert_match = False
                        if _a_pbr > 0:
                            try:
                                if float(r.get("pbr", 999)) > _a_pbr:
                                    _alert_match = False
                            except:
                                _alert_match = False
                        if _a_score > 0 and r["score"] < _a_score:
                            _alert_match = False
                        if _a_rsi > 0:
                            try:
                                if float(r.get("rsi", 50)) > _a_rsi:
                                    _alert_match = False
                            except:
                                pass
                        if _a_vol > 0:
                            try:
                                if float(r.get("vol_ratio", 0)) < _a_vol:
                                    _alert_match = False
                            except:
                                pass
                        if _a_pattern:
                            _reasons_all = " ".join(r.get("buy_reasons", []))
                            _has_pattern = any(_pat in _reasons_all for _pat in _a_pattern)
                            if not _has_pattern:
                                _alert_match = False
                        if not _alert_match:
                            continue
                        _alert_icon = "🔔 "
                    else:
                        _alert_icon = ""
                    with st.expander(f"{_alert_icon}{r.get('crown', '')} {r['name']} ({r['code']}) — {r['score']}점 {r['grade']} | {r['verdict']}", expanded=False):
                        show_card(r, f"scan_{skey}_{r['code']}", cfg)


        st.divider()

    # 개별 검색
    st.subheader("🔍 종목 검색")
    query = st.text_input("종목명 또는 코드 입력")
    if query:
        query = query.strip()
        # 티커 정확 매칭 우선
        exact_match = stocks[stocks["Code"].str.upper() == query.upper()]
        if not exact_match.empty:
            matched = exact_match
        else:
            matched = stocks[
                stocks["Name"].str.contains(query, case=False, na=False) |
                stocks["Code"].str.upper().str.contains(query.upper(), na=False)
            ]
            matched = matched.sort_values(
                by="Code",
                key=lambda x: x.str.upper() != query.upper()
            )
        if matched.empty:
            st.warning("검색 결과가 없습니다.")
        else:
            style_name = st.selectbox("분석 스타일", list(STYLES.keys()), key="search_style")
            cfg = STYLES[style_name]
            for _, row in matched.head(10).iterrows():
                code = str(row["Code"]).strip()
                name = str(row["Name"]).strip()
                with st.expander(f"📋 {name} ({code})", expanded=True):
                    df = fetch(code)
                    if df is not None:
                        r = analyze(df, code, name, cfg, market=market)
                        r["crown"] = ""
                        show_card(r, f"search_{code}", cfg)
                    else:
                        st.error(f"{name} 데이터를 불러올 수 없습니다.")

    st.divider()

    # ── 종목 비교 ──
    st.subheader("⚖️ 종목 비교")
    with st.expander("종목 2~3개를 비교해보세요", expanded=False):
        cmp_cols = st.columns(3)
        cmp_inputs = []
        for ci in range(3):
            with cmp_cols[ci]:
                label = ["첫 번째", "두 번째", "세 번째"][ci]
                val = st.text_input(f"{label} 종목", key=f"cmp_input_{ci}", placeholder="종목명 또는 코드")
                cmp_inputs.append(val.strip())

        if st.button("⚖️ 비교 분석", key="cmp_btn", use_container_width=True):
            cmp_results = []
            for q in cmp_inputs:
                if not q:
                    continue
                match = stocks[
                    stocks["Name"].str.contains(q, case=False, na=False) |
                    stocks["Code"].str.upper().str.contains(q.upper(), na=False)
                ]
                if match.empty:
                    st.warning(f"'{q}' 종목을 찾을 수 없습니다.")
                    continue
                row = match.iloc[0]
                code = str(row["Code"]).strip()
                name = str(row["Name"]).strip()
                df = fetch(code)
                if df is None:
                    st.warning(f"{name} 데이터를 불러올 수 없습니다.")
                    continue

                cfg = list(STYLES.values())[0]
                r = analyze(df, code, name, cfg, market=market)

                # EV/EBITDA 추가
                try:
                    is_us = market in ["NASDAQ", "NYSE", "S&P500", "AMEX"]
                    ev_data = get_ev_ebitda(code, is_us=is_us)
                    ev_val = ev_data.get("ev_ebitda")
                except:
                    ev_val = None

                cmp_results.append({
                    "name": name,
                    "code": code,
                    "price": r.get("price", 0),
                    "change": r.get("change", 0),
                    "score": r.get("score", 0),
                    "grade": r.get("grade", ""),
                    "verdict": r.get("verdict", ""),
                    "per": r.get("per", "-"),
                    "pbr": r.get("pbr", "-"),
                    "div_yield": r.get("div_yield", "-"),
                    "rsi": r.get("rsi", "-"),
                    "ev_ebitda": ev_val if ev_val else "-",
                    "vol_ratio": r.get("vol_ratio", "-"),
                    "support": r.get("support", 0),
                    "resist": r.get("tp_price", 0),
                    "buy_reasons": r.get("buy_reasons", []),
                    "sell_reasons": r.get("sell_reasons", []),
                })

            if len(cmp_results) >= 2:
                # 비교 테이블
                is_us = market in ["NASDAQ", "NYSE", "S&P500", "AMEX"]
                currency = "$" if is_us else ""
                unit = "" if is_us else "원"

                header = "<tr><th style='text-align:left;padding:8px;border-bottom:2px solid #555'>항목</th>"
                for cr in cmp_results:
                    header += f"<th style='text-align:center;padding:8px;border-bottom:2px solid #555'>{cr['name']}<br><span style='font-size:0.8em;color:#888'>{cr['code']}</span></th>"
                header += "</tr>"

                def make_row(label, key, fmt="", color_fn=None):
                    row_html = f"<tr><td style='padding:6px 8px;border-bottom:1px solid #333'>{label}</td>"
                    vals = [cr[key] for cr in cmp_results]
                    for i, cr in enumerate(cmp_results):
                        v = cr[key]
                        color = ""
                        if color_fn and v != "-":
                            try:
                                color = f"color:{color_fn(v, vals)};"
                            except:
                                pass
                        if isinstance(v, float):
                            txt = f"{v:{fmt}}" if fmt else f"{v}"
                        elif isinstance(v, int):
                            txt = f"{v:,}"
                        else:
                            txt = str(v)
                        row_html += f"<td style='text-align:center;padding:6px 8px;border-bottom:1px solid #333;font-weight:bold;{color}'>{txt}</td>"
                    row_html += "</tr>"
                    return row_html

                def best_score(v, vals):
                    nums = [x for x in vals if isinstance(x, (int, float))]
                    if nums and v == max(nums):
                        return "#4caf50"
                    return ""

                def low_is_good(v, vals):
                    nums = [x for x in vals if isinstance(x, (int, float)) and x > 0]
                    if nums and v == min(nums):
                        return "#4caf50"
                    return ""

                def high_is_good(v, vals):
                    nums = [x for x in vals if isinstance(x, (int, float)) and x > 0]
                    if nums and v == max(nums):
                        return "#4caf50"
                    return ""

                def change_color(v, vals):
                    if isinstance(v, (int, float)):
                        return "#ff1744" if v > 0 else "#2979ff" if v < 0 else ""
                    return ""

                table_html = f"""
                <div style="background:#1a1a2e;padding:15px;border-radius:12px;margin:10px 0;overflow-x:auto">
                <table style="width:100%;font-size:0.9em;border-collapse:collapse">
                {header}
                {make_row("💰 현재가", "price")}
                {make_row("📊 등락률", "change", "+.2f", change_color)}
                {make_row("🎯 점수", "score", "", best_score)}
                {make_row("🏅 등급", "grade")}
                {make_row("📋 판정", "verdict")}
                {make_row("📈 PER", "per", "", low_is_good)}
                {make_row("📉 PBR", "pbr", "", low_is_good)}
                {make_row("💵 배당률", "div_yield")}
                {make_row("🏢 EV/EBITDA", "ev_ebitda", "", low_is_good)}
                {make_row("📊 RSI", "rsi")}
                {make_row("📈 거래량비", "vol_ratio")}
                {make_row("🛡️ 지지선", "support")}
                {make_row("🎯 저항선", "resist")}
                </table>
                </div>
                """
                st.markdown(table_html, unsafe_allow_html=True)

                # 매수/매도 이유 비교
                for cr in cmp_results:
                    st.markdown(f"**{cr['name']}** ({cr['code']})")
                    if cr["buy_reasons"]:
                        st.success("매수: " + " | ".join(cr["buy_reasons"][:5]))
                    if cr["sell_reasons"]:
                        st.error("매도: " + " | ".join(cr["sell_reasons"][:5]))

                # Gemini AI 비교 분석
                if GEMINI_OK and len(cmp_results) >= 2:
                    if st.button("🧠 AI 비교 분석", key="cmp_ai_btn"):
                        with st.spinner("AI가 비교 분석 중..."):
                            cmp_info = ""
                            for cr in cmp_results:
                                cmp_info += (
                                    f"\n{cr['name']}({cr['code']}): "
                                    f"점수 {cr['score']}점, PER {cr['per']}, PBR {cr['pbr']}, "
                                    f"배당률 {cr['div_yield']}, EV/EBITDA {cr['ev_ebitda']}, "
                                    f"RSI {cr['rsi']}, 등락률 {cr['change']:+.2f}%"
                                    f"\n매수이유: {', '.join(cr['buy_reasons'][:3])}"
                                    f"\n매도이유: {', '.join(cr['sell_reasons'][:3])}"
                                )
                            ai_prompt = (
                                "주식 초보자를 위해 아래 종목들을 비교 분석하세요.\n"
                                f"{cmp_info}\n\n"
                                "1) 어떤 종목이 더 좋은지 결론 (이모지)\n"
                                "2) 이유 2~3줄 (쉬운 말로)\n"
                                "3) 주의할 점 1줄\n"
                                "한국어로 답하세요."
                            )
                            try:
                                ai_resp = gemini_model.generate_content(ai_prompt)
                                st.markdown(
                                    f'<div style="background:#1a2e1a;padding:14px;border-radius:10px;border-left:4px solid #4caf50;margin:10px 0">'
                                    f'🧠 <b>AI 비교 분석</b><br><br>'
                                    f'{ai_resp.text.strip()}'
                                    f'</div>', unsafe_allow_html=True)
                            except:
                                st.warning("AI 분석을 가져올 수 없습니다.")

            elif len(cmp_results) == 1:
                st.info("비교하려면 최소 2개 종목을 입력하세요.")

    st.divider()


    # 전체 스캔 버튼
    if st.button("🚀 전체 스캔 시작 (3개 스타일 동시)", type="primary", use_container_width=True):
        if stocks.empty:
            st.error("종목 리스트를 불러올 수 없습니다.")
        else:
            st.session_state["scan_running"] = True
            all_results, cached_data = run_full_scan(stocks)
            st.session_state["all_scan_results"] = all_results
            st.session_state["cached_data"] = cached_data
            st.session_state["scan_done"] = True
            st.session_state["scan_running"] = False
            save_perf_snapshot(all_results)

            # 히스토리 저장
            for skey, rlist in all_results.items():
                records = [{"code": r["code"], "name": r["name"], "score": r["score"], "verdict": r["verdict"], "crown": r.get("crown", "")} for r in rlist]
                if records:
                    add_to_history_direct(skey, records)
            # 섹터 동반 상승 저장
            surges = detect_sector_surge(all_results)
            if surges:
                SECTOR_FILE = "sector_history.json"
                if os.path.exists(SECTOR_FILE):
                    with open(SECTOR_FILE, "r", encoding="utf-8") as f:
                        sector_hist = json.load(f)
                else:
                    sector_hist = []
                sector_record = {
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "surges": surges,
                }
                sector_hist.append(sector_record)
                with open(SECTOR_FILE, "w", encoding="utf-8") as f:
                    json.dump(sector_hist, f, ensure_ascii=False, indent=2)



# ======================== 내 종목 감시 ========================
elif menu == "👀 내 종목 감시":
    st.header("👀 내 종목 감시")
    wl = load_wl()

    if not wl:
        st.info("감시 종목이 없습니다. 스캔/검색에서 ⭐ 버튼으로 추가하세요.")
    else:
        style_name = st.session_state.get("wl_style_side", list(STYLES.keys())[0])
        cfg = STYLES[style_name]

        # 사이드바 버튼 처리
        if st.session_state.get("wl_refresh_side"):
            st.rerun()
        if st.session_state.get("wl_clear_side"):
            save_wl([])
            st.rerun()
        if st.session_state.get("wl_del_side"):
            selected = st.session_state.get("selected_wl", [])
            if selected:
                new_wl = [w for w in wl if w["code"] not in selected]
                save_wl(new_wl)
                st.session_state["selected_wl"] = []
                st.rerun()

        if AUTOREFRESH_OK:
            auto_on = st.checkbox("⏱️ 자동 갱신 (60초)", value=False, key="wl_auto")
            if auto_on:
                st_autorefresh(interval=60000, key="wl_refresh")

        if "selected_wl" not in st.session_state:
            st.session_state["selected_wl"] = []

        st.caption(f"총 {len(wl)}개 감시 중 | 스타일: {style_name}")

        # 종목 분석
        summary_data = []
        for item in wl:
            code = str(item["code"]).strip()
            name = item["name"]
            df = fetch(code)
            if df is not None:
                r = analyze(df, code, name, cfg)
                r["crown"] = ""
                summary_data.append(r)
            else:
                summary_data.append(None)

        for idx, item in enumerate(wl):
            code = str(item["code"]).strip()
            name = item["name"]
            r = summary_data[idx]

            if r is None:
                st.error(f"{name} 데이터 불러오기 실패")
                continue

            score = r["score"]
            sc = "#4caf50" if score >= 80 else "#ff9800" if score >= 60 else "#f44336"
            chg_c = "#ff1744" if r["change"] >= 0 else "#2979ff"
            crown = r.get("crown", "")

            badges = ""
            if r.get("divergence"):
                div_c = "#4caf50" if "상승" in r["divergence"] else "#f44336"
                badges += f' <span style="background:{div_c};color:#fff;padding:1px 6px;border-radius:8px;font-size:0.75em">{r["divergence"]}</span>'
            if r.get("ma_align"):
                ma_c = "#4caf50" if "정배열" in r["ma_align"] else "#f44336" if "역배열" in r["ma_align"] else "#ff9800"
                badges += f' <span style="background:{ma_c};color:#fff;padding:1px 6px;border-radius:8px;font-size:0.75em">{r["ma_align"]}</span>'
            if r.get("vol_spike"):
                badges += f' <span style="background:#ff6f00;color:#fff;padding:1px 6px;border-radius:8px;font-size:0.75em">{r["vol_spike"]}</span>'

            col_chk, col_summary = st.columns([0.03, 0.97])
            with col_chk:
                checked = st.checkbox("", key=f"chk_wl_{code}", value=(code in st.session_state["selected_wl"]))
                if checked and code not in st.session_state["selected_wl"]:
                    st.session_state["selected_wl"].append(code)
                elif not checked and code in st.session_state["selected_wl"]:
                    st.session_state["selected_wl"].remove(code)
            with col_summary:
                verdict_emoji = {"적극 매수": "🔥", "매수 관심": "👀", "적극 매도": "🧊", "매도 관심": "⚠️", "중립 관망": "😐"}.get(r["verdict"], "")
                st.markdown(
                    f'<div style="background:#1a1a2e;padding:10px 14px;border-radius:10px;margin:4px 0;display:flex;justify-content:space-between;align-items:center">'
                    f'<div>'
                    f'<b>{crown} {name}</b> ({code}) '
                    f'<span style="color:{chg_c};font-weight:bold">{r["price"]:,}원 ({r["change"]:+.2f}%)</span>'
                    f'{badges}'
                    f'</div>'
                    f'<div style="text-align:right">'
                    f'<span style="font-size:1.4em;font-weight:bold;color:{sc}">{score}점</span> '
                    f'<span style="color:{sc}">{r["grade"]} {verdict_emoji}</span>'
                    f'</div>'
                    f'</div>', unsafe_allow_html=True
                )

                with st.expander("상세 분석 보기", expanded=False):
                    prev = st.session_state["prev_signals"].get(code, {"buy": [], "sell": []})
                    new_buy = [b for b in r["buy_reasons"] if b not in prev.get("buy", [])]
                    new_sell = [s for s in r["sell_reasons"] if s not in prev.get("sell", [])]
                    st.session_state["prev_signals"][code] = {"buy": r["buy_reasons"], "sell": r["sell_reasons"]}
                    is_new_buy = len(new_buy) > 0
                    is_new_sell = len(new_sell) > 0
                    if sound_on and (is_new_buy or is_new_sell):
                        st.session_state["sound_counter"] += 1
                        if is_new_buy:
                            st.markdown(f'<audio autoplay><source src="{SOUND_BUY}"></audio>', unsafe_allow_html=True)
                        if is_new_sell:
                            st.markdown(f'<audio autoplay><source src="{SOUND_SELL}"></audio>', unsafe_allow_html=True)
                    show_card(r, f"wl_{code}", cfg, is_new_buy=is_new_buy, is_new_sell=is_new_sell)


# ======================== 성과 리포트 ========================
elif menu == "📊 성과 리포트":
    st.header("📊 전일 추천 종목 성과 리포트")
    
    perf_data = load_perf_snapshot()
    if not perf_data:
        st.info("아직 스캔 기록이 없습니다. 전체 스캔을 먼저 실행하세요.")
    else:
        if st.button("📊 성과 분석 실행", type="primary", use_container_width=True):
            with st.spinner("전일 추천 종목 현재가 조회 중..."):
                report = generate_perf_report()
            if report:
                # 요약 카드
                win_color = "#4caf50" if report["win_rate"] >= 50 else "#f44336"
                pnl_color = "#4caf50" if report["avg_pnl"] >= 0 else "#f44336"
                st.markdown(
                    f'<div style="background:#1a1a2e;padding:20px;border-radius:12px;border:2px solid {win_color};margin:10px 0">'
                    f'<div style="font-size:1.3em;font-weight:bold;margin-bottom:10px">📅 스캔 시점: {report["date"]}</div>'
                    f'<div style="display:flex;justify-content:space-around;text-align:center">'
                    f'<div><span style="font-size:2em;font-weight:bold;color:{win_color}">{report["win_rate"]}%</span><br>승률</div>'
                    f'<div><span style="font-size:2em;font-weight:bold;color:{pnl_color}">{report["avg_pnl"]:+.2f}%</span><br>평균 수익률</div>'
                    f'<div><span style="font-size:1.5em;color:#4caf50">{report["wins"]}</span> / <span style="font-size:1.5em;color:#f44336">{report["losses"]}</span><br>승 / 패</div>'
                    f'<div><span style="font-size:1.2em">{report["total"]}개</span><br>추천 종목</div>'
                    f'</div>'
                    f'</div>', unsafe_allow_html=True
                )

                # 베스트 / 워스트
                col_best, col_worst = st.columns(2)
                with col_best:
                    b = report["best"]
                    st.markdown(
                        f'<div style="background:#1a2e1a;padding:14px;border-radius:10px;border:1px solid #4caf50">'
                        f'🏆 <b>최고 수익</b><br>'
                        f'<b>{b["name"]}</b> ({b["code"]})<br>'
                        f'{b["entry_price"]:,}원 → {b["current_price"]:,}원<br>'
                        f'<span style="font-size:1.5em;color:#4caf50">{b["pnl"]:+.2f}%</span>'
                        f'</div>', unsafe_allow_html=True
                    )
                with col_worst:
                    w = report["worst"]
                    w_color = "#4caf50" if w["pnl"] >= 0 else "#f44336"
                    st.markdown(
                        f'<div style="background:#2e1a1a;padding:14px;border-radius:10px;border:1px solid #f44336">'
                        f'💀 <b>최저 수익</b><br>'
                        f'<b>{w["name"]}</b> ({w["code"]})<br>'
                        f'{w["entry_price"]:,}원 → {w["current_price"]:,}원<br>'
                        f'<span style="font-size:1.5em;color:{w_color}">{w["pnl"]:+.2f}%</span>'
                        f'</div>', unsafe_allow_html=True
                    )

                # 개별 종목 리스트
                st.divider()
                st.subheader("📋 종목별 성과")
                for r in report["results"]:
                    pnl_c = "#4caf50" if r["pnl"] >= 0 else "#f44336"
                    st.markdown(
                        f'<div style="background:#1a1a2e;padding:10px 14px;border-radius:10px;margin:4px 0;display:flex;justify-content:space-between;align-items:center">'
                        f'<div>'
                        f'<b>{r["name"]}</b> ({r["code"]}) — {r["score"]}점 [{r["grade"]}]<br>'
                        f'추천가: {r["entry_price"]:,}원 → 현재: {r["current_price"]:,}원'
                        f'</div>'
                        f'<div style="text-align:right">'
                        f'<span style="font-size:1.4em;font-weight:bold;color:{pnl_c}">{r["pnl"]:+.2f}%</span><br>'
                        f'{r["status"]}'
                        f'</div>'
                        f'</div>', unsafe_allow_html=True
                    )


                # ── 지표별 승률 분석 ──
                st.divider()
                st.subheader("🔬 지표별 승률 분석")
                with st.spinner("지표별 성과 계산 중..."):
                    ind_report = generate_indicator_report()
                if ind_report:
                    for ind in ind_report:
                        wr = ind["win_rate"]
                        wr_color = "#4caf50" if wr >= 60 else "#ff9800" if wr >= 40 else "#f44336"
                        pnl_color = "#4caf50" if ind["avg_pnl"] >= 0 else "#f44336"
                        bar_width = int(wr)
                        st.markdown(
                            f'<div style="background:#1a1a2e;padding:10px 14px;border-radius:10px;margin:4px 0">'
                            f'<div style="display:flex;justify-content:space-between;align-items:center">'
                            f'<div><b>{ind["indicator"]}</b> — {ind["total"]}건</div>'
                            f'<div style="text-align:right">'
                            f'<span style="font-size:1.2em;font-weight:bold;color:{wr_color}">{wr}%</span> 승률 | '
                            f'<span style="color:{pnl_color}">{ind["avg_pnl"]:+.2f}%</span> 평균수익'
                            f'</div></div>'
                            f'<div style="background:#333;border-radius:4px;height:8px;margin-top:6px">'
                            f'<div style="background:{wr_color};width:{bar_width}%;height:8px;border-radius:4px"></div>'
                            f'</div></div>', unsafe_allow_html=True
                        )
                else:
                    st.info("아직 데이터가 부족합니다. 스캔을 2회 이상 실행하면 지표별 승률이 표시됩니다.")

            else:
                st.warning("성과 데이터를 생성할 수 없습니다. 전일 스캔 후 하루 이상 지나야 합니다.")

        # ── 백테스트 ──
        st.divider()
        st.subheader("🔬 백테스트 (과거 데이터 검증)")

        col_bt1, col_bt2, col_bt3 = st.columns(3)
        with col_bt1:
            bt_days = st.selectbox("분석 기간", [60, 120, 180, 250], index=1, key="bt_days")
        with col_bt2:
            bt_hold = st.selectbox("보유 기간(일)", [3, 5, 7, 10], index=1, key="bt_hold")
        with col_bt3:
            bt_stocks = st.selectbox("종목 수", [50, 100, 200, 300], index=1, key="bt_stocks")

        col_run, col_load = st.columns(2)
        with col_run:
            if st.button("🚀 백테스트 실행", key="run_bt"):
                prog = st.progress(0, text="백테스트 진행 중...")
                def update_prog(pct):
                    prog.progress(min(pct, 1.0), text=f"백테스트 진행 중... {int(pct*100)}%")
                bt_results = run_backtest(
                    days_back=bt_days,
                    hold_days=bt_hold,
                    max_stocks=bt_stocks,
                    _progress_callback=update_prog,
                    cfg=list(STYLES.values())[0]
                )
                prog.progress(1.0, text="완료!")
                st.session_state["bt_results"] = bt_results
        with col_load:
            if st.button("📂 저장된 결과 불러오기", key="load_bt"):
                bt_results = load_backtest()
                if bt_results:
                    st.session_state["bt_results"] = bt_results
                    st.success(f"{len(bt_results)}건 불러옴")
                else:
                    st.warning("저장된 백테스트 결과가 없습니다.")

        if "bt_results" in st.session_state and st.session_state["bt_results"]:
            bt = st.session_state["bt_results"]
            bt_wins = len([r for r in bt if r["win"]])
            bt_total = len(bt)
            bt_wr = round(bt_wins / bt_total * 100, 1) if bt_total else 0
            bt_avg = round(sum(r["pnl"] for r in bt) / bt_total, 2) if bt_total else 0
            wr_c = "#4caf50" if bt_wr >= 50 else "#f44336"
            pnl_c = "#4caf50" if bt_avg >= 0 else "#f44336"

            st.markdown(
                f'<div style="background:#1a1a2e;padding:16px;border-radius:12px;border:2px solid {wr_c};margin:10px 0">'
                f'<h3 style="margin:0">📊 백테스트 종합 결과</h3>'
                f'<div style="display:flex;gap:30px;margin-top:10px">'
                f'<div>총 <b>{bt_total}</b>건</div>'
                f'<div>승률 <b style="color:{wr_c};font-size:1.3em">{bt_wr}%</b></div>'
                f'<div>평균수익 <b style="color:{pnl_c};font-size:1.3em">{bt_avg:+.2f}%</b></div>'
                f'<div>{bt_wins}승 / {bt_total - bt_wins}패</div>'
                f'</div></div>', unsafe_allow_html=True
            )

            # 지표별 승률
            ind_report = analyze_backtest(bt)
            if ind_report:
                st.markdown("#### 📈 지표별 승률")
                for ind in ind_report:
                    wr = ind["win_rate"]
                    wr_color = "#4caf50" if wr >= 60 else "#ff9800" if wr >= 40 else "#f44336"
                    pnl_color = "#4caf50" if ind["avg_pnl"] >= 0 else "#f44336"
                    st.markdown(
                        f'<div style="background:#1a1a2e;padding:10px 14px;border-radius:10px;margin:4px 0">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<div><b>{ind["indicator"]}</b> — {ind["total"]}건</div>'
                        f'<div>'
                        f'<span style="font-size:1.2em;font-weight:bold;color:{wr_color}">{wr}%</span> 승률 | '
                        f'<span style="color:{pnl_color}">{ind["avg_pnl"]:+.2f}%</span> 평균수익'
                        f'</div></div>'
                        f'<div style="background:#333;border-radius:4px;height:8px;margin-top:6px">'
                        f'<div style="background:{wr_color};width:{int(wr)}%;height:8px;border-radius:4px"></div>'
                        f'</div></div>', unsafe_allow_html=True
                    )

            # 지표 조합별 승률
            combo_report = analyze_combo(bt)
            if combo_report:
                st.markdown("#### 🏆 지표 조합별 승률 (상위 15개)")
                for i, c in enumerate(combo_report[:15]):
                    wr = c["win_rate"]
                    wr_color = "#4caf50" if wr >= 60 else "#ff9800" if wr >= 40 else "#f44336"
                    pnl_color = "#4caf50" if c["avg_pnl"] >= 0 else "#f44336"
                    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                    st.markdown(
                        f'<div style="background:#1a2e1a;padding:10px 14px;border-radius:10px;margin:4px 0">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<div>{medal} <b>{c["combo"]}</b> — {c["total"]}건</div>'
                        f'<div>'
                        f'<span style="font-size:1.2em;font-weight:bold;color:{wr_color}">{wr}%</span> 승률 | '
                        f'<span style="color:{pnl_color}">{c["avg_pnl"]:+.2f}%</span> 평균수익'
                        f'</div></div>'
                        f'<div style="background:#333;border-radius:4px;height:8px;margin-top:6px">'
                        f'<div style="background:{wr_color};width:{int(wr)}%;height:8px;border-radius:4px"></div>'
                        f'</div></div>', unsafe_allow_html=True
                    )



        # 과거 스냅샷 목록
        st.divider()
        st.subheader("📜 과거 스캔 기록")
        if st.button("🗑️ 전체 기록 삭제", key="clear_all_perf"):
            with open(PERF_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
            st.success("전체 기록 삭제 완료!")
            st.rerun()
        for i, snap in enumerate(perf_data[:10]):
            with st.expander(f"📅 {snap['date']} — {len(snap['stocks'])}종목 추천", expanded=False):
                if st.button(f"🗑️ 이 기록 삭제", key=f"del_perf_{i}"):
                    perf_data.pop(i)
                    with open(PERF_FILE, "w", encoding="utf-8") as f:
                        json.dump(perf_data, f, ensure_ascii=False, indent=2)
                    st.success("삭제 완료!")
                    st.rerun()
                if st.button(f"📊 이 날짜 성과 분석", key=f"perf_hist_{i}"):
                    with st.spinner("성과 계산 중..."):
                        _results = []
                        for s in snap["stocks"]:
                            try:
                                _df = fetch(s["code"], days=10)
                                if _df is None or len(_df) < 2:
                                    continue
                                _cur = int(_df["Close"].iloc[-1])
                                _pnl = round((_cur - s["price"]) / (s["price"] + 1e-9) * 100, 2)
                                _hit_tp = _cur >= s.get("tp_price", 999999)
                                _hit_sl = _cur <= s.get("sl_price", 0)
                                _status = "🎯 목표달성" if _hit_tp else "🛑 손절" if _hit_sl else "📊 보유중"
                                _results.append({
                                    "name": s["name"], "code": s["code"],
                                    "score": s["score"], "grade": s["grade"],
                                    "entry": s["price"], "current": _cur,
                                    "pnl": _pnl, "status": _status,
                                })
                            except:
                                continue
                        if _results:
                            _wins = len([r for r in _results if r["pnl"] > 0])
                            _total = len(_results)
                            _wr = round(_wins / _total * 100, 1)
                            _avg = round(sum(r["pnl"] for r in _results) / _total, 2)
                            wr_color = "#4caf50" if _wr >= 50 else "#f44336"
                            pnl_color = "#4caf50" if _avg >= 0 else "#f44336"
                            st.markdown(
                                f'<div style="background:#1a1a2e;padding:14px;border-radius:10px;border:1px solid {wr_color}">'
                                f'승률: <b style="color:{wr_color}">{_wr}%</b> | '
                                f'평균수익: <b style="color:{pnl_color}">{_avg:+.2f}%</b> | '
                                f'{_wins}승 / {_total - _wins}패</div>', unsafe_allow_html=True
                            )
                            for r in sorted(_results, key=lambda x: x["pnl"], reverse=True):
                                _c = "#4caf50" if r["pnl"] >= 0 else "#f44336"
                                st.markdown(
                                    f'{r["status"]} **{r["name"]}** — {r["entry"]:,}원 → {r["current"]:,}원 '
                                    f'<span style="color:{_c}">{r["pnl"]:+.2f}%</span>', unsafe_allow_html=True
                                )
                        else:
                            st.warning("성과 데이터를 가져올 수 없습니다.")



# ======================== 스캔 기록 ========================
elif menu == "📜 스캔 기록":
    st.header("📜 스캔 기록")
    hist = load_history()

    if AUTOREFRESH_OK:
        auto_on = st.checkbox("⏱️ 자동 갱신 (60초)", value=False, key="hist_auto")
        if auto_on:
            st_autorefresh(interval=60000, key="hist_refresh")

    if not hist:
        st.info("스캔 기록이 없습니다.")
    else:
        if "selected_hist" not in st.session_state:
            st.session_state["selected_hist"] = []

        col_del1, col_del2 = st.columns([1, 1])
        with col_del1:
            if st.button("🗑️ 기록 전체 삭제"):
                clear_history()
                st.session_state["selected_hist"] = []
                st.rerun()
        with col_del2:
            if st.button("🗑️ 선택 삭제", key="del_selected_hist"):
                selected = st.session_state.get("selected_hist", [])
                if selected:
                    full_hist = load_history()
                    new_hist = [h for i, h in enumerate(full_hist) if i not in selected]
                    save_history(new_hist)
                    st.session_state["selected_hist"] = []
                    st.rerun()
                else:
                    st.warning("삭제할 기록을 선택하세요")

        tabs = st.tabs(["단타", "스윙", "중장기"])
        style_keys = ["short", "swing", "long"]
        for ti, skey in enumerate(style_keys):
            with tabs[ti]:
                filtered = [(i, h) for i, h in enumerate(hist) if isinstance(h, dict) and h.get("style") == skey]
                if not filtered:
                    st.info("해당 스타일 기록 없음")
                    continue
                cfg = STYLES[KEY_TO_STYLE[skey]]
                for orig_idx, entry in filtered:
                    col_check, col_label = st.columns([0.05, 0.95])
                    with col_check:
                        checked = st.checkbox("", key=f"chk_hist_{skey}_{orig_idx}", value=(orig_idx in st.session_state["selected_hist"]))
                        if checked and orig_idx not in st.session_state["selected_hist"]:
                            st.session_state["selected_hist"].append(orig_idx)
                        elif not checked and orig_idx in st.session_state["selected_hist"]:
                            st.session_state["selected_hist"].remove(orig_idx)
                    with col_label:
                        with st.expander(f"📅 {entry['date']} — {len(entry['stocks'])}종목", expanded=False):
                            for si, stock_info in enumerate(entry["stocks"]):
                                code = str(stock_info["code"]).strip()
                                name = stock_info["name"]
                                crown_text = stock_info.get("crown", "")
                                st.markdown(f"{crown_text} **{name}** ({code}) — 당시 {stock_info.get('score', '?')}점 {stock_info.get('verdict', '')}")
                                df = fetch(code)
                                if df is not None:
                                    r = analyze(df, code, name, cfg)
                                    r["crown"] = ""
                                    prev_key = f"hist_{skey}_{orig_idx}_{code}"
                                    prev = st.session_state["hist_prev_signals"].get(prev_key, {"buy": [], "sell": []})
                                    new_buy = [b for b in r["buy_reasons"] if b not in prev.get("buy", [])]
                                    new_sell = [s for s in r["sell_reasons"] if s not in prev.get("sell", [])]
                                    st.session_state["hist_prev_signals"][prev_key] = {"buy": r["buy_reasons"], "sell": r["sell_reasons"]}
                                    show_card(r, f"hist_{skey}_{orig_idx}_{code}", cfg, is_new_buy=len(new_buy) > 0, is_new_sell=len(new_sell) > 0, show_wl_btn=True)

elif menu == "🔥 섹터 동반 상승":
    st.header("🔥 섹터 동반 상승 기록")
    # ── 섹터 로테이션 테이블 ──
    st.subheader("📊 섹터 로테이션 (업종별 흐름)")
    with st.expander("📊 업종별 등락률 한눈에 보기", expanded=True):
        try:
            # 업종(테마) 매핑 - 종목코드 앞자리 기반
            _sector_map = {
                "반도체": ["삼성전자", "SK하이닉스", "한미반도체", "리노공업", "이오테크닉스", "주성엔지니어링", "테크윙", "하나마이크론"],
                "2차전지": ["LG에너지솔루션", "삼성SDI", "에코프로", "에코프로비엠", "포스코퓨처엠", "엘앤에프"],
                "바이오": ["삼성바이오로직스", "셀트리온", "유한양행", "알테오젠", "리가켐바이오", "HLB"],
                "자동차": ["현대차", "기아", "현대모비스", "만도", "HL만도"],
                "금융": ["KB금융", "신한지주", "하나금융지주", "우리금융지주", "삼성생명"],
                "IT/소프트웨어": ["카카오", "네이버", "크래프톤", "엔씨소프트", "카카오뱅크", "두나무"],
                "철강/화학": ["POSCO홀딩스", "LG화학", "롯데케미칼", "한화솔루션", "금호석유"],
                "유통/소비재": ["삼성물산", "CJ제일제당", "아모레퍼시픽", "LG생활건강", "오리온"],
                "건설": ["현대건설", "대우건설", "GS건설", "DL이앤씨", "HDC현대산업개발"],
                "엔터/미디어": ["하이브", "JYP Ent.", "SM", "에스엠", "와이지엔터테인먼트", "CJ ENM"],
            }

            # 최근 스캔 결과에서 섹터별 데이터 수집
            _sector_data = {}
            _scan_results = {}

            # all_results가 있으면 사용, 없으면 스캔 기록에서 가져오기
            if "scan_results_cache" in st.session_state:
                _scan_results = st.session_state["scan_results_cache"]
            elif "all_results" in st.session_state:
                _scan_results = st.session_state["all_results"]

            # 스캔 결과가 있으면 섹터별 분류
            if _scan_results:
                _all_stocks = []
                for _skey, _rlist in _scan_results.items():
                    _all_stocks.extend(_rlist)

                for _sector, _names in _sector_map.items():
                    _matched = []
                    for _r in _all_stocks:
                        if any(_n in _r.get("name", "") for _n in _names):
                            if not any(_m["code"] == _r["code"] for _m in _matched):
                                _matched.append(_r)
                    if _matched:
                        _avg_change = sum(_r.get("change", 0) for _r in _matched) / len(_matched)
                        _avg_score = sum(_r.get("score", 0) for _r in _matched) / len(_matched)
                        _sector_data[_sector] = {
                            "count": len(_matched),
                            "avg_change": _avg_change,
                            "avg_score": _avg_score,
                            "stocks": _matched
                        }

            if _sector_data:
                # 등락률 기준 정렬
                _sorted_sectors = sorted(_sector_data.items(), key=lambda x: x[1]["avg_change"], reverse=True)

                # 테이블 생성
                _header = (
                    '<tr>'
                    '<th style="text-align:left;padding:8px;border-bottom:2px solid #555">순위</th>'
                    '<th style="text-align:left;padding:8px;border-bottom:2px solid #555">섹터</th>'
                    '<th style="text-align:center;padding:8px;border-bottom:2px solid #555">종목수</th>'
                    '<th style="text-align:center;padding:8px;border-bottom:2px solid #555">평균 등락률</th>'
                    '<th style="text-align:center;padding:8px;border-bottom:2px solid #555">평균 점수</th>'
                    '<th style="text-align:center;padding:8px;border-bottom:2px solid #555">상태</th>'
                    '</tr>'
                )
                _rows = ""
                for _rank, (_sec_name, _sec_info) in enumerate(_sorted_sectors, 1):
                    _chg = _sec_info["avg_change"]
                    _scr = _sec_info["avg_score"]

                    # 색상/아이콘 결정
                    if _chg >= 3:
                        _color = "#ff1744"
                        _icon = "🔥"
                        _status = "급등"
                    elif _chg >= 1:
                        _color = "#ff6d00"
                        _icon = "📈"
                        _status = "상승"
                    elif _chg >= 0:
                        _color = "#4caf50"
                        _icon = "➡️"
                        _status = "보합"
                    elif _chg >= -2:
                        _color = "#2979ff"
                        _icon = "📉"
                        _status = "하락"
                    else:
                        _color = "#7b1fa2"
                        _icon = "💀"
                        _status = "급락"

                    # 순위 메달
                    if _rank == 1:
                        _medal = "🥇"
                    elif _rank == 2:
                        _medal = "🥈"
                    elif _rank == 3:
                        _medal = "🥉"
                    else:
                        _medal = f"{_rank}"

                    _rows += (
                        f'<tr>'
                        f'<td style="padding:6px 8px;border-bottom:1px solid #333;text-align:center">{_medal}</td>'
                        f'<td style="padding:6px 8px;border-bottom:1px solid #333;font-weight:bold">{_sec_name}</td>'
                        f'<td style="padding:6px 8px;border-bottom:1px solid #333;text-align:center">{_sec_info["count"]}개</td>'
                        f'<td style="padding:6px 8px;border-bottom:1px solid #333;text-align:center;color:{_color};font-weight:bold">{_chg:+.2f}%</td>'
                        f'<td style="padding:6px 8px;border-bottom:1px solid #333;text-align:center">{_scr:.0f}점</td>'
                        f'<td style="padding:6px 8px;border-bottom:1px solid #333;text-align:center">{_icon} {_status}</td>'
                        f'</tr>'
                    )

                st.markdown(
                    f'<div style="background:#1a1a2e;padding:15px;border-radius:12px;overflow-x:auto">'
                    f'<table style="width:100%;font-size:0.9em;border-collapse:collapse">'
                    f'{_header}{_rows}'
                    f'</table></div>',
                    unsafe_allow_html=True
                )

                # 요약 코멘트
                _top = _sorted_sectors[0]
                _bottom = _sorted_sectors[-1]
                st.markdown(
                    f'<div style="background:#1a2e1a;padding:12px;border-radius:8px;border-left:4px solid #4caf50;margin-top:10px">'
                    f'💡 <b>섹터 요약:</b> '
                    f'가장 강한 섹터는 <b style="color:#ff1744">{_top[0]}</b> ({_top[1]["avg_change"]:+.2f}%), '
                    f'가장 약한 섹터는 <b style="color:#2979ff">{_bottom[0]}</b> ({_bottom[1]["avg_change"]:+.2f}%)'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # AI 섹터 분석
                if GEMINI_OK:
                    if st.button("🧠 AI 섹터 로테이션 분석", key="sector_rotation_ai"):
                        with st.spinner("AI가 섹터를 분석 중..."):
                            _sec_info_txt = "\n".join([
                                f"{_s}: 등락률 {_d['avg_change']:+.2f}%, 평균점수 {_d['avg_score']:.0f}점, {_d['count']}종목"
                                for _s, _d in _sorted_sectors
                            ])
                            _sec_prompt = (
                                "주식 초보자를 위해 아래 섹터별 데이터를 분석하세요.\n\n"
                                f"{_sec_info_txt}\n\n"
                                "1) 지금 돈이 몰리는 섹터는? (1줄, 이모지)\n"
                                "2) 피해야 할 섹터는? (1줄, 이모지)\n"
                                "3) 다음 주 주목할 섹터 예측 (1줄, 이모지)\n"
                                "4) 초보자 추천 전략 (1줄, 이모지)\n"
                                "한국어로 답하세요."
                            )
                            try:
                                _sec_resp = gemini_model.generate_content(_sec_prompt)
                                _sec_txt = _sec_resp.text.strip().replace("\n", "<br>")
                                st.markdown(
                                    f'<div style="background:#1a2e1a;padding:14px;border-radius:10px;border:2px solid #4caf50;margin:10px 0">'
                                    f'🧠 <b>AI 섹터 분석</b><br><br>'
                                    f'<span style="font-size:0.95em;line-height:1.8">{_sec_txt}</span>'
                                    f'</div>', unsafe_allow_html=True)
                            except:
                                st.warning("AI 분석을 가져올 수 없습니다.")

            else:
                st.info("📡 전체 스캔을 먼저 실행하면 섹터 로테이션 데이터가 표시됩니다.")

        except Exception as _sec_err:
            st.error(f"섹터 로테이션 오류: {str(_sec_err)[:200]}")

    st.divider()

    SECTOR_FILE = "sector_history.json"
    def load_sector_history():
        if os.path.exists(SECTOR_FILE):
            with open(SECTOR_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    def save_sector_history(data):
        with open(SECTOR_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    sector_hist = load_sector_history()
    if not sector_hist:
        st.info("아직 섹터 동반 상승 기록이 없어요. 전체 스캔을 하면 자동 저장됩니다.")
    else:
        if "selected_sector" not in st.session_state:
            st.session_state["selected_sector"] = []

        col_sd1, col_sd2 = st.columns([1, 1])
        with col_sd1:
            if st.button("🗑️ 전체 삭제", key="sector_clear"):
                save_sector_history([])
                st.session_state["selected_sector"] = []
                st.rerun()
        with col_sd2:
            if st.button("🗑️ 선택 삭제", key="sector_del_selected"):
                selected = st.session_state.get("selected_sector", [])
                if selected:
                    sh = load_sector_history()
                    new_sh = [h for i, h in enumerate(sh) if i not in selected]
                    save_sector_history(new_sh)
                    st.session_state["selected_sector"] = []
                    st.rerun()
                else:
                    st.warning("삭제할 기록을 선택하세요")

        reversed_hist = list(reversed(sector_hist))
        for ri, record in enumerate(reversed_hist):
            real_idx = len(sector_hist) - 1 - ri
            scan_time = record.get("time", "")
            surges = record.get("surges", [])
            col_check, col_content = st.columns([0.05, 0.95])
            with col_check:
                checked = st.checkbox("", key=f"chk_sector_{real_idx}", value=(real_idx in st.session_state["selected_sector"]))
                if checked and real_idx not in st.session_state["selected_sector"]:
                    st.session_state["selected_sector"].append(real_idx)
                elif not checked and real_idx in st.session_state["selected_sector"]:
                    st.session_state["selected_sector"].remove(real_idx)
            with col_content:
                with st.expander(f"📅 {scan_time} — {len(surges)}개 섹터 감지", expanded=(ri == 0)):
                    for si, sg in enumerate(surges):
                        if sg["count"] >= 3:
                            icon = "🚀"
                            color = "#ff1744"
                            label = "강력"
                        else:
                            icon = "⚡"
                            color = "#ff9800"
                            label = "주의"
                        st.markdown(
                            f'<div style="background:{color};color:#fff;padding:12px 16px;border-radius:10px;margin:6px 0;font-size:1.1em">'
                            f'{icon} <b>[{label}] {sg["theme"]}</b> — {sg["count"]}종목 동시 신호 | '
                            f'평균 점수 {sg["avg_score"]}점 | 평균 등락률 {sg["avg_change"]:+.2f}%'
                            f'</div>', unsafe_allow_html=True
                        )
                        seen_codes = set()
                        unique_stocks = []
                        for s in sg.get("stocks", []):
                            if s["code"] not in seen_codes:
                                seen_codes.add(s["code"])
                                unique_stocks.append(s)
                        for s in unique_stocks:
                            st.write(f"• {s['name']} ({s['code']}) — {s['score']}점, {s['change']:+.2f}%")
                        for s in unique_stocks:
                            btn_key = f"sect_{real_idx}_{si}_{s['code']}"
                            if st.button(f"📊 {s['name']} 상세분석", key=btn_key):
                                df_t = fetch(s["code"])
                                if df_t is not None:
                                    style_cfg = list(STYLES.values())[0]
                                    result = analyze(df_t, s["code"], s["name"], style_cfg)
                                    result["crown"] = ""
                                    show_card(result, f"sect_card_{real_idx}_{si}_{s['code']}", style_cfg)
                                else:
                                    st.warning("데이터를 가져올 수 없습니다")

# ======================== 코인 선물 ========================
elif menu == "🪙 코인 선물":
    st.header("🪙 코인 선물 분석")
    st.caption("바이낸스 USDT 무기한 선물 | 실시간 분석")

    # ── TP/SL 자동 체크 ──
    check_paper_tpsl()

    coin_tabs = st.tabs(["📊 전체 스캔", "🔍 개별 분석", "⭐ 관심종목", "💰 가상매매", "🚀 코인 급등 사냥"])

    # ═══════════════════════════════════════════
    #  탭 1: 전체 스캔
    # ═══════════════════════════════════════════
    with coin_tabs[0]:
        st.subheader("📊 코인 선물 전체 스캔")
        cs_c1, cs_c2 = st.columns(2)
        with cs_c1:
            timeframe = st.selectbox("분석 기간", ["1m","3m","5m","15m","1h","4h","1d"], index=6, key="coin_tf")
        with cs_c2:
            min_score_coin = st.slider("최소 점수", 0, 100, 50, key="coin_min_score")

        if st.button("🔍 코인 전체 스캔 시작", key="coin_scan_btn", use_container_width=True):
            results = []
            prog = st.progress(0)
            status = st.empty()
            total = len(COIN_FUTURES)
            for i, symbol in enumerate(COIN_FUTURES):
                status.text(f"분석 중: {symbol} ({i+1}/{total})")
                prog.progress((i+1)/total)
                try:
                    df = get_coin_klines(symbol, interval=timeframe)
                    funding = get_funding_rate(symbol)
                    ls_ratio = get_long_short_ratio(symbol)
                    oi = get_open_interest(symbol)
                    r = analyze_coin(df, symbol, funding, ls_ratio, oi)
                    if r and r["score"] >= min_score_coin:
                        results.append(r)
                except:
                    pass
                time.sleep(0.1)
            prog.empty()
            status.empty()


            # 세션에 저장
            st.session_state["coin_scan_results"] = [r for r in results if r is not None and isinstance(r, dict)]
            results = st.session_state["coin_scan_results"]

            if results:
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
                # 텔레그램 보고
                msg = f"🪙 <b>코인 선물 스캔 완료</b>\n총 {len(results)}개 종목 감지\n\n"
                for r in results[:10]:
                    msg += f"{'🟢' if r.get('score',0)>=70 else '🟡'} {r.get('symbol','')} | 점수: {r.get('score',0)} | {r.get('position','')}\n"
                send_telegram(msg)
                st.success(f"✅ 스캔 완료! {len(results)}개 종목 감지")
            else:
                st.warning("조건에 맞는 종목이 없습니다.")


        # 세션에 저장된 스캔 결과 표시
        if "coin_scan_results" in st.session_state and st.session_state["coin_scan_results"]:
            results = st.session_state["coin_scan_results"]
            for r in results:
                if not isinstance(r, dict) or "score" not in r:
                    continue
                score_color = "#ff4444" if r["score"] >= 80 else "#ff8800" if r["score"] >= 60 else "#4488ff"
                with st.expander(f"{r['position']} {r['symbol']} | 점수: {r['score']} | 현재가: ${r['price']:,.4f} ({r['change']:+.2f}%)", expanded=False):
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("RSI", f"{r.get('rsi', '-')}")
                    m2.metric("점수", f"{r['score']}")
                    m3.metric("펀딩비", f"{r.get('funding', '-')}")
                    m4.metric("롱숏비", f"{r.get('ls_ratio', '-')}")
                    m5.metric("거래량비", f"{r.get('vol_ratio', '-')}x")
                    if r.get("buy"):
                        st.success("매수: " + " | ".join(r["buy"][:5]))
                    if r.get("sell"):
                        st.error("매도: " + " | ".join(r["sell"][:5]))
                    try:
                        fig = draw_coin_chart(r)
                        st.pyplot(fig)
                        plt.close(fig)
                    except:
                        pass

                    # 관심종목 추가 버튼
                    if st.button(f"⭐ 관심종목 추가", key=f"scan_add_wl_{r['symbol']}"):
                        wl = load_coin_wl()
                        if r["symbol"] not in wl:
                            wl.append(r["symbol"])
                            save_coin_wl(wl)
                            st.success(f"{r['symbol']} 관심종목 추가!")
                        else:
                            st.info("이미 관심종목에 있습니다.")

    # ═══════════════════════════════════════════
    #  탭 2: 개별 분석
    # ═══════════════════════════════════════════
    with coin_tabs[1]:
        st.subheader("🔍 코인 개별 분석")
        coin_input = st.text_input("코인 심볼 입력 (예: BTCUSDT)", value="BTCUSDT", key="coin_single")
        coin_tf2 = st.selectbox("분석 기간", ["1m","3m","5m","15m","1h","4h","1d"], index=6, key="coin_tf2")
        use_mtf = st.checkbox("📊 멀티 타임프레임 분석", value=False, key="use_mtf")

        if st.button("🔍 분석", key="coin_analyze"):
            symbol = coin_input.upper().strip()
            if not symbol.endswith("USDT"):
                symbol += "USDT"
            with st.spinner(f"{symbol} 분석 중..."):
                df = get_coin_klines(symbol, interval=coin_tf2)
                funding = get_funding_rate(symbol)
                ls_ratio = get_long_short_ratio(symbol)
                oi = get_open_interest(symbol)
                fund_hist = get_funding_history(symbol)
                oi_hist = get_oi_history(symbol)
                top_trader = get_top_trader_ratio(symbol)
                fund_trend_text, fund_trend_icon, fund_consec = get_funding_trend(symbol)
                oi_val, oi_chg = get_oi_change(symbol)

                # 멀티 타임프레임
                mtf_data = {}
                if use_mtf:
                    for mtf_tf in ["5m", "15m", "1h", "4h", "1d"]:
                        try:
                            mtf_df = get_coin_klines(symbol, interval=mtf_tf)
                            mtf_r = analyze_coin(mtf_df, symbol, funding, ls_ratio, oi)
                            if mtf_r:
                                mtf_data[mtf_tf] = {"score": mtf_r["score"], "position": mtf_r["position"], "rsi": mtf_r.get("rsi", "-")}
                        except:
                            pass
                        time.sleep(0.1)

            if df is None:
                st.error(f"{symbol} 데이터를 가져올 수 없습니다")
            else:
                print(f"📡 df 길이: {len(df)}, 컬럼: {list(df.columns)}")
                print(f"📡 funding: {funding}, ls_ratio: {ls_ratio}, oi: {oi}")
                r = analyze_coin(df, symbol, funding, ls_ratio, oi)
                print(f"📡 analyze_coin 결과: {r is not None}")
                if r is None:
                    st.error("분석 데이터 부족")
                else:
                    st.session_state["coin_detail"] = {
                        "r": r, "symbol": symbol, "funding": funding,
                        "ls_ratio": ls_ratio, "oi": oi,
                        "fund_hist": fund_hist, "oi_hist": oi_hist,
                        "top_trader": top_trader,
                        "fund_trend_text": fund_trend_text,
                        "fund_trend_icon": fund_trend_icon,
                        "fund_consec": fund_consec,
                        "oi_chg": oi_chg,
                        "mtf_data": mtf_data,
                    }

        # ── 세션에 저장된 결과 표시 ──
        if "coin_detail" in st.session_state:
            d = st.session_state["coin_detail"]
            r = d["r"]
            symbol = d["symbol"]
            funding = d["funding"]
            ls_ratio = d["ls_ratio"]
            oi = d["oi"]
            top_trader = d["top_trader"]
            fund_trend_text = d["fund_trend_text"]
            fund_trend_icon = d["fund_trend_icon"]
            fund_consec = d["fund_consec"]
            oi_chg = d["oi_chg"]
            fund_hist = d["fund_hist"]
            oi_hist = d["oi_hist"]
            mtf_data = d.get("mtf_data", {})

            # 헤더
            score_color = "#ff4444" if r["score"] >= 80 else "#ff8800" if r["score"] >= 60 else "#4488ff"
            chg_color = "#ff4444" if r["change"] > 0 else "#4488ff"
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px;border-radius:15px;margin-bottom:20px">
                <h2 style="color:white;margin:0">{symbol} <span style="color:{chg_color}">${r['price']:,.4f} ({r['change']:+.2f}%)</span></h2>
                <p style="color:{score_color};font-size:24px;margin:5px 0">점수: {r['score']} | {r['position']}</p>
            </div>
            """, unsafe_allow_html=True)

            # 지표 카드 5개
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            fund_color = "#00ff88" if funding and funding < 0 else "#ff4444" if funding and funding > 0.03 else "#ff8800"
            ls_color = "#ff4444" if ls_ratio and ls_ratio > 1.5 else "#00ff88" if ls_ratio and ls_ratio < 0.7 else "#ff8800"
            whale_color = "#ff4444" if top_trader and top_trader > 1.5 else "#00ff88" if top_trader and top_trader < 0.7 else "#ff8800"
            rsi_color = "#ff4444" if r.get("rsi",50) > 70 else "#00ff88" if r.get("rsi",50) < 30 else "#ff8800"

            with mc1:
                st.markdown(f"<div style='background:#1e1e2e;padding:15px;border-radius:10px;text-align:center'><div style='color:#888;font-size:12px'>펀딩비</div><div style='color:{fund_color};font-size:20px;font-weight:bold'>{funding if funding else 'N/A'}</div></div>", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"<div style='background:#1e1e2e;padding:15px;border-radius:10px;text-align:center'><div style='color:#888;font-size:12px'>롱숏비율</div><div style='color:{ls_color};font-size:20px;font-weight:bold'>{ls_ratio if ls_ratio else 'N/A'}</div></div>", unsafe_allow_html=True)
            with mc3:
                st.markdown(f"<div style='background:#1e1e2e;padding:15px;border-radius:10px;text-align:center'><div style='color:#888;font-size:12px'>미체결약정</div><div style='color:white;font-size:20px;font-weight:bold'>{oi if oi else 'N/A'}</div></div>", unsafe_allow_html=True)
            with mc4:
                st.markdown(f"<div style='background:#1e1e2e;padding:15px;border-radius:10px;text-align:center'><div style='color:#888;font-size:12px'>고래비율</div><div style='color:{whale_color};font-size:20px;font-weight:bold'>{top_trader if top_trader else 'N/A'}</div></div>", unsafe_allow_html=True)
            with mc5:
                st.markdown(f"<div style='background:#1e1e2e;padding:15px;border-radius:10px;text-align:center'><div style='color:#888;font-size:12px'>RSI</div><div style='color:{rsi_color};font-size:20px;font-weight:bold'>{r.get('rsi', 'N/A')}</div></div>", unsafe_allow_html=True)

            # 펀딩 추세 뱃지
            if fund_consec and fund_consec >= 3:
                st.markdown(f"<div style='background:#2a2a3e;padding:10px;border-radius:8px;margin:10px 0'>{fund_trend_icon} {fund_trend_text}</div>", unsafe_allow_html=True)

            # OI 변화율
            if oi_chg is not None:
                oi_icon = "📈" if oi_chg > 0 else "📉"
                st.markdown(f"{oi_icon} OI 변화율: **{oi_chg:+.2f}%**")

            # 멀티 타임프레임 결과
            if mtf_data:
                st.markdown("**📊 멀티 타임프레임 분석**")
                mtf_cols = st.columns(len(mtf_data))
                for i, (tf, info) in enumerate(mtf_data.items()):
                    with mtf_cols[i]:
                        st.markdown(f"<div style='background:#1e1e2e;padding:10px;border-radius:8px;text-align:center'><b>{tf}</b><br>점수: {info['score']}<br>{info['position']}<br>RSI: {info['rsi']}</div>", unsafe_allow_html=True)

            # 매수/매도 신호
            if r.get("buy"):
                st.success("🟢 매수 신호: " + " | ".join(r["buy"][:5]))
            if r.get("sell"):
                st.error("🔴 매도 신호: " + " | ".join(r["sell"][:5]))

            # ── Gemini AI 코인 심층 분석 ──
            if GEMINI_OK:
                if st.button("🧠 Gemini AI 코인 분석", key=f"gemini_coin_{symbol}"):
                    with st.spinner("🧠 Gemini AI가 코인 분석 중..."):
                        gm_coin = gemini_coin_judgment(
                            symbol,
                            r.get("price", 0),
                            r.get("change", 0),
                            r.get("rsi", 0),
                            r.get("mfi", 0),
                            r.get("adx", 0),
                            r.get("funding_rate", None),
                            r.get("long_short_ratio", None),
                            r.get("oi_change", None),
                            r.get("score", 0),
                            r.get("grade", ""),
                            r.get("buy", []),
                            r.get("sell", []),
                            r.get("support", 0),
                            r.get("resist", 0),
                        )
                    if gm_coin:
                        st.markdown(
                            f'<div style="background:#1a2e1a;padding:14px;border-radius:10px;margin:8px 0;border-left:4px solid #4caf50">'
                            f'🧠 <b>Gemini AI 코인 심층 분석</b><br><br>'
                            f'{gm_coin}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning("Gemini AI 응답을 가져올 수 없습니다.")

            # 코인글래스 외부 링크
            coin_name = symbol.replace("USDT", "")
            st.markdown(
                f'<div style="background:#1e1e2e;padding:14px;border-radius:10px;margin:10px 0;border:1px solid #ff9800">'
                f'🌐 <b>코인글래스 실시간 데이터</b><br><br>'
                f'<a href="https://www.coinglass.com/ko/pro/futures/LiquidationHeatMap?coin={coin_name}&type=symbol" target="_blank" style="color:#ff9800;font-size:1.05em;text-decoration:none">🗺️ 청산 히트맵</a> &nbsp;&nbsp;|&nbsp;&nbsp; '
                f'<a href="https://www.coinglass.com/ko/pro/futures/LiquidationMap/{coin_name}" target="_blank" style="color:#ff9800;font-size:1.05em;text-decoration:none">📊 청산맵</a> &nbsp;&nbsp;|&nbsp;&nbsp; '
                f'<a href="https://www.coinglass.com/ko/whale-alert" target="_blank" style="color:#ff9800;font-size:1.05em;text-decoration:none">🐋 고래 알림</a> &nbsp;&nbsp;|&nbsp;&nbsp; '
                f'<a href="https://www.coinglass.com/ko/FundingRate/{coin_name}" target="_blank" style="color:#ff9800;font-size:1.05em;text-decoration:none">💰 펀딩비</a> &nbsp;&nbsp;|&nbsp;&nbsp; '
                f'<a href="https://www.coinglass.com/ko/LongShortRatio/{coin_name}" target="_blank" style="color:#ff9800;font-size:1.05em;text-decoration:none">📈 롱숏비율</a> &nbsp;&nbsp;|&nbsp;&nbsp; '
                f'<a href="https://www.coinglass.com/ko/OpenInterest/{coin_name}" target="_blank" style="color:#ff9800;font-size:1.05em;text-decoration:none">📋 미결제약정</a>'
                f'</div>', unsafe_allow_html=True
            )


            # 청산가 테이블
            with st.expander("📋 레버리지별 예상 청산가"):
                liq_zones = calc_liquidation_zones(r["price"])
                liq_df = pd.DataFrame(liq_zones)
                st.dataframe(liq_df, use_container_width=True, hide_index=True)

            # 펀딩비 히스토리 차트
            if fund_hist:
                with st.expander("📊 펀딩비 히스토리"):
                    try:
                        fh_df = pd.DataFrame(fund_hist)
                        fh_df["fundingRate"] = fh_df["fundingRate"].astype(float)
                        fh_df["color"] = fh_df["fundingRate"].apply(lambda x: "green" if x < 0 else "red")
                        import plotly.graph_objects as go
                        fig_f = go.Figure()
                        fig_f.add_trace(go.Bar(y=fh_df["fundingRate"], marker_color=fh_df["color"]))
                        fig_f.update_layout(title="펀딩비 추이", height=300, template="plotly_dark")
                        st.plotly_chart(fig_f, use_container_width=True, key=f"fund_hist_{symbol}")
                    except:
                        st.info("펀딩비 차트를 그릴 수 없습니다.")

            # OI 히스토리 차트
            if oi_hist:
                with st.expander("📊 미체결약정 히스토리"):
                    try:
                        oi_df = pd.DataFrame(oi_hist)
                        oi_df["sumOpenInterest"] = oi_df["sumOpenInterest"].astype(float)
                        import plotly.graph_objects as go
                        fig_oi = go.Figure()
                        fig_oi.add_trace(go.Scatter(y=oi_df["sumOpenInterest"], mode="lines", line=dict(color="cyan")))
                        fig_oi.update_layout(title="미체결약정 추이", height=300, template="plotly_dark")
                        st.plotly_chart(fig_oi, use_container_width=True, key=f"oi_hist_{symbol}")
                    except:
                        st.info("OI 차트를 그릴 수 없습니다.")

            # 메인 차트
            try:
                fig = draw_coin_chart(r)
                st.pyplot(fig)
                plt.close(fig)
            except:
                pass

            # ── 가상매매 바로 진입 ──
            st.markdown("---")
            st.markdown("**⚡ 바로 가상매매 진입**")
            pt_c1, pt_c2 = st.columns(2)
            with pt_c1:
                pt_lev = st.selectbox("레버리지", [5, 10, 20, 25, 50, 100], index=1, key=f"pt_lev_{symbol}")
            with pt_c2:
                pt_qty = st.number_input("투자금 (USDT)", min_value=10.0, value=100.0, step=10.0, key=f"pt_qty_{symbol}")

            tp_c1, tp_c2 = st.columns(2)
            with tp_c1:
                pt_tp = st.number_input("목표가 (TP) - 0이면 없음", min_value=0.0, value=0.0, step=0.01, format="%.4f", key=f"pt_tp_{symbol}")
            with tp_c2:
                pt_sl = st.number_input("손절가 (SL) - 0이면 없음", min_value=0.0, value=0.0, step=0.01, format="%.4f", key=f"pt_sl_{symbol}")

            btn_long, btn_short = st.columns(2)
            with btn_long:
                if st.button("🟢 롱 진입", key=f"pt_long_{symbol}", use_container_width=True, type="primary"):
                    trade = open_paper_trade(symbol, "LONG", r["price"], pt_lev, pt_qty, pt_tp if pt_tp > 0 else None, pt_sl if pt_sl > 0 else None)
                    send_telegram(
                        f"🪙 <b>가상매매 진입</b>\n"
                        f"🟢 롱 {symbol}\n"
                        f"진입가: ${r['price']:,.4f}\n"
                        f"레버리지: {pt_lev}x | 투자금: ${pt_qty}"
                        + (f"\nTP: ${pt_tp:,.4f}" if pt_tp > 0 else "")
                        + (f" | SL: ${pt_sl:,.4f}" if pt_sl > 0 else "")
                    )
                    st.success(f"🟢 롱 진입 완료! 진입가: ${r['price']:,.4f}")
                    time.sleep(1)
                    st.rerun()
            with btn_short:
                if st.button("🔴 숏 진입", key=f"pt_short_{symbol}", use_container_width=True):
                    trade = open_paper_trade(symbol, "SHORT", r["price"], pt_lev, pt_qty, pt_tp if pt_tp > 0 else None, pt_sl if pt_sl > 0 else None)
                    send_telegram(
                        f"🪙 <b>가상매매 진입</b>\n"
                        f"🔴 숏 {symbol}\n"
                        f"진입가: ${r['price']:,.4f}\n"
                        f"레버리지: {pt_lev}x | 투자금: ${pt_qty}"
                        + (f"\nTP: ${pt_tp:,.4f}" if pt_tp > 0 else "")
                        + (f" | SL: ${pt_sl:,.4f}" if pt_sl > 0 else "")
                    )
                    st.success(f"🔴 숏 진입 완료! 진입가: ${r['price']:,.4f}")
                    time.sleep(1)
                    st.rerun()

    # ═══════════════════════════════════════════
    #  탭 3: 관심종목
    # ═══════════════════════════════════════════
    with coin_tabs[2]:
        st.subheader("⭐ 코인 관심종목")
        cwl = load_coin_wl()

        # 추가 UI
        cwl_c1, cwl_c2 = st.columns([3, 1])
        with cwl_c1:
            new_coin = st.text_input("종목 추가 (예: BTCUSDT)", key="cwl_add_input")
        with cwl_c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ 추가", key="cwl_add_btn"):
                sym = new_coin.upper().strip()
                if sym and not sym.endswith("USDT"):
                    sym += "USDT"
                if sym and sym not in cwl:
                    cwl.append(sym)
                    save_coin_wl(cwl)
                    st.success(f"{sym} 추가 완료!")
                    st.rerun()

        if not cwl:
            st.info("관심종목이 없습니다. 전체 스캔이나 개별 분석에서 추가하세요.")
        else:
            cwl_tf = st.selectbox("분석 기간", ["1m","3m","5m","15m","1h","4h","1d"], index=6, key="cwl_tf")

            # 자동 갱신
            if AUTOREFRESH_OK:
                cwl_auto = st.checkbox("⏱️ 자동 갱신 (30초)", value=False, key="cwl_auto")
                if cwl_auto:
                    st_autorefresh(interval=30000, key="cwl_refresh")

            if st.button("🔄 관심종목 분석", key="cwl_scan_btn", use_container_width=True):
                cwl_results = []
                prog = st.progress(0)
                for i, sym in enumerate(cwl):
                    prog.progress((i+1)/len(cwl))
                    try:
                        df = get_coin_klines(sym, interval=cwl_tf)
                        funding = get_funding_rate(sym)
                        ls_ratio = get_long_short_ratio(sym)
                        oi = get_open_interest(sym)
                        r = analyze_coin(df, sym, funding, ls_ratio, oi)
                        if r:
                            cwl_results.append(r)
                    except:
                        continue
                    time.sleep(0.1)
                prog.empty()
                st.session_state["cwl_results"] = cwl_results

            # 저장된 관심종목 결과 표시
            if "cwl_results" in st.session_state and st.session_state["cwl_results"]:
                for r in st.session_state["cwl_results"]:
                    with st.expander(f"{r['position']} {r['symbol']} | 점수: {r['score']} | ${r['price']:,.4f} ({r['change']:+.2f}%)", expanded=False):
                        wm1, wm2, wm3, wm4 = st.columns(4)
                        wm1.metric("점수", r["score"])
                        wm2.metric("RSI", r.get("rsi", "-"))
                        wm3.metric("펀딩비", r.get("funding", "-"))
                        wm4.metric("거래량비", f"{r.get('vol_ratio', '-')}x")
                        if r.get("buy"):
                            st.success("매수: " + " | ".join(r["buy"][:3]))
                        if r.get("sell"):
                            st.error("매도: " + " | ".join(r["sell"][:3]))
                        try:
                            fig = draw_coin_chart(r)
                            st.pyplot(fig)
                            plt.close(fig)
                        except:
                            pass
                        if st.button(f"❌ {r['symbol']} 삭제", key=f"cwl_del_{r['symbol']}"):
                            cwl = load_coin_wl()
                            if r["symbol"] in cwl:
                                cwl.remove(r["symbol"])
                                save_coin_wl(cwl)
                                st.success(f"{r['symbol']} 삭제 완료!")
                                st.rerun()


    # ── 탭 4: 가상매매 ──
    with coin_tabs[3]:
        st.subheader("💰 가상매매 (페이퍼 트레이딩)")
        if AUTOREFRESH_OK:
            coin_auto = st.checkbox("⏱️ 자동 갱신 (30초)", value=False, key="coin_trade_auto")
            if coin_auto:
                st_autorefresh(interval=30000, key="coin_trade_refresh")
        trades = load_coin_trades()
        open_trades = [t for t in trades if isinstance(t, dict) and t.get("status") == "open"]
        closed_trades = [t for t in trades if isinstance(t, dict) and t.get("status") == "closed"]
 

        # 통계
        stats = get_paper_stats()
        if stats is None:                                                          
            stats = {"total": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0}    
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        stat_c1.metric("총 거래", stats.get("total", 0))
        stat_c2.metric("승률", f"{stats.get('win_rate', 0):.1f}%")
        stat_c3.metric("총 수익", f"${stats.get('total_pnl', 0):,.2f}")
        stat_c4.metric("평균 수익", f"${stats.get('avg_pnl', 0):,.2f}")

        # ── 오픈 포지션 ──
        st.markdown("### 📂 진행 중인 포지션")
        if not open_trades:
            st.info("진행 중인 가상매매가 없습니다. 개별 분석에서 진입하세요.")
        else:
            for t in open_trades:
                sym = t["symbol"]
                side = t["side"]
                entry = t["entry_price"]
                lev = t["leverage"]
                qty = t.get("qty_usdt", 0)
                tp = t.get("tp_price")
                sl = t.get("sl_price")

                # 현재가 조회
                try:
                    url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}"
                    cur_price = float(requests.get(url, timeout=5).json()["price"])
                except:
                    cur_price = entry

                # 수익 계산
                if side == "LONG":
                    pnl_pct = round((cur_price - entry) / (entry + 1e-9) * 100 * lev, 2)
                else:
                    pnl_pct = round((entry - cur_price) / (entry + 1e-9) * 100 * lev, 2)
                pnl_usdt = round(qty * pnl_pct / 100, 2)
                pnl_color = "#00ff88" if pnl_pct >= 0 else "#ff4444"
                side_icon = "🟢" if side == "LONG" else "🔴"

                # 청산가 계산
                if side == "LONG":
                    liq_price = round(entry * (1 - 1/lev), 4)
                else:
                    liq_price = round(entry * (1 + 1/lev), 4)

                st.markdown(f"""
                <div style="background:#1e1e2e;padding:15px;border-radius:10px;margin:10px 0;border-left:4px solid {pnl_color}">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <span style="font-size:18px;font-weight:bold;color:white">{side_icon} {sym} {side} {lev}x</span><br>
                            <span style="color:#888">진입: ${entry:,.4f} | 현재: ${cur_price:,.4f} | 청산: ${liq_price:,.4f}</span><br>
                            <span style="color:#888">투자금: ${qty} | TP: {f'${tp:,.4f}' if tp else '없음'} | SL: {f'${sl:,.4f}' if sl else '없음'}</span>
                        </div>
                        <div style="text-align:right">
                            <span style="color:{pnl_color};font-size:24px;font-weight:bold">{pnl_pct:+.2f}%</span><br>
                            <span style="color:{pnl_color};font-size:16px">${pnl_usdt:+,.2f}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                trade_id = t.get("id", t.get("open_time", sym))
                if st.button(f"💰 {sym} 포지션 종료", key=f"close_{trade_id}"):
                    result = close_paper_trade(trade_id, cur_price)
                    send_telegram(
                        f"🪙 <b>가상매매 종료</b>\n"
                        f"{side_icon} {sym} {side}\n"
                        f"진입: ${entry:,.4f} → 종료: ${cur_price:,.4f}\n"
                        f"수익: {pnl_pct:+.2f}% (${pnl_usdt:+,.2f})"
                    )
                    st.success(f"{sym} 포지션 종료! 수익: {pnl_pct:+.2f}%")
                    time.sleep(1)
                    st.rerun()

        # ── 종료된 거래 내역 ──
        if closed_trades:
            st.markdown("### 📋 거래 내역")
            with st.expander(f"종료된 거래 {len(closed_trades)}건", expanded=False):
                for t in reversed(closed_trades[-20:]):
                    pnl = t.get("pnl_pct", 0)
                    pnl_icon = "✅" if pnl >= 0 else "❌"
                    side_icon = "🟢" if t["side"] == "LONG" else "🔴"
                    st.markdown(f"{pnl_icon} {side_icon} **{t['symbol']}** {t['side']} {t['leverage']}x | "
                               f"${t['entry_price']:,.4f} → ${t.get('exit_price', 0):,.4f} | "
                               f"**{pnl:+.2f}%** | {t.get('close_time', '')[:16]}")

            # 전체 삭제
            if st.button("🗑️ 거래 내역 초기화", key="clear_trades"):
                save_coin_trades([t for t in trades if t.get("status") == "open"])
                st.success("종료된 거래 내역 삭제 완료!")
                st.rerun()

    # ═══════════════════════════════════════════
    #  탭 5: 코인 급등 사냥
    # ═══════════════════════════════════════════
    with coin_tabs[4]:
        st.subheader("🚀 코인 급등 사냥 — 터지기 직전 코인 탐지")
        st.caption("거래량 폭발 · 펀딩비 극단 · OI 급증 · 기술적 스퀴즈 등 급등 직전 신호를 자동 감지합니다")

        coin_surge_tf = st.selectbox("분석 기간", ["5m", "15m", "1h", "4h", "1d"], index=4, key="coin_surge_tf")

        # 단일 코인 급등 분석
        st.markdown("### 🔍 단일 코인 급등 분석")
        coin_surge_query = st.text_input("코인 심볼 입력 (예: BTC)", key="coin_surge_query")
        if coin_surge_query:
            sym = coin_surge_query.upper().strip()
            if not sym.endswith("USDT"):
                sym += "USDT"
            with st.spinner(f"{sym} 급등 분석 중..."):
                df = get_coin_klines(sym, interval=coin_surge_tf)
                funding = get_funding_rate(sym)
                ls_ratio = get_long_short_ratio(sym)
                oi = get_open_interest(sym)
            if df is None or len(df) < 30:
                st.error(f"{sym} 데이터 부족")
            else:
                r = analyze_coin(df, sym, funding, ls_ratio, oi)
                if r:
                    # 급등 추가 점수
                    surge_score = 0
                    surge_signals = []
                    if r["vol_ratio"] >= 5:
                        surge_score += 20
                        surge_signals.append(f"🔥 거래량 {r['vol_ratio']}배 폭발")
                    elif r["vol_ratio"] >= 3:
                        surge_score += 12
                        surge_signals.append(f"📈 거래량 {r['vol_ratio']}배 급증")
                    if funding and funding < -0.03:
                        surge_score += 15
                        surge_signals.append(f"💰 펀딩비 음수({funding}%) → 숏스퀴즈 가능")
                    if funding and funding > 0.05:
                        surge_score += 10
                        surge_signals.append(f"⚠️ 펀딩비 과열({funding}%) → 롱스퀴즈 가능")
                    if ls_ratio and ls_ratio < 0.5:
                        surge_score += 15
                        surge_signals.append(f"🔥 숏 과밀({ls_ratio}) → 숏스퀴즈 임박")
                    if r.get("rsi", 50) < 25:
                        surge_score += 10
                        surge_signals.append(f"📊 RSI 극과매도({r['rsi']}) → 반등 임박")
                    # 볼린저 스퀴즈
                    c = df["Close"].astype(float)
                    is_sq, sq_pct = calc_squeeze(c)
                    if is_sq:
                        surge_score += 15
                        surge_signals.append(f"🔋 볼린저 스퀴즈 (밴드폭 하위 {sq_pct}%) → 곧 폭발")
                    # OI 변화
                    oi_val, oi_chg = get_oi_change(sym)
                    if oi_chg and oi_chg > 10:
                        surge_score += 10
                        surge_signals.append(f"📈 미결제약정 {oi_chg:+.1f}% 급증 → 큰돈 유입")

                    total_score = min(100, surge_score)
                    grade = "S" if total_score >= 70 else "A" if total_score >= 50 else "B" if total_score >= 30 else "C"
                    grade_emoji = {"S": "🔥", "A": "⚡", "B": "👀", "C": "🔍"}.get(grade, "")
                    chg_icon = "🔴" if r["change"] > 0 else "🔵" if r["change"] < 0 else "⚪"

                    st.markdown(f"""**{grade_emoji} {sym}** — 급등 점수: **{total_score}점** ({grade})

{chg_icon} **${r['price']:,.4f}** ({r['change']:+.2f}%) | 거래량 {r['vol_ratio']}x | RSI {r['rsi']}
""")
                    if surge_signals:
                        st.markdown("**📡 급등 신호:**")
                        for sig in surge_signals:
                            st.markdown(f"- {sig}")
                    else:
                        st.info("현재 급등 신호가 감지되지 않습니다.")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊 차트", key=f"coin_surge_chart_{sym}"):
                            fig = draw_coin_chart(r)
                            st.pyplot(fig)
                            plt.close(fig)
                    with col2:
                        if st.button("📡 정밀분석", key=f"coin_surge_detail_{sym}"):
                            st.markdown("**전체 지표 분석:**")
                            if r.get("buy"):
                                st.success("매수: " + " | ".join(r["buy"][:5]))
                            if r.get("sell"):
                                st.error("매도: " + " | ".join(r["sell"][:5]))

        st.divider()

        # 전체 코인 급등 스캔
        if st.button("🚀 코인 급등 사냥 시작!", use_container_width=True, type="primary", key="coin_surge_start"):
            results = []
            prog = st.progress(0)
            status = st.empty()
            total = len(COIN_FUTURES)
            for idx, sym in enumerate(COIN_FUTURES):
                status.text(f"분석 중: {sym} ({idx+1}/{total})")
                prog.progress((idx+1)/total)
                try:
                    df = get_coin_klines(sym, interval=coin_surge_tf)
                    if df is None or len(df) < 30:
                        continue
                    funding = get_funding_rate(sym)
                    ls_ratio = get_long_short_ratio(sym)
                    oi = get_open_interest(sym)
                    r = analyze_coin(df, sym, funding, ls_ratio, oi)
                    if not r:
                        continue

                    surge_score = 0
                    surge_signals = []
                    if r["vol_ratio"] >= 5:
                        surge_score += 20
                        surge_signals.append(f"🔥 거래량 {r['vol_ratio']}배 폭발")
                    elif r["vol_ratio"] >= 3:
                        surge_score += 12
                        surge_signals.append(f"📈 거래량 {r['vol_ratio']}배 급증")
                    if funding and funding < -0.03:
                        surge_score += 15
                        surge_signals.append(f"💰 펀딩비 음수({funding}%) → 숏스퀴즈 가능")
                    if funding and funding > 0.05:
                        surge_score += 10
                        surge_signals.append(f"⚠️ 펀딩비 과열({funding}%)")
                    if ls_ratio and ls_ratio < 0.5:
                        surge_score += 15
                        surge_signals.append(f"🔥 숏 과밀({ls_ratio})")
                    if r.get("rsi", 50) < 25:
                        surge_score += 10
                        surge_signals.append(f"📊 RSI 극과매도({r['rsi']})")
                    c = df["Close"].astype(float)
                    is_sq, sq_pct = calc_squeeze(c)
                    if is_sq:
                        surge_score += 15
                        surge_signals.append(f"🔋 볼린저 스퀴즈 ({sq_pct}%)")
                    oi_val, oi_chg = get_oi_change(sym)
                    if oi_chg and oi_chg > 10:
                        surge_score += 10
                        surge_signals.append(f"📈 OI {oi_chg:+.1f}% 급증")

                    if surge_score >= 20:
                        grade = "S" if surge_score >= 70 else "A" if surge_score >= 50 else "B" if surge_score >= 30 else "C"
                        results.append({
                            "symbol": sym,
                            "price": r["price"],
                            "change": r["change"],
                            "vol_ratio": r["vol_ratio"],
                            "rsi": r.get("rsi", 50),
                            "funding": funding,
                            "ls_ratio": ls_ratio,
                            "surge_score": surge_score,
                            "grade": grade,
                            "signals": surge_signals,
                            "coin_r": r,
                        })
                except:
                    pass
                time.sleep(0.1)
            prog.empty()
            status.empty()

            if results:
                results.sort(key=lambda x: x["surge_score"], reverse=True)
                st.session_state["coin_surge_results"] = results
                st.success(f"🎯 코인 급등 후보 {len(results)}개 발견!")
            else:
                st.info("현재 급등 신호가 감지된 코인이 없습니다.")

        # 저장된 결과 표시
        if "coin_surge_results" in st.session_state and st.session_state["coin_surge_results"]:
            for i, item in enumerate(st.session_state["coin_surge_results"]):
                grade_emoji = {"S": "🔥", "A": "⚡", "B": "👀", "C": "🔍"}.get(item["grade"], "")
                chg_icon = "🔴" if item["change"] > 0 else "🔵" if item["change"] < 0 else "⚪"

                st.markdown(f"""**{i+1}. {grade_emoji} {item['symbol']}** — 급등 점수: **{item['surge_score']}점** ({item['grade']})

{chg_icon} **${item['price']:,.4f}** ({item['change']:+.2f}%) | 거래량 {item['vol_ratio']}x | RSI {item['rsi']} | 펀딩비 {item['funding']}%
""")
                with st.expander(f"📡 신호 ({len(item['signals'])}개)", expanded=(i < 3)):
                    for sig in item["signals"]:
                        st.markdown(f"- {sig}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📊 차트", key=f"coin_surge_res_chart_{item['symbol']}_{i}"):
                            fig = draw_coin_chart(item["coin_r"])
                            st.pyplot(fig)
                            plt.close(fig)
                    with col2:
                        if st.button("⭐ 관심종목", key=f"coin_surge_res_wl_{item['symbol']}_{i}"):
                            cwl = load_coin_wl()
                            if item["symbol"] not in cwl:
                                cwl.append(item["symbol"])
                                save_coin_wl(cwl)
                                st.success(f"{item['symbol']} 추가!")
                    with col3:
                        if st.button("📡 정밀분석", key=f"coin_surge_res_detail_{item['symbol']}_{i}"):
                            r = item["coin_r"]
                            if r.get("buy"):
                                st.success("매수: " + " | ".join(r["buy"][:5]))
                            if r.get("sell"):
                                st.error("매도: " + " | ".join(r["sell"][:5]))
                st.divider()


# ======================== 매매 ========================
elif menu == "💰 매매":
    st.header("💰 매매")
    if not KIS_OK:
        st.error("KIS API가 연결되지 않았습니다. kis_config.yaml을 확인하세요.")
    else:
        tabs = st.tabs(["매수", "매도", "주문내역", "잔고"])

        with tabs[0]:
            st.subheader("📈 매수 주문")
            buy_code = st.text_input("종목코드", key="buy_code")
            buy_type = st.selectbox("주문유형", ["시장가", "지정가"], key="buy_type")
            buy_price = 0
            if buy_type == "지정가":
                buy_price = st.number_input("매수가격", min_value=0, step=100, key="buy_price")
            buy_qty = st.number_input("수량", min_value=1, step=1, key="buy_qty")
            if st.button("🟢 매수 실행", type="primary", key="buy_exec"):
                try:
                    if buy_type == "시장가":
                        result = _broker.create_market_buy_order(buy_code, buy_qty)
                    else:
                        result = _broker.create_limit_buy_order(buy_code, buy_price, buy_qty)
                    st.success(f"매수 주문 완료: {result}")
                except Exception as e:
                    st.error(f"매수 실패: {e}")

        with tabs[1]:
            st.subheader("📉 매도 주문")
            sell_code = st.text_input("종목코드", key="sell_code")
            sell_type = st.selectbox("주문유형", ["시장가", "지정가"], key="sell_type")
            sell_price = 0
            if sell_type == "지정가":
                sell_price = st.number_input("매도가격", min_value=0, step=100, key="sell_price")
            sell_qty = st.number_input("수량", min_value=1, step=1, key="sell_qty")
            if st.button("🔴 매도 실행", type="primary", key="sell_exec"):
                try:
                    if sell_type == "시장가":
                        result = _broker.create_market_sell_order(sell_code, sell_qty)
                    else:
                        result = _broker.create_limit_sell_order(sell_code, sell_price, sell_qty)
                    st.success(f"매도 주문 완료: {result}")
                except Exception as e:
                    st.error(f"매도 실패: {e}")

        with tabs[2]:
            st.subheader("📋 주문 내역")
            if st.button("조회", key="order_fetch"):
                try:
                    orders = _broker.fetch_open_orders()
                    if orders:
                        st.dataframe(pd.DataFrame(orders))
                    else:
                        st.info("주문 내역 없음")
                except Exception as e:
                    st.error(f"조회 실패: {e}")

        with tabs[3]:
            st.subheader("💰 잔고 조회")
            if st.button("조회", key="balance_fetch"):
                try:
                    balance = _broker.fetch_balance()
                    if balance:
                        st.dataframe(pd.DataFrame(balance))
                    else:
                        st.info("잔고 없음")
                except Exception as e:
                    st.error(f"조회 실패: {e}")

# ═══════════════════════════════════════════════════════
#  🚀 급등 사냥 모드
# ═══════════════════════════════════════════════════════
elif menu == "🚀 급등 사냥":
    st.markdown("## 🚀 급등 사냥 — 터지기 직전 종목 탐지")
    st.caption("거래대금 급증 · 볼린저 스퀴즈 · 스텔스 매집 · 바닥 돌파 등 급등 직전 신호를 자동 감지합니다")

    if not FDR_OK:
        st.error("FinanceDataReader가 설치되지 않았습니다.")
    else:
        surge_country = st.radio("시장 선택", ["🇰🇷 한국 (KOSPI+KOSDAQ)", "🇺🇸 미국 (전체)"], horizontal=True, key="surge_country")

        # ── 단일 종목 급등 분석 ──
        st.subheader("🔍 단일 종목 급등 분석")
        surge_query = st.text_input("종목명 또는 코드 입력", key="surge_single_query")
        if surge_query:
            surge_query = surge_query.strip()
            found = None
            found_market = ""
            for mkt in ["KOSPI", "KOSDAQ", "NASDAQ"]:
                stocks_df = get_stocks(mkt)
                if stocks_df.empty:
                    continue
                match = stocks_df[stocks_df["Name"].str.contains(surge_query, case=False, na=False)]
                if not match.empty:
                    found = match.iloc[0]
                    found_market = mkt
                    break
                match2 = stocks_df[stocks_df["Code"].str.contains(surge_query, case=False, na=False)]
                if not match2.empty:
                    found = match2.iloc[0]
                    found_market = mkt
                    break
            if found is None:
                st.warning(f"'{surge_query}' 종목을 찾을 수 없습니다.")
            else:
                code = str(found["Code"]).strip()
                name = str(found["Name"]).strip()
                is_us = found_market in ["NASDAQ", "NYSE", "AMEX"]
                with st.spinner(f"{name} 급등 분석 중..."):
                    df = fetch(code, days=300)
                if df is None or len(df) < 60:
                    st.error(f"{name} 데이터가 부족합니다.")
                else:
                    r = analyze_surge(df, code, name)
                    if r is None or r["score"] < 10:
                        st.info(f"{name}은 현재 급등 신호가 감지되지 않습니다.")
                    else:
                        r["market"] = found_market
                        r["is_us"] = is_us
                        grade_emoji = {"S": "🔥", "A": "⚡", "B": "👀", "C": "🔍"}.get(r["grade"], "")
                        currency = "$" if is_us else ""
                        unit = "" if is_us else "원"
                        tv_unit = "만$" if is_us else "억"
                        chg_icon = "🔴" if r["change"] > 0 else "🔵" if r["change"] < 0 else "⚪"

                        st.markdown(f"""**{grade_emoji} {r['name']}** ({r['code']}) `{found_market}`

{chg_icon} **{currency}{r['price']:,}{unit}** ({r['change']:+.2f}%) | 거래량 {r['vol_ratio']}x | 거래대금 {r['trade_val']:.0f}{tv_unit} | RSI {r['rsi']}

**등급: {r['grade']} ({r['score']}점)** — {r['verdict']}
""")
                        st.markdown("**📡 감지 신호:**")
                        for sig in r["signals"]:
                            st.markdown(f"- {sig}")

                        if is_us:
                            st.markdown(f"**지지** ${r['support']:,} | **저항** ${r['resist']:,} | **손절** ${r['stop_loss']:,}")
                        else:
                            st.markdown(f"**지지** {r['support']:,}원 | **저항** {r['resist']:,}원 | **손절** {r['stop_loss']:,}원")

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if st.button("📊 차트", key=f"surge_single_chart_{code}"):
                                cfg_temp = {"tp": 10, "sl": 5}
                                fig = draw_chart(r, cfg_temp)
                                st.pyplot(fig)
                                plt.close(fig)
                        with col_b:
                            if st.button("⭐ 관심종목 추가", key=f"surge_single_wl_{code}"):
                                add_to_wl(code, name)
                                st.success(f"{name} 관심종목 추가!")
                        with col_c:
                            if st.button("📡 정밀분석", key=f"surge_single_detail_{code}"):
                                with st.spinner(f"{name} 스캔/검색 지표로 재분석 중..."):
                                    detail_df = fetch(code, days=300)
                                    if detail_df is not None:
                                        mkt = found_market
                                        for style_name, cfg in STYLES.items():
                                            detail_r = analyze(detail_df, code, name, cfg, market=mkt)
                                            detail_r["crown"] = ""
                                            st.markdown(f"**{style_name}**")
                                            show_card(detail_r, f"surge_single_detail_{code}_{cfg['key']}", cfg)
                                    else:
                                        st.error("데이터를 가져올 수 없습니다")

        st.divider()

        if st.button("🚀 급등 사냥 시작!", use_container_width=True, type="primary"):


            # ── 종목 리스트 가져오기 ──
            all_stocks = pd.DataFrame()

            if "한국" in surge_country:
                for mkt in ["KOSPI", "KOSDAQ"]:
                    tmp = get_stocks(mkt)
                    if not tmp.empty:
                        tmp["market"] = mkt
                        all_stocks = pd.concat([all_stocks, tmp], ignore_index=True)
                is_us = False
                max_price = 500000
                min_trade_val = 5
                min_vol_avg = 10000
            else:
                # 미국 전체 상장 종목
                try:
                    for mkt in ["NASDAQ"]:
                        tmp = get_stocks(mkt)
                        if not tmp.empty:
                            tmp["market"] = mkt
                            all_stocks = pd.concat([all_stocks, tmp], ignore_index=True)
                except:
                    pass
                is_us = True
                max_price = 50  # $50 이하 중소형주
                min_trade_val = 500  # $500만 이상 (만달러 단위)
                min_vol_avg = 50000

            if all_stocks.empty:
                st.warning("종목 리스트를 불러올 수 없습니다.")
            else:
                total = len(all_stocks)
                st.info(f"📋 총 {total:,}개 종목 로드 완료 → 자동 필터링 후 급등 분석 시작")
                progress = st.progress(0)
                status = st.empty()
                results = []
                skipped = 0
                filtered = 0

                for idx, row in all_stocks.iterrows():
                    code = str(row["Code"]).strip()
                    name = str(row["Name"]).strip()
                    mkt_name = row["market"]
                    pct = (idx + 1) / total
                    if idx % 50 == 0 or idx == total - 1:
                        progress.progress(pct)
                        status.text(f"분석 중... {idx+1:,}/{total:,} | [{mkt_name}] {name} | 후보 {len(results)}개 발견")

                    try:
                        df = fetch(code, days=300)
                        if df is None or len(df) < 60:
                            continue

                        price_val = df["Close"].iloc[-1]
                        volume = df["Volume"]
                        vol_mean = volume.rolling(20).mean().iloc[-1]

                        # ── 자동 필터 ──
                        # 1) 가격 필터
                        if is_us:
                            if price_val > max_price or price_val < 1:
                                continue
                        else:
                            if price_val > max_price or price_val < 300:
                                continue

                        # 2) 거래량 평균 대비 1.5배 미만 스킵
                        vol_ratio = volume.iloc[-1] / (vol_mean + 1e-9)
                        if vol_ratio < 1.5:
                            continue

                        # 3) 거래대금 필터
                        if is_us:
                            trade_val = price_val * volume.iloc[-1] / 1e4  # 만달러
                            if trade_val < min_trade_val:
                                continue
                        else:
                            trade_val = price_val * volume.iloc[-1] / 1e8  # 억원
                            if trade_val < min_trade_val:
                                continue

                        # 4) 평균 거래량 자체가 너무 적으면 스킵
                        if vol_mean < min_vol_avg:
                            continue

                        filtered += 1

                        r = analyze_surge(df, code, name)
                        if r and r["score"] >= 40:
                            r["market"] = mkt_name
                            r["is_us"] = is_us
                            results.append(r)
                    except:
                        skipped += 1
                        continue

                progress.empty()
                status.empty()

                st.caption(f"전체 {total:,}개 중 필터 통과 {filtered:,}개 → 급등 후보 {len(results)}개")
                if not results:
                    st.info("오늘은 급등 후보가 없습니다. 장 마감 후보다 장중에 다시 시도해 보세요.")
                else:
                    results.sort(key=lambda x: x["score"], reverse=True)
                    # 자동 저장
                    country_label = "한국" if "한국" in surge_country else "미국"
                    add_surge_record(results, country_label)
                    st.success(f"🎯 급등 후보 {len(results)}개 발견!")



                    for i, r in enumerate(results):
                        grade_color = {"S": "#ff1744", "A": "#ff9100", "B": "#ffc107", "C": "#90a4ae"}.get(r["grade"], "#888")
                        chg_color = "#ff1744" if r["change"] > 0 else "#2979ff" if r["change"] < 0 else "#888"
                        is_us_stock = r.get("is_us", False)
                        currency = "$" if is_us_stock else ""
                        unit = "" if is_us_stock else "원"
                        tv_unit = "만$" if is_us_stock else "억"

                        theme_txt = f" [{r['theme']}]" if r.get('theme') else ""
                        mkt_txt = r.get('market', '')
                        tv_unit = "만$" if is_us_stock else "억"
                        chg_icon = "🔴" if r['change'] > 0 else "🔵" if r['change'] < 0 else "⚪"
                        grade_emoji = {"S": "🔥", "A": "⚡", "B": "👀", "C": "🔍"}.get(r['grade'], "")

                        st.markdown(f"""**{i+1}. {grade_emoji} {r['name']}** ({r['code']}) `{mkt_txt}`{theme_txt}

{chg_icon} **{currency}{r['price']:,}{unit}** ({r['change']:+.2f}%) | 거래량 {r['vol_ratio']}x | 거래대금 {r['trade_val']:.0f}{tv_unit} | RSI {r['rsi']}

**등급: {r['grade']} ({r['score']}점)** — {r['verdict']}
""")
                        with st.expander(f"📡 감지 신호 ({len(r['signals'])}개)", expanded=(i < 3)):
                            for sig in r["signals"]:
                                st.markdown(f"- {sig}")
                            if is_us_stock:
                                st.markdown(f"**지지** ${r['support']:,} | **저항** ${r['resist']:,} | **손절** ${r['stop_loss']:,}")
                            else:
                                st.markdown(f"**지지** {r['support']:,}원 | **저항** {r['resist']:,}원 | **손절** {r['stop_loss']:,}원")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button(f"📊 차트", key=f"surge_chart_{r['code']}_{i}"):
                                cfg_temp = {"tp": 10, "sl": 5}
                                fig = draw_chart(r, cfg_temp)
                                st.pyplot(fig)
                                plt.close(fig)
                        with col_b:
                            wl = []
                            if os.path.exists(WL_FILE):
                                with open(WL_FILE, "r", encoding="utf-8") as f:
                                    wl = json.load(f)
                            if st.button(f"⭐ 관심종목", key=f"surge_wl_{r['code']}_{i}"):
                                item = {"code": r["code"], "name": r["name"]}
                                if item not in wl:
                                    wl.append(item)
                                    with open(WL_FILE, "w", encoding="utf-8") as f:
                                        json.dump(wl, f, ensure_ascii=False, indent=2)
                                    st.success(f"{r['name']} 관심종목 추가!")
                                else:
                                    st.info("이미 관심종목에 있어요")

                        st.divider()

                        st.divider()

        # ── 급등 사냥 기록 ──
        st.divider()
        st.subheader("📜 급등 사냥 기록")
        surge_hist = load_surge_history()
        if not surge_hist:
            st.info("아직 급등 사냥 기록이 없습니다.")
        else:
            # 현재가 자동 갱신 버튼
            if st.button("🔄 현재가 업데이트", key="surge_hist_refresh"):
                with st.spinner("기록 종목 현재가 조회 중..."):
                    from pykrx import stock as pkstock
                    from datetime import datetime, timedelta
                    today = datetime.now().strftime("%Y%m%d")
                    ago = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
                    updated = 0
                    failed = []
                    for rec in surge_hist[:5]:
                        if rec.get("country") == "미국":
                            continue
                        for s in rec["stocks"]:
                            try:
                                tmp = pkstock.get_market_ohlcv(ago, today, s["code"])
                                if tmp is not None and len(tmp) > 0:
                                    now_price = int(tmp["종가"].iloc[-1])
                                    old_price = s["price"]
                                    if old_price > 0:
                                        pnl = round((now_price / old_price - 1) * 100, 2)
                                        s["now_price"] = now_price
                                        s["pnl"] = pnl
                                        updated += 1
                                else:
                                    failed.append(s["name"])
                            except Exception as e:
                                failed.append(s["name"])
                    save_surge_history(surge_hist)
                st.session_state["surge_refresh_msg"] = f"업데이트 완료! 성공 {updated}개"
                if failed:
                    st.session_state["surge_refresh_msg"] += f" / 실패: {', '.join(failed[:10])}"
                st.rerun()

            if "surge_refresh_msg" in st.session_state:
                st.success(st.session_state.pop("surge_refresh_msg"))


            for si, rec in enumerate(surge_hist[:10]):
                flag = "🇰🇷" if rec["country"] == "한국" else "🇺🇸"
                with st.expander(f"{flag} {rec['date']} — {rec['count']}개 발견", expanded=(si == 0)):
                    for sj, s in enumerate(rec["stocks"]):
                        grade_emoji = {"S": "🔥", "A": "⚡", "B": "👀", "C": "🔍"}.get(s["grade"], "")

                        # 수익률 표시
                        pnl_text = ""
                        if "now_price" in s and "pnl" in s:
                            pnl = s["pnl"]
                            now_p = s["now_price"]
                            if pnl > 0:
                                pnl_text = f" → 현재 {now_p:,} (**+{pnl}%** 🟢)"
                            elif pnl < 0:
                                pnl_text = f" → 현재 {now_p:,} (**{pnl}%** 🔴)"
                            else:
                                pnl_text = f" → 현재 {now_p:,} (0% ⚪)"

                        st.markdown(
                            f"{grade_emoji} **{s['name']}** ({s['code']}) `{s.get('market','')}` — "
                            f"{s['grade']} {s['score']}점 | 발견가 {s['price']:,} ({s['change']:+.2f}%){pnl_text}"
                        )
                        for sig in s.get("signals", [])[:3]:
                            st.caption(f"  {sig}")
                        # ── 기록 종목 버튼 ──
                        hcol1, hcol2, hcol3 = st.columns(3)
                        with hcol1:
                            if st.button("📊 차트", key=f"hist_chart_{si}_{sj}"):
                                hist_df = fetch(s["code"], days=200)
                                if hist_df is not None:
                                    cfg_temp = {"tp": 10, "sl": 5}
                                    mkt = s.get("market", "KOSDAQ")
                                    hist_r = analyze(hist_df, s["code"], s["name"], cfg_temp, market=mkt)
                                    fig = draw_chart(hist_r, cfg_temp)
                                    st.pyplot(fig)
                                    plt.close(fig)
                                else:
                                    st.error("데이터를 가져올 수 없습니다")
                        with hcol2:
                            if st.button("📡 정밀분석", key=f"hist_detail_{si}_{sj}"):
                                with st.spinner(f"{s['name']} 재분석 중..."):
                                    hist_df = fetch(s["code"], days=300)
                                    if hist_df is not None:
                                        mkt = s.get("market", "KOSDAQ")
                                        for style_name, cfg in STYLES.items():
                                            detail_r = analyze(hist_df, s["code"], s["name"], cfg, market=mkt)
                                            detail_r["crown"] = ""
                                            st.markdown(f"**{style_name}**")
                                            show_card(detail_r, f"hist_detail_{si}_{sj}_{cfg['key']}", cfg)
                                    else:
                                        st.error("데이터를 가져올 수 없습니다")
                        with hcol3:
                            if st.button("⭐ 관심종목", key=f"hist_wl_{si}_{sj}"):
                                add_to_wl(s["code"], s["name"])
                                st.success(f"{s['name']} 추가!")
            if st.button("🗑️ 급등 기록 전체 삭제", key="surge_hist_clear"):
                save_surge_history([])
                st.rerun()


elif menu == "🌊 외국인 수급 추적":
    st.markdown("## 🌊 외국인 수급 추적 (전인구 전략)")
    st.caption("외국인이 야금야금 모으는 종목 & 대장주 이탈을 자동 탐지합니다")

    if st.button("🔍 외국인 수급 스캔 시작", use_container_width=True, type="primary"):
        with st.spinner("외국인 수급 데이터 수집 중..."):
            try:
                from bs4 import BeautifulSoup

                url = "https://www.truefriend.com/main/research/research/Sell.jsp"
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                resp.encoding = "utf-8"
                soup = BeautifulSoup(resp.text, "html.parser")

                tables = soup.find_all("table")

                def parse_truefriend_table(table):
                    results = []
                    rows = table.find_all("tr")
                    for row in rows:
                        cols = row.find_all("td")
                        if len(cols) >= 4:
                            sell_name = cols[0].get_text(strip=True)
                            sell_amount = cols[1].get_text(strip=True)
                            buy_name = cols[2].get_text(strip=True)
                            buy_amount = cols[3].get_text(strip=True)
                            if sell_name and buy_name:
                                results.append({
                                    "sell_name": sell_name,
                                    "sell_amount": sell_amount,
                                    "buy_name": buy_name,
                                    "buy_amount": buy_amount
                                })
                    return results

                all_data = []
                for t in tables:
                    parsed = parse_truefriend_table(t)
                    if len(parsed) >= 3:
                        all_data.append(parsed)

                if len(all_data) >= 2:
                    inst_data = all_data[0]
                    foreign_data = all_data[1]

                    # ── 4개 컬럼 가로 배치 ──
                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        st.markdown("### 📈 외국인 순매수")
                        for item in foreign_data:
                            name = item["buy_name"]
                            amount = item["buy_amount"]
                            st.markdown(
                                f'<div style="background:#1a2e1a;padding:8px;'
                                f'border-radius:8px;margin:4px 0;'
                                f'border-left:4px solid #4caf50;font-size:13px">'
                                f'🟢 <b>{name}</b><br>'
                                f'<span style="color:#4caf50">{amount}백만</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    with c2:
                        st.markdown("### 📉 외국인 순매도")
                        for item in foreign_data:
                            name = item["sell_name"]
                            amount = item["sell_amount"]
                            st.markdown(
                                f'<div style="background:#2e1a1a;padding:8px;'
                                f'border-radius:8px;margin:4px 0;'
                                f'border-left:4px solid #f44336;font-size:13px">'
                                f'🔴 <b>{name}</b><br>'
                                f'<span style="color:#f44336">{amount}백만</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    with c3:
                        st.markdown("### 🏢 기관 순매수")
                        for item in inst_data:
                            name = item["buy_name"]
                            amount = item["buy_amount"]
                            st.markdown(
                                f'<div style="background:#1a1a2e;padding:8px;'
                                f'border-radius:8px;margin:4px 0;'
                                f'border-left:4px solid #2196f3;font-size:13px">'
                                f'🟦 <b>{name}</b><br>'
                                f'<span style="color:#2196f3">{amount}억</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    with c4:
                        st.markdown("### 📉 기관 순매도")
                        for item in inst_data:
                            name = item["sell_name"]
                            amount = item["sell_amount"]
                            st.markdown(
                                f'<div style="background:#2e1a1a;padding:8px;'
                                f'border-radius:8px;margin:4px 0;'
                                f'border-left:4px solid #ff5722;font-size:13px">'
                                f'🟠 <b>{name}</b><br>'
                                f'<span style="color:#ff5722">{amount}억</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    # ── 동시 매수/매도 시그널 (하단 전체 폭) ──
                    st.markdown("---")

                    overlap_buy = []
                    for item in foreign_data:
                        for inst_item in inst_data:
                            if item["buy_name"] == inst_item["buy_name"]:
                                overlap_buy.append(item["buy_name"])

                    overlap_sell = []
                    for item in foreign_data:
                        for inst_item in inst_data:
                            if item["sell_name"] == inst_item["sell_name"]:
                                overlap_sell.append(item["sell_name"])

                    sig1, sig2 = st.columns(2)
                    with sig1:
                        if overlap_buy:
                            names = ", ".join(overlap_buy)
                            st.markdown(
                                f'<div style="background:#1a3a1a;padding:15px;'
                                f'border-radius:10px;'
                                f'border:2px solid #4caf50">'
                                f'🎯 <b>외국인+기관 동시 순매수</b><br>'
                                f'<span style="font-size:18px;color:#4caf50">{names}</span><br>'
                                f'<small>→ 전인구 전략: 강력 매수 시그널!</small>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.info("외국인+기관 동시 순매수 종목 없음")

                    with sig2:
                        if overlap_sell:
                            names = ", ".join(overlap_sell)
                            st.markdown(
                                f'<div style="background:#3a1a1a;padding:15px;'
                                f'border-radius:10px;'
                                f'border:2px solid #f44336">'
                                f'⚠️ <b>외국인+기관 동시 순매도</b><br>'
                                f'<span style="font-size:18px;color:#f44336">{names}</span><br>'
                                f'<small>→ 대장주 이탈 경고!</small>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.info("외국인+기관 동시 순매도 종목 없음")

                    st.success("✅ 외국인 수급 스캔 완료! (데이터: 한국투자증권)")
                else:
                    st.warning("데이터를 파싱할 수 없습니다.")

            except Exception as e:
                st.error(f"❌ 스캔 오류: {e}")


# ── 푸터 ──
st.divider()
st.caption(f"🔥 급등 예측 탐색기 v24.0 PRO | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | ⚠️ 본 앱은 참고용이며 투자 판단의 책임은 본인에게 있습니다.")

