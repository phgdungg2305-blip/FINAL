import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import os
import io

# ==============================================================================
# 1. CẤU HÌNH TRANG WEB & THEME
# ==============================================================================
st.set_page_config(
    page_title="Quantitative Trading: SMA + OBV Strategy Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại phong cách FinTech
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #546E7A;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .highlight-box {
        background-color: #E3F2FD;
        border-radius: 8px;
        padding: 14px 18px;
        border: 1px solid #90CAF9;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. HẰNG SỐ MẶC ĐỊNH & THAM SỐ ĐÃ TỐI ƯU TỪ NOTEBOOK
# ==============================================================================
DEFAULT_TRAIN_START = "2020-01-01"
DEFAULT_TRAIN_END   = "2021-12-31"
DEFAULT_TEST_START  = "2022-01-01"
DEFAULT_TEST_END    = "2022-12-30"

DEFAULT_PRESETS = {
    "DIG": {"ma_short": 55, "ma_long": 120, "obv_window": 10},
    "DPM": {"ma_short": 30, "ma_long": 290, "obv_window": 55},
    "KBC": {"ma_short": 40, "ma_long": 130, "obv_window": 15},
    "MSN": {"ma_short": 60, "ma_long": 100, "obv_window": 5},
    "DCM": {"ma_short": 35, "ma_long": 260, "obv_window": 15}
}

DEFAULT_TOP5 = ["DIG", "DPM", "KBC", "MSN", "DCM"]


# ==============================================================================
# 3. HÀM TÍNH TOÁN KỸ THUẬT & LOGIC CHIẾN LƯỢC (QUANT ENGINE)
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_and_clean_data(file_source):
    """
    Tải và làm sạch dữ liệu giá sàn HOSE.
    Hỗ trợ cả đường dẫn file hoặc UploadedFile từ Streamlit.
    """
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            return None
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)
    else:
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)

    df.columns = df.columns.str.strip().str.lower()
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc: {col}")

    # Chuyển đổi định dạng ngày và số
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Loại bỏ missing values và duplicates
    df = df.dropna(subset=required)
    df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    return df


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Tính On-Balance Volume (OBV) chuẩn."""
    price_change = close.diff()
    direction = np.where(price_change > 0, 1.0, np.where(price_change < 0, -1.0, 0.0))
    direction[0] = 0.0
    obv = (direction * volume).cumsum()
    return pd.Series(obv, index=close.index, name="OBV")


def calculate_ta_features(df_train: pd.DataFrame, min_obs: int = 400, trading_days: int = 252) -> pd.DataFrame:
    """
    Sàng lọc cổ phiếu theo 4 trụ cột kỹ thuật:
    1. Trend (SMA50 vs SMA200, Price vs SMA200)
    2. Momentum (6 tháng ~ 126 phiên)
    3. Volume (Liquidity TB & OBV Confirmation 20 phiên)
    4. Risk (Volatility & Max Drawdown)
    """
    results = []
    for ticker, g in df_train.groupby("ticker"):
        g = g.sort_values("date").dropna(subset=["close", "volume"])
        if len(g) < min_obs:
            continue

        close = g["close"].astype(float)
        volume = g["volume"].astype(float)

        # 1. Trend
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        trend_strength = (sma50 > sma200).mean()
        price_vs_sma200 = (close.iloc[-1] / sma200.iloc[-1] - 1) if pd.notna(sma200.iloc[-1]) and sma200.iloc[-1] != 0 else np.nan

        # 2. Momentum
        momentum_6m = (close.iloc[-1] / close.iloc[-126] - 1) if len(close) >= 126 else np.nan

        # 3. Volume
        avg_trading_val = (close * volume).mean()
        obv = calculate_obv(close, volume)
        if len(obv) >= 20:
            vol20 = volume.iloc[-20:].sum()
            obv_conf = (obv.iloc[-1] - obv.iloc[-20]) / vol20 if vol20 != 0 else np.nan
        else:
            obv_conf = np.nan

        # 4. Risk
        ret = close.pct_change().dropna()
        annual_vol = ret.std() * np.sqrt(trading_days)
        equity = (1 + ret).cumprod()
        mdd = (equity / equity.cummax() - 1).min()

        results.append({
            "Ticker": ticker,
            "N": len(g),
            "Trend_Strength": trend_strength,
            "Price_vs_SMA200": price_vs_sma200,
            "Momentum_6M": momentum_6m,
            "Avg_Trading_Value": avg_trading_val,
            "OBV_Confirmation": obv_conf,
            "Volatility": annual_vol,
            "Max_Drawdown": mdd
        })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        return res_df
    res_df = res_df.dropna()
    return res_df


def rank_ta_scores(features_df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa và tính điểm TA Score tổng hợp."""
    df = features_df.copy()
    if df.empty:
        return df

    # 1. Trend Score (30%)
    trend_rank1 = df["Trend_Strength"].rank(pct=True)
    trend_rank2 = df["Price_vs_SMA200"].rank(pct=True)
    df["Trend_Score"] = 0.5 * trend_rank1 + 0.5 * trend_rank2

    # 2. Momentum Score (25%)
    df["Momentum_Score"] = df["Momentum_6M"].rank(pct=True)

    # 3. Volume Score (25%)
    liq_rank = df["Avg_Trading_Value"].rank(pct=True)
    obv_rank = df["OBV_Confirmation"].rank(pct=True)
    df["Volume_Score"] = 0.6 * liq_rank + 0.4 * obv_rank

    # 4. Risk Score (20%) - Volatility thấp hơn rank cao hơn; MDD gần 0 hơn rank cao hơn
    vol_rank = df["Volatility"].rank(pct=True, ascending=False)
    mdd_rank = df["Max_Drawdown"].rank(pct=True)
    df["Risk_Score"] = 0.5 * vol_rank + 0.5 * mdd_rank

    # Total Score
    df["TA_Score"] = (
        0.30 * df["Trend_Score"] +
        0.25 * df["Momentum_Score"] +
        0.25 * df["Volume_Score"] +
        0.20 * df["Risk_Score"]
    )
    df = df.sort_values("TA_Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df


def find_position_sma(df: pd.DataFrame, ma_short: int, ma_long: int) -> pd.Series:
    """Tín hiệu SMA: BUY khi cắt lên, SELL khi cắt xuống."""
    position = pd.Series(0.0, index=df.index, name="SMA_signal")
    if ma_short >= ma_long:
        return position

    sma_s = df["Close"].rolling(window=ma_short).mean()
    sma_l = df["Close"].rolling(window=ma_long).mean()

    buy = (sma_s > sma_l) & (sma_s.shift(1) <= sma_l.shift(1))
    sell = (sma_s < sma_l) & (sma_s.shift(1) >= sma_l.shift(1))

    position.loc[buy] = 1.0
    position.loc[sell] = -1.0
    return position


def find_position_obv(df: pd.DataFrame, obv_window: int) -> pd.Series:
    """Tín hiệu OBV: BUY khi OBV cắt lên OBV-MA, SELL khi cắt xuống."""
    position = pd.Series(0.0, index=df.index, name="OBV_signal")
    obv = calculate_obv(df["Close"], df["Volume"])
    obv_ma = obv.rolling(window=obv_window).mean()

    buy = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    position.loc[buy] = 1.0
    position.loc[sell] = -1.0
    return position


def find_position_or(df: pd.DataFrame, ma_short: int, ma_long: int, obv_window: int) -> pd.Series:
    """
    Kết hợp tín hiệu SMA và OBV theo logic OR:
    - BUY nếu SMA BUY hoặc OBV BUY
    - SELL nếu SMA SELL hoặc OBV SELL
    - Nếu cả 2 xung đột cùng ngày thì tín hiệu = 0
    """
    sma_sig = find_position_sma(df, ma_short, ma_long)
    obv_sig = find_position_obv(df, obv_window)

    position = pd.Series(0.0, index=df.index, name="SMA_OBV_OR")
    buy = (sma_sig == 1.0) | (obv_sig == 1.0)
    sell = (sma_sig == -1.0) | (obv_sig == -1.0)
    conflict = buy & sell

    position.loc[buy & ~conflict] = 1.0
    position.loc[sell & ~conflict] = -1.0
    return position


def events_to_holding(events: pd.Series) -> pd.Series:
    """Chuyển đổi tín hiệu thành vị thế nắm giữ liên tục (0: Tiền mặt, 1: Nắm giữ cổ phiếu)."""
    holding = pd.Series(0.0, index=events.index)
    current = 0.0
    for i, sig in enumerate(events):
        if sig == 1.0:
            current = 1.0
        elif sig == -1.0:
            current = 0.0
        holding.iloc[i] = current
    return holding


def strategy_returns(df: pd.DataFrame, events: pd.Series, commission: float = 0.0):
    """
    Tính tỷ suất sinh lời của chiến lược:
    - Tránh look-ahead bias: Tín hiệu ngày t chỉ giao dịch và sinh lời ở phiên t+1
    - Khấu trừ phí giao dịch dựa trên turnover
    """
    asset_ret = df["Close"].pct_change().fillna(0.0)
    holding = events_to_holding(events)
    executed_holding = holding.shift(1).fillna(0.0)
    strat_ret = executed_holding * asset_ret
    turnover = executed_holding.diff().abs().fillna(executed_holding.abs())
    strat_ret = strat_ret - turnover * commission
    return strat_ret, executed_holding


def performance_stats(returns: pd.Series, trading_days: int = 252) -> dict:
    """Tính toán toàn diện các chỉ số hiệu quả định lượng."""
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return {
            "Mean Daily Return [%]": np.nan,
            "Total Return [%]": np.nan,
            "Annual Return [%]": np.nan,
            "Annual Volatility [%]": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown [%]": np.nan,
            "Win Rate [%]": np.nan,
            "Profit Factor": np.nan
        }

    equity = (1 + r).cumprod()
    tot_ret = (equity.iloc[-1] - 1) * 100
    years = len(r) / trading_days
    ann_ret = ((equity.iloc[-1] ** (1 / years) - 1) * 100) if (years > 0 and equity.iloc[-1] > 0) else np.nan
    ann_vol = (r.std() * np.sqrt(trading_days)) * 100
    sharpe = (r.mean() / r.std() * np.sqrt(trading_days)) if r.std() != 0 else np.nan
    dd = (equity / equity.cummax() - 1) * 100
    max_dd = dd.min()

    # Thêm Win Rate & Profit Factor
    active_trades = r[r != 0]
    win_rate = (len(active_trades[active_trades > 0]) / len(active_trades) * 100) if len(active_trades) > 0 else 0.0
    gross_profits = active_trades[active_trades > 0].sum()
    gross_losses = abs(active_trades[active_trades < 0].sum())
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else np.nan

    return {
        "Mean Daily Return [%]": r.mean() * 100,
        "Total Return [%]": tot_ret,
        "Annual Return [%]": ann_ret,
        "Annual Volatility [%]": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": max_dd,
        "Win Rate [%]": win_rate,
        "Profit Factor": profit_factor
    }


def evaluate_or(df: pd.DataFrame, ma_short: int, ma_long: int, obv_window: int, commission: float = 0.0):
    events = find_position_or(df, ma_short, ma_long, obv_window)
    ret, holding = strategy_returns(df, events, commission=commission)
    stats = performance_stats(ret)
    return {
        "events": events,
        "holding": holding,
        "returns": ret,
        "stats": stats
    }


def evaluate_test_with_warmup(
    full_stock_df: pd.DataFrame,
    ma_short: int,
    ma_long: int,
    obv_window: int,
    test_start: str,
    test_end: str,
    warmup_periods: int = 349,
    commission: float = 0.0
):
    """
    Quy tắc warm-up cho Test:
    Lấy đúng 349 phiên giao dịch trước TEST_START để tính SMA350 từ phiên đầu năm Test.
    Tuyệt đối không tính lợi nhuận warm-up vào kết quả Test.
    """
    df = full_stock_df.sort_index().copy()
    test_mask = df.index >= pd.Timestamp(test_start)
    if not test_mask.any():
        return None

    first_test_pos = np.where(test_mask)[0][0]
    warmup_start_pos = max(0, first_test_pos - warmup_periods)
    warmup_and_test = df.iloc[warmup_start_pos:].copy()
    warmup_and_test = warmup_and_test.loc[warmup_and_test.index <= pd.Timestamp(test_end)].copy()

    full_res = evaluate_or(warmup_and_test, ma_short, ma_long, obv_window, commission=commission)
    test_events = full_res["events"].loc[test_start:test_end]
    test_holding = full_res["holding"].loc[test_start:test_end]
    test_returns = full_res["returns"].loc[test_start:test_end]

    return {
        "events": test_events,
        "holding": test_holding,
        "returns": test_returns,
        "stats": performance_stats(test_returns),
        "warmup_and_test_df": warmup_and_test
    }


# ==============================================================================
# 4. TỐI ƯU HÓA DANH MỤC HIỆN ĐẠI (MPT - MARKOWITZ)
# ==============================================================================
def optimize_mpt_sharpe(returns_matrix: pd.DataFrame, risk_free_rate: float = 0.0, trading_days: int = 252):
    """Tối ưu hóa trọng số danh mục theo Markowitz (SLSQP Maximize Sharpe)."""
    n = returns_matrix.shape[1]
    mean_daily = returns_matrix.mean().values
    cov_daily = returns_matrix.cov().values

    def neg_sharpe(weights):
        p_ret = float(weights @ mean_daily * trading_days)
        p_var = float(weights @ (cov_daily * trading_days) @ weights)
        p_vol = float(np.sqrt(max(p_var, 0.0)))
        if p_vol <= 1e-8:
            return 1e9
        return -(p_ret - risk_free_rate) / p_vol

    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        return res.x
    return x0


# ==============================================================================
# 5. SIDEBAR: ĐIỀU KHIỂN & CẤU HÌNH THAM SỐ
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.title("Bảng Điều Khiển")
    st.markdown("---")

    # 1. Nguồn Dữ Liệu
    st.subheader("1. Dữ Liệu Đầu Vào")
    default_files = ["HOSE_2020_2023_in.csv", "HOSE_2020_2023_in (1).csv"]
    existing_file = next((f for f in default_files if os.path.exists(f)), None)

    data_option = st.radio(
        "Chọn nguồn file:",
        options=["File mặc định trong hệ thống", "Tải file CSV lên"] if existing_file else ["Tải file CSV lên"],
        index=0
    )

    uploaded_file = None
    if data_option == "Tải file CSV lên":
        uploaded_file = st.file_uploader("Upload CSV dữ liệu HOSE:", type=["csv"])
        active_file_source = uploaded_file
    else:
        active_file_source = existing_file

    st.markdown("---")

    # 2. Thiết lập giai đoạn
    st.subheader("2. Giai Đoạn Kiểm Định")
    train_c1, train_c2 = st.columns(2)
    with train_c1:
        train_start = st.text_input("Train Start", value=DEFAULT_TRAIN_START)
    with train_c2:
        train_end = st.text_input("Train End", value=DEFAULT_TRAIN_END)

    test_c1, test_c2 = st.columns(2)
    with test_c1:
        test_start = st.text_input("Test Start", value=DEFAULT_TEST_START)
    with test_c2:
        test_end = st.text_input("Test End", value=DEFAULT_TEST_END)

    warmup_periods = st.number_input("Số phiên Warm-up trước Test", min_value=100, max_value=500, value=349, step=10)

    st.markdown("---")

    # 3. Quản lý vốn & Chi phí
    st.subheader("3. Vốn & Chi Phí")
    initial_capital = st.number_input("Vốn ban đầu (VND)", min_value=100_000, value=1_000_000, step=100_000, format="%d")
    commission_pct = st.number_input("Phí giao dịch mỗi vòng (%)", min_value=0.0, max_value=2.0, value=0.0, step=0.05, format="%.2f")
    commission_rate = commission_pct / 100.0

    st.markdown("---")
    st.caption("🚀 Phát triển bởi Antigravity Quant Team • Python & Streamlit")


# ==============================================================================
# 6. TẢI DỮ LIỆU & KIỂM TRA TÍNH TOÀN VẸN
# ==============================================================================
if active_file_source is None:
    st.info("👋 Chào mừng bạn đến với Ứng Dụng Kiểm Định Chiến Lược SMA + OBV!")
    st.warning("Vui lòng tải lên file dữ liệu CSV (HOSE) ở thanh bên trái để bắt đầu phân tích.")
    st.stop()

with st.spinner("Đang xử lý dữ liệu lịch sử sàn HOSE..."):
    try:
        df_full = load_and_clean_data(active_file_source)
    except Exception as e:
        st.error(f"Lỗi khi đọc file CSV: {str(e)}")
        st.stop()

if df_full is None or df_full.empty:
    st.error("Không tìm thấy dữ liệu hợp lệ trong file. Vui lòng kiểm tra định dạng!")
    st.stop()

all_available_tickers = sorted(df_full["ticker"].unique().tolist())


# ==============================================================================
# 7. GIAO DIỆN CHÍNH: TIÊU ĐỀ & CÁC TAB CHỨC NĂNG
# ==============================================================================
st.markdown('<div class="main-header">📈 Hệ Thống Kiểm Định Định Lượng: Chiến Lược Kết Hợp SMA + OBV</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Phân tích kỹ thuật định lượng • Tối ưu hóa siêu tham số • Phân bổ danh mục Markowitz (MPT) • Kiểm định thực nghiệm Out-Of-Sample trên sàn HOSE</div>', unsafe_allow_html=True)

# Lọc dữ liệu Train
train_mask = (df_full["date"] >= pd.Timestamp(train_start)) & (df_full["date"] <= pd.Timestamp(train_end))
train_df_full = df_full[train_mask].copy()

# Tabs
tab_overview, tab_screening, tab_single_stock, tab_portfolio, tab_report = st.tabs([
    "📊 1. Tổng Quan & Dữ Liệu",
    "🎯 2. Sàng Lọc Kỹ Thuật (TA Score)",
    "⚡ 3. Kiểm Định Từng Cổ Phiếu",
    "💼 4. Danh Mục MPT vs Equal Weight",
    "📑 5. Báo Cáo & Kết Luận Thực Nghiệm"
])


# ==============================================================================
# TAB 1: TỔNG QUAN & DỮ LIỆU
# ==============================================================================
with tab_overview:
    st.markdown("### 📌 Bối Cảnh Nghiên Cứu & Phương Pháp Luận")
    st.markdown("""
    Hệ thống này kiểm định tính hiệu quả của chiến lược giao dịch kết hợp **Chỉ báo Xu hướng (SMA)** và **Chỉ báo Dòng tiền Khối lượng (OBV)** trên thị trường chứng khoán Việt Nam:
    - **Logic kết hợp OR**: Tín hiệu MUA kích hoạt khi *SMA MUA hoặc OBV MUA*; Tín hiệu BÁN kích hoạt khi *SMA BÁN hoặc OBV BÁN*. Giúp nắm bắt dòng tiền nhạy bén và cắt lỗ quyết liệt.
    - **Kiểm định Out-of-Sample nghiêm ngặt**: Huấn luyện & tối ưu hóa trên giai đoạn **Uptrend (2020 - 2021)**, kiểm tra thực nghiệm tính phòng thủ trên giai đoạn **Downtrend khắc nghiệt (2022)**.
    - **Loại bỏ Look-ahead Bias**: Tín hiệu ngày $t$ chỉ được thực thi từ ngày $t+1$; Sử dụng cơ chế **349 phiên Warm-up** để chỉ báo SMA dài hạn (lên tới 350) hoạt động ngay từ ngày đầu năm 2022.
    """)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Tổng Số Mã Cổ Phiếu", f"{len(all_available_tickers):,}")
    with col_m2:
        st.metric("Tổng Số Dòng Dữ Liệu", f"{len(df_full):,}")
    with col_m3:
        st.metric("Thời Gian Bắt Đầu", df_full['date'].min().strftime('%d/%m/%Y'))
    with col_m4:
        st.metric("Thời Gian Kết Thúc", df_full['date'].max().strftime('%d/%m/%Y'))

    st.markdown("#### 🔍 Xem trước mẫu dữ liệu:")
    st.dataframe(df_full.head(10), use_container_width=True)


# ==============================================================================
# TAB 2: SÀNG LỌC KỸ THUẬT (TA SCREENING)
# ==============================================================================
with tab_screening:
    st.markdown("### 🎯 Sàng Lọc Cổ Phiếu Dựa Trên 4 Trụ Cột Kỹ Thuật (Train 2020 - 2021)")
    st.markdown("""
    *Nguyên tắc khoa học*: Tuyệt đối không dùng dữ liệu Test để lọc cổ phiếu. Chỉ sử dụng dữ liệu trong giai đoạn Train:
    1. **Trend (30%)**: Sức mạnh xu hướng SMA50 > SMA200 và Vị thế giá so với SMA200.
    2. **Momentum (25%)**: Tỷ suất sinh lời 6 tháng (126 phiên).
    3. **Volume (25%)**: Thanh khoản giao dịch trung bình (60%) và Độ xác nhận OBV 20 phiên (40%).
    4. **Risk (20%)**: Mức biến động hàng năm (thấp hơn tốt hơn) và Mức sụt giảm tối đa (Max Drawdown).
    """)

    min_obs_val = st.slider("Số phiên giao dịch tối thiểu trong kỳ Train:", min_value=200, max_value=500, value=400, step=20)

    with st.spinner("Đang tính toán các chỉ số kỹ thuật và xếp hạng..."):
        ta_feat = calculate_ta_features(train_df_full, min_obs=min_obs_val)
        ta_ranked = rank_ta_scores(ta_feat)

    if not ta_ranked.empty:
        top_n_count = st.slider("Số lượng cổ phiếu trong Top danh mục:", min_value=3, max_value=10, value=5)
        top_candidates = ta_ranked.head(top_n_count).copy()

        st.subheader(f"🏆 Top {top_n_count} Cổ Phiếu Có Điểm TA Score Cao Nhất")
        display_cols = ["Rank", "Ticker", "TA_Score", "Trend_Score", "Momentum_Score", "Volume_Score", "Risk_Score", "Trend_Strength", "Avg_Trading_Value"]
        st.dataframe(
            top_candidates[display_cols].style.format({
                "TA_Score": "{:.3f}",
                "Trend_Score": "{:.3f}",
                "Momentum_Score": "{:.3f}",
                "Volume_Score": "{:.3f}",
                "Risk_Score": "{:.3f}",
                "Trend_Strength": "{:.1%}",
                "Avg_Trading_Value": "{:,.0f}"
            }),
            use_container_width=True
        )

        # Plotly Bar Chart so sánh điểm các thành phần
        fig_score = go.Figure()
        fig_score.add_trace(go.Bar(name='Trend (30%)', x=top_candidates['Ticker'], y=top_candidates['Trend_Score'], marker_color='#1E88E5'))
        fig_score.add_trace(go.Bar(name='Momentum (25%)', x=top_candidates['Ticker'], y=top_candidates['Momentum_Score'], marker_color='#43A047'))
        fig_score.add_trace(go.Bar(name='Volume (25%)', x=top_candidates['Ticker'], y=top_candidates['Volume_Score'], marker_color='#FB8C00'))
        fig_score.add_trace(go.Bar(name='Risk (20%)', x=top_candidates['Ticker'], y=top_candidates['Risk_Score'], marker_color='#E53935'))
        fig_score.update_layout(
            barmode='group',
            title=f"So Sánh Các Điểm Thành Phần của Top {top_n_count} Cổ Phiếu",
            xaxis_title="Cổ Phiếu",
            yaxis_title="Percentile Score (0 -> 1)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_score, use_container_width=True)
    else:
        st.warning("Không có cổ phiếu nào thỏa mãn điều kiện số phiên tối thiểu.")


# ==============================================================================
# TAB 3: KIỂM ĐỊNH TỪNG CỔ PHIẾU (INDIVIDUAL STOCK BACKTEST)
# ==============================================================================
with tab_single_stock:
    st.markdown("### ⚡ Kiểm Định Từng Mã: Chiến Lược SMA + OBV (OR) vs Buy & Hold")

    # Chọn mã để kiểm tra
    col_sel1, col_sel2 = st.columns([2, 3])
    with col_sel1:
        selected_ticker = st.selectbox(
            "Chọn mã cổ phiếu kiểm định:",
            options=all_available_tickers,
            index=all_available_tickers.index("DIG") if "DIG" in all_available_tickers else 0
        )

    # Lấy thông số tối ưu mặc định nếu có
    preset_params = DEFAULT_PRESETS.get(selected_ticker, {"ma_short": 30, "ma_long": 150, "obv_window": 20})

    with col_sel2:
        use_preset = st.checkbox("Sử dụng bộ tham số tối ưu (Nghiên cứu gốc)", value=selected_ticker in DEFAULT_PRESETS)

    # Bộ điều khiển tham số
    p_c1, p_c2, p_c3 = st.columns(3)
    with p_c1:
        ma_s = st.slider("SMA Ngắn (ma_short):", 5, 80, preset_params["ma_short"] if use_preset else 30, step=5)
    with p_c2:
        ma_l = st.slider("SMA Dài (ma_long):", 80, 350, preset_params["ma_long"] if use_preset else 150, step=10)
    with p_c3:
        obv_w = st.slider("Cửa sổ OBV (obv_window):", 5, 80, preset_params["obv_window"] if use_preset else 20, step=5)

    # Chuẩn bị dữ liệu cho mã
    stock_df = df_full[df_full["ticker"] == selected_ticker].sort_values("date").set_index("date")

    # Đánh giá Test với Warm-up 349 phiên
    eval_test = evaluate_test_with_warmup(
        stock_df, ma_s, ma_l, obv_w,
        test_start=test_start, test_end=test_end,
        warmup_periods=warmup_periods, commission=commission_rate
    )

    if eval_test is None or eval_test["returns"].empty:
        st.error(f"Không đủ dữ liệu cho mã {selected_ticker} trong giai đoạn Test.")
    else:
        test_rets = eval_test["returns"]
        test_holding = eval_test["holding"]
        test_events = eval_test["events"]
        test_data_slice = stock_df.loc[test_start:test_end]
        bh_rets = test_data_slice["Close"].pct_change().fillna(0.0)

        strat_stats = eval_test["stats"]
        bh_stats = performance_stats(bh_rets)

        # Hiển thị Metrics so sánh
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        with m_c1:
            diff_ret = strat_stats["Total Return [%]"] - bh_stats["Total Return [%]"]
            st.metric("Total Return (Chiến lược)", f"{strat_stats['Total Return [%]']:.2f}%", delta=f"{diff_ret:+.2f}% vs B&H")
        with m_c2:
            st.metric("Total Return (Buy & Hold)", f"{bh_stats['Total Return [%]']:.2f}%")
        with m_c3:
            diff_sharpe = strat_stats["Sharpe Ratio"] - bh_stats["Sharpe Ratio"]
            st.metric("Sharpe Ratio", f"{strat_stats['Sharpe Ratio']:.3f}", delta=f"{diff_sharpe:+.3f} vs B&H")
        with m_c4:
            st.metric("Max Drawdown (Chiến lược)", f"{strat_stats['Max Drawdown [%]']:.2f}%")
        with m_c5:
            st.metric("Max Drawdown (Buy & Hold)", f"{bh_stats['Max Drawdown [%]']:.2f}%")

        # Biểu đồ giá kỹ thuật với Tín hiệu Mua/Bán
        st.markdown("#### 📈 Biểu Đồ Kỹ Thuật & Tín Hiệu Giao Dịch (Test 2022)")
        sub_df = test_data_slice.copy()
        sub_df["SMA_Short"] = sub_df["Close"].rolling(ma_s).mean()
        sub_df["SMA_Long"] = sub_df["Close"].rolling(ma_l).mean()
        obv_series = calculate_obv(sub_df["Close"], sub_df["Volume"])
        sub_df["OBV"] = obv_series
        sub_df["OBV_MA"] = obv_series.rolling(obv_w).mean()

        fig_tech = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=[f"Giá {selected_ticker} & Tín Hiệu Giao Dịch", "Chỉ Báo On-Balance Volume (OBV)", "Vị Thế Nắm Giữ (1: Long, 0: Cash)"]
        )

        # Giá & SMA
        fig_tech.add_trace(go.Scatter(x=sub_df.index, y=sub_df["Close"], name="Close", line=dict(color="#263238", width=1.5)), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=sub_df.index, y=sub_df["SMA_Short"], name=f"SMA {ma_s}", line=dict(color="#1E88E5", width=1.5)), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=sub_df.index, y=sub_df["SMA_Long"], name=f"SMA {ma_l}", line=dict(color="#E53935", width=1.5)), row=1, col=1)

        # Markers Mua / Bán
        buy_pts = sub_df[test_events == 1.0]
        sell_pts = sub_df[test_events == -1.0]
        fig_tech.add_trace(go.Scatter(
            x=buy_pts.index, y=buy_pts["Close"],
            mode="markers", name="Tín hiệu MUA",
            marker=dict(symbol="triangle-up", size=11, color="#00C853", line=dict(width=1, color="black"))
        ), row=1, col=1)

        fig_tech.add_trace(go.Scatter(
            x=sell_pts.index, y=sell_pts["Close"],
            mode="markers", name="Tín hiệu BÁN",
            marker=dict(symbol="triangle-down", size=11, color="#D50000", line=dict(width=1, color="black"))
        ), row=1, col=1)

        # OBV & OBV MA
        fig_tech.add_trace(go.Scatter(x=sub_df.index, y=sub_df["OBV"], name="OBV", line=dict(color="#8E24AA", width=1.2)), row=2, col=1)
        fig_tech.add_trace(go.Scatter(x=sub_df.index, y=sub_df["OBV_MA"], name=f"OBV MA {obv_w}", line=dict(color="#FB8C00", width=1.2)), row=2, col=1)

        # Vị thế nắm giữ
        fig_tech.add_trace(go.Scatter(
            x=test_holding.index, y=test_holding,
            name="Vị thế", line=dict(color="#00897B", width=1.5), fill="tozeroy"
        ), row=3, col=1)

        fig_tech.update_layout(height=650, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_tech, use_container_width=True)

        # Biểu đồ Đường Cong Tăng Trưởng Vốn (Equity Curve)
        st.markdown("#### 💰 Tăng Trưởng Vốn (Equity Curve): Chiến Lược vs Buy & Hold")
        eq_strat = initial_capital * (1 + test_rets).cumprod()
        eq_bh = initial_capital * (1 + bh_rets).cumprod()

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=eq_strat.index, y=eq_strat, name="Chiến Lược SMA+OBV (OR)", line=dict(color="#1E88E5", width=2.5)))
        fig_eq.add_trace(go.Scatter(x=eq_bh.index, y=eq_bh, name="Mua & Nắm Giữ (Buy & Hold)", line=dict(color="#78909C", width=2, dash="dot")))
        fig_eq.update_layout(
            title=f"Tăng Trưởng Vốn Đầu Tư {selected_ticker} (Vốn ban đầu: {initial_capital:,.0f} VND)",
            xaxis_title="Thời gian",
            yaxis_title="Giá trị tài sản (VND)",
            height=420
        )
        st.plotly_chart(fig_eq, use_container_width=True)


# ==============================================================================
# TAB 4: DANH MỤC MPT VS EQUAL WEIGHT (PORTFOLIO ALLOCATION)
# ==============================================================================
with tab_portfolio:
    st.markdown("### 💼 Phân Bổ Danh Mục: Equal Weight vs Tối Ưu MPT (Markowitz)")
    st.markdown("""
    Đánh giá sức mạnh danh mục trên **Test 2022**:
    - **Equal Weight (Đẳng trọng số)**: 20% chia đều cho 5 mã cổ phiếu.
    - **MPT (Modern Portfolio Theory)**: Tối đa hóa Sharpe Ratio từ tỷ suất sinh lời của chiến lược trên **Train** (SLSQP).
    - **Equal Weight Buy & Hold Benchmark**: Danh mục đối chứng 20% mỗi mã mua và nắm giữ suốt năm 2022.
    """)

    # Chọn danh sách mã cho danh mục
    default_selected = [t for t in DEFAULT_TOP5 if t in all_available_tickers]
    portfolio_tickers = st.multiselect(
        "Chọn danh sách cổ phiếu đưa vào danh mục:",
        options=all_available_tickers,
        default=default_selected if len(default_selected) >= 3 else all_available_tickers[:5]
    )

    if len(portfolio_tickers) < 2:
        st.warning("Vui lòng chọn ít nhất 2 cổ phiếu để xây dựng danh mục.")
    else:
        # Chuẩn bị ma trận lợi nhuận
        train_returns_dict = {}
        test_returns_dict = {}
        bh_test_returns_dict = {}

        for ticker in portfolio_tickers:
            stk_df = df_full[df_full["ticker"] == ticker].sort_values("date").set_index("date")
            p = DEFAULT_PRESETS.get(ticker, {"ma_short": 30, "ma_long": 150, "obv_window": 20})

            # Train
            train_stk = stk_df.loc[train_start:train_end]
            eval_tr = evaluate_or(train_stk, p["ma_short"], p["ma_long"], p["obv_window"], commission=commission_rate)
            train_returns_dict[ticker] = eval_tr["returns"]

            # Test
            eval_te = evaluate_test_with_warmup(
                stk_df, p["ma_short"], p["ma_long"], p["obv_window"],
                test_start=test_start, test_end=test_end,
                warmup_periods=warmup_periods, commission=commission_rate
            )
            test_returns_dict[ticker] = eval_te["returns"]

            # B&H Test
            te_slice = stk_df.loc[test_start:test_end]
            bh_test_returns_dict[ticker] = te_slice["Close"].pct_change().fillna(0.0)

        train_mat = pd.DataFrame(train_returns_dict).dropna()
        test_mat = pd.DataFrame(test_returns_dict).dropna()
        bh_test_mat = pd.DataFrame(bh_test_returns_dict).dropna()

        # Tính trọng số
        n_assets = len(portfolio_tickers)
        ew_weights = np.repeat(1.0 / n_assets, n_assets)
        mpt_weights = optimize_mpt_sharpe(train_mat)

        # Lợi nhuận danh mục
        ew_test_ret = (test_mat * ew_weights).sum(axis=1)
        mpt_test_ret = (test_mat * mpt_weights).sum(axis=1)
        bh_test_ret = (bh_test_mat * ew_weights).sum(axis=1)

        # Thống kê
        bench_stats = pd.DataFrame({
            "Equal Weight Buy & Hold": performance_stats(bh_test_ret),
            "SMA+OBV Equal Weight": performance_stats(ew_test_ret),
            "SMA+OBV MPT": performance_stats(mpt_test_ret)
        }).T

        st.subheader("📋 Bảng So Sánh Hiệu Quả Danh Mục Trên Test (2022)")
        st.dataframe(
            bench_stats.style.format({
                "Mean Daily Return [%]": "{:.4f}%",
                "Total Return [%]": "{:.2f}%",
                "Annual Return [%]": "{:.2f}%",
                "Annual Volatility [%]": "{:.2f}%",
                "Sharpe Ratio": "{:.4f}",
                "Max Drawdown [%]": "{:.2f}%",
                "Win Rate [%]": "{:.2f}%",
                "Profit Factor": "{:.2f}"
            }),
            use_container_width=True
        )

        # Đồ thị so sánh Equity Curves
        eq_bh_port = initial_capital * (1 + bh_test_ret).cumprod()
        eq_ew_port = initial_capital * (1 + ew_test_ret).cumprod()
        eq_mpt_port = initial_capital * (1 + mpt_test_ret).cumprod()

        fig_port_eq = go.Figure()
        fig_port_eq.add_trace(go.Scatter(
            x=eq_bh_port.index, y=eq_bh_port,
            name="Equal Weight Buy & Hold Benchmark",
            line=dict(color="#B0BEC5", width=2, dash="dash")
        ))
        fig_port_eq.add_trace(go.Scatter(
            x=eq_ew_port.index, y=eq_ew_port,
            name="SMA+OBV OR Equal Weight",
            line=dict(color="#1E88E5", width=2.5)
        ))
        fig_port_eq.add_trace(go.Scatter(
            x=eq_mpt_port.index, y=eq_mpt_port,
            name="SMA+OBV OR MPT Tối Ưu",
            line=dict(color="#43A047", width=3)
        ))
        fig_port_eq.update_layout(
            title="Đường Tăng Trưởng Tài Sản Danh Mục Trên Test 2022 (VND)",
            xaxis_title="Thời Gian",
            yaxis_title="Giá Trị Danh Mục",
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_port_eq, use_container_width=True)

        # So sánh cơ cấu trọng số
        w_df = pd.DataFrame({
            "Ticker": portfolio_tickers,
            "Equal Weight": ew_weights,
            "MPT Weight": mpt_weights
        })
        col_w1, col_w2 = st.columns([1, 1])
        with col_w1:
            st.markdown("#### ⚖️ Cơ Cấu Trọng Số Danh Mục")
            st.dataframe(w_df.style.format({"Equal Weight": "{:.2%}", "MPT Weight": "{:.2%}"}), use_container_width=True)

        with col_w2:
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(name='Equal Weight', x=w_df['Ticker'], y=w_df['Equal Weight'], marker_color='#90CAF9'))
            fig_w.add_trace(go.Bar(name='MPT Tối Ưu', x=w_df['Ticker'], y=w_df['MPT Weight'], marker_color='#2E7D32'))
            fig_w.update_layout(
                barmode='group',
                title="So Sánh Tỷ Trọng Phân Bổ (Equal Weight vs MPT)",
                yaxis_title="Tỷ trọng (%)",
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_w, use_container_width=True)


# ==============================================================================
# TAB 5: BÁO CÁO & KẾT LUẬN THỰC NGHIỆM
# ==============================================================================
with tab_report:
    st.markdown("### 📑 Báo Cáo Tóm Tắt & Kết Luận Thực Nghiệm")

    st.markdown("""
    <div class="highlight-box">
        <h4>💡 Phát Hiện Cốt Lõi Từ Dữ Liệu Nghiên Cứu Sàn HOSE:</h4>
        <ul>
            <li><strong>Khả năng phòng vệ vượt trội trong Down Market:</strong> Năm 2022 là giai đoạn thị trường chứng khoán giảm sâu, danh mục Mua & Nắm Giữ (Buy & Hold) giảm hơn <strong>-50.41%</strong> với Max Drawdown tới <strong>-64.30%</strong>. Trong khi đó, chiến lược kết hợp SMA + OBV bảo toàn vốn ngoạn mục (Equal Weight chỉ giảm -0.85%, còn MPT thậm chí đạt <strong>Lợi nhuận dương +8.08%</strong>).</li>
            <li><strong>Sức mạnh của Logic OR:</strong> Việc kết hợp tín hiệu theo cổng OR thay vì AND giúp nhà đầu tư bắt nhịp dòng tiền sớm (OBV) đồng thời thoát khỏi thị trường nhanh chóng ngay khi xu hướng giá hoặc dòng tiền gãy đổ.</li>
            <li><strong>Vai trò của Phân Bổ MPT:</strong> Mô hình Markowitz lọc bỏ tỷ trọng của các cổ phiếu suy yếu (như DPM về 0%) và tập trung vốn vào các mã có tỷ lệ Sharpe cao (DCM, DIG, MSN, KBC), giúp danh mục sinh lời ngay trong năm thị trường chung sụp đổ.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📥 Xuất Dữ Liệu Báo Cáo")
    csv_buffer = io.StringIO()
    if 'bench_stats' in locals():
        bench_stats.to_csv(csv_buffer)
        st.download_button(
            label="Tải Bảng Kết Quả Kiểm Định (CSV)",
            data=csv_buffer.getvalue(),
            file_name="backtest_summary_report.csv",
            mime="text/csv"
        )
