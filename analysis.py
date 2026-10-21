"""
마우나로아 대기 CO2 농도(월별) 시계열 분석
실행: python analysis.py
결과: images/*.png, results/summary.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "co2_mlo_monthly_raw.csv"
IMG = BASE / "images"
OUT = BASE / "results"
IMG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.unicode_minus": False})
# 한글 폰트: Windows(맑은 고딕) / macOS(AppleGothic) / Linux(Noto CJK) 순서로 있는 것 사용
from matplotlib import font_manager
_avail = {f.name for f in font_manager.fontManager.ttflist}
for _f in ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"]:
    if _f in _avail:
        plt.rcParams["font.family"] = _f
        break
summary = {}

# ------------------------------------------------------------------
# 1. 데이터 로드
#    원본 CSV의 헤더 이름이 실제 열 의미와 어긋나 있어서(열 7개, 헤더 6개 의미 혼재)
#    NOAA 원본 형식(월평균, 계절보정값, 측정일수, 표준편차, 불확도)대로 이름을 다시 붙인다.
# ------------------------------------------------------------------
df = pd.read_csv(RAW, header=0,
                 names=["date", "decimal_date", "co2", "co2_deseason",
                        "ndays", "sdev", "unc"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

# ------------------------------------------------------------------
# 2. 데이터 기본 정보 / 결측치 확인
# ------------------------------------------------------------------
full_range = pd.date_range(df.index.min(), df.index.max(), freq="MS")
summary["n_points"] = int(len(df))
summary["period"] = [str(df.index.min().date()), str(df.index.max().date())]
summary["missing_months"] = int(len(full_range.difference(df.index)))
summary["co2_null"] = int(df["co2"].isna().sum())
# NOAA는 측정 정보가 없을 때 -1, -9.99, -0.99 를 '결측 코드'로 쓴다 → NaN 처리
for col, code in [("ndays", -1), ("sdev", -9.99), ("unc", -0.99)]:
    summary[f"{col}_missing_code"] = int((df[col] == code).sum())
    df.loc[df[col] == code, col] = np.nan
summary["ndays_missing_period"] = [
    str(df[df["ndays"].isna()].index.min().date()),
    str(df[df["ndays"].isna()].index.max().date())]
# 측정일수가 적은 달(<10일)은 대표성이 약할 수 있음 → 제거하지 않고 표시만
summary["low_ndays_months"] = int((df["ndays"] < 10).sum())

# ------------------------------------------------------------------
# 3. 이상치 탐지: 12개월 중심 이동평균(추세+계절 제거용) 대비 잔차의 |z| > 3
#    계절 패턴을 먼저 빼야 '여름마다 떨어지는 것'을 이상치로 오판하지 않는다.
# ------------------------------------------------------------------
ma12c = df["co2"].rolling(12, center=True).mean()
detr = df["co2"] - ma12c
seasonal_profile = detr.groupby(detr.index.month).transform("mean")
resid = detr - seasonal_profile
z = (resid - resid.mean()) / resid.std()
outliers = df[z.abs() > 3]
summary["outliers"] = [{"date": str(i.date()), "co2": float(r.co2),
                        "z": round(float(z[i]), 2), "ndays": None if pd.isna(r.ndays) else int(r.ndays)}
                       for i, r in outliers.iterrows()]
# 처리 기준: 물리적으로 가능한 범위(300~450ppm) 안이고 단발성이면 실제 관측으로 보고 유지

# ------------------------------------------------------------------
# 4. 시계열 분석 기법
#   (1) 이동평균  (2) 전년 동월 대비 변화율(YoY)  (3) 월별 집계(계절성)
#   (4) 10년 구간별 통계  (5) STL 분해 (보너스)
# ------------------------------------------------------------------
df["ma12"] = df["co2"].rolling(12).mean()
df["ma60"] = df["co2"].rolling(60).mean()
df["yoy"] = df["co2"].diff(12)                 # ppm/년
df["yoy_pct"] = df["co2"].pct_change(12) * 100  # %

df["decade"] = (df.index.year // 10) * 10
dec = df.groupby("decade").agg(mean_co2=("co2", "mean"),
                               mean_yoy=("yoy", "mean"),
                               std_yoy=("yoy", "std"),
                               n=("co2", "size")).round(2)
summary["decade_stats"] = dec.reset_index().to_dict(orient="records")

first, last = df["co2"].iloc[0], df["co2"].iloc[-1]
summary["first_value"], summary["last_value"] = float(first), float(last)
summary["total_increase_ppm"] = round(float(last - first), 2)
summary["total_increase_pct"] = round(float((last / first - 1) * 100), 1)

# 선형 vs 2차 추세 적합(가속 여부 확인)
t = (df.index.year + (df.index.month - 1) / 12).values - 1958
p1 = np.polyfit(t, df["co2"], 1)
p2 = np.polyfit(t, df["co2"], 2)
rmse1 = float(np.sqrt(np.mean((np.polyval(p1, t) - df["co2"]) ** 2)))
rmse2 = float(np.sqrt(np.mean((np.polyval(p2, t) - df["co2"]) ** 2)))
summary["linear_fit"] = {"slope_ppm_per_year": round(p1[0], 3), "rmse": round(rmse1, 2)}
summary["quad_fit"] = {"a": round(p2[0], 5), "rmse": round(rmse2, 2)}

# 월별 계절성 (최근 10년, 추세 제거 후)
recent = df.loc[df.index >= df.index.max() - pd.DateOffset(years=10)].copy()
recent["detr"] = recent["co2"] - recent["co2"].rolling(12, center=True).mean()
monthly = recent.groupby(recent.index.month)["detr"].mean().round(2)
summary["seasonal_profile_recent10y"] = monthly.to_dict()
summary["seasonal_amplitude_recent10y"] = round(float(monthly.max() - monthly.min()), 2)

# 계절 진폭이 시간에 따라 달라졌는지: 연도별 (최대-최소)
full_years = df.groupby(df.index.year).filter(lambda g: len(g) == 12)
amp = full_years.groupby(full_years.index.year)["co2"].apply(lambda s: s.max() - s.min())
summary["amp_1960s_mean"] = round(float(amp.loc[1960:1969].mean()), 2)
summary["amp_2015_2025_mean"] = round(float(amp.loc[2015:2025].mean()), 2)

# YoY 최고/최저
yoy = df["yoy"].dropna()
summary["yoy_top5"] = [{"date": str(i.date()), "yoy": round(float(v), 2)}
                       for i, v in yoy.nlargest(5).items()]
summary["yoy_bottom5_after1980"] = [{"date": str(i.date()), "yoy": round(float(v), 2)}
                                    for i, v in yoy[yoy.index >= "1980"].nsmallest(5).items()]
summary["yoy_2020"] = round(float(df.loc["2020", "yoy"].mean()), 2)
summary["yoy_2019"] = round(float(df.loc["2019", "yoy"].mean()), 2)
summary["yoy_2021"] = round(float(df.loc["2021", "yoy"].mean()), 2)
summary["yoy_1998"] = round(float(df.loc["1998", "yoy"].mean()), 2)
summary["yoy_2016"] = round(float(df.loc["2016", "yoy"].mean()), 2)
summary["yoy_2024"] = round(float(df.loc["2024", "yoy"].mean()), 2)

# 400ppm 첫 돌파
summary["first_month_over_400"] = str(df[df["co2"] >= 400].index.min().date())
summary["first_year_mean_over_400"] = int(
    df.groupby(df.index.year)["co2"].mean().loc[lambda s: s >= 400].index.min())

# ------------------------------------------------------------------
# 5. 시각화
# ------------------------------------------------------------------
# 01 원 시계열 + 이동평균
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df.index, df["co2"], lw=0.7, color="#9aa5b1", label="월평균 (원본)")
ax.plot(df.index, df["ma12"], lw=1.8, color="#1f6feb", label="12개월 이동평균")
ax.plot(df.index, np.polyval(p1, t), "--", lw=1, color="#d1242f", label="선형 추세(비교용)")
ax.axhline(400, color="k", lw=0.6, ls=":")
ax.text(df.index[5], 401.5, "400 ppm", fontsize=9)
ax.set_title("마우나로아 대기 CO2 농도 (1958-03 ~ 2026-08)")
ax.set_ylabel("CO2 (ppm)")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(IMG / "01_trend_moving_average.png"); plt.close(fig)

# 02 전년 동월 대비 증가량(YoY)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(df.index, df["yoy"], width=25, color="#8fb8ff")
ax.plot(df.index, df["yoy"].rolling(24, center=True).mean(), color="#0b3d91", lw=1.8,
        label="24개월 이동평균")
for yr, lab in [(1998, "1997-98\n엘니뇨"), (2016, "2015-16\n엘니뇨"), (2024, "2023-24\n엘니뇨"),
                (1992, "1991\n피나투보 분화 후")]:
    ax.annotate(lab, (pd.Timestamp(f"{yr}-06-01"), df.loc[str(yr), "yoy"].max()),
                xytext=(0, 18), textcoords="offset points", ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.6))
ax.set_title("전년 동월 대비 CO2 증가량 (ppm/년)")
ax.set_ylabel("ppm / 년")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(IMG / "02_yoy_growth.png"); plt.close(fig)

# 03 월별 계절성 (최근 10년)
mon_lbl = ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]
fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ["#d1242f" if v > 0 else "#1f6feb" for v in monthly.values]
ax.bar(mon_lbl, monthly.values, color=colors)
ax.axhline(0, color="k", lw=0.8)
for i, v in enumerate(monthly.values):
    ax.text(i, v + (0.1 if v > 0 else -0.3), f"{v:+.1f}", ha="center", fontsize=8)
ax.set_title("월별 계절 편차 (최근 10년, 12개월 중심이동평균 대비)")
ax.set_ylabel("편차 (ppm)")
fig.tight_layout(); fig.savefig(IMG / "03_monthly_seasonality.png"); plt.close(fig)

# 04 10년 단위 평균 증가 속도
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(dec.index.astype(str) + "s", dec["mean_yoy"], color="#57a773",
       yerr=dec["std_yoy"], capsize=4, error_kw={"lw": 0.8})
for i, v in enumerate(dec["mean_yoy"]):
    if not np.isnan(v):
        ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
ax.set_title("10년 단위 평균 증가 속도 (오차막대 = YoY 표준편차)")
ax.set_ylabel("ppm / 년")
fig.tight_layout(); fig.savefig(IMG / "04_decade_growth.png"); plt.close(fig)

# 05 (보너스 A) STL 분해
stl = STL(df["co2"], period=12, robust=True).fit()
fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
for a, s, name in zip(axes, [df["co2"], stl.trend, stl.seasonal, stl.resid],
                      ["관측값", "추세(Trend)", "계절성(Seasonal)", "잔차(Noise)"]):
    a.plot(s.index, s, lw=0.8)
    a.set_ylabel(name)
axes[0].set_title("STL 시계열 분해 (period=12)")
fig.tight_layout(); fig.savefig(IMG / "05_stl_decomposition.png"); plt.close(fig)
var = df["co2"].var()
summary["stl_var_share"] = {
    "trend": round(float(stl.trend.var() / var * 100), 2),
    "seasonal": round(float(stl.seasonal.var() / var * 100), 2),
    "resid": round(float(stl.resid.var() / var * 100), 3)}
summary["stl_resid_std"] = round(float(stl.resid.std()), 3)

# 06 (보너스 B) 베이스라인 예측: 마지막 24개월은 숨기고 예측 → 실제와 비교
h = 24
train, test = df["co2"].iloc[:-h], df["co2"].iloc[-h:]
# (a) 계절 나이브: 1년 전 같은 달 값
naive = train.iloc[-12:].values
naive = np.concatenate([naive, naive])[:h]
# (b) 계절 나이브 + 추세: 1년 전 값 + 최근 3년 평균 YoY 증가량
drift = (train.iloc[-1] - train.iloc[-37]) / 36  # 월당 증가
snd = np.array([train.iloc[-12 + (k % 12)] + drift * 12 * (k // 12 + 1) for k in range(h)])
mae = lambda p: float(np.mean(np.abs(p - test.values)))
summary["forecast"] = {"horizon_months": h, "test_period": [str(test.index[0].date()), str(test.index[-1].date())],
                       "mae_seasonal_naive": round(mae(naive), 2),
                       "mae_seasonal_naive_drift": round(mae(snd), 2)}
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(train.index[-60:], train.iloc[-60:], color="#555", lw=1, label="학습 구간")
ax.plot(test.index, test, color="k", lw=1.8, label="실제")
ax.plot(test.index, naive, "--", color="#d1242f", label=f"계절 나이브 (MAE {mae(naive):.2f})")
ax.plot(test.index, snd, "--", color="#1f6feb", label=f"계절 나이브+추세 (MAE {mae(snd):.2f})")
ax.axvline(test.index[0], color="k", lw=0.5, ls=":")
ax.set_title(f"베이스라인 예측 검증 (마지막 {h}개월 홀드아웃)")
ax.set_ylabel("CO2 (ppm)")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(IMG / "06_baseline_forecast.png"); plt.close(fig)

# 정제된 데이터 저장
df.drop(columns="decade").to_csv(OUT / "co2_clean.csv", float_format="%.3f")
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                  encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
