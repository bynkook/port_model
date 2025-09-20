# %%
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

OUTPUT_DIR = Path("./output_chi_test")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 분석 데이터셋 컬럼 정의
COL_TIME = ['사용년월', '전체일수', '작업가능일수']
COL_WEATHER = ['최고기온', '최저기온', '강우', '풍속', '강설', '안개', '미세먼지']

def load_and_prepare(path_csv: str = "data.csv") -> pd.DataFrame:
    """ CSV 로드 후 년, 월 파생변수 컬럼을 생성. 필요없는 컬럼 삭제 후 data2.csv 저장 """
    # csv 파일 로드
    df = pd.read_csv(path_csv, sep='\t')    # csv 파일은 tab 으로 구분되어 있음

    # 전역변수에 정의된 데이터셋의 컬럼만 사용함
    df = df[COL_TIME + COL_WEATHER]

    # datetime 형식 변환, year & month 생성
    df['datetime'] = pd.to_datetime(df['사용년월'], errors='raise', format="%Y년%m월")  # '2025년03월' 문자열을format 을 지정하여 읽음
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month

    # 분석에 필요없는 데이터 컬럼 삭제
    df.drop(columns=['사용년월', 'datetime'], inplace=True, errors='raise')

    return df

def create_weather_score(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    data = df.copy()
    weather_features = [c for c in COL_WEATHER if c in data.columns]
    weather_data = data[weather_features].fillna(0).to_numpy(dtype=float)

    # ---- weights는 사용자가 수정해야함 ----
    weights = [0.15, 0.1, 0.3, 0.2, 0.05, 0.15, 0.05] # 2025-09-19 설정 저장

    # weather 데이터 컬럼들과 가중치의 곱
    risk_score = weather_data.dot(weights)

    percentiles = np.percentile(risk_score, [20, 40, 60, 80])

    def assign(score: float) -> int:
        if score <= percentiles[0]:
            return 5
        elif score <= percentiles[1]:
            return 4
        elif score <= percentiles[2]:
            return 3
        elif score <= percentiles[3]:
            return 2
        else:
            return 1

    # weather_score 컬럼에 날씨점수 저장
    data['weather_score'] = [assign(s) for s in risk_score]

    # percentiles 는 저장해두고 새로운 데이터 들어올때 사용해야 한다.
    return data, percentiles

# %%

# data load, 변환, 파생변수컬럼 생성
df = load_and_prepare('data.csv')

# 파생변수 컬럼 추가 생성
df, weather_percentiles = create_weather_score(df)

# 컬럼명 순서 재지정
column_name_order = ['year', 'month', '전체일수', '작업가능일수', 'weather_score', '최고기온', '최저기온', '강우', '풍속', '강설', '안개', '미세먼지']
df = df[column_name_order]

# 파생변수를 포함한 데이터셋 저장
df.to_csv(OUTPUT_DIR / 'data2.csv', index=False)

# %%
from typing import Literal
from dataclasses import dataclass
import warnings
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

@dataclass
class TestOutputs:
    glm_summary_path: str
    calibration_png: str
    trend_json: str
    chi_json: str
    corr_json: str

def _working_rate_bins(x: pd.Series, k: int = 5) -> pd.Series:
    """작업가능율을 k분위 이산화(카이제곱용)."""
    rate = x.clip(0, 1)
    r = pd.qcut(rate.rank(method="first"), q=k, labels=[f"Q{i}" for i in range(1, k+1)])
    return r

def fit_binomial_glm(
    df: pd.DataFrame,
    treat_score_as: Literal["ordinal","numeric"] = "ordinal"
) -> Tuple[object, pd.DataFrame, dict]:
    """
    GLM(Binomial): 작업가능일수 ~ Binomial(전체일수, p), logit(p) = β0 + f(weather_score)
    반환: (결과객체, 캘리브레이션DF, 요약딕트)
    """
    data = df.copy()
    data = data.dropna(subset=["전체일수","작업가능일수","weather_score"])
    data = data.loc[data["전체일수"] > 0].copy()
    data["rate"] = data["작업가능일수"] / data["전체일수"]

    if treat_score_as == "ordinal":
        data["weather_score"] = data["weather_score"].astype("category")
        formula = "rate ~ C(weather_score)"
    else:
        formula = "rate ~ weather_score"

    model = smf.glm(formula=formula, data=data,
                    family=sm.families.Binomial(), freq_weights=data["전체일수"])
    res = model.fit()

    # 캘리브레이션: 관측율 vs 예측율(가중평균)
    data["pred"] = res.predict(data)
    calib = (
        data[["weather_score","전체일수","작업가능일수","pred"]]
        .groupby("weather_score", observed=True)
        .apply(
            lambda g: pd.Series({
                "n_months": len(g),
                "total_days": g["전체일수"].sum(),
                "total_work": g["작업가능일수"].sum(),
                "obs_rate": g["작업가능일수"].sum() / g["전체일수"].sum(),
                "pred_rate_weighted": np.average(g["pred"], weights=g["전체일수"])
            }),
            include_groups=False
        )
        .reset_index()
    )

    # 플롯 저장
    fig = plt.figure(figsize=(5,4))
    xs = calib["weather_score"].astype(str)
    plt.plot(xs, calib["obs_rate"], marker="o", label="관측율")
    plt.plot(xs, calib["pred_rate_weighted"], marker="s", label="예측율")
    plt.xlabel("weather_score"); plt.ylabel("작업가능율"); plt.title("캘리브레이션(관측 vs 예측)")
    plt.legend(); fig.tight_layout()
    cal_path = str(OUTPUT_DIR / "calibration_weather_score.png")
    fig.savefig(cal_path, dpi=150); plt.close(fig)

    # 요약 저장 + LR test
    summ_path = str(OUTPUT_DIR / "glm_binomial_weather_score.txt")
    with open(summ_path, "w", encoding="utf-8") as f:
        f.write(res.summary().as_text())

    base = smf.glm("rate ~ 1", data=data,
                   family=sm.families.Binomial(),
                   freq_weights=data["전체일수"]).fit()
    lr_stat = float(2*(res.llf - base.llf))
    df_diff = int(res.df_model - base.df_model)
    lr_p = float(1 - st.chi2.cdf(lr_stat, df_diff))
    with open(summ_path, "a", encoding="utf-8") as f:
        f.write("\n\n[Likelihood Ratio Test vs Intercept-only]\n")
        f.write(f"LR stat={lr_stat:.3f}, df={df_diff}, p-value={lr_p:.6f}\n")
        f.write(f"AIC(full)={res.aic:.3f}, AIC(base)={base.aic:.3f}\n")

    info = {
        "formula": formula,
        "aic_full": float(res.aic),
        "aic_base": float(base.aic),
        "lr_stat": lr_stat,
        "lr_df": df_diff,
        "lr_p": lr_p,
        "n_obs": int(len(data)),
    }

    # 보조 저장
    (OUTPUT_DIR / "calibration_by_score.csv").write_text(
        calib.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8"
    )

    return res, calib, info

def cochran_armitage_trend(df: pd.DataFrame) -> dict:
    """Cochran–Armitage 추세검정(직접 구현). 결과 JSON 저장."""
    g = df.groupby("weather_score", observed=True).agg(
        success=("작업가능일수","sum"),
        trials=("전체일수","sum")
    ).sort_index()
    g["fail"] = g["trials"] - g["success"]

    k = g.shape[0]
    scores = np.arange(1, k+1, dtype=float)
    n = float(g["trials"].sum())
    p_hat = float(g["success"].sum() / n)

    num = float(np.sum(scores * (g["success"] - g["trials"]*p_hat)))
    denom = float(np.sqrt(p_hat*(1-p_hat) * (np.sum(g["trials"]*scores**2) - (np.sum(g["trials"]*scores)**2)/n)))
    z_stat = num / denom if denom > 0 else np.nan
    p_val = float(2 * (1 - st.norm.cdf(abs(z_stat))))

    out = {"statistic": z_stat, "p_value": p_val, "method": "Cochran-Armitage trend test (manual)", "k": int(k)}
    (OUTPUT_DIR / "trend_test.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def chi_square_independence(df: pd.DataFrame, k_bins: int = 5) -> dict:
    """카이제곱 독립성 + Cramér’s V. 결과 JSON 저장."""
    tmp = df.copy()
    tmp["rate"] = tmp["작업가능일수"] / tmp["전체일수"]
    tmp = tmp.dropna(subset=["rate","weather_score"])
    tmp = tmp.loc[tmp["전체일수"] > 0]
    tmp["rate_bin"] = _working_rate_bins(tmp["rate"], k=k_bins)

    ct = pd.crosstab(tmp["weather_score"], tmp["rate_bin"])
    chi2, p, dof, _ = st.chi2_contingency(ct.values)

    n = ct.values.sum()
    r, c = ct.shape
    cramers_v = float(np.sqrt((chi2/n) / (min(r-1, c-1))))

    out = {"chi2": float(chi2), "p_value": float(p), "dof": int(dof),
           "cramers_v": cramers_v, "n": int(n), "k_bins": int(k_bins)}
    (OUTPUT_DIR / "chi_square_independence.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def monotone_correlations(df: pd.DataFrame) -> dict:
    """Spearman, Kendall tau-b. 결과 JSON 저장."""
    x = df["weather_score"].astype(float)
    y = (df["작업가능일수"] / df["전체일수"]).astype(float)
    mask = x.notna() & y.notna() & (df["전체일수"] > 0)
    x, y = x[mask], y[mask]

    spear = st.spearmanr(x, y)
    kend = st.kendalltau(x, y, variant="b")
    out = {
        "spearman_rho": float(spear.correlation),
        "spearman_p": float(spear.pvalue),
        "kendall_tau_b": float(kend.statistic),
        "kendall_p": float(kend.pvalue)
    }
    (OUTPUT_DIR / "ordinal_correlations.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def _print_console_summary(glm_info: dict, trend: dict, chi: dict, corr: dict, res_obj) -> None:
    """콘솔 요약 출력."""
    print("\n=== [요약] weather_score 적합성 검정 ===")
    # GLM
    print("[GLM Binomial]")
    print(f"  formula     : {glm_info['formula']}")
    print(f"  n_obs       : {glm_info['n_obs']}")
    print(f"  AIC full    : {glm_info['aic_full']:.3f}   | AIC base: {glm_info['aic_base']:.3f}")
    print(f"  LR test     : stat={glm_info['lr_stat']:.3f}, df={glm_info['lr_df']}, p={glm_info['lr_p']:.3g}")
    # 주요 계수 요약(상위 몇 개만)
    try:
        coefs = res_obj.summary2().tables[1]
        top = min(6, len(coefs))
        print("  Coeff head  :")
        for i in range(top):
            row = coefs.iloc[i]
            print(f"    {coefs.index[i]:<18} coef={row['Coef.']:.4f}  z={row['z']:.2f}  p={row['P>|z|']:.3g}")
    except Exception:
        pass

    # 추세검정
    print("[Cochran-Armitage trend]")
    print(f"  Z={trend['statistic']:.3f}, p={trend['p_value']:.3g}, k={trend['k']}")

    # 카이제곱
    print("[Chi-square independence]")
    print(f"  chi2={chi['chi2']:.3f}, df={chi['dof']}, p={chi['p_value']:.3g}, Cramér's V={chi['cramers_v']:.3f}, n={chi['n']}")

    # 순서상관
    print("[Ordinal correlations]")
    print(f"  Spearman rho={corr['spearman_rho']:.3f}, p={corr['spearman_p']:.3g}")
    print(f"  Kendall tau-b={corr['kendall_tau_b']:.3f}, p={corr['kendall_p']:.3g}")
    print("=== 끝 ===\n")

def run_all_tests(df: pd.DataFrame) -> TestOutputs:
    """
    1) GLM + LR test + 캘리브레이션 파일/그림 저장
    2) Cochran–Armitage 추세검정 JSON 저장
    3) 카이제곱 독립성 + Cramér’s V JSON 저장
    4) Spearman/Kendall JSON 저장
    5) 콘솔 요약 출력
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    res, calib, glm_info = fit_binomial_glm(df, treat_score_as="ordinal")
    trend = cochran_armitage_trend(df)
    chi = chi_square_independence(df, k_bins=5)
    corr = monotone_correlations(df)

    # 콘솔 경로 안내 + 요약
    print("[OK] Binomial GLM summary :", str(OUTPUT_DIR / "glm_binomial_weather_score.txt"))
    print("[OK] Calibration plot     :", str(OUTPUT_DIR / "calibration_weather_score.png"))
    print("[OK] Trend test JSON      :", str(OUTPUT_DIR / "trend_test.json"))
    print("[OK] Chi-square JSON      :", str(OUTPUT_DIR / "chi_square_independence.json"))
    print("[OK] Ordinal corr JSON    :", str(OUTPUT_DIR / "ordinal_correlations.json"))
    # 핵심 요약
    _print_console_summary(glm_info, trend, chi, corr, res)

    return TestOutputs(
        glm_summary_path=str(OUTPUT_DIR / "glm_binomial_weather_score.txt"),
        calibration_png=str(OUTPUT_DIR / "calibration_weather_score.png"),
        trend_json=str(OUTPUT_DIR / "trend_test.json"),
        chi_json=str(OUTPUT_DIR / "chi_square_independence.json"),
        corr_json=str(OUTPUT_DIR / "ordinal_correlations.json"),
    )

# === 실행 ===
tests_out = run_all_tests(df)
# %%
'''
[OK] Binomial GLM summary : output_chi_test\glm_binomial_weather_score.txt
[OK] Calibration plot     : output_chi_test\calibration_weather_score.png
[OK] Trend test JSON      : output_chi_test\trend_test.json
[OK] Chi-square JSON      : output_chi_test\chi_square_independence.json
[OK] Ordinal corr JSON    : output_chi_test\ordinal_correlations.json

=== [요약] weather_score 적합성 검정 ===
[GLM Binomial]
  formula     : rate ~ C(weather_score)
  n_obs       : 180
  AIC full    : 4744.116   | AIC base: 5040.281
  LR test     : stat=304.165, df=4, p=0
  Coeff head  :
    Intercept          coef=-0.4214  z=-6.66  p=2.68e-11
    C(weather_score)[T.2] coef=0.6050  z=6.99  p=2.67e-12
    C(weather_score)[T.3] coef=0.9066  z=10.07  p=7.32e-24
    C(weather_score)[T.4] coef=1.0853  z=12.15  p=5.6e-34
    C(weather_score)[T.5] coef=1.4649  z=15.77  p=5.17e-56
[Cochran-Armitage trend]
  Z=16.932, p=0, k=5
[Chi-square independence]
  chi2=185.282, df=16, p=7.34e-31, Cramér's V=0.507, n=180
[Ordinal correlations]
  Spearman rho=0.848, p=5.6e-51
  Kendall tau-b=0.717, p=4.91e-38
=== 끝 ===

1. GLM Binomial 결과

공식: rate ~ C(weather_score)

샘플 수: 180개월

AIC 감소: base모형(5040.3) → full모형(4744.1). AIC가 크게 낮아져 weather_score가 설명력을 상당히 높임.

LR 검정: χ²=304.2, df=4, p≈0.0 → 귀무가설(설명력 없음) 기각.

계수: score=2~5에서 계수가 모두 양수이며, p<1e-11 수준으로 유의. 즉 weather_score가 높을수록 작업가능율이 유의하게 증가.

2. Cochran-Armitage 추세검정

Z=16.9, p≈0 → 1~5 순서대로 작업가능율이 단조적으로 증가하는 강한 추세 존재.

3. 카이제곱 독립성 검정

χ²=185.3, df=16, p≈7e-31 → 독립 아님. weather_score와 작업가능율 분위 그룹 간에 뚜렷한 연관성.

Cramér’s V=0.507 → 효과크기 기준(0.1=소, 0.3=중, 0.5=대)에서 큰 효과.

4. 순서상관

Spearman rho=0.848, Kendall tau-b=0.717 → 매우 강한 양의 상관. p값은 극소.

종합 결론

네 가지 접근(GLM, 추세검정, 카이제곱, 상관분석) 모두 일관되게 강한 양의 관계를 확인.

weather_score는 단순히 이론적 지표가 아니라 실제 작업가능일수 예측 변수로 매우 잘 작동함.

특히 GLM 계수와 AIC 개선폭, Cramér’s V 0.5 이상의 큰 효과크기가 이를 뒷받침.

즉, 현재 정의된 방식의 weather_score는 통계적으로도 적합하고, 예측모형에 강력하게 기여하는 변수입니다.

👉 이 결과를 바탕으로 하향식 모델링(예: 작업가능율 ~ weather_score + 기타 기상변수)로 확장하면, 다른 기상 변수 대비 weather_score의 종합적 대표성도 검증할 수 있습니다.

'''