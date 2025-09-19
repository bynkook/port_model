#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project-2: ML 모델 구축 및 저장

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import optuna

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

THIS_YEAR = 2025
RANDOM_STATE = 42

# --- 사용자 지정 전역변수 ----
MODEL_ID = 4
USE_OPTUNA = False
STACK_PASSTHROUGH = True
N_TRIALS = 10
# ------------------------------

INPUT_DIR = Path("./input")
OUTPUT_DIR = Path("./output_model") / f"model_{MODEL_ID}"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
INPUT_DIR.mkdir(exist_ok=True, parents=True)

# 분석 데이터셋 컬럼 정의
COL_TIME = ['사용년월', '전체일수', '작업가능일수', '중복일수', '공휴일']
COL_WEATHER = ['최고기온', '최저기온', '강우', '풍속', '강설', '안개', '미세먼지']
COL_OPERATION = ['NIS', 'NOS', 'FIS', 'FOS', 'CIS', 'COS', 'NIGT', 'NOGT', 'FIGT', 'FOGT', 'CIGT', 'COGT']


def load_and_prepare(path_csv: str = "data.csv") -> pd.DataFrame:
    """ CSV 로드 후 년, 월 파생변수 컬럼을 생성. 필요없는 컬럼 삭제 후 data2.csv 저장 """
    # csv 파일 로드
    df = pd.read_csv(path_csv, sep='\t')    # csv 파일은 tab 으로 구분되어 있음

    # 전역변수에 정의된 데이터셋의 컬럼만 사용함
    df = df[COL_TIME + COL_WEATHER + COL_OPERATION]

    # datetime 형식 변환, year & month 생성
    df['datetime'] = pd.to_datetime(df['사용년월'], errors='raise', format="%Y년%m월")  # '2025년03월' 문자열을format 을 지정하여 읽음
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month

    # 파생변수 생성
    df['weather_bad_ratio'] = df[COL_WEATHER].sum(axis=1) / df['전체일수']
    df['weather_bad_ratio'] = df['weather_bad_ratio'].round(4)

    # 분석에 필요없는 데이터 컬럼 삭제
    df.drop(columns=['사용년월', 'datetime'], inplace=True, errors='raise')

    # 파일로 저장(사용자 확인용)
    df.to_csv("data2.csv", index=False)

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

def create_weather_score_for_predict_new(df: pd.DataFrame, percentiles: np.ndarray) -> pd.DataFrame:
    data = df.copy()
    weather_features = [c for c in COL_WEATHER if c in data.columns]
    weather_data = data[weather_features].fillna(0).to_numpy(dtype=float)

    # ---- weights는 사용자가 수정해야함 ----
    weights = [0.15, 0.1, 0.3, 0.2, 0.05, 0.15, 0.05] # 2025-09-19 설정 저장

    # weather 데이터 컬럼들과 가중치의 곱
    risk_score = weather_data.dot(weights)

    # percentiles 훈련시 값을 가져와서 사용
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

    return data


def create_data_age(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    age = THIS_YEAR - data['year'].astype(int)

    # 2010 ~ 2015, 2016 ~ 2020, 2021 ~ 2024
    percentiles = np.percentile(age, [33, 66])

    def assign(score: float) -> int:
        if score <= percentiles[0]:
            return 5
        elif score <= percentiles[1]:
            return 5
        else:
            return 1

    # data_age 컬럼에 점수 저장
    data['data_age_score'] = [assign(s) for s in age]

    return data


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    scale_cols = COL_OPERATION  # operation 데이터는 모두 StandardScaler 로 scaling

    transformers = []
    if scale_cols:
        transformers.append(("scaled_numeric",
                             Pipeline([("imputer", SimpleImputer(strategy="median")),
                                       ("scaler", StandardScaler())]),
                             scale_cols))
    # 이건 사실 사용되지 않음
    if cat_cols:
        transformers.append(("categorical",
                             Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                             cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def make_models(model_id: int):
    if model_id == 1:
        base = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=300)
    elif model_id == 2:
        base = XGBRegressor(random_state=RANDOM_STATE, n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9)
    elif model_id == 3:
        base = GradientBoostingRegressor(random_state=RANDOM_STATE)
    elif model_id == 4:
        base = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    else:
        raise ValueError("MODEL_ID 는 1~4")
    meta = Ridge(alpha=1.0)
    return base, meta


def make_stacking(model_id: int, passthrough: bool):
    base, meta = make_models(model_id)
    estimators = [('base', base)]
    stack = StackingRegressor(estimators=estimators, final_estimator=meta, passthrough=passthrough)
    return stack


def get_param_space(model_id: int) -> Dict[str, Any]:
    if model_id == 1:
        return {
            "base__n_estimators": (50, 300),
            "base__max_depth": (3, 20),
            "base__min_samples_split": (2, 20),
            "base__min_samples_leaf": (1, 10),
            "final_estimator__alpha": (0.01, 100)
        }
    elif model_id == 2:
        return {
            "base__n_estimators": (50, 300),
            "base__max_depth": (3, 12),
            "base__learning_rate": (1e-3, 0.3),
            "base__subsample": (0.6, 1.0),
            "base__colsample_bytree": (0.6, 1.0),
            "final_estimator__alpha": (0.01, 100)
        }
    elif model_id == 3:
        return {
            "base__n_estimators": (50, 300),
            "base__learning_rate": (1e-3, 0.3),
            "base__max_depth": (2, 6),
            "base__subsample": (0.6, 1.0),
            "final_estimator__alpha": (0.01, 100)
        }
    elif model_id == 4:
        return {
            "base__learning_rate": (1e-3, 0.5),
            "base__max_depth": (2, 16),
            "base__max_bins": (64, 256),
            "base__l2_regularization": (0.001, 2.0),
            "final_estimator__alpha": (0.01, 100)
        }
    else:
        raise ValueError("MODEL_ID 는 1~4")


def suggest_params(trial: optuna.trial.Trial, space: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    params = {}
    for k, v in space.items():
        low, high = v
        if "learning_rate" in k or "l2_regularization" in k or "subsample" in k or "colsample_bytree" in k:
            params[k] = trial.suggest_float(k, low, high, log=True)
        elif "max_depth" in k or "max_bins" in k or "n_estimators" in k or "min_samples" in k:
            params[k] = trial.suggest_int(k, int(low), int(high))
        else:
            params[k] = trial.suggest_float(k, low, high)
    return params


def evaluate(y_true, y_pred) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main():
    # data load, 변환, 파생변수컬럼 생성
    df = load_and_prepare('data.csv')
    # 파생변수 컬럼 추가 생성
    df, weather_percentiles = create_weather_score(df)
    df = create_data_age(df)    # year 컬럼을 사용함
    # year 컬럼 삭제
    df.drop(columns='year', inplace=True, errors='raise')
    # 파생변수를 포함한 데이터셋 저장 -> 사용자가 파생변수값을 확인할 수 있다.
    df.to_csv('data3.csv')

    ####
    #### 분석에 사용하는 파생변수: weather_bad_ratio(한달간 날씨 안좋은날 비율), weather_score(날씨가 얼마나 안좋은지 점수화), data_age(데이터 연도별 가중치)
    ####

    # X, y 저장
    y = df.pop("작업가능일수").values
    X = df

    pre = build_preprocessor(X)
    model = make_stacking(MODEL_ID, STACK_PASSTHROUGH)
    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    best_params = {}
    if USE_OPTUNA:
        study = optuna.create_study(direction="maximize")   # R2 score 최대화
        space = get_param_space(MODEL_ID)

        def objective(trial):
            params = suggest_params(trial, space)
            pipe.set_params(**{f"model__{k}": v for k, v in params.items()})
            scores = cross_val_score(pipe, X_train, y_train, scoring="r2", cv=kf, n_jobs=None)
            return float(np.mean(scores))

        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        best_params = study.best_params     # 최적 훈련파라메터 저장
        pipe.set_params(**{f"model__{k}": v for k, v in best_params.items()})   # 최적파라메터를 pipe 에 입력
        with open(OUTPUT_DIR / f"bestparam_model_{MODEL_ID}.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)     # 최적 파라메터를 json 파일에 저장(/input 폴더에 복사해놓고 수정한다음 USE_OPTUNA=False 로 수동훈련 가능)

        # optuna 최적 파라메터 탐색 history plot
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            # Many environments require kaleido for write_image. Fallback handled below.
            fig.write_image(str(OUTPUT_DIR / f"optuna_history_model_{MODEL_ID}.png"))
        except Exception:
            hist = [t.value for t in study.trials if t.value is not None]
            plt.figure(figsize=(6, 3))
            plt.plot(hist)
            plt.xlabel("trial")
            plt.ylabel("r2")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"optuna_history_model_{MODEL_ID}.png", dpi=160)
            plt.close()

    else:
        param_path = INPUT_DIR / f"bestparam_model_{MODEL_ID}.json"
        if not param_path.exists():
            raise FileNotFoundError(f"{param_path} 가 존재하지 않습니다. USE_OPTUNA=True 로 먼저 실행해 저장하세요.")
        with open(param_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)
        pipe.set_params(**{f"model__{k}": v for k, v in best_params.items()})

    # 최종 훈련된 파라메터와 전체 훈련데이터로 model fit
    pipe.fit(X_train, y_train)

    y_pred_tr = pipe.predict(X_train)
    y_pred_te = pipe.predict(X_test)
    metrics_train = evaluate(y_train, y_pred_tr)
    metrics_test = evaluate(y_test, y_pred_te)

    print("[TRAIN]", metrics_train)
    print("[TEST ]", metrics_test)

    pd.DataFrame([metrics_train]).to_csv(OUTPUT_DIR / "metrics_train.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics_test]).to_csv(OUTPUT_DIR / "metrics_test.csv", index=False, encoding="utf-8-sig")

    joblib.dump(pipe, OUTPUT_DIR / f"stacking_model_{MODEL_ID}.joblib")
    print(f"모델 저장 완료: {OUTPUT_DIR / f'stacking_model_{MODEL_ID}.joblib'}")

    meta = {
        "MODEL_ID": MODEL_ID,
        "USE_OPTUNA": USE_OPTUNA,
        "STACK_PASSTHROUGH": STACK_PASSTHROUGH,
        "best_params": best_params
    }
    with open(OUTPUT_DIR / "run_meta.json", "w", encoding="utf-8-sig") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)    # run_meta.json 은 방금 실행한 run 관련 정보를 담고 있음. 매 실행시 덮어쓰기됨.

    return weather_percentiles


def main2():
    # 모델 파일 불러오기
    model_loaded = joblib.load(OUTPUT_DIR / f"stacking_model_{MODEL_ID}.joblib")
    df = load_and_prepare('data_new.csv')
    # weather_score, data_age_score 파생변수 컬럼 생성
    df = create_weather_score_for_predict_new(df, weather_percentiles)
    df['data_age_score'] = 5.0  # 최신데이터라면 모두 5점 준다.
    # year 컬럼 삭제
    df.drop(columns='year', inplace=True, errors='raise')
    # 파생변수를 포함한 데이터셋 저장 -> 사용자가 파생변수값을 확인할 수 있다.
    df.to_csv('data3_new.csv')
    # X, y 저장
    y_new = df.pop("작업가능일수").values
    X_new = df
    # 새로운 데이터로 predict 한 결과를 출력
    y_pred = model_loaded.predict(X_new)
    metrics_predict = evaluate(y_new, y_pred)
    print("[PREDICT ]", metrics_predict)
    pd.DataFrame([metrics_predict]).to_csv(OUTPUT_DIR / "metrics_predict.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    weather_percentiles = main()
    main2()
