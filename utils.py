# utils.py --------------------------------------------------------------
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def train_models(df: pd.DataFrame):
    X = df.drop(columns=["First_Month_Spend"])
    y = df["First_Month_Spend"]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    models = {
        "K-NN": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42)
    }
    params = {
        "K-NN": {"model__n_neighbors": [3,5,7]},
        "Decision Tree": {"model__max_depth": [None,5,10]},
        "Random Forest": {"model__n_estimators": [100,250],
                          "model__max_depth": [None,10]}
    }
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("model", model)])
        gs = GridSearchCV(pipe, params[name], cv=3, n_jobs=-1)
        gs.fit(X, y)
        y_hat = gs.predict(X)
        results[name] = {
            "best": gs.best_estimator_,
            "R²": round(r2_score(y, y_hat), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y, y_hat)), 2)
        }
    best_name = max(results, key=lambda k: results[k]["R²"])
    return results, best_name

def forecast_city_revenue(df, best_est):
    X = df.drop(columns=["First_Month_Spend"])
    df["Pred_Spend"] = best_est.predict(X)

    # month 1 … 12 revenue = predicted spend · (renewal_prob)^(m-1)
    for m in range(1, 13):
        df[f"Month_{m}"] = df["Pred_Spend"] * (df["Renewal_Probability"]**(m-1))

    return (df.groupby("City")[ [f"Month_{m}" for m in range(1,13)] ]
              .sum()
              .round(0)
              .reset_index())
