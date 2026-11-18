# Task 04: Supervised Classification / Regression Modeling & Tuning
# Run: python task04_modeling.py
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

SEED = 42
dataset = load_breast_cancer(as_frame=True)
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = {
    "Logistic Regression": (Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, random_state=SEED))]), {"model__C":[0.1,1.0,10.0]}),
    "Random Forest": (Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(random_state=SEED, class_weight="balanced"))]), {"model__n_estimators":[150,300], "model__max_depth":[None,5,10]}),
    "Gradient Boosting": (Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", GradientBoostingClassifier(random_state=SEED))]), {"model__n_estimators":[100,150], "model__learning_rate":[0.05,0.1]}),
    "Extra Trees": (Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesClassifier(random_state=SEED, class_weight="balanced"))]), {"model__n_estimators":[150,300], "model__max_depth":[None,10]})
}
rows, searches = [], {}
for name, (pipe, grid) in models.items():
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True).fit(X_train, y_train)
    searches[name] = gs
    p, prob = gs.predict(X_test), gs.predict_proba(X_test)[:,1]
    rows.append({"Model":name, "Best CV ROC-AUC":gs.best_score_, "Accuracy":accuracy_score(y_test,p),
                 "Precision":precision_score(y_test,p), "Recall":recall_score(y_test,p),
                 "F1":f1_score(y_test,p), "Test ROC-AUC":roc_auc_score(y_test,prob),
                 "Best Parameters":str(gs.best_params_)})
results = pd.DataFrame(rows).sort_values("Best CV ROC-AUC", ascending=False)
winner = results.iloc[0]["Model"]
joblib.dump(searches[winner].best_estimator_, "champion_model.pkl")
results.to_csv("model_comparison.csv", index=False)
print(results.to_string(index=False))
print("\\nChampion:", winner)
print(classification_report(y_test, searches[winner].predict(X_test), target_names=dataset.target_names))
