# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# veri okuma
df = pd.read_csv("AllData.csv")

# veri temizleme
df = df.fillna(df.median(numeric_only=True))
df = df.drop_duplicates()
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# feature & label
y = df["IsStealer"]
X = df.drop("IsStealer", axis=1)
X = X.select_dtypes(include=[np.number])

# feature engineering
X["avg_daily_consumption"] = X.mean(axis=1)
X["peak_usage"] = X.max(axis=1)
X["consumption_variance"] = X.var(axis=1)

X = X[
    [
        "avg_daily_consumption",
        "peak_usage",
        "consumption_variance",
    ]
]

# outlier
for col in X.columns:
    X[col] = X[col].clip(X[col].quantile(0.01), X[col].quantile(0.99))

# log transform
X = np.log1p(X)

# grafikler
X_df = pd.DataFrame(
    X,
    columns=[
        "avg_daily_consumption",
        "peak_usage",
        "consumption_variance",
    ],
)

X_df.hist(figsize=(8, 6))
plt.suptitle("Feature Dagilimlari (Log Scale)")
plt.show()

plt.scatter(X_df["avg_daily_consumption"], X_df["peak_usage"], c=y)
plt.title("Scatter Plot")
plt.show()

sns.heatmap(X_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42
)

# smote
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# naive bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\nNaive Bayes Performans:")
print("Accuracy :", accuracy_score(y_test, nb_pred))
print("Precision:", precision_score(y_test, nb_pred))
print("Recall   :", recall_score(y_test, nb_pred))
print("F1 Score :", f1_score(y_test, nb_pred))

cm_nb = confusion_matrix(y_test, nb_pred)
plt.figure()
sns.heatmap(cm_nb, annot=True, fmt="d")
plt.title("Confusion Matrix (Naive Bayes)")
plt.show()

# logistic regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLogistic Regression Performans:")
print("Accuracy :", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall   :", recall_score(y_test, lr_pred))
print("F1 Score :", f1_score(y_test, lr_pred))

cm_lr = confusion_matrix(y_test, lr_pred)
plt.figure()
sns.heatmap(cm_lr, annot=True, fmt="d")
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()

# random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Performans:")
print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1 Score :", f1_score(y_test, rf_pred))

cm_rf = confusion_matrix(y_test, rf_pred)
plt.figure()
sns.heatmap(cm_rf, annot=True, fmt="d")
plt.title("Confusion Matrix (Random Forest)")
plt.show()

# decision tree (ana model)
model = DecisionTreeClassifier(max_depth=3, class_weight="balanced")
model.fit(X_train, y_train)

print("\nFeature Importance:")
for name, val in zip(X_df.columns, model.feature_importances_):
    print(name, ":", val)

# threshold optimization
y_proba = model.predict_proba(X_test)[:, 1]

best_thresh = 0.5
best_f1 = 0

for t in np.arange(0.2, 0.8, 0.05):
    y_pred_temp = (y_proba > t).astype(int)
    f1 = f1_score(y_test, y_pred_temp)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("\nEn iyi threshold:", best_thresh)

# final prediction
y_pred = (y_proba > best_thresh).astype(int)

print("\nDecision Tree Performans:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

# ROC-AUC
auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", auc)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# cross validation
cv_scores = cross_val_score(model, X_df, y, cv=5, scoring="f1")
print("Cross-Validation F1 Ortalama:", cv_scores.mean())

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix (Optimized Decision Tree)")
plt.show()

print("\n--- TÜM MODELLER KARŞILAŞTIRILDI ---")