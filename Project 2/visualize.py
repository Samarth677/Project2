import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import pandas as pd

# Add subfolders to path
sys.path.append(os.path.abspath("models"))
sys.path.append(os.path.abspath("data"))

# Now import your modules from subfolders
from gradient_boosting import GradientBoostingClassifier
from synthetic_data import generate_synthetic_data


# Generate and fit model
X, y = generate_synthetic_data(n_samples=500, n_classes=2)
model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=2)
model.fit(X, y)
preds = model.predict(X)
report = classification_report(y, preds, output_dict=True)

# --- 1. Decision Boundary
def plot_decision_boundary(model, X, y):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(["#FFAAAA", "#AAAAFF"]))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=ListedColormap(["#FF0000", "#0000FF"]))
    plt.title("Decision Boundary")
    plt.grid(True)
    plt.show()

# --- 2. PCA Visualization
def plot_pca_predictions(X, y_true, y_pred):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df = pd.DataFrame({"PCA1": X_pca[:, 0], "PCA2": X_pca[:, 1], "True": y_true, "Pred": y_pred})
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="True", ax=axs[0], palette="coolwarm", edgecolor='k')
    axs[0].set_title("True Labels")
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Pred", ax=axs[1], palette="coolwarm", edgecolor='k')
    axs[1].set_title("Predicted Labels")
    plt.tight_layout()
    plt.show()

# --- 3. Probability Heatmap
def plot_probability_heatmap(model, X, y):
    h = .05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    prob = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, prob, cmap="coolwarm", alpha=0.8)
    plt.colorbar(contour)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#FF0000", "#0000FF"]), edgecolor='k')
    plt.title("Probability Heatmap")
    plt.show()

# --- 4. Classification Report Heatmap
def plot_classification_report_heatmap(report):
    df = pd.DataFrame(report).iloc[:-1, :].T
    sns.heatmap(df, annot=True, cmap="YlGnBu")
    plt.title("Classification Report Heatmap")
    plt.show()

# --- 5. Predicted Probability Distribution
def plot_predicted_probabilities(model, X, y):
    probas = model.predict_proba(X)
    df = pd.DataFrame(probas, columns=[f"Class {i}" for i in range(probas.shape[1])])
    df["True Label"] = y
    df_melted = df.melt(id_vars="True Label", var_name="Class", value_name="Probability")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_melted, x="Probability", hue="Class", multiple="stack", bins=20)
    plt.title("Predicted Probability Distribution")
    plt.show()

# Run all plots
plot_decision_boundary(model, X, y)
plot_pca_predictions(X, y, preds)
plot_probability_heatmap(model, X, y)
plot_classification_report_heatmap(report)
plot_predicted_probabilities(model, X, y)
