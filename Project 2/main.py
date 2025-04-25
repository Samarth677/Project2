import numpy as np
from models.gradient_boosting import GradientBoostingClassifier
from data.synthetic_data import generate_synthetic_data
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

X, y = generate_synthetic_data(n_samples=500, n_classes=2)
model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=2)
model.fit(X, y)
preds = model.predict(X)
print(classification_report(y, preds))

def plot_decision_boundary(model, X, y):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(model, X, y)
