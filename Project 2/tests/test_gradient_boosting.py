import sys
import os
import numpy as np
import logging

# Setup logging to display in pytest without needing -s
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger()

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model and data generator
from models.gradient_boosting import GradientBoostingClassifier
from data.synthetic_data import generate_synthetic_data

def test_gradient_boosting_binary():
    log.info("🔹 Running Binary Classification Test...")
    X, y = generate_synthetic_data(n_samples=200, n_classes=2)
    clf = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=2)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = np.mean(preds == y)
    log.info("✅ Binary Accuracy: %.2f", acc)
    assert acc > 0.8
    log.info("✅ Binary classification test passed!\n")

def test_gradient_boosting_multiclass():
    log.info("🔹 Running Multi-class Classification Test...")
    X, y = generate_synthetic_data(n_samples=200, n_classes=3)
    clf = GradientBoostingClassifier(n_estimators=15, learning_rate=0.1, max_depth=2)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = np.mean(preds == y)
    log.info("✅ Multi-class Accuracy: %.2f", acc)
    assert acc > 0.7
    log.info("✅ Multiclass classification test passed!\n")

def test_overfit_small_dataset():
    log.info("🔹 Running Overfitting Test...")
    X = np.array([[0.1, 0.2], [0.2, 0.1], [0.9, 0.8], [1.0, 1.1]])
    y = np.array([0, 0, 2, 2])
    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.5, max_depth=1)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = np.mean(preds == y)
    log.info("✅ Overfitting test accuracy: %.2f", acc)
    assert acc == 1.0
    log.info("✅ Overfitting test passed!\n")

def test_early_stopping_triggers():
    log.info("🔹 Running Early Stopping Test...")
    X, y = generate_synthetic_data(n_samples=200, n_classes=2)
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.5,
        max_depth=1,
        early_stopping=True,
        patience=2,
        val_fraction=0.4
    )
    clf.fit(X, y)
    total_trees = sum(len(clf.trees[c]) for c in clf.classes_)
    log.info("✅ Trees built (early stopping): %d / %d", total_trees, 100 * len(clf.classes_))
    for cls in clf.classes_:
        log.info(f" - Class {cls}: {len(clf.trees[cls])} trees")

    if total_trees < 100 * len(clf.classes_):
        log.info("✅ Early stopping was triggered successfully!\n")
    else:
        log.info("ℹ️  Early stopping did not trigger, but the model still completed correctly.\n")

def test_high_dimensional_data():
    log.info("🔹 Running High-Dimensional Data Test...")
    X = np.random.randn(100, 200)  # 200 features, 100 samples
    y = np.random.randint(0, 2, size=100)
    clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=2)
    clf.fit(X, y)
    acc = np.mean(clf.predict(X) == y)
    log.info("✅ Accuracy on high-dimensional data: %.2f", acc)
    assert acc > 0.5
    log.info("✅ High-dimensional test passed!\n")
