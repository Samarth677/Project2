import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=1, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(y) < self.min_samples_split or depth == self.max_depth:
            return np.mean(y)

        best_split = None
        best_mse = float("inf")
        best_left, best_right = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = ~left_mask
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                mse = self._mse(y[left_mask], y[right_mask])
                if mse < best_mse:
                    best_mse = mse
                    best_split = (feature, t)
                    best_left, best_right = left_mask, right_mask

        if best_split is None:
            return np.mean(y)

        left_tree = self._build_tree(X[best_left], y[best_left], depth + 1)
        right_tree = self._build_tree(X[best_right], y[best_right], depth + 1)
        return (best_split[0], best_split[1], left_tree, right_tree)

    def _mse(self, left_y, right_y):
        left_mse = np.var(left_y) * len(left_y)
        right_mse = np.var(right_y) * len(right_y)
        return (left_mse + right_mse) / (len(left_y) + len(right_y))

    def _predict_tree(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left, right = node
        return self._predict_tree(x, left) if x[feature] <= threshold else self._predict_tree(x, right)

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=1,
                 early_stopping=False, patience=5, val_fraction=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.early_stopping = early_stopping
        self.patience = patience
        self.val_fraction = val_fraction
        self.trees = []
        self.classes_ = None
        self.F0 = {}

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.trees = {cls: [] for cls in self.classes_}
        self.F0 = {}

        if self.early_stopping:
            split = int(len(X) * (1 - self.val_fraction))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
        else:
            X_train, y_train = X, y

        for cls in self.classes_:
            y_bin = np.where(y_train == cls, 1, -1)
            self.F0[cls] = 0.5 * np.log((1 + y_bin.mean()) / (1 - y_bin.mean()))
            Fm = np.full(len(y_train), self.F0[cls])
            best_loss = float('inf')
            rounds_without_improvement = 0

            for m in range(self.n_estimators):
                residual = 2 * y_bin / (1 + np.exp(2 * y_bin * Fm))
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X_train, residual)
                update = tree.predict(X_train)
                Fm += self.learning_rate * update
                self.trees[cls].append(tree)

                # Early stopping logic
                if self.early_stopping:
                    F_val = np.full(len(X_val), self.F0[cls])
                    for t in self.trees[cls]:
                        F_val += self.learning_rate * t.predict(X_val)
                    probs_val = self._sigmoid(F_val)
                    y_val_bin = np.where(y_val == cls, 1, 0)
                    loss = -np.mean(y_val_bin * np.log(probs_val + 1e-10) + (1 - y_val_bin) * np.log(1 - probs_val + 1e-10))

                    if loss < best_loss:
                        best_loss = loss
                        rounds_without_improvement = 0
                    else:
                        rounds_without_improvement += 1

                    if rounds_without_improvement >= self.patience:
                        print(f"[Early Stopping] Class {cls}: Stopped at iteration {m}")
                        break

    def predict_proba(self, X):
        scores = []
        for cls in self.classes_:
            Fm = np.full((X.shape[0],), self.F0[cls])
            for tree in self.trees[cls]:
                Fm += self.learning_rate * tree.predict(X)
            scores.append(self._sigmoid(Fm))
        scores = np.vstack(scores).T
        scores /= scores.sum(axis=1, keepdims=True)
        return scores

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

