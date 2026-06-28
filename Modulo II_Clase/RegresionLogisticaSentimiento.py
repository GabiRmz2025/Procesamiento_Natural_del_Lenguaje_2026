#Clase Regresión Logística Sentimiento (vista en clase)
class RegresionLogisticaSentimiento:
    """Regresion logistica binaria con gradiente descendente batch."""

    def __init__(self, lr=0.1, epochs=500, lambda_=0.01, umbral=0.5):
        self.lr = lr; self.epochs = epochs
        self.lambda_ = lambda_; self.umbral = umbral
        self.w = None; self.b = 0.0; self.historial = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _perdida(self, y_hat, y):
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        ce = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return ce + (self.lambda_ / 2) * np.dot(self.w, self.w)

    def fit(self, X, y, verbose=True):
        N, D = X.shape
        self.w = np.zeros(D); self.b = 0.0; self.historial = []
        for epoch in range(self.epochs):
            y_hat = self._sigmoid(X.dot(self.w) + self.b)
            loss = self._perdida(y_hat, y)
            self.historial.append(loss)
            error = y_hat - y
            grad_w = X.T.dot(error) / N + self.lambda_ * self.w
            grad_b = np.mean(error)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            if verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                acc = accuracy_score(y, self.predecir(X))
                print(f"  Epoch {epoch:4d} | Perdida: {loss:.4f} | Acc train: {acc:.4f}")
        return self

    def predecir_proba(self, X):
        return self._sigmoid(X.dot(self.w) + self.b)

    def predecir(self, X):
        return (self.predecir_proba(X) >= self.umbral).astype(int)

    def palabras_importantes(self, vocab, top_n=10):
        idx_pos = np.argsort(self.w)[::-1][:top_n]
        idx_neg = np.argsort(self.w)[:top_n]
        return [(vocab[i], self.w[i]) for i in idx_pos], [(vocab[i], self.w[i]) for i in idx_neg]

print("Clase RegresionLogisticaSentimiento lista.")