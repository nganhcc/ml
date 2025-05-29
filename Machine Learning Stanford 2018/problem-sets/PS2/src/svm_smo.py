import numpy as np

class SVM_SMO:
    def __init__(self, C=1.0, tol=1e-3, max_passes=5, kernel=None, gamma=1.0):
        self.C = C  # Regularization parameter
        self.tol = tol  # Tolerance for stopping criterion
        self.max_passes = max_passes  # Maximum number of passes without changes
        self.gamma = gamma  # Parameter for the Gaussian (RBF) kernel
        self.kernel = kernel if kernel else self.gaussian_kernel  # Default Gaussian kernel
        self.alphas = None  # Lagrange multipliers
        self.b = 0  # Bias term
        self.X = None  # Training data
        self.y = None  # Training labels

    def gaussian_kernel(self, x1, x2):
        """Gaussian (RBF) kernel function."""
        return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * self.gamma**2))

    def compute_kernel_matrix(self, X):
        """Compute the kernel matrix using the kernel function."""
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])
        return K

    def fit(self, X, y):
        """Train the SVM using Sequential Minimal Optimization (SMO)."""
        n_samples, n_features = X.shape
        self.X = X
        self.y = y
        self.alphas = np.zeros(n_samples)
        self.b = 0
        K = self.compute_kernel_matrix(X)

        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(n_samples):
                E_i = self._error(i, K)

                if (self.y[i] * E_i < -self.tol and self.alphas[i] < self.C) or \
                   (self.y[i] * E_i > self.tol and self.alphas[i] > 0):

                    # Randomly choose j different from i
                    j = np.random.choice([x for x in range(n_samples) if x != i])
                    E_j = self._error(j, K)

                    # Save old alphas
                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]

                    # Compute bounds L and H
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])

                    if L == H:
                        continue

                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    self.alphas[j] -= self.y[j] * (E_i - E_j) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)

                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    self.alphas[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])

                    # Update b
                    b1 = self.b - E_i - self.y[i] * (self.alphas[i] - alpha_i_old) * K[i, i] \
                         - self.y[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - self.y[i] * (self.alphas[i] - alpha_i_old) * K[i, j] \
                         - self.y[j] * (self.alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            passes = passes + 1 if num_changed_alphas == 0 else 0

    def _error(self, i, K):
        """Compute the error for sample i."""
        f_xi = np.sum(self.alphas * self.y * K[:, i]) + self.b
        return f_xi - self.y[i]

    def predict(self, X):
        """Predict labels for the given test set X."""
        n_samples = X.shape[0]
        y_pred = []
        for i in range(n_samples):
            prediction = np.sum(self.alphas * self.y * np.array([self.kernel(x, X[i]) for x in self.X])) + self.b
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)


# Example usage
if __name__ == "__main__":
    # Training data
    X = np.array([[2, 3], [1, 1], [2, 2], [3, 3], [4, 1]])
    y = np.array([1, 1, -1, -1, 1])  # Labels (+1, -1)

    # Train SVM with Gaussian (RBF) kernel
    svm = SVM_SMO(C=1.0, tol=1e-3, max_passes=10, gamma=0.5)  # You can adjust gamma
    svm.fit(X, y)

    # Make predictions
    predictions = svm.predict(X)
    print("Predictions:", predictions)
