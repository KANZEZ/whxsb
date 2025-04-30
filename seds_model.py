import numpy as np

class SEDSModel:
    def __init__(self, Priors, Mu, Sigma, att, M):
        self.Priors = Priors
        self.Mu = Mu
        self.Sigma = Sigma
        self.att = att
        self.M = M
        self.D = Mu.shape[0]

    @staticmethod
    def load(path):
        data = np.load(path)
        return SEDSModel(data["Priors"], data["Mu"], data["Sigma"], data["att"], int(data["M"]))

    def predict(self, x):  # x: (2,) or (N,2)
        x = np.atleast_2d(x) - self.att
        N = x.shape[0]
        K = len(self.Priors)
        vel = np.zeros((N, self.M))
        probs = np.zeros((K, N))

        # === 预计算每个 component 的 inv(sigma_x) 和 A 矩阵 ===
        mu_x_list = []
        mu_dx_list = []
        sigma_x_inv_list = []
        A_list = []

        for k in range(K):
            mu_x = self.Mu[:self.M, k]
            mu_dx = self.Mu[self.M:, k]
            sigma_x = self.Sigma[:self.M, :self.M, k]
            sigma_xdx = self.Sigma[:self.M, self.M:, k]
            sigma_x_inv = np.linalg.inv(sigma_x)
            A = sigma_xdx.T @ sigma_x_inv

            mu_x_list.append(mu_x)
            mu_dx_list.append(mu_dx)
            sigma_x_inv_list.append(sigma_x_inv)
            A_list.append(A)

        # === 计算每个点在每个 GMM component 中的责任概率 ===
        for k in range(K):
            mu_x = mu_x_list[k]
            sigma_x_inv = sigma_x_inv_list[k]
            det_sigma_x = np.linalg.det(np.linalg.inv(sigma_x_inv))  # = det(sigma_x)

            for i in range(N):
                xi = x[i]
                diff = xi - mu_x
                probs[k, i] = self.Priors[k] * np.exp(-0.5 * diff @ sigma_x_inv @ diff) / \
                            np.sqrt((2 * np.pi)**self.M * det_sigma_x)

        h = probs / np.sum(probs, axis=0, keepdims=True)

        # === 加权 GMR 输出 ===
        for k in range(K):
            mu_x = mu_x_list[k]
            mu_dx = mu_dx_list[k]
            A = A_list[k]

            for i in range(N):
                xi = x[i]
                dx_k = mu_dx + A @ (xi - mu_x)
                vel[i] += h[k, i] * dx_k

        return vel if len(vel) > 1 else vel[0]

    def lyapunov(self, x):
        x = np.atleast_2d(x) - self.att
        return np.sum(x * x, axis=1)  # V(x) = ||x - att||^2

    def lyapunov_dot(self, x):
        x = np.atleast_2d(x)
        v = self.predict(x)
        x_shift = x - self.att
        return np.sum(2 * x_shift * v, axis=1)  # \dot{V}(x) = 2(x-att)^T * f(x)
