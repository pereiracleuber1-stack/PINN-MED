import numpy as np

class PINN_SINDy_Extractor:
    @staticmethod
    def build_library(X):
        P, N, C, D, PAM = [X[:, i:i+1] for i in range(5)]
        theta_list = [
            np.ones_like(P), P, N, C, D, PAM,
            P**2, N**2, C**2,
            P * N, C * PAM, (P + D),
            P / (1.0 + 0.5 * P),
            C / (1.0 + 0.2 * C)
        ]
        names = ["1", "P", "N", "C", "D", "PAM", "P^2", "N^2", "C^2", "P*N", "C*PAM", "(P+D)", "P/(1+0.5P)", "C/(1+0.2C)"]
        return np.hstack(theta_list), names

    @staticmethod
    def fit_sr3(Theta, dX_dt, lambda_reg=0.08, kappa=1.0, max_iter=50, tol=1e-5):
        n_features = Theta.shape[1]
        n_targets = dX_dt.shape[1]
        
        Xi = np.linalg.lstsq(Theta, dX_dt, rcond=None)[0]
        W = Xi.copy()
        
        A = Theta.T @ Theta + (1.0 / kappa) * np.eye(n_features)
        b_base = Theta.T @ dX_dt
        
        for _ in range(max_iter):
            Xi_prev = Xi.copy()
            Xi = np.linalg.solve(A, b_base + (1.0 / kappa) * W)
            
            threshold = lambda_reg * kappa
            W = np.sign(Xi) * np.maximum(np.abs(Xi) - threshold, 0.0)
            
            # Aplica a restrição na coluna da PAM apenas se o sistema completo (5 variáveis) estiver presente
            if n_targets > 4:
                W[0, 4] = 0.0
                
            if np.linalg.norm(Xi - Xi_prev) / (np.linalg.norm(Xi_prev) + 1e-8) < tol:
                break
                
        return W
