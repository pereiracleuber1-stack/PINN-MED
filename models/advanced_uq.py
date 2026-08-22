import numpy as np
import torch

class NeuralSDE_EulerMaruyama:
    """
    Motor Estocástico Physics-Informed Neural SDE (Cálculo de Itô)
    dx_t = [f(x_t) + N_phi(x_t)] dt + g_psi(x_t) dW_t
    """
    def __init__(self, dt=0.2, t_max=48.0):
        self.dt = dt
        self.t_max = t_max
        self.time_steps = int(t_max / dt) + 1
        self.t_grid = np.linspace(0, t_max, self.time_steps)
        
    def simulate_ensemble(self, x0, k_pg=0.45, c_pn=0.22, mu_c=0.14, diff_scale=0.03, num_paths=100, seed=42):
        np.random.seed(seed)
        paths = np.zeros((num_paths, self.time_steps, 5))
        paths[:, 0, :] = x0
        sq_dt = np.sqrt(self.dt)
        
        for k in range(self.time_steps - 1):
            x_curr = paths[:, k, :]
            P = np.clip(x_curr[:, 0], 0.0, 5.0)
            N = np.clip(x_curr[:, 1], 0.1, 5.0)
            C = np.clip(x_curr[:, 2], 0.0, 5.0)
            D = np.clip(x_curr[:, 3], 0.0, 5.0)
            Lac = np.clip(x_curr[:, 4], 0.8, 12.0)
            
            # Deriva Fisiológica Determinística f(x)
            dP = k_pg * P * (1.0 - P / 3.0) - (c_pn * P * N) / (1.0 + 0.3 * P)
            dN = 0.10 + (0.30 * C) / (1.0 + 0.5 * C) - 0.20 * N
            dC = (0.20 * P + 0.15 * D) / (1.0 + 0.5 * C) - mu_c * C
            dD = (0.08 * C) / (1.0 + 0.2 * D) - 0.10 * D
            dLac = 0.30 * D + 0.15 * C - 0.25 * (Lac - 1.8)
            
            drift = np.stack([dP, dN, dC, dD, dLac], axis=-1)
            
            # Matriz de Difusão g_psi(x) - Ruído Biológico Não-Estacionário de Itô
            g = diff_scale * np.stack([
                0.10 * np.sqrt(P + 0.01),
                0.08 * np.sqrt(N + 0.01),
                0.12 * np.sqrt(C + 0.01),
                0.08 * np.sqrt(D + 0.01),
                0.15 * np.sqrt(Lac + 0.01)
            ], axis=-1)
            
            # Incremento do Processo de Wiener
            dW = np.random.normal(0, 1, size=(num_paths, 5))
            
            # Atualização de Euler-Maruyama
            x_next = x_curr + drift * self.dt + g * sq_dt * dW
            paths[:, k+1, :] = np.clip(x_next, 0.0, 15.0)
            
        return self.t_grid, paths

class EpistemicAleatoricDecomposer:
    """
    Decomposição Formal de Incerteza Heteroscedástica
    sigma_total^2(t) = sigma_aleatoric^2(t) + sigma_epistemic^2(t)
    """
    @staticmethod
    def decompose(sde_engine, x0, k_candidates=[0.42, 0.45, 0.48, 0.51], sensor_noise_std=0.12):
        ensemble_means = []
        for k_val in k_candidates:
            _, p = sde_engine.simulate_ensemble(x0=x0, k_pg=k_val, num_paths=30, seed=42)
            ensemble_means.append(np.mean(p[:, :, 4], axis=0))
            
        ensemble_means = np.array(ensemble_means)
        mu_global = np.mean(ensemble_means, axis=0)
        
        # Incerteza Epistêmica vs. Aleatória
        var_epistemic = np.var(ensemble_means, axis=0)
        var_aleatoric = np.full_like(var_epistemic, sensor_noise_std**2) + 0.03 * (mu_global / 2.0)**2
        var_total = var_epistemic + var_aleatoric
        
        return mu_global, np.sqrt(var_total), np.sqrt(var_epistemic), np.sqrt(var_aleatoric)

class PhysicsConformalCalibrator:
    """
    Functional Conformal Prediction com Restrição Física
    Garante cobertura da trajetória inteira: P(forall t, Y(t) in C_{1-alpha}(t)) >= 1 - alpha
    """
    @staticmethod
    def calibrate_cohort(sde_engine, N_cal=250, alpha=0.05):
        t_grid, cal_paths = sde_engine.simulate_ensemble(
            x0=[1.2, 0.8, 0.4, 0.1, 2.2], k_pg=0.48, c_pn=0.22, mu_c=0.14, num_paths=N_cal, seed=123
        )
        y_cal_true = cal_paths[:, :, 4]
        mu_cal = np.mean(y_cal_true, axis=0)
        std_cal = np.std(y_cal_true, axis=0) + 0.02
        
        # Score de Supremo Funcional (Pior desvio na série temporal inteira de cada paciente)
        pointwise_scores = np.abs(y_cal_true - np.tile(mu_cal, (N_cal, 1))) / np.tile(std_cal, (N_cal, 1))
        supremum_scores = np.max(pointwise_scores, axis=1)
        
        # Quantil Funcional Conforme com Correção de Amostra Finita
        k_idx = int(np.ceil((N_cal + 1) * (1.0 - alpha))) - 1
        k_idx = min(max(k_idx, 0), N_cal - 1)
        q_hat = float(np.sort(supremum_scores)[k_idx])
        
        return q_hat, supremum_scores, mu_cal, std_cal
        
    @staticmethod
    def evaluate_test_coverage(sde_engine, q_hat, mu_pred, std_pred, N_test=200):
        _, test_paths = sde_engine.simulate_ensemble(
            x0=[1.2, 0.8, 0.4, 0.1, 2.2], k_pg=0.48, c_pn=0.22, mu_c=0.14, num_paths=N_test, seed=999
        )
        y_test = test_paths[:, :, 4]
        
        lower_conf = np.maximum(mu_pred - q_hat * std_pred, 0.5)
        upper_conf = mu_pred + q_hat * std_pred
        
        # 1. Cobertura da Trajetória Inteira (Simultânea)
        in_conf_traj = np.all((y_test >= lower_conf) & (y_test <= upper_conf), axis=1)
        cov_traj = float(np.mean(in_conf_traj))
        
        # 2. Cobertura Média Ponto a Ponto (Pointwise)
        in_conf_point = (y_test >= lower_conf) & (y_test <= upper_conf)
        cov_point = float(np.mean(in_conf_point))
        
        # Comparativo Gaussiano Tradicional (1.96)
        lower_naive = np.maximum(mu_pred - 1.96 * std_pred, 0.5)
        upper_naive = mu_pred + 1.96 * std_pred
        in_naive_traj = np.all((y_test >= lower_naive) & (y_test <= upper_naive), axis=1)
        cov_naive_traj = float(np.mean(in_naive_traj))
        
        return cov_traj, cov_point, cov_naive_traj, lower_conf, upper_conf
