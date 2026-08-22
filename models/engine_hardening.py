import torch
import torch.nn as nn
import numpy as np

class AdaptiveLossWeighting:
    """
    Equilíbrio Dinâmico de Gradientes (NTK / GradNorm Proxy)
    Garante convergência estável sem ajuste empírico manual de hiperparâmetros de perda.
    """
    @staticmethod
    def balance_losses(loss_data, loss_pde, loss_ic, weights, alpha=0.9):
        # Atualização exponencial suave de pesos relativos
        with torch.no_grad():
            w_d = 1.0 / (loss_data.item() + 1e-6)
            w_p = 1.0 / (loss_pde.item() + 1e-6)
            w_i = 1.0 / (loss_ic.item() + 1e-6)
            total = w_d + w_p + w_i
            new_weights = torch.tensor([w_d/total, w_p/total, w_i/total], dtype=torch.float32)
            updated_weights = alpha * weights + (1 - alpha) * new_weights
        return updated_weights

class SobolSensitivityAnalyzer:
    """
    Análise de Sensibilidade Global (GSA) de Sobol para Parâmetros Fisiológicos Críticos.
    """
    @staticmethod
    def compute_sobol_indices(param_bounds, eval_function, N=256):
        num_vars = len(param_bounds)
        # Amostragem quase-aleatória de Monte Carlo
        A = np.random.uniform(0, 1, (N, num_vars))
        B = np.random.uniform(0, 1, (N, num_vars))
        
        # Escala para limites reais
        lows = np.array([b[0] for b in param_bounds])
        highs = np.array([b[1] for b in param_bounds])
        A_real = lows + A * (highs - lows)
        B_real = lows + B * (highs - lows)
        
        y_A = np.array([eval_function(x) for x in A_real])
        y_B = np.array([eval_function(x) for x in B_real])
        var_total = np.var(np.concatenate([y_A, y_B])) + 1e-8
        
        # Índices de Primeira Ordem (S_i) e Totais (S_Ti)
        S_i = []
        S_Ti = []
        for i in range(num_vars):
            AB_i = np.copy(A_real)
            AB_i[:, i] = B_real[:, i]
            y_AB_i = np.array([eval_function(x) for x in AB_i])
            
            # Estimador Jansen/Saltelli
            s_first = np.mean(y_B * (y_AB_i - y_A)) / var_total
            s_total = 0.5 * np.mean((y_A - y_AB_i)**2) / var_total
            S_i.append(np.clip(s_first, 0.0, 1.0))
            S_Ti.append(np.clip(s_total, 0.0, 1.0))
            
        return np.array(S_i), np.array(S_Ti)
