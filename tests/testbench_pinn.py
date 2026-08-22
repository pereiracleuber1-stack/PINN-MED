import unittest
import torch
import numpy as np
from models.model_a_upinn import UniversalPINN_Residual
from models.model_b_inverse import ConditionalPatientPINN
from models.model_c_sindy import PINN_SINDy_Extractor

class TestEnterprisePINN(unittest.TestCase):
    def test_model_a_causal_and_rad(self):
        model = UniversalPINN_Residual()
        t = torch.linspace(0, 24, 40).view(-1, 1)
        profile = model.compute_residual_profile(t)
        self.assertEqual(profile.shape, (40, 1), "Falha no cálculo do perfil de colocação adaptativa.")
        loss, _, _, _ = model.compute_loss(t)
        self.assertFalse(torch.isnan(loss), "Perda causal retornou NaN.")

    def test_model_b_bayesian_ci(self):
        model = ConditionalPatientPINN(static_dim=4)
        z = torch.tensor([[65.0, 5.0, 2.0, 26.0]])
        mu, std, ic_low, ic_high = model.sample_parameters(z, num_samples=50)
        self.assertTrue(torch.all(ic_high >= ic_low), "Falha na consistência do intervalo de confiança.")

    def test_model_c_sr3_convergence(self):
        X = np.random.uniform(0.1, 2.0, (100, 5))
        Theta, _ = PINN_SINDy_Extractor.build_library(X)
        dX = 0.4 * X[:, 0:1] # Derivada com 1 termo real
        W = PINN_SINDy_Extractor.fit_sr3(Theta, dX, lambda_reg=0.05)
        # O termo correto (índice 1 correspondente a P) deve ser o dominante
        self.assertGreater(np.abs(W[1, 0]), 0.1, "SR3 não identificou o termo governante principal.")

if __name__ == "__main__":
    unittest.main()
