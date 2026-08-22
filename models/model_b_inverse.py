import torch
import torch.nn as nn

class ConditionalPatientPINN(nn.Module):
    """
    PINN Bayesiana Hierárquica com Modelagem de Efeitos Mistos (NLME).
    Infere distribuições de parâmetros q(theta | z) = N(mu_theta, sigma_theta^2).
    """
    def __init__(self, static_dim=4):
        super().__init__()
        # Rede de Parâmetros com Inferência Variacional
        self.encoder_mu = nn.Sequential(
            nn.Linear(static_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 4), nn.Softplus() # Parâmetros Médios (mu)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(static_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 4)                # Incerteza (log(sigma^2))
        )
        
        # Trajetória de Estado Condicional
        self.state_net = nn.Sequential(
            nn.Linear(1 + static_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 5)
        )

    def sample_parameters(self, z, num_samples=100):
        """Amostragem Monte Carlo para estimar Intervalos de Confiança (IC 95%)."""
        mu = self.encoder_mu(z)
        logvar = self.encoder_logvar(z)
        std = torch.exp(0.5 * logvar)
        
        eps = torch.randn(num_samples, *mu.shape, device=z.device)
        samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        
        # Percentis 2.5% e 97.5% (IC 95%)
        ic_lower = torch.quantile(samples, 0.025, dim=0)
        ic_upper = torch.quantile(samples, 0.975, dim=0)
        return mu, std, ic_lower, ic_upper

    def forward(self, t, z):
        inputs = torch.cat([t, z], dim=1)
        return torch.nn.functional.softplus(self.state_net(inputs)) + 1e-4

    def get_patient_parameters(self, z):
        return self.encoder_mu(z)
