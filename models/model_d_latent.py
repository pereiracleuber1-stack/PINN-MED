import torch
import torch.nn as nn

class PhysiologicalLatentODE(nn.Module):
    """
    Neural Jump-ODE com tratamento contínuo-discreto de intervenções médicas (Bolus).
    """
    def __init__(self, observed_dim=15, latent_dim=5):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder de Séries Temporais Irregulares
        self.encoder_rnn = nn.GRU(observed_dim, 64, batch_first=True)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Campo Vetorial de Drift Contínuo
        self.latent_drift = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, latent_dim)
        )
        
        # Operador de Salto Discreto (Jump Operator para intervenções e medicações)
        self.jump_net = nn.Sequential(
            nn.Linear(latent_dim + 1, 32), nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder Clínico
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, observed_dim)
        )

    def apply_intervention_jump(self, z, dose_bolus):
        """Aplica descontinuidade instantânea no estado latente devido a medicação."""
        jump_input = torch.cat([z, dose_bolus], dim=-1)
        delta_z = self.jump_net(jump_input)
        return z + delta_z

    def integrate_rk4_step(self, z, dt):
        """Passo Runge-Kutta 4 clássico para evolução temporal contínua."""
        k1 = self.latent_drift(z)
        k2 = self.latent_drift(z + 0.5 * dt * k1)
        k3 = self.latent_drift(z + 0.5 * dt * k2)
        k4 = self.latent_drift(z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
