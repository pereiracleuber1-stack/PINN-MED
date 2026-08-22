import torch
import torch.nn as nn
import numpy as np

class DeepONetPhysiology(nn.Module):
    """
    Deep Operator Network (DeepONet) para Modelagem Fisiológica Contínua
    Aprende o operador que mapeia o histórico de sinais vitais u(t) 
    para o estado metabólico futuro s(t) = [P, N, C, L](t) em < 2 milissegundos.
    """
    def __init__(self, branch_dim=6, trunk_dim=1, latent_dim=64, output_dim=4):
        super(DeepONetPhysiology, self).__init__()
        
        # Branch Net: Processa o vetor de sinais vitais e biomarcadores
        self.branch_net = nn.Sequential(
            nn.Linear(branch_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim * output_dim)
        )
        
        # Trunk Net: Processa a coordenada temporal contínua t
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim)
        )
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, u_branch, t_trunk):
        b_out = self.branch_net(u_branch).view(-1, self.output_dim, self.latent_dim)
        t_out = self.trunk_net(t_trunk)
        out = torch.einsum('bdl,tl->btd', b_out, t_out) + self.bias
        return torch.relu(out)

    def predict_trajectory_fast(self, vital_vector, time_horizon_hours=24, n_steps=100):
        self.eval()
        with torch.no_grad():
            u_t = torch.tensor(vital_vector, dtype=torch.float32).unsqueeze(0)
            t_t = torch.linspace(0, time_horizon_hours, n_steps, dtype=torch.float32).unsqueeze(1)
            pred = self.forward(u_t, t_t).squeeze(0).cpu().numpy()
            t_axis = t_t.squeeze().cpu().numpy()
        return t_axis, pred
