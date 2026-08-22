import torch
import torch.nn as nn

class UniversalPINN_Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 5)
        )
        self.discovery_net = nn.Sequential(
            nn.Linear(5, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 5)
        )
        self.k_pg = 0.40
        self.c_pn = 0.20
        self.s_nr = 0.10
        self.mu_c = 0.10
        self.mu_d = 0.02
        self.pam_0 = 80.0

    def forward(self, t):
        return torch.nn.functional.softplus(self.state_net(t)) + 1e-4

    def compute_residual_profile(self, t_dense):
        t_dense.requires_grad_(True)
        x = self.forward(t_dense)
        
        dx_dt = []
        for i in range(5):
            g = torch.autograd.grad(
                outputs=x[:, i],
                inputs=t_dense,
                grad_outputs=torch.ones_like(x[:, i]),
                create_graph=False,
                retain_graph=True
            )[0]
            dx_dt.append(g)
        dx_dt = torch.cat(dx_dt, dim=1)
        
        P, N, C, D, PAM = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4], x[:, 4:5]
        f_known = torch.cat([
            self.k_pg * P - self.c_pn * N * P,
            self.s_nr * C - 0.05 * N,
            0.3 * (P + D) - self.mu_c * C,
            0.1 * N - self.mu_d * D,
            -0.05 * C * PAM - 0.01 * (PAM - self.pam_0)
        ], dim=1)
        
        n_discovered = self.discovery_net(x.detach())
        residual_norm = torch.norm(dx_dt - (f_known + n_discovered), p=2, dim=1, keepdim=True)
        return residual_norm.detach()

    def compute_loss(self, t_colloc, t_obs=None, x_obs=None, causal_eps=0.05, w_data=1.0, w_ode=0.1, w_sparse=1e-4):
        loss_data = torch.tensor(0.0, device=t_colloc.device)
        if t_obs is not None and x_obs is not None:
            x_pred_obs = self.forward(t_obs)
            loss_data = torch.mean((x_pred_obs - x_obs) ** 2)
            
        t_colloc.requires_grad_(True)
        x = self.forward(t_colloc)
        P, N, C, D, PAM = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4], x[:, 4:5]
        
        dx_dt = []
        for i in range(5):
            g = torch.autograd.grad(
                outputs=x[:, i],
                inputs=t_colloc,
                grad_outputs=torch.ones_like(x[:, i]),
                create_graph=True,
                retain_graph=True
            )[0]
            dx_dt.append(g)
        dx_dt = torch.cat(dx_dt, dim=1)
        
        f_known = torch.cat([
            self.k_pg * P - self.c_pn * N * P,
            self.s_nr * C - 0.05 * N,
            0.3 * (P + D) - self.mu_c * C,
            0.1 * N - self.mu_d * D,
            -0.05 * C * PAM - 0.01 * (PAM - self.pam_0)
        ], dim=1)
        
        n_discovered = self.discovery_net(x)
        residual = dx_dt - (f_known + n_discovered)
        
        causal_weights = torch.exp(-causal_eps * t_colloc.detach())
        loss_ode = torch.mean(causal_weights * (residual ** 2))
        loss_sparse = torch.mean(torch.abs(n_discovered))
        
        total_loss = w_data * loss_data + w_ode * loss_ode + w_sparse * loss_sparse
        return total_loss, loss_data, loss_ode, loss_sparse
