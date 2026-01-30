import torch
import torch.nn as nn
import numpy as np
import math
from collections import namedtuple
from torch_scatter import scatter_sum, scatter_mean

import models.torch_utils as torch_utils
from models.torch_utils import *


class NeuroCore(nn.Module):
    def __init__(self, opts, cfg):
        super(NeuroCore, self).__init__()
        self.cfg = cfg
        self.L_updates = nn.ModuleList()
        self.C_updates = nn.ModuleList()

        self.l_layer_norms = nn.ModuleList()
        self.c_layer_norms = nn.ModuleList()

        if cfg['repeat_layers']:
            for _ in range(cfg['n_rounds']):
                self.l_layer_norms.append(nn.LayerNorm(cfg['d']))
                self.c_layer_norms.append(nn.LayerNorm(cfg['d']))

                self.L_updates.append(MLP(cfg=cfg, num_layers=cfg['n_update_layers'],
                                          input_dim=2 * cfg['d'] + cfg['d'],
                                          hidden_dim=cfg['d'],
                                          output_dim=cfg['d'], activation=cfg['mlp_transfer_fn']))
                self.C_updates.append(MLP(cfg=cfg, num_layers=cfg['n_update_layers'],
                                          input_dim=cfg['d'] + cfg['d'],
                                          hidden_dim=cfg['d'],
                                          output_dim=cfg['d'], activation=cfg['mlp_transfer_fn']))
        else:
            l_mlp = MLP(cfg=cfg, num_layers=cfg['n_update_layers'],
                        input_dim=2 * cfg['d'] + cfg['d'],
                        hidden_dim=cfg['d'],
                        output_dim=cfg['d'], activation=cfg['mlp_transfer_fn'])
            c_mlp = MLP(cfg=cfg, num_layers=cfg['n_update_layers'],
                        input_dim=cfg['d'] + cfg['d'],
                        hidden_dim=cfg['d'],
                        output_dim=cfg['d'], activation=cfg['mlp_transfer_fn'])

            for _ in range(cfg['n_rounds']):
                self.l_layer_norms.append(nn.LayerNorm(cfg['d']))
                self.c_layer_norms.append(nn.LayerNorm(cfg['d']))
                self.L_updates.append(l_mlp)
                self.C_updates.append(c_mlp)

        self.L_init_scale = nn.Parameter(torch.tensor(
            1.0 / math.sqrt(cfg['d']), dtype=torch.float32))
        self.C_init_scale = nn.Parameter(torch.tensor(
            1.0 / math.sqrt(cfg['d']), dtype=torch.float32))
        self.LC_scale = nn.Parameter(torch.tensor(
            cfg['LC_scale'], dtype=torch.float32))
        self.CL_scale = nn.Parameter(torch.tensor(
            cfg['CL_scale'], dtype=torch.float32))

        _v_act_fn = cfg['mlp_transfer_fn'] if opts.v_act_fn else None
        self.V_score = MLP(cfg=cfg, num_layers=cfg['n_score_layers'], input_dim=2 * cfg['d'],
                           hidden_dim=cfg['d'], output_dim=1, activation=_v_act_fn)

    def forward(self, n_vars, n_clauses, clause_index, literal_index, l_batch=None, c_batch=None):
        n_vars = n_vars.sum().item()
        n_lits = 2 * n_vars
        n_clauses = n_clauses.sum().item()

        L = torch.ones(
            size=(n_lits, self.cfg['d']), dtype=torch.float32).to(literal_index.device) * self.L_init_scale
        C = torch.ones(
            size=(n_clauses, self.cfg['d']), dtype=torch.float32).to(clause_index.device) * self.C_init_scale

        # def flip(lits): return torch.cat([lits[n_vars:], lits[:n_vars]], dim=0)
        def flip(lits):
            pl_emb, ul_emb = torch.chunk(lits.reshape(n_lits // 2, -1), 2, 1)
            return torch.cat([ul_emb, pl_emb], dim=1).reshape(n_lits, -1)

        for t in range(self.cfg['n_rounds']):
            C_old, L_old = C, L

            # aggregate literal to clause messages
            l_msg_feat = L[literal_index]
            LC_msgs = scatter_sum(l_msg_feat, clause_index,
                                  dim=0, dim_size=n_clauses) * self.LC_scale
            C = self.C_updates[t](torch.cat([C, LC_msgs], dim=-1))
            check_numerics(C, message=f"C after update {t}")
            if self.cfg['res_layers']:
                C = C + C_old

            # aggregate clause/negated literal to literal messages
            c_msg_feat = C[clause_index]
            CL_msgs = scatter_sum(c_msg_feat, literal_index,
                                  dim=0, dim_size=n_lits) * self.CL_scale
            L = self.L_updates[t](torch.cat([L, CL_msgs, flip(L)], dim=-1))
            check_numerics(L, message=f"L after update {t}")
            if self.cfg['res_layers']:
                L = L + L_old

        # V = torch.cat([L[:n_vars], L[n_vars:]], dim=1)

        V = L.reshape(n_vars, -1)
        V_scores = self.V_score(V)  # (n_vars, 1)
        pi_core_var_logits = V_scores.squeeze(-1)

        return NeuroCoreGuesses(pi_core_var_logits=pi_core_var_logits)
