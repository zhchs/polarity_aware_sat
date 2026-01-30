import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from collections import namedtuple

from torch_scatter import scatter_sum, scatter_mean, scatter_add
from torch_geometric.nn.inits import glorot, zeros

import models.torch_utils as torch_utils
from models.torch_utils import *


class PASAT(nn.Module):
    """  
        PASAT (Polarity-Aware SAT)
    """

    def __init__(self, opts, cfg):
        super(PASAT, self).__init__()
        self.cfg = cfg
        self.opts = opts

        # ----------------- ND2_3 Components (Decoupling) -----------------
        self.f_even = nn.ModuleList()
        self.f_odd = nn.ModuleList()
        self.f_inv = nn.ModuleList()
        self.f_s = nn.ModuleList()
        self.f_d = nn.ModuleList()
        
        # ----------------- NHD1_0_0 Components (HyperConv) -----------------
        self.hyperconv_lins = nn.ModuleList()
        self.hyperconv_biases = nn.ParameterList()
        self.L_updates = nn.ModuleList()
        self.l_layer_norms = nn.ModuleList()
        self.c_layer_norms = nn.ModuleList()

        for i in range(cfg['n_rounds']):
            self.l_layer_norms.append(nn.LayerNorm(cfg['d']))
            self.c_layer_norms.append(nn.LayerNorm(cfg['d']))

            # --- Decoupling MLPs ---
            self.f_even.append(MLP(cfg=cfg, num_layers=cfg['n_even_layers'],
                                   input_dim=2 * cfg['d'], hidden_dim=cfg['d'], output_dim=cfg['d'],
                                   activation=cfg['mlp_transfer_fn']))
            self.f_odd.append(MLP(cfg=cfg, num_layers=cfg['n_odd_layers'],
                                  input_dim=2 * cfg['d'], hidden_dim=cfg['d'], output_dim=cfg['d'],
                                  activation=cfg['mlp_transfer_fn']))

            self.f_inv.append(MLP(cfg=cfg, num_layers=cfg['n_inv_layers'],
                                  input_dim=cfg['d'], hidden_dim=cfg['d'], output_dim=cfg['d'],
                                  activation=cfg['mlp_transfer_fn']))
            self.f_s.append(MLP(cfg=cfg, num_layers=cfg['n_s_layers'],
                                input_dim=cfg['d'], hidden_dim=cfg['d'], output_dim=cfg['d'],
                                activation=cfg['mlp_transfer_fn']))
            self.f_d.append(MLP(cfg=cfg, num_layers=cfg['n_d_layers'],
                                input_dim=cfg['d'], hidden_dim=cfg['d'], output_dim=cfg['d'],
                                activation=cfg['mlp_transfer_fn']))

            # --- HyperConv Layers ---
            lin = nn.Linear(cfg['d'], cfg['d'], bias=False)
            bias = nn.Parameter(torch.zeros(cfg['d']))
            self.hyperconv_lins.append(lin)
            self.hyperconv_biases.append(bias)

            # Update L using [L, CL_msgs, flip(L)]
            self.L_updates.append(MLP(cfg=cfg, num_layers=cfg['n_update_layers'],
                                      input_dim=3 * cfg['d'], hidden_dim=cfg['d'],
                                      output_dim=cfg['d'],
                                      activation=cfg['mlp_transfer_fn']))

        # ----------------- Scales & Params -----------------
        self.V_init_scale = nn.Parameter(torch.tensor(
            1.0 / math.sqrt(2 * cfg['d']), dtype=torch.float32))
        
        # Used in NHD1_0_0 for CL messages
        self.CL_scale = nn.Parameter(torch.tensor(
            cfg['CL_scale'], dtype=torch.float32))

        # ----------------- CC (Clause-Clause) Interaction -----------------
        self.cc_lin = nn.Linear(cfg['d'], cfg['d'], bias=False)
        self.cc_alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        
        if cfg['mlp_transfer_fn'] == 'relu':
            self.cc_activation = F.relu
        elif cfg['mlp_transfer_fn'] == 'tanh':
            self.cc_activation = torch.tanh
        elif cfg['mlp_transfer_fn'] == 'relu6':
            self.cc_activation = F.relu6
        else:
            self.cc_activation = lambda x: x

        # ----------------- Readout -----------------
        _v_act_fn = cfg['mlp_transfer_fn'] if opts.v_act_fn else None

        self.V_inv_score = MLP(cfg=cfg, num_layers=cfg['n_score_layers'],
                               input_dim=cfg['d'], hidden_dim=cfg['d'], output_dim=1,
                               activation=_v_act_fn)
        self.V_pair_score = MLP(cfg=cfg, num_layers=cfg['n_score_layers'],
                                input_dim=2 * cfg['d'], hidden_dim=cfg['d'], output_dim=1,
                                activation=_v_act_fn)

        self.res_gnn = opts.res_gnn
        self._device = opts.device

        self.reset_parameters()

    def reset_parameters(self):
        for lin, bias in zip(self.hyperconv_lins, self.hyperconv_biases):
            glorot(lin.weight)
            zeros(bias)

        glorot(self.cc_lin.weight)
        nn.init.constant_(self.V_init_scale, 1.0 / math.sqrt(2 * self.cfg['d']))

    def forward(self, n_vars, n_clauses, clause_index, literal_index,
                cc_edge_index=None, cc_edge_weight=None,
                l_batch=None, c_batch=None):
        n_vars = n_vars.sum().item()
        n_lits = 2 * n_vars
        n_clauses = n_clauses.sum().item()

        # Initialize V (Variables)
        V = torch.ones(size=(n_vars, 2 * self.cfg['d']),
                       dtype=torch.float32).to(literal_index.device) * self.V_init_scale

        hyperedge_index = torch.vstack([literal_index, clause_index])

        def flip(lits):
            pl_emb, ul_emb = torch.chunk(lits.reshape(n_lits // 2, -1), 2, 1)
            return torch.cat([ul_emb, pl_emb], dim=1).reshape(n_lits, -1)

        s_list, d_list = [], []
        s_tilde_list, d_tilde_list = [], []
        var_inv_list, var_pair_list = [], []

        for t in range(self.cfg['n_rounds']):
            # 1. Decouple V -> s, d -> L
            var_s, var_d = self.f_even[t](V), self.f_odd[t](V)
            s_list.append(var_s)
            d_list.append(var_d)
            
            L_pos = var_s + var_d
            L_neg = var_s - var_d
            L = torch.stack([L_pos, L_neg], dim=1).reshape(n_lits, -1)
            
            if self.training:
                check_numerics(L, message=f"L before message passing {t}")

            V_pre = V
            L_old = L

            # 2. Hypergraph Message Passing (NHD1_0_0 Style)
            # ---------------- HypergraphConv ----------------
            x_proj = self.hyperconv_lins[t](L)

            hyper_nodes, hyper_edges = hyperedge_index

            # L -> C Aggregation
            B = scatter_add(x_proj.new_ones(hyper_edges.size(0)),
                            hyper_edges, dim=0, dim_size=n_clauses)
            B_inv = B.reciprocal()
            B_inv[~torch.isfinite(B_inv)] = 0

            hyperedge_feat = scatter_sum(B_inv[hyper_edges].unsqueeze(-1) * x_proj[hyper_nodes],
                                         hyper_edges, dim=0, dim_size=n_clauses)
            
            # Inner message-passing among clauses (CC)
            if cc_edge_index is not None and cc_edge_index.numel() > 0:
                # [v1_0_3L Change]: No Pre-Norm before CC branch
                # cc_feat = self.c_layer_norms[t](hyperedge_feat) 
                cc_x = self.cc_lin(hyperedge_feat)
                cc_row, cc_col = cc_edge_index

                if cc_edge_weight is not None:
                    deg = scatter_add(cc_edge_weight, cc_row, dim=0, dim_size=n_clauses)
                else:
                    deg = scatter_add(cc_x.new_ones(cc_row.size(0)), cc_row, dim=0, dim_size=n_clauses)
                
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[~torch.isfinite(deg_inv_sqrt)] = 0

                norm = deg_inv_sqrt[cc_row] * deg_inv_sqrt[cc_col]

                # Use sparse matrix multiplication to avoid OOM from edge expansion (E x d tensor)
                vals = norm
                if cc_edge_weight is not None:
                    vals = vals * cc_edge_weight
                
                # Indices: (row=target, col=source) -> (cc_col, cc_row)
                adj = torch.sparse_coo_tensor(torch.stack([cc_col, cc_row]), vals, (n_clauses, n_clauses))
                delta_C = torch.sparse.mm(adj.coalesce(), cc_x)

                delta_C = self.cc_activation(delta_C)

                # [v1_0_3L Change]: Post-Norm at residual connection
                # hyperedge_feat = hyperedge_feat + self.cc_alpha * delta_C
                # Applying LayerNorm(x + delta)
                hyperedge_feat = self.c_layer_norms[t](hyperedge_feat + self.cc_alpha * delta_C)

            # C -> L Aggregation
            D = scatter_add(x_proj.new_ones(hyper_edges.size(0)),
                            hyper_nodes, dim=0, dim_size=n_lits)
            D_inv = D.reciprocal()
            D_inv[~torch.isfinite(D_inv)] = 0

            CL_msgs = scatter_sum(D_inv[hyper_nodes].unsqueeze(-1) * hyperedge_feat[hyper_edges],
                                   hyper_nodes, dim=0, dim_size=n_lits)

            CL_msgs = CL_msgs + self.hyperconv_biases[t]
            # ------------------------------------------------

            CL_msgs = CL_msgs * self.CL_scale

            # Update L
            L = self.L_updates[t](torch.cat([L, CL_msgs, flip(L)], dim=-1))
            check_numerics(L, message=f"L after update {t}")

            L = self.l_layer_norms[t](L)
            # L = torch_utils.normalize(
            #    L, axis=self.cfg['norm_axis'], eps=self.cfg['norm_eps'])
            check_numerics(L, message=f"L after norm {t}")

            if self.cfg.get('res_layers', False):
                L = L + L_old

            # 3. Recouple L -> s_tilde, d_tilde -> V
            pl_emb, ul_emb = torch.chunk(L.reshape(n_lits // 2, -1), 2, 1)
            var_s_tilde = 0.5 * (pl_emb + ul_emb)
            var_d_tilde = 0.5 * (pl_emb - ul_emb)
            s_tilde_list.append(var_s_tilde)
            d_tilde_list.append(var_d_tilde)

            var_inv = self.f_inv[t](var_s_tilde)
            var_pair = torch.cat([self.f_s[t](var_s_tilde),
                                  self.f_d[t](var_d_tilde)], dim=1)
            
            if self.training:
                check_numerics(var_inv, message=f"var_inv {t}")
                check_numerics(var_pair, message=f"var_pair {t}")

            var_inv_list.append(var_inv)
            var_pair_list.append(var_pair)

            V = var_pair
            if self.training:
                check_numerics(V, message=f"V after to_var {t}")

            if self.res_gnn:
                V = V + V_pre
                if self.training:
                    check_numerics(V, message=f"V after res {t}")

        # 4. Readout
        V_inv_scores = self.V_inv_score(var_inv_list[-1])
        V_inv_scores = V_inv_scores.squeeze(-1)
        pi_assign_logits = self.V_pair_score(var_pair_list[-1]).squeeze(-1)

        return PASATGuesses(pi_var_logits=V_inv_scores,
                                  pi_assign_logits=pi_assign_logits,
                                  var_s=s_list if self.training else None,
                                  var_d=d_list if self.training else None,
                                  var_s_tilde=s_tilde_list if self.training else None,
                                  var_d_tilde=d_tilde_list if self.training else None,
                                  gate_reg=None
                                  )
