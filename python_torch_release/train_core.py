import glob
import os
import random
import time
import argparse
import numpy as np
import json
import collections
import sys
from pathlib import Path
import datetime

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR, PolynomialLR, CosineAnnealingLR
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.utils import softmax

import util
import dataset.dataloader_bench as dataloader_bench

from models.neurocore import NeuroCore
from models.PASAT import PASAT


from models.torch_utils import check_numerics

from test_module import run_test_satbench
from test_module import run_test_satbench_batch_version

# torch.autograd.set_detect_anomaly(True)

''' ---------- model selection ---------- '''

model_baselines = ["neurocore"]
model_lcg_ccg_series = ["PASAT"]
model_neuro_diff_series = ["PASAT"]
model_1_series = []

model_select = {
    "neurocore": NeuroCore,
    "PASAT": PASAT,
}

''' ------------------------------------- '''


def flip_data(data, y_flip=False):
    d_data = data.clone()
    d_data.l_edge_index = d_data.l_edge_index ^ 1
    if y_flip and hasattr(d_data, 'y_var') and d_data.y_var is not None:
        mask1 = (d_data.y_var == 1)
        mask0 = (d_data.y_var == 0)
        d_data.y_var[mask1] = 0
        d_data.y_var[mask0] = 1
    return d_data


def kl_loss(logits, pi_targets, batch, batch_size):
    # batch/graph-wise averaged KL divergence
    # N = batch_size
    mask_sum_targets = scatter_sum(pi_targets, batch,
                                   dim=0, dim_size=batch_size)
    norm_mask_sum = mask_sum_targets[batch].clamp(min=1e-8)
    pi_targets = pi_targets / norm_mask_sum
    prob_per_graph = softmax(logits, batch)
    log_prob_per_graph = torch.log(prob_per_graph + 1e-10)
    kl_div_vec = pi_targets * \
        (torch.log(pi_targets + 1e-10) - log_prob_per_graph)
    kl_graph_sum = scatter_sum(
        kl_div_vec, batch, dim=0, dim_size=batch_size)
    # num_valid = (mask_sum_targets > 0).sum().item()
    return kl_graph_sum.mean()


def kl_loss_top_k(logits, pi_targets, batch, batch_size, top_k_ratio=0.5):
    """
    【新增】Top-K Hard Example Mining for KL Divergence.
    解决简单样本（Core易识别）掩盖困难样本的问题。
    """
    # 1. 归一化 Target (保持原有逻辑)
    mask_sum_targets = scatter_sum(
        pi_targets, batch, dim=0, dim_size=batch_size)
    norm_mask_sum = mask_sum_targets[batch].clamp(min=1e-8)
    pi_targets_norm = pi_targets / norm_mask_sum

    # 2. 计算预测分布
    prob_per_graph = softmax(logits, batch)
    log_prob_per_graph = torch.log(prob_per_graph + 1e-10)

    # 3. 计算节点级 KL 散度
    kl_div_vec = pi_targets_norm * \
        (torch.log(pi_targets_norm + 1e-10) - log_prob_per_graph)

    # 4. 聚合到图级别 (Shape: [Batch_Size])
    kl_graph_sum = scatter_sum(kl_div_vec, batch, dim=0, dim_size=batch_size)

    # 5. Top-K 筛选: 只取 Loss 最大的前 K% 个图
    # 至少取 1 个，防止 batch_size 极小时出错
    k = max(1, int(batch_size * top_k_ratio))

    # 找出最大的 k 个值
    top_k_losses, _ = torch.topk(kl_graph_sum, k)

    # 6. 只对这些难样本求平均
    return top_k_losses.mean()


def bce_loss(logits, targets, v_batch, batch_size):
    # node-wise averaged BCE loss
    # N = valid nodes (y=0/1)
    mask = (targets < 2)            # y = 0/1
    y01 = targets[mask].float()     # [M]
    logits_01 = logits[mask]
    v_batch_01 = v_batch[mask]

    u0 = (y01 == 0).sum().item()
    u1 = (y01 == 1).sum().item()
    u01 = u0 + u1                   # = M

    w = torch.zeros_like(y01, device=logits_01.device)
    w[y01 == 0] = u01 / (2 * (u0 + 1))
    w[y01 == 1] = u01 / (2 * (u1 + 1))

    crit = torch.nn.BCEWithLogitsLoss(weight=w)
    # crit = torch.nn.BCELoss(weight=w)
    loss = crit(logits_01.view(-1), y01.view(-1))
    return loss, u01


def bce_loss_batch(logits, targets, v_batch, batch_size):
    # batch/graph-wise averaged BCE loss
    # N = batch_size
    mask = (targets != 2)           # y = 0/1
    y01 = targets[mask].float()     # [M]
    logits_sel = logits[mask]
    v_batch_sel = v_batch[mask]

    u0 = (y01 == 0).sum().item()
    u1 = (y01 == 1).sum().item()
    u01 = u0 + u1

    w = torch.zeros_like(y01, device=logits_sel.device)
    w[y01 == 0] = u01 / (2 * (u0 + 1))
    w[y01 == 1] = u01 / (2 * (u1 + 1))

    crit = torch.nn.BCEWithLogitsLoss(weight=w, reduction='none')
    # crit = crit = torch.nn.BCELoss(weight=w, reduction='none')
    loss = crit(logits_sel.view(-1), y01.view(-1))  # [M]
    loss_per_graph = scatter_mean(
        loss, v_batch_sel, dim=0, dim_size=batch_size)  # [batch_size]
    loss_mean = loss_per_graph.mean()
    return loss_mean


def fro_nroms_loss(L_pos, L_neg, l_batch, n_vars):
    # batch/graph-wise averaged Frobenius norms
    # N = batch_size
    l_size = n_vars.sum().item() * 2
    l_batch_pos, l_batch_neg = torch.chunk(
        l_batch.reshape(l_size // 2, -1), 2, 1)
    fro_norms = 0
    for i in range(l_batch.max().item() + 1):
        node_idx = (l_batch_pos == i).nonzero(
            as_tuple=True)[0]
        assert torch.equal(l_batch_pos, l_batch_neg)
        l_pos_i, l_neg_i = L_pos[node_idx], L_neg[node_idx]
        l_fro_norm = torch.norm(l_pos_i + l_neg_i, p='fro')
        fro_norms += l_fro_norm
    fro_norms = fro_norms / (int(l_batch.max().item()) + 1)
    return fro_norms


def calc_orth_loss(opts, guesses, v_batch, batch_size):
    _n = opts.n_rounds
    var_s_tilde_list, var_d_tilde_list = guesses.var_s_tilde, guesses.var_d_tilde
    orth_loss = 0
    count_n = 0
    for i in range(opts.group_loss_begin, _n):
        var_s_tilde, var_d_tilde = var_s_tilde_list[i], var_d_tilde_list[i]
        inner = (var_s_tilde * var_d_tilde).sum(dim=-1)
        orth_loss_batch = scatter_mean(
            inner ** 2, v_batch, dim=0, dim_size=batch_size)
        orth_loss += orth_loss_batch.mean()
        count_n += 1
    orth_loss = orth_loss / max(count_n, 1)
    return orth_loss


def calc_recon_loss(opts, guesses, v_batch, batch_size):
    _n = opts.n_rounds
    var_s_list, var_d_list = guesses.var_s, guesses.var_d
    var_s_tilde_list, var_d_tilde_list = guesses.var_s_tilde, guesses.var_d_tilde
    recon_loss = 0
    count_n = 0
    for i in range(opts.group_loss_begin, _n):
        var_s, var_d = var_s_list[i], var_d_list[i]
        var_s_tilde, var_d_tilde = var_s_tilde_list[i], var_d_tilde_list[i]
        loss_recon = F.mse_loss(var_s, var_s_tilde, reduction='none') + \
            F.mse_loss(var_d, var_d_tilde,
                       reduction='none')
        loss_recon_batch = scatter_mean(loss_recon, v_batch,
                                        dim=0, dim_size=batch_size)
        recon_loss += loss_recon_batch.mean()
        count_n += 1
    recon_loss = recon_loss / max(count_n, 1)
    return recon_loss


def main(opts, cfg):
    train_log = os.path.join(
        opts.train_id_dir, f"train_log.core.{opts.train_num}.txt")
    train_exp = os.path.join(
        opts.train_id_dir, f"train_exp.core.{opts.train_num}.txt")
    train_stat = os.path.join(
        opts.train_id_dir, f"train_stat.core.{opts.train_num}.txt")

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", opts.device)
    print("train_id", cfg['train_id'])
    print()
    with open(train_log, opts.log_mode) as f:
        f.write(f"Using device: {opts.device}\n")
        f.write(f"train_id: {cfg['train_id']}\n\n")

    if opts.data_type in ['SATBench', 'SR', '3-SAT']:
        opts.batch_size = max(opts.batch_size // 2,
                              1) if opts.dual else opts.batch_size

        difficulty, dataset_name = tuple(os.path.abspath(
            opts.train_dir).split(os.path.sep)[-3:-1])
        _flag_file_name = f"train_{dataset_name}_{difficulty}.info"
        _flag_file_path = os.path.join(
            opts.train_id_dir, _flag_file_name)
        open(_flag_file_path, 'a').close()

        dataloader, opts.__len_train_data__ = dataloader_bench.get_dataloader(
            opts.train_dir, opts.train_splits, opts.train_sample_size,
            opts, 'train')

        print(f"len of dataloader: {len(dataloader)}")
        print(f"__len_train_data__: {opts.__len_train_data__}")
        with open(train_log, opts.log_mode) as f:
            f.write(f"len of dataloader: {len(dataloader)}\n")
            f.write(f"__len_train_data__: {opts.__len_train_data__}\n")

        valid_loader = None
        if opts.valid_dir is not None:
            # valid_loader, opts.__len_valid_data__ = dataloader_bench.get_dataloader(
            #     opts.valid_dir, opts.valid_splits, opts.valid_sample_size,
            #     opts, 'valid')
            valid_loader_batch, opts.__len_valid_data__ = dataloader_bench.get_dataloader(
                opts.valid_dir, opts.valid_splits, opts.valid_sample_size,
                opts, 'valid_batch')
            print(f"__len_valid_data__: {opts.__len_valid_data__}")
            with open(train_log, opts.log_mode) as f:
                f.write(f"__len_valid_data__: {opts.__len_valid_data__}\n\n")

        test_loader = None
        if opts.test_dir is not None:
            test_loader, opts.__len_test_data__ = dataloader_bench.get_dataloader(
                opts.test_dir, opts.test_splits, opts.test_sample_size,
                opts, 'valid_batch')
            print(f"__len_test_data__: {opts.__len_test_data__}")
            with open(train_log, opts.log_mode) as f:
                f.write(f"__len_test_data__: {opts.__len_test_data__}\n\n")

        # _d_idx = 0
        # for i, d in enumerate(dataloader):
        #     print(d)
        #     print(d.n_vars)
        #     print(d.n_clauses)
        #     print(d.l_edge_index.shape)
        #     print(d.c_edge_index.shape)
        #     _d_idx += 1
        #     if _d_idx >= 100:
        #         break
        #     # break

    elif opts.data_type in ['SATBench_v2']:
        pass

    else:
        print(f" ! Error: unknown data_type: {opts.data_type}\n")
        raise NotImplementedError

    # return

    if opts.model in model_select:
        model = model_select[opts.model](opts, cfg=cfg).to(opts.device)
    else:
        raise NotImplementedError

    # MOVED: Checkpoint loading is now AFTER optimizer creation
    # init_step = 0
    # if opts.train_id > 0:
    #     ...

    # build_learning_rate
    lr = opts.lr
    optimizer, scheduler = None, None
    if isinstance(lr, (int, float)):
        if opts.group_optimize:
            # group parameters: head vs base
            # head_names = {"V_core_score", "V_bb_score"}
            head_names = opts.optim_head_names
            head_params, base_params = [], []
            for n, p in model.named_parameters():
                if any(n.startswith(h) for h in head_names):
                    head_params.append(p)
                else:
                    base_params.append(p)

            optimizer = torch.optim.AdamW(
                [
                    {"params": base_params, "lr": opts.lr},
                    {"params": head_params, "lr": opts.lr * opts.lr_rate},
                ],
                lr=opts.lr, weight_decay=cfg['l2_loss_scale'],
            )

            with open(train_log, opts.log_mode) as f:
                f.write(f"\n[optimizer] Using group optimize: "
                        f"head lr ({head_names}) * {opts.lr_rate}\n\n")

        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr, weight_decay=cfg['l2_loss_scale'])
            with open(train_log, opts.log_mode) as f:
                f.write(f"\n[optimizer] Using single lr: {lr}\n\n")

    # --- MOVED AND UPDATED: Checkpoint Loading with Optimizer Support ---
    init_step = 0
    start_epoch = 0
    if opts.train_id > 0:
        if opts.checkpoint is None:
            init_step, start_epoch = load_latest_checkpoint(
                model, optimizer, None, util.checkpoint_dir(train_id=cfg['train_id']), opts)
        else:
            init_step, start_epoch = load_checkpoint(
                model, optimizer, None, util.checkpoint_dir(train_id=cfg['train_id']), opts.checkpoint, opts)
            assert init_step >= 0
    # ------------------------------------------------------------------

    st = (opts.scheduler_type or "none").lower()
    scheduler_steps = 0
    if opts.init_scheduler_steps > 0:
        scheduler_steps = opts.init_scheduler_steps
    if optimizer is None:
        raise RuntimeError("no optimizer")

    if st in ("none", "", "off"):
        scheduler = None
    # elif st == "poly":
    #     scheduler = PolynomialLR(
    #         optimizer,
    #         total_iters=opts.lr_iters_steps,
    #         power=opts.lr_power,
    #     )
    elif st == "exp":
        gamma = float(opts.lr_gamma)
        if not (0.0 < gamma < 1.0):
            raise ValueError(
                "lr_gamma should be between 0 and 1 for ExponentialLR")
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif st == "exp_inner":
        gamma = float(opts.lr_gamma)
        if not (0.0 < gamma < 1.0):
            raise ValueError(
                "lr_gamma should be between 0 and 1 for ExponentialLR")
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif st == "cosine":
        # 1. SATBench / SR / 3-SAT 场景
        # 这些数据集返回的是标准 DataLoader，直接读取 len(dataloader) 最准确
        steps_per_epoch = len(dataloader)
        print(f" [Scheduler] Using len(dataloader) for SATBench:"
              f" {steps_per_epoch} steps/epoch")

        # 计算总步数：Epochs * (Steps_per_Epoch / Accum_Step)
        if start_epoch > 0:
            total_steps = int((cfg['n_epochs'] - start_epoch)
                              * steps_per_epoch / opts.accum_step)
            print(f" [Scheduler] Resuming from epoch {start_epoch}. "
                  f"Total steps updated to {total_steps} (remaining epochs).")
        else:
            total_steps = int(cfg['n_epochs'] *
                              steps_per_epoch / opts.accum_step)

        scheduler = CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=float(opts.lr_min))
    else:
        # raise Exception("lr_decay_type must be 'none', 'poly' or 'exp'")
        raise Exception("lr_decay_type must be 'none' or 'exp'")

    global_step = init_step
    break_signal = False
    n_flag = True
    _init_train_num = opts.train_num
    with open(train_log, opts.log_mode) as f, open(train_exp, opts.log_mode) as f_exp, \
            open(train_stat, opts.log_mode) as f_stat:
        optimizer.zero_grad()

        report_step = 0
        report_batch = 0
        report_loss = 0.
        report_cons_loss = 0.
        report_time = 0.
        report_loss_list = [0.] * 5
        report_count_list = [0] * 5

        # diagnostic accumulators (dual tilde stability)
        diag_report_sum_share = 0.0
        diag_report_sum_diff = 0.0
        diag_report_sum_P = 0.0
        diag_report_count = 0
        diag_epoch_sum_share = 0.0
        diag_epoch_sum_diff = 0.0
        diag_epoch_sum_P = 0.0
        diag_epoch_count = 0

        loss_best = 100000
        valid_best_precision, valid_best_pr_auc, valid_best_roc_auc = 0, 0, 0

        accum_step = 0
        accum_batch = 0
        accum_loss = 0.
        accum_cons_loss = 0.
        accum_loss_list = [0.] * 5
        accum_count_list = [0] * 5

        last_save_step = -1

        def eval_model(opts, model):

            f.write(f"\n1 pass.\n\n")
            f.flush()

            if opts.data_type in ['SATBench', 'SATBench_v2', 'SR', '3-SAT']:
                t_eval_start = time.time()
                valid_pre, valid_pr_auc, valid_roc_auc = \
                    run_test_satbench_batch_version(
                        opts, model, valid_loader_batch, out_file=f_stat)
                t_eval = time.time() - t_eval_start
            else:
                valid_pre, valid_pr_auc, valid_roc_auc = 0.0, 0.0, 0.0

            f.write(f"\n2 pass.\n\n")
            f.flush()

            nonlocal valid_best_precision, valid_best_pr_auc, valid_best_roc_auc
            nonlocal last_save_step

            # UPDATED: passed optimizer, scheduler, epoch
            if valid_pre > valid_best_precision:
                valid_best_precision = valid_pre
                save_best_precision_checkpoint(opts,
                                               model, optimizer, scheduler, global_step, epoch,
                                               util.checkpoint_dir(train_id=cfg['train_id']))
                last_save_step = global_step
                f.write(f"Saved the best precision checkpoint at "
                        f"Epoch {epoch}, Step: {global_step}\n")

            if not np.isnan(valid_pr_auc) and valid_pr_auc > valid_best_pr_auc:
                valid_best_pr_auc = valid_pr_auc
                save_best_pr_auc_checkpoint(opts,
                                            model, optimizer, scheduler, global_step, epoch,
                                            util.checkpoint_dir(train_id=cfg['train_id']))
                last_save_step = global_step
                f.write(f"Saved the best PR-AUC checkpoint at "
                        f"Epoch {epoch}, Step: {global_step}\n")

            if not np.isnan(valid_roc_auc) and valid_roc_auc > valid_best_roc_auc:
                valid_best_roc_auc = valid_roc_auc
                save_best_roc_auc_checkpoint(opts,
                                             model, optimizer, scheduler, global_step, epoch,
                                             util.checkpoint_dir(train_id=cfg['train_id']))
                last_save_step = global_step
                f.write(f"Saved the best ROC-AUC checkpoint at "
                        f"Epoch {epoch}, Step: {global_step}\n")

            f.write(f"\n3 pass.\n\n")
            f.flush()

            return valid_pre, valid_pr_auc, valid_roc_auc

        def step_optimizer():
            nonlocal scheduler_steps
            # 1. 优先使用 Norm Clipping (通常效果更好)
            if cfg.get('clip_norm_val', 0) > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(cfg['clip_norm_val'])
                )

                if opts.debug:
                    print(f"\n - [debug] Epoch {epoch}/{cfg['n_epochs']-1}, Step: {global_step}, "
                          f"total_norm: {total_norm:.8f}, "
                          f"accum num: {accum_batch}, "
                          f"tid: {cfg['train_id']}")
                    f.write(f"\n - [debug] Epoch {epoch}/{cfg['n_epochs']-1}, Step: {global_step}, "
                            f"total_norm: {total_norm:.8f}, "
                            f"accum num: {accum_batch}, "
                            f"tid: {cfg['train_id']}\n")

            # 2. 如果没用 Norm Clip 且配置了 Value Clip，则使用 Value Clip (互斥)
            elif cfg.get('clip_val_val', 0) > 0:
                # 使用官方 API 替代手动循环
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), clip_value=float(cfg['clip_val_val']))

            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                if st in ["cosine"]:
                    scheduler.step()
                    opts.train_num = _init_train_num + epoch
                    print(f" > Scheduler step done. "
                          f"New LR: {optimizer.param_groups[0]['lr']:.12f}, "
                          f"Init_train_num: {_init_train_num}, "
                          f"train_num: {opts.train_num}\n")
                    f.write(f" > Scheduler step done. "
                            f"New LR: {optimizer.param_groups[0]['lr']:.12f}, "
                            f"Init_train_num: {_init_train_num}, "
                            f"train_num: {opts.train_num}\n\n")
                elif st in ["exp_inner"]:
                    scheduler_steps += 1
                    if scheduler_steps >= opts.scheduler_steps:
                        scheduler.step()
                        opts.train_num = _init_train_num + epoch
                        scheduler_steps = 0
                        print(f"\n > Scheduler step done. "
                              f"New LR: {optimizer.param_groups[0]['lr']:.12f}, "
                              f"Init_train_num: {_init_train_num}, "
                              f"train_num: {opts.train_num}\n")
                        f.write(f"\n > Scheduler step done. "
                                f"New LR: {optimizer.param_groups[0]['lr']:.12f}, "
                                f"Init_train_num: {_init_train_num}, "
                                f"train_num: {opts.train_num}\n\n")

        # UPDATED: Use start_epoch to resume training
        for epoch in range(start_epoch, cfg['n_epochs']):
            epoch_loss = 0.
            epoch_batch = 0
            epoch_loss_list = [0.] * 5
            epoch_count_list = [0] * 5
            idx = 0

            last_save_step = -1

            for data in dataloader:
                model.train()

                _n_vars = data.n_vars.sum().detach().item()
                _n_clauses = data.n_clauses.sum().detach().item()
                data = data.to(opts.device)

                tilde_loss_val = None

                if opts.dual:
                    data_dual = flip_data(data)
                    data_dual = data_dual.to(opts.device)

                global_step += 1
                idx += 1
                l_batch = data.l_batch
                v_batch = l_batch[0::2]
                c_batch = data.c_batch
                batch_size = l_batch.max().cpu().detach().item() + 1

                batch_loss = [0.] * 5
                batch_count = [0] * 5
                # assert batch_size == opts.batch_size
                # print(f"Epoch {epoch}, Batch size: {batch_size}")

                start_time = time.time()

                if opts.graph == 'LCG':
                    guesses = model(data.n_vars, data.n_clauses,
                                    data.c_edge_index, data.l_edge_index,
                                    l_batch=l_batch, c_batch=c_batch)
                    if opts.dual:
                        guesses_dual = model(data_dual.n_vars, data_dual.n_clauses,
                                             data_dual.c_edge_index, data_dual.l_edge_index,
                                             l_batch=data_dual.l_batch, c_batch=data_dual.c_batch)
                elif opts.graph == 'LCG_CCG':
                    guesses = model(data.n_vars, data.n_clauses,
                                    data.c_edge_index, data.l_edge_index,
                                    cc_edge_index=data.cc_edge_index,
                                    cc_edge_weight=data.cc_edge_weight,
                                    l_batch=l_batch, c_batch=c_batch)
                    if opts.dual:
                        guesses_dual = model(data_dual.n_vars, data_dual.n_clauses,
                                             data_dual.c_edge_index, data_dual.l_edge_index,
                                             cc_edge_index=data_dual.cc_edge_index,
                                             cc_edge_weight=data_dual.cc_edge_weight,
                                             l_batch=data_dual.l_batch, c_batch=data_dual.c_batch)
                else:
                    raise NotImplementedError

                if opts.debug and global_step % opts.debug_step == 0:
                    with torch.no_grad():
                        pass

                end_time = time.time()
                n_secs = end_time - start_time

                logits = None
                if hasattr(guesses, 'pi_core_var_logits'):
                    logits = guesses.pi_core_var_logits
                elif hasattr(guesses, 'pi_var_logits'):
                    logits = guesses.pi_var_logits

                if opts.pair_score and hasattr(guesses, 'pi_assign_logits') \
                        and guesses.pi_assign_logits is not None:
                    logits = guesses.pi_assign_logits

                if logits is None:
                    raise Exception("logits not found in guesses")

                check_numerics(logits, message=f"logits")

                if opts.dual:
                    if hasattr(guesses_dual, 'pi_core_var_logits'):
                        logits_dual = guesses_dual.pi_core_var_logits
                    elif hasattr(guesses_dual, 'pi_var_logits'):
                        logits_dual = guesses_dual.pi_var_logits
                    else:
                        raise Exception(
                            "logits_dual not found in guesses_dual")

                    if opts.pair_score and hasattr(guesses_dual, 'pi_assign_logits'):
                        logits_dual = guesses_dual.pi_assign_logits

                    check_numerics(
                        logits_dual, message=f"logits_dual")

                ''' ---------------------------------------------------------------------- '''
                ''' 1. Prediction Loss '''
                pi_v_targets = data.core_var_mask.float()

                # 【修改】使用 kl_loss_top_k 替代原本的 kl_loss
                # top_k_ratio=0.5: 只对 batch 中最难的 50% 样本计算 Loss
                cv_loss = kl_loss_top_k(
                    logits, pi_v_targets, v_batch, batch_size, top_k_ratio=opts.top_k_ratio)
                # cv_loss = kl_loss(logits, pi_v_targets, v_batch, batch_size)

                cv_loss = cv_loss

                if opts.dual:
                    pi_v_targets_dual = data_dual.core_var_mask.float()
                    # 【修改】Dual 部分也同步使用 kl_loss_top_k
                    cv_loss_dual = kl_loss_top_k(
                        logits_dual, pi_v_targets_dual, v_batch, batch_size, top_k_ratio=opts.top_k_ratio)
                    # cv_loss_dual = kl_loss(
                    #     logits_dual, pi_v_targets_dual, v_batch, batch_size)

                    cv_loss_dual = cv_loss_dual
                    cv_loss = cv_loss + cv_loss_dual

                loss = cv_loss
                batch_loss[0] += cfg['cv_loss_scale'] * cv_loss
                batch_count[0] += batch_size if not opts.dual else 2 * batch_size

                ''' 2. Auxiliary Losses '''
                # if opts.constraint_loss:
                #     if hasattr(guesses, 'L_pos') and \
                #             hasattr(guesses, 'L_neg'):

                #         fro_norms = fro_nroms_loss(
                #             guesses.L_pos, guesses.L_neg, l_batch, data.n_vars)
                #         fro_norms = fro_norms * cfg['constraint_loss_lambda']

                #         fro_norms_dual = fro_nroms_loss(
                #             guesses_dual.L_pos, guesses_dual.L_neg, l_batch, data.n_vars)
                #         fro_norms_dual = fro_norms_dual * \
                #             cfg['constraint_loss_lambda']
                #         loss += fro_norms + fro_norms_dual

                #     else:
                #         print(" ! Warning: L_pos and L_neg not found in guesses\n")

                if opts.model in model_neuro_diff_series:
                    _add_batch_flag = False
                    if opts.orth_loss:
                        orth_loss = calc_orth_loss(
                            opts, guesses, v_batch, batch_size)
                        if opts.dual:
                            orth_loss_dual = calc_orth_loss(
                                opts, guesses_dual, v_batch, batch_size)
                            orth_loss = orth_loss + orth_loss_dual
                        batch_loss[2] += opts.orth_loss_lambda * orth_loss
                        _add_batch_flag = True

                    if opts.recon_loss:
                        recon_loss = calc_recon_loss(
                            opts, guesses, v_batch, batch_size)
                        if opts.dual:
                            recon_loss_dual = calc_recon_loss(
                                opts, guesses_dual, v_batch, batch_size)
                            recon_loss = recon_loss + recon_loss_dual
                        batch_loss[2] += opts.recon_loss_lambda * recon_loss
                        _add_batch_flag = True

                    if _add_batch_flag:
                        batch_count[2] += batch_size if not opts.dual else 2 * batch_size

                ''' 3. Dual Loss '''
                """
                TODO: flip_consistency_loss
                # 不变性: core/backbone 在翻转前后相等
                inv_loss = F.mse_loss(p_core, p_core_f) + F.mse_loss(p_bb, p_bb_f)
                # 等变性: p_val 翻转后应互补 (p -> 1-p)
                equiv_loss = F.mse_loss(p_val, 1.0 - p_val_f)
                """

                if opts.dual:
                    if opts.consistency_type == "sym-kl":
                        T = opts.consistency_T
                        p1 = softmax(logits / T, v_batch)
                        p2 = softmax(logits_dual / T, v_batch)

                        # teacher-student style:
                        p2 = p2.detach()

                        eps = 1e-10
                        kl12 = p1 * (torch.log(p1 + eps) - torch.log(p2 + eps))
                        kl21 = p2 * (torch.log(p2 + eps) - torch.log(p1 + eps))
                        sym_kl_node = 0.5 * (kl12 + kl21)

                        sym_kl_graph = scatter_mean(
                            sym_kl_node, v_batch, dim=0, dim_size=batch_size)
                        cons_loss = sym_kl_graph.mean()

                        batch_loss[3] += (opts.consistency_loss_lambda * cons_loss)
                        batch_count[3] += batch_size

                    elif opts.consistency_type == "js":
                        T = opts.consistency_T
                        p1 = softmax(logits / T, v_batch)
                        p2 = softmax(logits_dual / T, v_batch)

                        m = 0.5 * (p1 + p2)
                        eps = 1e-8
                        # KL(p||m) = sum p * (log p - log m)
                        kl_p1_m = (
                            p1 * (torch.log(p1 + eps) - torch.log(m + eps)))
                        kl_p2_m = (
                            p2 * (torch.log(p2 + eps) - torch.log(m + eps)))

                        # node -> graph
                        js_node = 0.5 * (kl_p1_m + kl_p2_m)
                        cons_loss = scatter_mean(
                            js_node, v_batch, dim=0, dim_size=batch_size).mean()

                        batch_loss[3] += (opts.consistency_loss_lambda * cons_loss)
                        batch_count[3] += batch_size

                    elif opts.consistency_type == "logits-center-mse":
                        logits_g_mean = scatter_mean(
                            logits, v_batch, dim=0, dim_size=batch_size)[v_batch]
                        logits_dual_g_mean = scatter_mean(
                            logits_dual, v_batch, dim=0, dim_size=batch_size)[v_batch]
                        l1 = logits - logits_g_mean
                        l2 = logits_dual - logits_dual_g_mean
                        cons_node = (l1 - l2).pow(2)
                        cons_loss = scatter_mean(
                            cons_node, v_batch, dim=0, dim_size=batch_size).mean()

                        batch_loss[3] += (opts.consistency_loss_lambda * cons_loss)
                        batch_count[3] += batch_size

                    elif opts.consistency_type == "logits-mse":
                        cons_node = (logits - logits_dual).pow(2)
                        cons_loss = scatter_mean(
                            cons_node, v_batch, dim=0, dim_size=batch_size).mean()
                        batch_loss[3] += opts.consistency_loss_lambda * cons_loss
                        batch_count[3] += batch_size

                    elif opts.consistency_type == "logits-mse-mu":
                        cons_node = (logits - logits_dual).pow(2)
                        base_cons = scatter_mean(
                            cons_node, v_batch, dim=0, dim_size=batch_size).mean()
                        mu = scatter_mean(
                            logits, v_batch, dim=0, dim_size=batch_size)
                        mu_dual = scatter_mean(
                            logits_dual, v_batch, dim=0, dim_size=batch_size)
                        mean_diff = (mu - mu_dual).pow(2).mean()
                        cons_loss = base_cons + opts.consistency_mu_alpha * mean_diff
                        batch_loss[3] += opts.consistency_loss_lambda * cons_loss
                        batch_count[3] += batch_size

                    elif opts.consistency_type is None:
                        pass

                    # tilde equivariance/anti-equivariance:
                    # s should match, d should negate under flip
                    if (
                        opts.tilde_consistency
                        and hasattr(guesses, 'var_s_tilde') and hasattr(guesses, 'var_d_tilde')
                        and hasattr(guesses_dual, 'var_s_tilde') and hasattr(guesses_dual, 'var_d_tilde')
                        and guesses.var_s_tilde is not None and guesses.var_d_tilde is not None
                        and guesses_dual.var_s_tilde is not None and guesses_dual.var_d_tilde is not None
                    ):
                        if type(guesses.var_s_tilde) is list:
                            var_s_tilde = guesses.var_s_tilde[-1]
                            var_d_tilde = guesses.var_d_tilde[-1]
                            var_s_tilde_dual = guesses_dual.var_s_tilde[-1]
                            var_d_tilde_dual = guesses_dual.var_d_tilde[-1]
                        else:
                            var_s_tilde = guesses.var_s_tilde
                            var_d_tilde = guesses.var_d_tilde
                            var_s_tilde_dual = guesses_dual.var_s_tilde
                            var_d_tilde_dual = guesses_dual.var_d_tilde

                        share_node = (var_s_tilde - var_s_tilde_dual).pow(2)
                        diff_node = (var_d_tilde + var_d_tilde_dual).pow(2)
                        share_loss = scatter_mean(
                            share_node, v_batch, dim=0, dim_size=batch_size).mean()
                        diff_loss = scatter_mean(
                            diff_node, v_batch, dim=0, dim_size=batch_size).mean()
                        tilde_loss = (share_loss + diff_loss) * \
                            opts.decomp_loss_lambda

                        tilde_loss_val = tilde_loss

                        batch_loss[4] += tilde_loss
                        batch_count[4] += batch_size

                diag_share = diag_diff = diag_P = None
                can_diag_tilde = (
                    opts.dual
                    and hasattr(guesses, 'var_s_tilde') and hasattr(guesses, 'var_d_tilde')
                    and hasattr(guesses_dual, 'var_s_tilde') and hasattr(guesses_dual, 'var_d_tilde')
                    and guesses.var_s_tilde is not None and guesses.var_d_tilde is not None
                    and guesses_dual.var_s_tilde is not None and guesses_dual.var_d_tilde is not None
                )
                if can_diag_tilde:
                    with torch.no_grad():
                        if type(guesses.var_s_tilde) is list:
                            var_s_tilde = guesses.var_s_tilde[-1]
                            var_d_tilde = guesses.var_d_tilde[-1]
                            var_s_tilde_dual = guesses_dual.var_s_tilde[-1]
                            var_d_tilde_dual = guesses_dual.var_d_tilde[-1]
                        else:
                            var_s_tilde = guesses.var_s_tilde
                            var_d_tilde = guesses.var_d_tilde
                            var_s_tilde_dual = guesses_dual.var_s_tilde
                            var_d_tilde_dual = guesses_dual.var_d_tilde

                        delta_share = 1.0 - F.cosine_similarity(
                            var_s_tilde, var_s_tilde_dual, dim=-1, eps=1e-8)
                        delta_diff = 1.0 - F.cosine_similarity(
                            var_d_tilde, var_d_tilde_dual, dim=-1, eps=1e-8)

                        g_delta_share = scatter_mean(
                            delta_share, v_batch, dim=0, dim_size=batch_size)
                        g_delta_diff = scatter_mean(
                            delta_diff, v_batch, dim=0, dim_size=batch_size)

                        advantage = scatter_mean(
                            (delta_share < delta_diff).float(),
                            v_batch,
                            dim=0,
                            dim_size=batch_size,
                        )

                        diag_share = g_delta_share.sum().item()
                        diag_diff = g_delta_diff.sum().item()
                        diag_P = advantage.sum().item()

                if diag_share is not None:
                    diag_report_sum_share += diag_share
                    diag_report_sum_diff += diag_diff
                    diag_report_sum_P += diag_P
                    diag_report_count += batch_size
                    diag_epoch_sum_share += diag_share
                    diag_epoch_sum_diff += diag_diff
                    diag_epoch_sum_P += diag_P
                    diag_epoch_count += batch_size

                # epoch_loss += loss.item() * batch_size
                # epoch_batch += 2 * batch_size

                batch_sum_loss = [batch_loss[i] * batch_count[i]
                                  for i in range(5)]
                real_batch_loss = 0.0
                for bi in range(5):
                    if batch_count[bi] > 0:
                        real_batch_loss += batch_loss[bi]
                real_batch_loss = real_batch_loss / opts.accum_step
                real_batch_loss.backward()  # !!!!!!!!!!!!!!!!!!!!!!!!!!!

                _sum_loss = sum(batch_sum_loss)
                _sum_loss = _sum_loss.cpu().detach().item()

                epoch_batch += batch_size
                epoch_loss += _sum_loss
                for i in range(5):
                    if batch_count[i] > 0:
                        try:
                            epoch_loss_list[i] += batch_sum_loss[i].cpu().detach().item()
                            epoch_count_list[i] += batch_count[i]
                        except Exception as e:
                            print(f"Exception: {e}")
                            print(f"Error Group: {i}")
                            print(f"batch_sum_loss[{i}]: {batch_sum_loss[i]}")
                            print(f"batch_count[{i}]: {batch_count[i]}")
                            raise e

                ''' ---------------------------------------------------------------------- '''
                ''' optimize and output '''

                _lr = optimizer.param_groups[0]['lr']
                # print(f"Epoch {epoch}, Batch: {idx}, Step: {global_step}, LR: {_lr:.12f}, "
                #       f"Loss: {loss.cpu().item():.9f}, time: {n_secs:.4f}s")

                # accum loss for optimizer
                accum_step += 1
                accum_batch += batch_size
                accum_loss += _sum_loss
                for bi in range(5):
                    if batch_count[bi] > 0:
                        accum_loss_list[bi] += batch_sum_loss[bi].cpu().detach().item()
                        accum_count_list[bi] += batch_count[bi]

                # report status for print
                report_step += 1
                report_batch += batch_size
                report_time += n_secs
                report_loss += _sum_loss
                for bi in range(5):
                    if batch_count[bi] > 0:
                        report_loss_list[bi] += batch_sum_loss[bi].cpu().detach().item()
                        report_count_list[bi] += batch_count[bi]

                # opts.accum_step should be 1, then
                # every step(batch) we do optimize
                if accum_step % opts.accum_step == 0:
                    loss_mean_value = 0.  # only for print
                    for i in range(5):
                        if accum_count_list[i] > 0:
                            mean_i = \
                                accum_loss_list[i] / accum_count_list[i]
                            loss_mean_value += mean_i

                    # if cfg['clip_val_val'] > 0:
                    #     clip_val = float(cfg['clip_val_val'])
                    #     for param in model.parameters():
                    #         if param.grad is not None:
                    #             param.grad.data.clamp_(-clip_val, clip_val)
                    # if cfg['clip_norm_val'] > 0:
                    #     torch.nn.utils.clip_grad_norm_(
                    #         model.parameters(), max_norm=float(cfg['clip_norm_val']))

                    step_optimizer()

                    if loss_mean_value < loss_best:
                        loss_best = loss_mean_value
                        save_best_loss_checkpoint(opts,
                                                  model, optimizer, scheduler, global_step, epoch,
                                                  util.checkpoint_dir(train_id=cfg['train_id']))
                        last_save_step = global_step
                        f.write(f"\nSaved the best checkpoint at "
                                f"Epoch {epoch}, Step: {global_step}\n\n")

                    print(f" - Epoch {epoch}/{cfg['n_epochs']-1}, Step: {global_step}, LR: {_lr:.12f}, "
                          f"Loss: {loss_mean_value:.5f}, "
                          f"accum num: {accum_batch}, "
                          f"tid: {cfg['train_id']}")

                    print(f" - n_vars={_n_vars}, n_clauses={_n_clauses}, ")
                    f.write(f" - Epoch {epoch}/{cfg['n_epochs']-1}, Step: {global_step}, LR: {_lr:.12f}  "
                            f"Loss: {loss_mean_value:.5f}, "
                            f"accum num: {accum_batch}, "
                            f"tid: {cfg['train_id']}\n")

                    if opts.debug:
                        print(f"   cv_loss: {cv_loss.cpu().item():.10f}")
                        f.write(f"   cv_loss: {cv_loss.cpu().item():.10f}\n")

                    if opts.debug and opts.dual and opts.consistency_type is not None:
                        print(f"   Cons_loss: {cons_loss.cpu().item():.10f}")
                        f.write(
                            f"   Cons_loss: {cons_loss.cpu().item():.10f}\n")

                    if opts.debug and opts.dual and opts.tilde_consistency and tilde_loss_val is not None:
                        _tval = tilde_loss_val.detach().cpu().item()
                        print(f"   decomp_cons_loss: {_tval:.10f}")
                        f.write(f"   decomp_cons_loss: {_tval:.10f}\n")

                    if opts.debug and opts.orth_loss and opts.model in model_neuro_diff_series:
                        print(f"   Orth_loss: {orth_loss.cpu().item():.10f}")
                        f.write(
                            f"   Orth_loss: {orth_loss.cpu().item():.10f}\n")

                    if opts.debug and opts.recon_loss and opts.model in model_neuro_diff_series:
                        print(f"   Recon_loss: {recon_loss.cpu().item():.10f}")
                        f.write(
                            f"   Recon_loss: {recon_loss.cpu().item():.10f}\n")

                    f.flush()

                    accum_step = 0
                    accum_batch = 0
                    accum_loss = 0.
                    accum_loss_list = [0.] * 5
                    accum_count_list = [0] * 5

                if report_step % opts.report_step == 0:
                    mean_report_loss = 0
                    for i in range(5):
                        if report_count_list[i] > 0:
                            mean_i = report_loss_list[i] / \
                                report_count_list[i]
                            mean_report_loss += mean_i
                    mean_report_time = report_time / report_step

                    f.write(f"\n0 pass.\n\n")
                    f.flush()

                    # === 新增逻辑：根据 opts.no_eval_in_report 控制是否评估 ===
                    if not opts.no_eval_in_report:
                        eval_start = time.time()
                        valid_pre, valid_pr_auc, valid_roc_auc = \
                            eval_model(opts, model)
                        eval_end = time.time()
                        eval_time = eval_end - eval_start

                        out_str = (f" : Epoch {epoch}/{cfg['n_epochs']-1}, Step: {global_step}, LR: {_lr:.12f}, "
                                   f"Loss: {mean_report_loss:.5f}, "
                                   f"time: {mean_report_time:.4f}s, "
                                   f"eval time: {eval_time:.4f}s, "
                                   f"batch num: {report_batch}, "
                                   f"tid:{cfg['train_id']}\n"
                                   f"   Precision: {valid_pre:.5f}, "
                                   f"PR-AUC: {valid_pr_auc:.5f}, "
                                   f"ROC-AUC: {valid_roc_auc:.5f}")
                    else:
                        out_str = (f" : Epoch {epoch}/{cfg['n_epochs']-1}, Step: {global_step}, LR: {_lr:.12f}, "
                                   f"Loss: {mean_report_loss:.5f}, "
                                   f"time: {mean_report_time:.4f}s, "
                                   f"batch num: {report_batch}, "
                                   f"tid:{cfg['train_id']}\n"
                                   f"   Precision: N/A, "
                                   f"PR-AUC: N/A, "
                                   f"ROC-AUC: N/A")

                    print()
                    print(out_str)

                    f.write("\n")
                    f.write(out_str + "\n\n")

                    f_exp.write("\n")
                    f_exp.write(out_str + "\n\n")

                    f.write(f"\n10 pass.\n\n")
                    f.flush()

                    if opts.debug and opts.dual and opts.tilde_consistency and tilde_loss_val is not None:
                        _tval = tilde_loss_val.detach().cpu().item()
                        print(
                            f"   [report] decomp_cons_loss (last batch): {_tval:.10f}")
                        f.write(
                            f"   [report] decomp_cons_loss (last batch): {_tval:.10f}\n")
                        f_exp.write(
                            f"   [report] decomp_cons_loss (last batch): {_tval:.10f}\n")

                    if diag_report_count > 0:
                        diag_mean_share = diag_report_sum_share / diag_report_count
                        diag_mean_diff = diag_report_sum_diff / diag_report_count
                        diag_mean_P = diag_report_sum_P / diag_report_count

                        diag_out = ("[diag][dual-tilde][report]\n"
                                    f"  delta_share = {diag_mean_share:.9f}\n"
                                    f"  delta_diff  = {diag_mean_diff:.9f}\n"
                                    f"  P           = {diag_mean_P:.9f}")
                        print(diag_out)
                        f.write(diag_out + "\n")
                        f_exp.write(diag_out + "\n")

                    f.write(f"\n11 pass.\n\n")
                    f.flush()

                    diag_report_sum_share = 0.0
                    diag_report_sum_diff = 0.0
                    diag_report_sum_P = 0.0
                    diag_report_count = 0

                    report_step = report_batch = 0
                    report_loss = report_time = 0.
                    report_loss_list = [0.] * 5
                    report_count_list = [0] * 5

                    f.flush()
                    f_exp.flush()

                if global_step % 2000 == 0:
                    # UPDATED: passed epoch and scheduler
                    save_checkpoint(opts, model, optimizer, scheduler, global_step, epoch,
                                    util.checkpoint_dir(train_id=cfg['train_id']))
                    last_save_step = global_step
                    f.write(f"\nSaved checkpoint at "
                            f"Epoch {epoch}, Step: {global_step}\n\n")

            # 保存当前 epoch 末尾尚未满 accum_step 的残余累积
            if accum_step > 0 and accum_batch > 0:
                loss_mean_value = 0.
                for i in range(5):
                    if accum_count_list[i] > 0:
                        mean_i = \
                            accum_loss_list[i] / accum_count_list[i]
                        loss_mean_value += mean_i

                step_optimizer()

                if not n_flag:
                    print()
                    f.write("\n")
                    n_flag = True

                print(f" - Epoch {epoch}/{cfg['n_epochs']-1}, Step: {global_step}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.12f}, "
                      f"Loss: {loss_mean_value:.5f}, "
                      f"accum num: {accum_batch}, "
                      f"tid: {cfg['train_id']}")
                f.write(f" - Epoch {epoch}/{cfg['n_epochs']-1}, Step: {global_step}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.12f}, "
                        f"Loss: {loss_mean_value:.5f}, "
                        f"accum num: {accum_batch}\n")

                # 清空累积器
                accum_step = 0
                accum_batch = 0
                accum_loss = 0.
                accum_loss_list = [0.] * 5
                accum_count_list = [0] * 5

            if last_save_step != global_step:
                # UPDATED: passed epoch and scheduler
                save_checkpoint(opts, model, optimizer, scheduler, global_step, epoch,
                                util.checkpoint_dir(train_id=cfg['train_id']))
                f.write(f"\nSaved checkpoint (latest fallback) at "
                        f"Epoch {epoch}, Step: {global_step}\n\n")
                last_save_step = global_step

            # save_checkpoint(opts, model, optimizer, global_step,
            #                 util.checkpoint_dir(train_id=cfg['train_id']))
            # f.write(f"\nSaved checkpoint at "
            #         f"Epoch {epoch}, Step: {global_step}\n\n")
            # last_save_step = global_step

            save_epoch_checkpoint(opts, model, optimizer, scheduler,
                                  epoch, global_step,
                                  util.checkpoint_dir(train_id=cfg['train_id']))
            f.write(f"\nSaved epoch checkpoint at "
                    f"Epoch {epoch}, Step: {global_step}\n\n")

            eval_start = time.time()
            valid_pre, valid_pr_auc, valid_roc_auc = \
                eval_model(opts, model)
            eval_end = time.time()
            eval_time = eval_end - eval_start

            print()
            out_str = (f" * Epoch {epoch}/{cfg['n_epochs']-1} finished. \n"
                       f"   Step: {global_step}, "
                       f"tid:{cfg['train_id']}")
            print(out_str)
            f.write("\n")
            f.write(out_str + "\n")
            f_exp.write("\n")
            f_exp.write(out_str + "\n")

            _lr = optimizer.param_groups[0]['lr']
            out_str = (f" * Epoch {epoch}/{cfg['n_epochs']-1}, Step: {global_step}, LR: {_lr:.12f}, "
                       f"eval time: {eval_time:.4f}s, "
                       f"tid:{cfg['train_id']}\n"
                       f"   Precision: {valid_pre:.5f}, "
                       f"PR-AUC: {valid_pr_auc:.5f}, "
                       f"ROC-AUC: {valid_roc_auc:.5f}")

            print(out_str)
            f.write("\n")
            f.write(out_str + "\n\n")
            f_exp.write("\n")
            f_exp.write(out_str + "\n")
            f_exp.write(f"Epoch finished.\n\n")

            avg_epoch_loss = 0.
            for i in range(5):
                if epoch_count_list[i] > 0:
                    mean_i = epoch_loss_list[i] / epoch_count_list[i]
                    avg_epoch_loss += mean_i

            print(f"\n * Epoch {epoch}, Average Loss: "
                  f"{avg_epoch_loss:.9f} \n")
            f.write(f"\n * Epoch {epoch}, Average Loss: "
                    f"{avg_epoch_loss:.9f} \n\n")

            if diag_epoch_count > 0:
                diag_epoch_mean_share = diag_epoch_sum_share / diag_epoch_count
                diag_epoch_mean_diff = diag_epoch_sum_diff / diag_epoch_count
                diag_epoch_mean_P = diag_epoch_sum_P / diag_epoch_count

                diag_epoch_out = ("[diag][dual-tilde][epoch]\n"
                                  f"  delta_share = {diag_epoch_mean_share:.9f}\n"
                                  f"  delta_diff  = {diag_epoch_mean_diff:.9f}\n"
                                  f"  P           = {diag_epoch_mean_P:.9f}")
                print(diag_epoch_out)
                f.write(diag_epoch_out + "\n")
                f_exp.write(diag_epoch_out + "\n")

            if opts.data_type in ['SATBench', 'SATBench_v2', 'SR', '3-SAT'] and test_loader is not None:
                _tz = datetime.timezone(datetime.timedelta(hours=8))
                _now = datetime.datetime.now(_tz)
                _now_str = _now.strftime('%Y%m%d_%H%M%S')
                log_file_dir = os.path.join(opts.train_id_dir, "logs")
                os.makedirs(log_file_dir, exist_ok=True)
                test_difficulty, test_dataset_name = tuple(os.path.abspath(
                    opts.test_dir).split(os.path.sep)[-3:-1])
                test_log_file_name = (f"{_now_str}-{opts.train_num}-E{epoch}-"
                                      f"{test_dataset_name}-{test_difficulty}.log")
                test_log_file_path = os.path.join(
                    log_file_dir, test_log_file_name)
                test_pre, test_pr_auc, test_roc_auc = \
                    run_test_satbench_batch_version(
                        opts, model, test_loader, test_log_file_path)

                out_str = (f" * Epoch {epoch}, Test Set Results: \n"
                           f"   Precision: {test_pre:.5f}, "
                           f"PR-AUC: {test_pr_auc:.5f}, "
                           f"ROC-AUC: {test_roc_auc:.5f}\n")
                print(out_str)
                f.write(out_str + "\n")

            _lr = optimizer.param_groups[0]['lr']
            if scheduler is not None and st in ["exp"] and _lr > opts.lr_min:
                scheduler_steps += 1
                if scheduler_steps >= opts.scheduler_steps:
                    scheduler.step()
                    opts.train_num += 1
                    f.write(f"Scheduler step done. "
                            f"New LR: {optimizer.param_groups[0]['lr']:.12f}, "
                            f"train_num: {opts.train_num}\n\n")
                    scheduler_steps = 0

            f.flush()
            f_exp.flush()

            if break_signal:
                break


def save_checkpoint(opts, model, optimizer, scheduler, step, epoch, dir_path, max_keep=5):
    global CHECKPOINT_QUEUE
    global MAX_CHECKPOINTS
    os.makedirs(dir_path, exist_ok=True)
    pt_path = os.path.join(dir_path, f"checkpoint_{step}.pt")

    # UPDATED: Added optimizer, scheduler, and epoch
    save_dict = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(save_dict, pt_path)

    if len(CHECKPOINT_QUEUE) == MAX_CHECKPOINTS:
        # 弹出并删除最老的
        oldest = CHECKPOINT_QUEUE.popleft()
        if os.path.exists(oldest):
            os.remove(oldest)
            # print(f" > Deleted old checkpoint: {oldest}")

    CHECKPOINT_QUEUE.append(pt_path)


def save_epoch_checkpoint(opts, model, optimizer, scheduler, epoch, step, dir_path):
    dir_path = os.path.join(dir_path, "epoch_checkpoints")
    os.makedirs(dir_path, exist_ok=True)
    pt_path = os.path.join(
        dir_path, f"checkpoint_{opts.train_num}-E{epoch}.pt")

    # UPDATED: Added optimizer, scheduler
    save_dict = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(save_dict, pt_path)


def save_best_loss_checkpoint(opts, model, optimizer, scheduler, step, epoch, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    pt_path = os.path.join(
        dir_path, f"checkpoint.best.loss.{opts.train_num}.pt")

    # UPDATED: Added optimizer, scheduler, epoch
    save_dict = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(save_dict, pt_path)


def save_best_pr_auc_checkpoint(opts, model, optimizer, scheduler, step, epoch, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    pt_path = os.path.join(
        dir_path, f"checkpoint.best.pr.auc.{opts.train_num}.pt")

    # UPDATED: Added optimizer, scheduler, epoch
    save_dict = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(save_dict, pt_path)


def save_best_roc_auc_checkpoint(opts, model, optimizer, scheduler, step, epoch, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    pt_path = os.path.join(
        dir_path, f"checkpoint.best.roc.auc.{opts.train_num}.pt")

    # UPDATED: Added optimizer, scheduler, epoch
    save_dict = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(save_dict, pt_path)


def save_best_precision_checkpoint(opts, model, optimizer, scheduler, step, epoch, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    pt_path = os.path.join(
        dir_path, f"checkpoint.best.precision.{opts.train_num}.pt")

    # UPDATED: Added optimizer, scheduler, epoch
    save_dict = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(save_dict, pt_path)


def load_latest_checkpoint(model, optimizer, scheduler, dir_path, opts):
    pts = sorted(
        glob.glob(os.path.join(dir_path, "checkpoint_*.pt")),
        key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if not pts:
        print(f"No checkpoint found in {dir_path}")
        return 0, 0  # Return 0 step, 0 epoch

    latest_pt = pts[-1]
    print('Loading model checkpoint from %s..' % latest_pt)
    if opts.device.type == 'cpu':
        checkpoint = torch.load(latest_pt, map_location='cpu')
    else:
        checkpoint = torch.load(latest_pt)

    model.load_state_dict(checkpoint['model_state_dict'])

    # UPDATED: Load optimizer and scheduler if present and objects are provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded.")

    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded.")

    step = checkpoint.get('step', 0)
    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint: {latest_pt} (step {step}, epoch {epoch})")

    return step, epoch


def load_checkpoint(model, optimizer, scheduler, dir_path, pt_name, opts):
    pt_path = os.path.join(dir_path, pt_name)

    if not os.path.isfile(pt_path):
        print(f"No checkpoint found at {pt_path}")
        return -1, 0

    print('Loading model checkpoint from %s..' % pt_path)
    if opts.device.type == 'cpu':
        checkpoint = torch.load(pt_path, map_location='cpu')
    else:
        checkpoint = torch.load(pt_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    # UPDATED: Load optimizer and scheduler if present and objects are provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded.")

    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded.")

    step = checkpoint.get('step', 0)
    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint: {pt_path} (step {step}, epoch {epoch})")

    return step, epoch


def init():
    parser = argparse.ArgumentParser()

    # ----------------------------------------------------------------------
    parser.add_argument("--train_id", help='train_id',
                        type=int, default=0)
    parser.add_argument("--train_num", type=int,  default=1)

    # ----------------------------------------------------------------------
    # model and graph selection
    _model_candicates = ["neurocore", "PASAT"]
    parser.add_argument("--model", type=str, default="neurocore",
                        choices=_model_candicates)
    parser.add_argument("--graph", type=str, default="LCG_CCG",
                        choices=["LCG", "LCG+", "LCG_CCG"])
    parser.add_argument("--task", type=str, default="unsatcore",
                        choices=["backbone", "unsatcore"])

    # ----------------------------------------------------------------------
    parser.add_argument('--data_type', type=str, default='SATBench',
                        choices=['SATBench', 'SR', '3-SAT'])

    # ----------------------------------------------------------------------
    # args for generated datasets (SR, 3-SAT, ...)
    # follow G4SATBench
    parser.add_argument('--train_dir', type=str, default=None,
                        help='Directory with training data')
    parser.add_argument('--train_splits', type=str, nargs='+',
                        choices=['sat', 'unsat',
                                 'augmented_sat', 'augmented_unsat'],
                        default=None, help='Category of the training data')
    parser.add_argument('--train_sample_size', type=int, default=None,
                        help='The number of instance in each training splits')

    parser.add_argument('--valid_dir', type=str, default=None,
                        help='Directory with validating data')
    parser.add_argument('--valid_splits', type=str, nargs='+', choices=[
                        'sat', 'unsat', 'augmented_sat', 'augmented_unsat'], default=None, help='Category of the validating data')
    parser.add_argument('--valid_sample_size', type=int, default=None,
                        help='The number of instance in each validating splits')

    parser.add_argument('--test_dir', type=str, default=None,
                        help='Directory with testing data')
    parser.add_argument('--test_splits', type=str, nargs='+', choices=[
                        'sat', 'unsat', 'augmented_sat', 'augmented_unsat'], default=None, help='Validation splits')
    parser.add_argument('--test_sample_size', type=int, default=None,
                        help='The number of instance in each testing splits')

    parser.add_argument('--in_memory', action='store_true', default=False,
                        help='Load the dataset into memory')
    parser.add_argument('--force_process', action='store_true', default=False,)
    parser.add_argument('--graph_build_mode', type=str, default=None,
                        choices=[None, 'static', 'dynamic'],
                        help='Graph building mode for generated datasets')

    # ----------------------------------------------------------------------
    # parameters for dataloader
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", help='batch_size',
                        type=int, default=1)
    parser.add_argument("--num_workers",  help='n_parallel_reads',
                        type=int, default=4)
    parser.add_argument("--file_pre_id", help='file_pre_id',
                        type=int, default=20)
    parser.add_argument("--shardshuffle", help='shardshuffle',
                        type=int, default=20000)
    parser.add_argument("--min_n_nodes", type=int, default=-1)
    parser.add_argument("--max_n_nodes", type=int, default=-1)
    # parser.add_argument("--max_n_cells", type=int, default=3000000)
    parser.add_argument("--max_n_cells", type=int, default=-1)

    parser.add_argument("--dual", action='store_true', default=False)
    parser.add_argument("--max_k_neighbors", type=int, default=-1)

    # ----------------------------------------------------------------------
    # parameters for models
    # base cfg: {project-root}/configs/train/train_torch_v3.json
    parser.add_argument("--cfg_name",  help='cfg_path', type=str,
                        default='train_torch_v3.json')
    # additional configs:
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=-1)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--attn_dropout", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=-1)
    parser.add_argument("--total_data_items", type=int, default=120000)

    # ------ for lr scheduler
    parser.add_argument("--scheduler_type", type=str, default=None)
    parser.add_argument("--scheduler_steps", type=int, default=2)
    parser.add_argument("--init_scheduler_steps", type=int, default=-1)
    parser.add_argument("--lr_iters_steps", type=int, default=100)
    parser.add_argument("--lr_power", type=float, default=2)
    parser.add_argument("--lr_gamma", type=float, default=0.933)
    # 0.933 : 10 step to half lr
    # 0.871 : 5 step to half lr
    # 0.7937: 3 step to half lr
    parser.add_argument("--lr_min", type=float, default=1e-6)

    parser.add_argument("--group_optimize",
                        action='store_true', default=False)
    # opts.optim_head_names = {"V_core_score", "V_bb_score"}
    parser.add_argument("--optim_head_names", nargs='+', type=str,
                        default=["V_core_score", "V_bb_score"])
    parser.add_argument("--lr_rate", type=float, default=3.0)
    # ------

    parser.add_argument("--constraint_loss",
                        action='store_true', default=False)
    parser.add_argument('--constraint_loss_lambda', type=float, default=2e-3)

    parser.add_argument("--clip_val_val", type=float, default=-1)
    parser.add_argument("--clip_norm_val", type=float, default=-1)

    parser.add_argument("--d", type=int, default=-1)
    parser.add_argument("--n_mlp_layers", type=int, default=2)
    parser.add_argument("--n_update_layers", type=int, default=-1)
    parser.add_argument("--n_score_layers", type=int, default=-1)
    parser.add_argument("--n_rounds", type=int, default=-1)

    parser.add_argument("--weight_reparam", action='store_true',
                        default=False)                          # default: True in cfg!

    parser.add_argument("--mlp_transfer_fn", type=str, default=None)
    parser.add_argument("--v_act_fn", action='store_true', default=False)

    parser.add_argument("--no_norm", action='store_true', default=False)
    parser.add_argument("--norm_axis", type=int, default=None)

    parser.add_argument("--cv_layer_norm", action='store_true', default=False)
    parser.add_argument("--pair_score", action='store_true', default=False)

    parser.add_argument("--normalize_until_round", type=int, default=None)

    # ----------------------------------------------------------------------
    # parameters for conv.HypergraphConv
    parser.add_argument("--n_heads", type=int, default=1)

    # ----------------------------------------------------------------------
    # parameters for prototype model 1
    parser.add_argument("--n_even_layers", type=int, default=2)
    parser.add_argument("--n_odd_layers", type=int, default=2)
    parser.add_argument("--n_inv_layers", type=int,
                        default=2)                              # invariant/equivariant
    parser.add_argument("--n_s_layers", type=int, default=2)    # symmetric
    parser.add_argument("--n_d_layers", type=int, default=2)    # difference
    parser.add_argument("--n_map_layers", type=int, default=2)

    parser.add_argument("--n_core_layers", type=int, default=2)
    parser.add_argument("--n_bb_layers", type=int, default=2)
    parser.add_argument("--n_p_layers", type=int, default=2)

    parser.add_argument("--init_type", type=str, default="zeros",
                        choices=["randn", "ones", "zeros",  "learnable"])

    parser.add_argument("--res_gnn", action='store_true', default=False)

    parser.add_argument("--res_v", action='store_true', default=False)
    parser.add_argument("--res_l", action='store_true', default=False)
    parser.add_argument("--res_type", type=str, default="cat",
                        choices=["add", "cat"])  # for 1_4_2 update layer

    parser.add_argument("--top_k_ratio", type=float, default=0.5)

    parser.add_argument("--orth_loss", action='store_true', default=False)
    parser.add_argument("--orth_loss_lambda",  type=float, default=0.1)

    parser.add_argument("--consistency_type", type=str, default=None,
                        choices=["sym-kl", "js",
                                 "logits-mse", "logits-center-mse", "logits-mse-mu"])
    parser.add_argument("--consistency_T",  type=float, default=1)
    parser.add_argument("--consistency_loss_lambda",  type=float, default=0.1)
    parser.add_argument("--consistency_mu_alpha", type=float, default=0.1)
    parser.add_argument("--tilde_consistency",
                        action='store_true', default=False)
    parser.add_argument("--decomp_loss_lambda", type=float, default=0.05)

    parser.add_argument("--recon_loss", action='store_true', default=False)
    parser.add_argument("--recon_loss_lambda",  type=float, default=0.1)

    parser.add_argument("--group_loss_begin", type=int, default=0)

    # ----------------------------------------------------------------------
    # parameters for training
    parser.add_argument("--report_step", type=int, default=1000)
    parser.add_argument("--accum_step", type=int, default=1)
    parser.add_argument("--valid_key", type=str, default='PR-AUC')

    parser.add_argument("--valid_batch_size", type=int, default=8)
    parser.add_argument("--valid_max_samples", type=int, default=-1)
    parser.add_argument("--valid_shuffle", action='store_true', default=False)

    parser.add_argument("--no_eval_in_report",
                        action='store_true', default=False)
    parser.add_argument("--debug_step", type=int, default=100)

    # ----------------------------------------------------------------------
    # checkpoint management
    # Correct argument name: --max_checkpoints
    # Keep the old misspelled name --max_checkpionts as a deprecated alias
    parser.add_argument("--max_checkpoints", help='max_checkpoints (preferred)',
                        type=int, default=None)
    parser.add_argument("--max_checkpionts", help='max_checkpionts (DEPRECATED misspelling)',
                        type=int, default=None)
    parser.add_argument("--checkpoint", help="checkpoint",
                        type=str, default=None)

    # ----------------------------------------------------------------------
    # others
    parser.add_argument("--record_offset", type=int, default=0)
    parser.add_argument("--file_per_comp", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=300000)

    parser.add_argument("--run_name", type=str, default="default_run")

    parser.add_argument("--debug", action='store_true', default=False)

    opts = parser.parse_args()

    # ======================================================================
    # main:

    if opts.graph == 'LCG_CCG':
        assert opts.model in model_lcg_ccg_series, \
            f"LCG_CCG graph only supports models: {model_lcg_ccg_series}"

    pwd = os.getcwd()
    cmd = 'python ' + ' '.join(sys.argv)
    print(f'{cmd}\n')
    report_cmd = f"{pwd}\t{cmd}"
    _home_dir = Path.home()

    global CHECKPOINT_QUEUE
    global MAX_CHECKPOINTS
    # Normalize checkpoint arg names: prefer --max_checkpoints, but accept
    # the old misspelled --max_checkpionts for backward compatibility.
    if getattr(opts, 'max_checkpoints', None) is None:
        if getattr(opts, 'max_checkpionts', None) is not None:
            opts.max_checkpoints = opts.max_checkpionts
        else:
            # default fallback
            opts.max_checkpoints = 10

    MAX_CHECKPOINTS = opts.max_checkpoints
    CHECKPOINT_QUEUE = collections.deque(maxlen=MAX_CHECKPOINTS)

    opts.cfg_path = util.train_config_dir(opts.cfg_name)
    with open(opts.cfg_path, 'r') as f:
        cfg = json.load(f)

    dummy_cfg = {"gd_id": 43, "n_rounds": 4}
    if opts.train_id == 0:
        # cfg['train_id'] = util.db_insert(
        #     table='train_runs', git_commit="origin", **dummy_cfg)
        cfg['train_id'] = util.gen_id(cmd=report_cmd)
        opts.log_mode = "w"
    else:
        cfg['train_id'] = opts.train_id
        opts.log_mode = "a"

    if opts.data_type in ['SATBench', 'SATBench_v2', 'SR', '3-SAT']:
        # goto generated dataset settings
        assert opts.train_dir is not None, f"'--data_dir' must be specified."
        assert os.path.isdir(opts.train_dir), \
            f"train_dir {opts.train_dir} not found."
        assert opts.valid_dir is not None, f"'--valid_dir' must be specified."
        assert os.path.isdir(opts.valid_dir), \
            f"valid_dir {opts.valid_dir} not found."

        opts.label = 'core_variable'
        opts.data_fetching = 'parallel'
        opts.file_list = opts.train_dir
        # goto generated dataset settings
        assert opts.train_dir is not None, f"'--data_dir' must be specified."
        assert os.path.isdir(opts.train_dir), \
            f"train_dir {opts.train_dir} not found."
        assert opts.valid_dir is not None, f"'--valid_dir' must be specified."
        assert os.path.isdir(opts.valid_dir), \
            f"valid_dir {opts.valid_dir} not found."

        opts.label = 'core_variable'
        opts.data_fetching = 'parallel'
        opts.file_list = opts.train_dir

    train_id_dir = os.path.join(util.checkpoint_dir(train_id=cfg['train_id']))

    if not os.path.exists(train_id_dir):
        os.makedirs(train_id_dir)

    infopath = os.path.join(train_id_dir, "info.core.txt")
    with open(infopath,  opts.log_mode) as f:
        f.write(__file__ + "\n")
        f.write(f"{opts.run_name}\n\n")
        f.write(cmd + '\n\n')

        for key in opts.__dict__:
            if key != "file_list":
                f.write(f"{key}: {opts.__dict__[key]}\n")

        f.write("\n")

        if opts.min_n_nodes >= 0:
            cfg['min_n_nodes'] = opts.min_n_nodes
            print("min_n_nodes is changed to ", opts.min_n_nodes)
            f.write(f"min_n_nodes is changed to {opts.min_n_nodes}\n\n")

        if opts.max_n_nodes > 0:
            cfg['max_n_nodes'] = opts.max_n_nodes
            print("max_n_nodes is changed to ", opts.max_n_nodes)
            f.write(f"max_n_nodes is changed to {opts.max_n_nodes}\n\n")

        cfg['n_epochs'] = opts.epochs

        if opts.lr > 0:
            cfg['learning_rate'] = opts.lr
            print("learning rate is changed to ", opts.lr)
            f.write(f"learning rate is changed to {opts.lr}\n\n")

        cfg['dropout'] = opts.dropout
        cfg['attn_dropout'] = opts.attn_dropout

        if opts.weight_decay > 0:
            cfg['l2_loss_scale'] = opts.weight_decay
            print("weight decay is changed to ", opts.weight_decay)
            f.write(f"weight decay is changed to {opts.weight_decay}\n\n")

        cfg['constraint_loss'] = opts.constraint_loss
        cfg['constraint_loss_lambda'] = opts.constraint_loss_lambda

        if opts.clip_val_val > 0:
            cfg['clip_val_val'] = opts.clip_val_val
            print("clip_val_val is changed to ", opts.clip_val_val)
            f.write(f"clip_val_val is changed to {opts.clip_val_val}\n\n")

        if opts.clip_norm_val > 0:
            cfg['clip_norm_val'] = opts.clip_norm_val
            print("clip_norm_val is changed to ", opts.clip_norm_val)
            f.write(f"clip_norm_val is changed to {opts.clip_norm_val}\n\n")

        if opts.d > 0:
            cfg['d'] = opts.d
            print("hidden dim is changed to ", opts.d)
            f.write(f"hidden dim is changed to {opts.d}\n\n")

        cfg['n_mlp_layers'] = opts.n_mlp_layers

        if opts.n_update_layers > 0:
            cfg['n_update_layers'] = opts.n_update_layers
            print("n_update_layers is changed to ", opts.n_update_layers)
            f.write(
                f"n_update_layers is changed to {opts.n_update_layers}\n\n")

        if opts.n_score_layers > 0:
            cfg['n_score_layers'] = opts.n_score_layers
            print("n_score_layers is changed to ", opts.n_score_layers)
            f.write(f"n_score_layers is changed to {opts.n_score_layers}\n\n")

        if opts.n_rounds > 0:
            cfg['n_rounds'] = opts.n_rounds
            print("n_rounds is changed to ", opts.n_rounds)
            f.write(f"n_rounds is changed to {opts.n_rounds}\n\n")
        else:
            opts.n_rounds = cfg['n_rounds']

        cfg['weight_reparam'] = opts.weight_reparam
        print("weight_reparam is set to ", opts.weight_reparam)
        f.write(f"weight_reparam is set to {opts.weight_reparam}\n\n")

        if opts.mlp_transfer_fn is not None:
            cfg['mlp_transfer_fn'] = opts.mlp_transfer_fn
            print("mlp_transfer_fn is changed to ", opts.mlp_transfer_fn)
            f.write(
                f"mlp_transfer_fn is changed to {opts.mlp_transfer_fn}\n\n")

        if opts.norm_axis is not None:
            cfg['norm_axis'] = opts.norm_axis
            print("norm_axis is changed to ", opts.norm_axis)
            f.write(f"norm_axis is changed to {opts.norm_axis}\n\n")

        cfg['n_heads'] = opts.n_heads

        cfg['n_even_layers'] = opts.n_even_layers
        cfg['n_odd_layers'] = opts.n_odd_layers
        cfg['n_inv_layers'] = opts.n_inv_layers
        cfg['n_s_layers'] = opts.n_s_layers
        cfg['n_d_layers'] = opts.n_d_layers
        cfg['n_map_layers'] = opts.n_map_layers
        cfg['n_core_layers'] = opts.n_core_layers
        cfg['n_bb_layers'] = opts.n_bb_layers
        cfg['n_p_layers'] = opts.n_p_layers

        for key in cfg.keys():
            f.write(f"{key}: {cfg[key]}\n")

        f.write('\n\n')

        for line in opts.file_list:
            f.write(f"file_list: {line}\n")

    opts.infopath = infopath
    opts.train_id_dir = train_id_dir

    main(opts, cfg=cfg)

    print(f'{cmd}\n')


if __name__ == "__main__":
    init()
