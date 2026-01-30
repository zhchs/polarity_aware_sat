import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def run_test_satbench(opts, model, valid_dataloader):
    model.eval()
    avg_pre = 0
    avg_pr_auc = 0
    ave_roc_auc = 0
    step = 0

    roc_sum = 0
    roc_count = 0

    with torch.no_grad():
        L = opts.__len_valid_data__
        for _i, data in enumerate(valid_dataloader):

            tfd_data = data.to(opts.device)

            if opts.graph == "LCG":
                guesses = model(tfd_data.n_vars, tfd_data.n_clauses,
                                tfd_data.c_edge_index, tfd_data.l_edge_index)
            elif opts.graph == "LCG_CCG":
                guesses = model(tfd_data.n_vars, tfd_data.n_clauses,
                                tfd_data.c_edge_index, tfd_data.l_edge_index,
                                tfd_data.cc_edge_index, tfd_data.cc_edge_weight)
            else:
                raise NotImplementedError(
                    f"Graph type {opts.graph} not supported in test")

            if hasattr(guesses, "pi_core_var_logits"):
                logits = guesses.pi_core_var_logits.squeeze().detach().cpu().numpy()
            elif hasattr(guesses, "pi_var_logits"):
                logits = guesses.pi_var_logits.squeeze().detach().cpu().numpy()

            if hasattr(opts, "pair_score") and opts.pair_score and \
                    hasattr(guesses, "pi_assign_logits") and \
                    guesses.pi_assign_logits is not None:
                logits = guesses.pi_assign_logits.squeeze().detach().cpu().numpy()

            np_core_var_mask = data.core_var_mask.long().cpu().numpy()
            if _i % 1000 == 0:
                print(f"Evaluating step {_i} ...")

            total = np.sum(np_core_var_mask)
            if total == 0:
                continue

            sorted_indices = np.argsort(-logits)[0:total]
            elements = np_core_var_mask[sorted_indices]
            num = int(np.sum(elements))

            acc_pre = num / total / L
            pr_auc = average_precision_score(np_core_var_mask, logits)

            avg_pre += acc_pre
            avg_pr_auc += pr_auc / L

            if np.unique(np_core_var_mask).size >= 2:
                roc_auc = roc_auc_score(np_core_var_mask, logits)
                roc_sum += roc_auc
                roc_count += 1

            step += 1

    if roc_count > 0:
        ave_roc_auc = roc_sum / roc_count
    else:
        ave_roc_auc = 0.0

    return avg_pre, avg_pr_auc, ave_roc_auc


def run_test_satbench_batch_version(opts, model, valid_dataloader, out_file=None):
    model.eval()

    f_out = None
    should_close = False
    if out_file is not None:
        if isinstance(out_file, str):
            f_out = open(out_file, "a")
            should_close = True
        else:
            f_out = out_file

    pre_sum = 0.0
    pre_cnt = 0
    pr_auc_sum = 0.0
    pr_auc_cnt = 0
    roc_sum = 0.0
    roc_cnt = 0

    batch_idx = 0
    with torch.no_grad():
        for _i, data in enumerate(valid_dataloader):
            batch_idx += 1
            tfd_data = data.to(opts.device)

            l_batch = tfd_data.l_batch
            v_batch = l_batch[0::2]
            v_batch = v_batch.cpu().numpy()
            B = v_batch.max().item() + 1

            if opts.graph == "LCG":
                guesses = model(
                    tfd_data.n_vars,
                    tfd_data.n_clauses,
                    tfd_data.c_edge_index,
                    tfd_data.l_edge_index
                )
            elif opts.graph == "LCG_CCG":
                guesses = model(
                    tfd_data.n_vars,
                    tfd_data.n_clauses,
                    tfd_data.c_edge_index,
                    tfd_data.l_edge_index,
                    tfd_data.cc_edge_index,
                    tfd_data.cc_edge_weight
                )
            else:
                raise NotImplementedError(
                    f"Graph type {opts.graph} not supported in test")

            if hasattr(guesses, "pi_core_var_logits"):
                logits = guesses.pi_core_var_logits.squeeze().detach().cpu().numpy()
            elif hasattr(guesses, "pi_var_logits"):
                logits = guesses.pi_var_logits.squeeze().detach().cpu().numpy()

            if getattr(opts, "pair_score", False) and getattr(guesses, "pi_assign_logits", None) is not None:
                logits = guesses.pi_assign_logits.squeeze().detach().cpu().numpy()

            core_mask = data.core_var_mask.long().cpu().numpy()

            for g in range(B):
                mask_g = (v_batch == g)
                logits_g = logits[mask_g]
                y_g = core_mask[mask_g]

                if logits_g.size == 0:
                    continue
                
                precision_g = -1.0
                pr_auc_g = -1.0
                roc_auc_g = -1.0

                total_g = int(y_g.sum())
                if total_g > 0:
                    sorted_idx = np.argsort(-logits_g)[:total_g]
                    correct = int(y_g[sorted_idx].sum())
                    precision_g = correct / total_g
                    pre_sum += precision_g
                    pre_cnt += 1

                pr_auc_g = average_precision_score(y_g, logits_g)
                pr_auc_sum += pr_auc_g
                pr_auc_cnt += 1

                if np.unique(y_g).size == 2:
                    roc_auc_g = roc_auc_score(y_g, logits_g)
                    roc_sum += roc_auc_g
                    roc_cnt += 1
                
                if f_out is not None:
                    n_vars_g = data.n_vars[g].item() if hasattr(data, "n_vars") else -1
                    n_clauses_g = data.n_clauses[g].item() if hasattr(data, "n_clauses") else -1
                    n_cells_g = -1
                    if hasattr(data, "_slice_dict") and "l_edge_index" in data._slice_dict:
                        slices = data._slice_dict["l_edge_index"]
                        n_cells_g = (slices[g+1] - slices[g]).item()
                    f_out.write(f"Batch_{batch_idx}_Graph_{g} {n_vars_g} {n_clauses_g} {n_cells_g} {precision_g:.4f} {pr_auc_g:.4f} {roc_auc_g:.4f}")

            if f_out is not None:
                avg_pre = pre_sum / pre_cnt if pre_cnt > 0 else 0.0
                avg_pr_auc = pr_auc_sum / pr_auc_cnt if pr_auc_cnt > 0 else 0.0
                avg_roc_auc = roc_sum / roc_cnt if roc_cnt > 0 else 0.0
                f_out.write(f"Batch {batch_idx}: average precision: {avg_pre:.4f}, average PR-AUC: {avg_pr_auc:.4f}, average ROC-AUC: {avg_roc_auc:.4f}")

    avg_pre = pre_sum / pre_cnt if pre_cnt > 0 else 0.0
    avg_pr_auc = pr_auc_sum / pr_auc_cnt if pr_auc_cnt > 0 else 0.0
    avg_roc_auc = roc_sum / roc_cnt if roc_cnt > 0 else 0.0

    if f_out is not None:
        f_out.write(f"average precision: {avg_pre:.4f}, average PR-AUC: {avg_pr_auc:.4f}, average ROC-AUC: {avg_roc_auc:.4f}")
        if should_close:
            f_out.close()

    return avg_pre, avg_pr_auc, avg_roc_auc