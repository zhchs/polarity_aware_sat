import os
import gzip
import itertools
import pickle
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.data import Batch

from tqdm import tqdm

from dataset.utils_bench import parse_cnf_file, clean_clauses, literal2l_idx


class SATBenchDataset(Dataset):
    def __init__(self, data_dir, splits, sample_size, use_contrastive_learning, opts):
        self.opts = opts
        self.splits = splits
        self.sample_size = sample_size
        self.data_dir = data_dir
        self.all_files = self._get_files(data_dir)
        self.split_len = self._get_split_len()
        # if self.opts.label != 'core_variable':
        #     self.all_labels = self._get_labels(data_dir)
        self.use_contrastive_learning = use_contrastive_learning
        if self.use_contrastive_learning:
            self.positive_indices = self._get_positive_indices()

        super().__init__(data_dir)

    def _get_files(self, data_dir):
        files = {}
        for split in self.splits:
            split_files = list(
                sorted(glob.glob(data_dir + f'/{split}/*.cnf', recursive=True)))
            if self.sample_size is not None and len(split_files) > self.sample_size:
                split_files = split_files[:self.sample_size]
            files[split] = split_files
        return files

    def __get_labels(self, data_dir):
        labels = {}
        if self.opts.label == 'satisfiability':
            for split in self.splits:
                if split == 'sat' or split == 'augmented_sat':
                    labels[split] = [torch.tensor(
                        1., dtype=torch.float)] * self.split_len
                else:
                    # split == 'unsat' or split == 'augmented_unsat'
                    labels[split] = [torch.tensor(
                        0., dtype=torch.float)] * self.split_len
        elif self.opts.label == 'assignment':
            for split in self.splits:
                assert split == 'sat' or split == 'augmented_sat'
                labels[split] = []
                # for cnf_filepath in self.all_files[split]:
                for cnf_filepath in tqdm(
                        self.all_files[split],
                        desc=f"Loading assignments for '{split}'",
                        dynamic_ncols=True):
                    filename = os.path.splitext(
                        os.path.basename(cnf_filepath))[0]
                    assignment_file = os.path.join(os.path.dirname(
                        cnf_filepath), filename + '_assignment.pkl')
                    with open(assignment_file, 'rb') as f:
                        assignment = pickle.load(f)
                    labels[split].append(torch.tensor(
                        assignment, dtype=torch.float))
        elif self.opts.label == 'core_variable':
            for split in self.splits:
                assert split == 'unsat' or split == 'augmented_unsat'
                labels[split] = []
                # for cnf_filepath in self.all_files[split]:
                for cnf_filepath in tqdm(
                    self.all_files[split],
                    desc=f"Loading core_variable for '{split}'",
                    dynamic_ncols=True
                ):
                    filename = os.path.splitext(
                        os.path.basename(cnf_filepath))[0]
                    assignment_file = os.path.join(os.path.dirname(
                        cnf_filepath), filename + '_core_variable.pkl')
                    with open(assignment_file, 'rb') as f:
                        core_variable = pickle.load(f)
                    labels[split].append(torch.tensor(
                        core_variable, dtype=torch.float))
        else:
            assert self.opts.label == None
            for split in self.splits:
                labels[split] = [None] * self.split_len

        return labels

    def _get_label(self, split, cnf_filepath):
        if self.opts.label == 'satisfiability':
            assert split is not None
            if split == 'sat' or split == 'augmented_sat':
                _label = torch.tensor(1., dtype=torch.float)
            else:
                # split == 'unsat' or split == 'augmented_unsat'
                _label = torch.tensor(0., dtype=torch.float)

        elif self.opts.label == 'assignment':
            assert split == 'sat' or split == 'augmented_sat'
            filename = os.path.splitext(
                os.path.basename(cnf_filepath))[0]
            assignment_file = os.path.join(os.path.dirname(
                cnf_filepath), filename + '_assignment.pkl')
            with open(assignment_file, 'rb') as f:
                assignment = pickle.load(f)
            _label = torch.tensor(assignment, dtype=torch.float)

        elif self.opts.label == 'core_variable':
            assert split == 'unsat' or split == 'augmented_unsat'
            # for cnf_filepath in self.all_files[split]:
            filename = os.path.splitext(
                os.path.basename(cnf_filepath))[0]
            assignment_file = os.path.join(os.path.dirname(
                cnf_filepath), filename + '_core_variable.pkl')
            with open(assignment_file, 'rb') as f:
                core_variable = pickle.load(f)
            _label = torch.tensor(core_variable, dtype=torch.float)
        else:
            assert self.opts.label == None
            _label = None

        return _label

    def _get_split_len(self):
        lens = [len(self.all_files[split]) for split in self.splits]
        assert len(set(lens)) == 1
        return lens[0]

    def _get_file_name(self, split, cnf_filepath):
        filename = os.path.splitext(os.path.basename(cnf_filepath))[0]
        if self.opts.label != 'core_variable':
            return f'{split}/{filename}.sample.pt'
        else:
            return f'{split}/{filename}.core.sample.pt'

    def _get_positive_indices(self):
        # calculate the index to map the original instance to its augmented one, and vice versa.
        positive_indices = []
        for offset, split in enumerate(self.splits):
            if split == 'sat':
                positive_indices.append(torch.tensor(
                    self.splits.index('augmented_sat')-offset, dtype=torch.long))
            elif split == 'augmented_sat':
                positive_indices.append(torch.tensor(
                    self.splits.index('sat')-offset, dtype=torch.long))
            elif split == 'unsat':
                positive_indices.append(torch.tensor(self.splits.index(
                    'augmented_unsat')-offset, dtype=torch.long))
            elif split == 'augmented_unsat':
                positive_indices.append(torch.tensor(
                    self.splits.index('unsat')-offset, dtype=torch.long))
        return positive_indices

    @property
    def processed_file_names(self):
        names = []
        for split in self.splits:
            for cnf_filepath in self.all_files[split]:
                names.append(self._get_file_name(split, cnf_filepath))
        return names

    def _save_data(self, split, cnf_filepath):
        file_name = self._get_file_name(split, cnf_filepath)
        saved_path = os.path.join(self.processed_dir, file_name)
        if os.path.exists(saved_path):
            return

        n_vars, clauses, learned_clauses = parse_cnf_file(
            cnf_filepath, split_clauses=True)

        # limit the size of the learned clauses to 1000
        if len(learned_clauses) > 1000:
            clauses = clauses + learned_clauses[:1000]
        else:
            clauses = clauses + learned_clauses

        clauses = clean_clauses(clauses)

        # if self.opts.graph == 'lcg':
        #     data = construct_lcg(n_vars, clauses)
        # elif self.opts.graph == 'vcg':
        #     data = construct_vcg(n_vars, clauses)

        l_edge_index_list = []
        c_edge_index_list = []

        for c_idx, clause in enumerate(clauses):
            for literal in clause:
                l_idx = literal2l_idx(literal)
                l_edge_index_list.append(l_idx)
                c_edge_index_list.append(c_idx)

        n_lits = n_vars * 2
        n_clauses = len(clauses)

        sample = {
            "n_vars":    torch.tensor(n_vars, dtype=torch.int32),
            "n_lits":    torch.tensor(n_lits, dtype=torch.int32),
            "n_clauses": torch.tensor(n_clauses, dtype=torch.int32),
            "l_edge_index": torch.as_tensor(l_edge_index_list, dtype=torch.int32).contiguous(),
            "c_edge_index": torch.as_tensor(c_edge_index_list, dtype=torch.int32).contiguous(),
        }

        if self.opts.label == 'core_variable':
            filename = os.path.splitext(
                os.path.basename(cnf_filepath))[0]
            assignment_file = os.path.join(os.path.dirname(
                cnf_filepath), filename + '_core_variable.pkl')
            with open(assignment_file, 'rb') as f:
                core_variable = pickle.load(f)
            label = torch.tensor(core_variable, dtype=torch.float)
            sample['label'] = label

        torch.save(sample, saved_path)

    def __process(self):
        for split in self.splits:
            os.makedirs(os.path.join(self.processed_dir, split), exist_ok=True)

        # for split in self.splits:
        #     for cnf_filepath in self.all_files[split]:
        #         self._save_data(split, cnf_filepath)

        for split in self.splits:
            file_list = self.all_files[split]

            for cnf_filepath in tqdm(
                file_list,
                desc=f"Processing split '{split}'",
                dynamic_ncols=True
            ):
                self._save_data(split, cnf_filepath)

    def process(self):
        pass

    def len(self):
        if self.opts.data_fetching == 'parallel':
            return self.split_len
        else:
            # self.opts.data_fetching == 'sequential'
            return self.split_len * len(self.splits)

    def _get_cnf(self, split, cnf_filepath):
        n_vars, clauses, learned_clauses = parse_cnf_file(
            cnf_filepath, split_clauses=True)

        # limit the size of the learned clauses to 1000
        if len(learned_clauses) > 1000:
            clauses = clauses + learned_clauses[:1000]
        else:
            clauses = clauses + learned_clauses

        clauses = clean_clauses(clauses)

        l_edge_index_list = []
        c_edge_index_list = []

        for c_idx, clause in enumerate(clauses):
            for literal in clause:
                l_idx = literal2l_idx(literal)
                l_edge_index_list.append(l_idx)
                c_edge_index_list.append(c_idx)

        n_lits = n_vars * 2
        n_clauses = len(clauses)

        sample = {
            "n_vars":    torch.tensor(n_vars, dtype=torch.int32),
            "n_lits":    torch.tensor(n_lits, dtype=torch.int32),
            "n_clauses": torch.tensor(n_clauses, dtype=torch.int32),
            "l_edge_index": torch.as_tensor(l_edge_index_list, dtype=torch.int32).contiguous(),
            "c_edge_index": torch.as_tensor(c_edge_index_list, dtype=torch.int32).contiguous(),
        }

        return sample

    def get(self, idx):
        if self.opts.data_fetching == 'parallel':
            data_list = []
            for split_idx, split in enumerate(self.splits):
                cnf_filepath = self.all_files[split][idx]
                sample = self._get_cnf(split, cnf_filepath)
                label = self._get_label(split, cnf_filepath)

                n_vars = sample["n_vars"].long()
                n_clauses = sample["n_clauses"].long()
                l = sample["l_edge_index"].long()
                c = sample["c_edge_index"].long()

                if self.opts.label == "core_variable":
                    core_var_mask = label
                    data = LCG(
                        n_vars=n_vars,
                        n_clauses=n_clauses,
                        l_edge_index=l,
                        c_edge_index=c,
                        core_var_mask=core_var_mask,
                        core_clause_mask=None,
                        l_batch=torch.zeros(n_vars * 2, dtype=torch.long),
                        c_batch=torch.zeros(n_clauses, dtype=torch.long)
                    )
                else:
                    label = self.all_labels[split][idx]
                    raise NotImplementedError(f"Label type {self.opts.label}"
                                              f" not fully implemented.")

                if self.use_contrastive_learning:
                    data.positive_index = self.positive_indices[split_idx]
                data_list.append(data)
            return data_list
        else:
            # self.opts.data_fetching == 'sequential'
            for split in self.splits:
                if idx >= self.split_len:
                    idx -= self.split_len
                else:
                    cnf_filepath = self.all_files[split][idx]

                    sample = self._get_cnf(split, cnf_filepath)
                    label = self._get_label(split, cnf_filepath)

                    n_vars = sample["n_vars"].long()
                    n_clauses = sample["n_clauses"].long()
                    l = sample["l_edge_index"].long()
                    c = sample["c_edge_index"].long()

                    if self.opts.label == "core_variable":
                        core_var_mask = label
                        data = LCG(
                            n_vars=n_vars,
                            n_clauses=n_clauses,
                            l_edge_index=l,
                            c_edge_index=c,
                            core_var_mask=core_var_mask,
                            core_clause_mask=None,
                            l_batch=torch.zeros(n_vars * 2, dtype=torch.long),
                            c_batch=torch.zeros(n_clauses, dtype=torch.long)
                        )
                    else:
                        label = self.all_labels[split][idx]
                        raise NotImplementedError(f"Label type {self.opts.label}"
                                                  f" not fully implemented.")

                    return [data]

    def __get(self, idx):
        if self.opts.data_fetching == 'parallel':
            data_list = []
            for split_idx, split in enumerate(self.splits):
                cnf_filepath = self.all_files[split][idx]
                # label = self.all_labels[split][idx]
                file_name = self._get_file_name(split, cnf_filepath)
                saved_path = os.path.join(self.processed_dir, file_name)

                sample = torch.load(saved_path)

                n_vars = sample["n_vars"].long()
                n_clauses = sample["n_clauses"].long()
                l = sample["l_edge_index"].long()
                c = sample["c_edge_index"].long()

                if self.opts.label == "core_variable":
                    core_var_mask = sample["label"]
                    data = LCG(
                        n_vars=n_vars,
                        n_clauses=n_clauses,
                        l_edge_index=l,
                        c_edge_index=c,
                        core_var_mask=core_var_mask,
                        core_clause_mask=None,
                        l_batch=torch.zeros(n_vars * 2, dtype=torch.long),
                        c_batch=torch.zeros(n_clauses, dtype=torch.long)
                    )
                else:
                    label = self.all_labels[split][idx]
                    raise NotImplementedError(f"Label type {self.opts.label}"
                                              f" not fully implemented.")

                if self.use_contrastive_learning:
                    data.positive_index = self.positive_indices[split_idx]
                data_list.append(data)
            return data_list
        else:
            # self.opts.data_fetching == 'sequential'
            for split in self.splits:
                if idx >= self.split_len:
                    idx -= self.split_len
                else:
                    cnf_filepath = self.all_files[split][idx]
                    # label = self.all_labels[split][idx]
                    file_name = self._get_file_name(split, cnf_filepath)
                    saved_path = os.path.join(self.processed_dir, file_name)

                    sample = torch.load(saved_path)

                    n_vars = sample["n_vars"].long()
                    n_clauses = sample["n_clauses"].long()
                    l = sample["l_edge_index"].long()
                    c = sample["c_edge_index"].long()

                    if self.opts.label == "core_variable":
                        core_var_mask = sample["label"]
                        data = LCG(
                            n_vars=n_vars,
                            n_clauses=n_clauses,
                            l_edge_index=l,
                            c_edge_index=c,
                            core_var_mask=core_var_mask,
                            core_clause_mask=None,
                            l_batch=torch.zeros(n_vars * 2, dtype=torch.long),
                            c_batch=torch.zeros(n_clauses, dtype=torch.long)
                        )
                    else:
                        label = self.all_labels[split][idx]
                        raise NotImplementedError(f"Label type {self.opts.label}"
                                                  f" not fully implemented.")

                    return [data]


class LCG(Data):
    def __init__(self,
                 n_vars=None,
                 n_clauses=None,
                 l_edge_index=None,
                 c_edge_index=None,

                 core_var_mask=None,
                 core_clause_mask=None,
                 l_batch=None,
                 c_batch=None
                 ):

        super().__init__()
        self.n_vars = n_vars
        self.n_clauses = n_clauses
        self.l_edge_index = l_edge_index
        self.c_edge_index = c_edge_index

        self.core_var_mask = core_var_mask
        self.core_clause_mask = core_clause_mask
        self.l_batch = l_batch
        self.c_batch = c_batch

    @property
    def num_edges(self):
        return self.c_edge_index.size(0)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'l_edge_index':
            return self.n_vars * 2
        elif key == 'c_edge_index':
            return self.n_clauses
        elif key == 'l_batch' or key == 'c_batch' or key == 'positive_index':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)
