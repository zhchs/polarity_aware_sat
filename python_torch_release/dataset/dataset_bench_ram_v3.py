import os
import itertools
import pickle
import glob

import torch
from torch_geometric.data import Data, Dataset, Batch

from tqdm import tqdm

from dataset.utils_bench import parse_cnf_file, clean_clauses, literal2l_idx


class SATBenchDataset_InMemory(Dataset):
    def __init__(self, data_dir, splits, sample_size, use_contrastive_learning, opts, force_process=False):
        self.opts = opts
        self.splits = splits
        self.sample_size = sample_size
        self.data_dir = data_dir
        self.all_files = self._get_files(data_dir)
        self.split_len = self._get_split_len()
        self._cached_data = {}
        self._label_tag = self._compute_label_tag()
        self._force_process = force_process

        super().__init__(data_dir)
        self.process()
        self._load_cached_data()

    def _get_files(self, data_dir):
        files = {}
        for split in self.splits:
            split_files = list(
                sorted(glob.glob(data_dir + f'/{split}/*.cnf', recursive=True)))
            if self.sample_size is not None and len(split_files) > self.sample_size:
                split_files = split_files[:self.sample_size]
            files[split] = split_files
        return files

    def _compute_label_tag(self):
        return str(self.opts.label) if self.opts.label is not None else 'None'

    def __get_labels(self, data_dir):
        labels = {}
        if self.opts.label == 'satisfiability':
            for split in self.splits:
                if split == 'sat' or split == 'augmented_sat':
                    labels[split] = [torch.tensor(
                        1., dtype=torch.float)] * self.split_len
                else:
                    labels[split] = [torch.tensor(
                        0., dtype=torch.float)] * self.split_len
        elif self.opts.label == 'assignment':
            for split in self.splits:
                assert split == 'sat' or split == 'augmented_sat'
                labels[split] = []
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

    def _get_label(self, split, cnf_filepath, sample=None):
        if self.opts.label == 'satisfiability':
            assert split is not None
            if split == 'sat' or split == 'augmented_sat':
                _label = torch.tensor(1., dtype=torch.float)
            else:
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
            filename = os.path.splitext(
                os.path.basename(cnf_filepath))[0]
            assignment_file = os.path.join(os.path.dirname(
                cnf_filepath), filename + '_core_variable.pkl')
            with open(assignment_file, 'rb') as f:
                core_variable = pickle.load(f)

            core_clause_mask = None
            if sample is not None:
                core_filepath = os.path.join(
                    os.path.dirname(cnf_filepath), filename + '.core')
                if os.path.exists(core_filepath):
                    with open(core_filepath, 'r') as f:
                        lines = f.readlines()
                        core_clauses_set = set()
                        for line in lines[1:]:
                            tokens = line.strip().split()
                            lits = [int(t) for t in tokens[:-1]]
                            lits = sorted(list(set(lits)))
                            core_clauses_set.add(tuple(lits))

                    clauses = sample['clauses']
                    n_clauses = len(clauses)
                    core_clause_mask = torch.zeros(n_clauses, dtype=torch.long)
                    for i, clause in enumerate(clauses):
                        lits = sorted(list(set(clause)))
                        if tuple(lits) in core_clauses_set:
                            core_clause_mask[i] = 1
                    if self.opts.debug:
                        # Debug print to verify mask generation
                        if core_clause_mask is not None and core_clause_mask.sum() > 0:
                            print(f"[DEBUG] Processed {filename}: Found {core_clause_mask.sum()} core clauses out of {n_clauses}")
                        elif core_clause_mask is not None:
                            print(f"[DEBUG] Processed {filename}: No core clauses matched! (Check matching logic)")

            _label = {
                'core_var_mask': torch.tensor(core_variable, dtype=torch.long),
                'core_clause_mask': core_clause_mask
            }
        else:
            assert self.opts.label == None
            _label = None

        return _label

    def _get_split_len(self):
        lens = [len(self.all_files[split]) for split in self.splits]
        assert len(set(lens)) == 1
        return lens[0]

    def _get_positive_indices(self):
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

    @property
    def processed_file_names(self):
        return [self._get_cache_file_name(split) for split in self.splits]

    def _get_cache_file_name(self, split):
        # Always use the base LCG cache file, regardless of opts.graph
        # This ensures we load the lightweight data and compute CCG on the fly if needed
        return f'{split}.data.{self._label_tag}.pt'

    def _get_cache_path(self, split):
        return os.path.join(self.processed_dir, self._get_cache_file_name(split))

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        for split_idx, split in enumerate(self.splits):
            cache_path = self._get_cache_path(split)
            if not self._force_process and os.path.exists(cache_path):
                continue

            processed_samples = []
            file_list = self.all_files[split]
            for cnf_filepath in tqdm(
                file_list,
                desc=f"Processing split '{split}'",
                dynamic_ncols=True
            ):
                sample = self._get_cnf(split, cnf_filepath)
                label = self._get_label(split, cnf_filepath, sample)
                processed_samples.append(
                    self._build_data_from_sample(split_idx, sample, label)
                )

            torch.save(processed_samples, cache_path)

    def _load_cached_data(self):
        for split in self.splits:
            cache_path = self._get_cache_path(split)
            self._cached_data[split] = torch.load(cache_path)

    def len(self):
        if self.opts.data_fetching == 'parallel':
            return self.split_len
        else:
            return self.split_len * len(self.splits)

    def _get_cnf(self, split, cnf_filepath):
        n_vars, clauses, learned_clauses = parse_cnf_file(
            cnf_filepath, split_clauses=True)

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
            "clauses": clauses
        }

        return sample

    def _build_data_from_sample(self, split_idx, sample, label):
        n_vars = sample["n_vars"].long()
        n_clauses = sample["n_clauses"].long()
        l = sample["l_edge_index"].long()
        c = sample["c_edge_index"].long()

        if self.opts.label == "core_variable":
            if isinstance(label, dict):
                core_var_mask = label['core_var_mask']
                core_clause_mask = label['core_clause_mask']
            else:
                core_var_mask = label
                core_clause_mask = None

            # Always build basic LCG here. CCG is computed on-the-fly in get()
            data = LCG(
                n_vars=n_vars,
                n_clauses=n_clauses,
                l_edge_index=l,
                c_edge_index=c,
                core_var_mask=core_var_mask,
                core_clause_mask=core_clause_mask,
                l_batch=torch.zeros(n_vars * 2, dtype=torch.long),
                c_batch=torch.zeros(n_clauses, dtype=torch.long)
            )
        else:
            raise NotImplementedError(
                f"Label type {self.opts.label} not fully implemented.")

        return data

    def _convert_to_ccg(self, data):
        n_clauses = data.n_clauses
        n_lits = data.n_vars * 2
        l_edge_index = data.l_edge_index
        c_edge_index = data.c_edge_index

        max_k_neighbors = getattr(self.opts, 'max_k_neighbors', None)
        cc_edge_index, cc_edge_weight = compute_line_graph(
            n_clauses, n_lits, l_edge_index, c_edge_index, max_k_neighbors=max_k_neighbors)

        # Handle optional core_clause_mask
        core_clause_mask = None
        if hasattr(data, 'core_clause_mask'):
            core_clause_mask = data.core_clause_mask

        return LCG_CCG(
            n_vars=data.n_vars,
            n_clauses=data.n_clauses,
            l_edge_index=data.l_edge_index,
            c_edge_index=data.c_edge_index,
            core_var_mask=data.core_var_mask,
            core_clause_mask=core_clause_mask,
            l_batch=data.l_batch,
            c_batch=data.c_batch,
            cc_edge_index=cc_edge_index,
            cc_edge_weight=cc_edge_weight
        )

    def get(self, idx):
        if self.opts.data_fetching == 'parallel':
            res = []
            for split in self.splits:
                data = self._cached_data[split][idx].clone()
                if hasattr(self.opts, 'graph') and self.opts.graph == 'LCG_CCG':
                    data = self._convert_to_ccg(data)
                res.append(data)
            return res

        for split_idx, split in enumerate(self.splits):
            if idx >= self.split_len:
                idx -= self.split_len
                continue
            data = self._cached_data[split][idx].clone()
            if hasattr(self.opts, 'graph') and self.opts.graph == 'LCG_CCG':
                data = self._convert_to_ccg(data)
            return [data]

    def __get(self, idx):
        return self.get(idx)


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


def compute_line_graph(n_clauses, n_lits, l_edge_index, c_edge_index, max_k_neighbors=None):
    # Ensure inputs are on CPU for processing
    # l_edge_index: [E] (literal indices)
    # c_edge_index: [E] (clause indices)

    # Construct sparse incidence matrix H: [n_clauses, n_lits]
    # H[c, l] = 1 if clause c contains literal l
    indices = torch.stack([c_edge_index, l_edge_index])
    values = torch.ones(c_edge_index.size(0), dtype=torch.float)

    # Use sparse tensor for efficient matrix multiplication
    # Shape: (n_clauses, n_lits)
    H = torch.sparse_coo_tensor(
        indices, values, (n_clauses, n_lits)).coalesce()

    # Compute intersection matrix: A = H @ H.t()
    # A[u, v] = number of shared literals between clause u and clause v
    # This operation is efficient on sparse tensors and produces a symmetric matrix (undirected graph)
    intersection_matrix = torch.sparse.mm(H, H.t()).coalesce()

    # Extract indices and values (intersection counts)
    indices = intersection_matrix.indices()
    counts = intersection_matrix.values()

    # Filter out self-loops (diagonal elements)
    # We only want edges between different clauses
    mask = indices[0] != indices[1]
    cc_edge_index = indices[:, mask]
    intersection_counts = counts[mask]

    # Calculate degrees (number of literals per clause)
    # degree[u] = |N(u)|
    degrees = torch.bincount(c_edge_index, minlength=n_clauses).float()

    # Get degrees for source and target nodes of the new edges
    deg_u = degrees[cc_edge_index[0]]
    deg_v = degrees[cc_edge_index[1]]

    # Calculate Jaccard Similarity
    # J(u, v) = |N(u) n N(v)| / |N(u) u N(v)|
    # |N(u) u N(v)| = |N(u)| + |N(v)| - |N(u) n N(v)|
    union_counts = deg_u + deg_v - intersection_counts

    # Avoid division by zero
    cc_edge_weight = intersection_counts / (union_counts + 1e-9)

    if max_k_neighbors is not None and max_k_neighbors > 0:
        # Sort by weight descending
        perm = torch.argsort(cc_edge_weight, descending=True)
        cc_edge_index = cc_edge_index[:, perm]
        cc_edge_weight = cc_edge_weight[perm]

        # Sort by source ascending (stable sort to preserve weight order)
        perm = torch.argsort(cc_edge_index[0], stable=True)
        cc_edge_index = cc_edge_index[:, perm]
        cc_edge_weight = cc_edge_weight[perm]

        # Now compute rank within each group
        row = cc_edge_index[0]
        unique_sources, counts = torch.unique_consecutive(
            row, return_counts=True)

        cumulative_counts = torch.cumsum(counts, dim=0)
        # Prepend 0
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=row.device),
                            cumulative_counts[:-1]])

        # Expand starts to match row length
        starts_expanded = torch.repeat_interleave(starts, counts)

        current_indices = torch.arange(row.size(0), device=row.device)
        ranks = current_indices - starts_expanded

        mask = ranks < max_k_neighbors

        cc_edge_index = cc_edge_index[:, mask]
        cc_edge_weight = cc_edge_weight[mask]

    return cc_edge_index, cc_edge_weight


class LCG_CCG(LCG):
    def __init__(self,
                 n_vars=None,
                 n_clauses=None,
                 l_edge_index=None,
                 c_edge_index=None,
                 core_var_mask=None,
                 core_clause_mask=None,
                 l_batch=None,
                 c_batch=None,
                 cc_edge_index=None,
                 cc_edge_weight=None
                 ):
        super().__init__(
            n_vars=n_vars,
            n_clauses=n_clauses,
            l_edge_index=l_edge_index,
            c_edge_index=c_edge_index,
            core_var_mask=core_var_mask,
            core_clause_mask=core_clause_mask,
            l_batch=l_batch,
            c_batch=c_batch
        )
        self.cc_edge_index = cc_edge_index
        self.cc_edge_weight = cc_edge_weight

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cc_edge_index':
            return self.n_clauses
        else:
            return super().__inc__(key, value, *args, **kwargs)


def collate_fn(batch):
    return Batch.from_data_list([s for s in list(itertools.chain(*batch))])
