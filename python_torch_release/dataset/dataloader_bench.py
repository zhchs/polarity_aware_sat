import itertools

from dataset.dataset_bench import SATBenchDataset
from dataset.dataset_bench_ram_v2 import SATBenchDataset_InMemory as SATBenchDataset_InMemory_v2
from dataset.dataset_bench_ram_v3 import SATBenchDataset_InMemory as SATBenchDataset_InMemory_v3
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
import copy


def collate_fn(batch):
    return Batch.from_data_list([s for s in list(itertools.chain(*batch))])


def _get_dataset_instance(data_dir, splits, sample_size, use_contrastive_learning, opts):
    if not opts.in_memory:
        if opts.graph != 'LCG':
            raise NotImplementedError(
                "Only LCG is supported for out-of-memory training.")
        return SATBenchDataset(
            data_dir, splits, sample_size, use_contrastive_learning, opts)

    # In-memory
    force_process = getattr(opts, 'force_process', False)

    if opts.graph == 'LCG_CCG':
        if opts.graph_build_mode == 'static':
            return SATBenchDataset_InMemory_v2(
                data_dir, splits, sample_size, use_contrastive_learning, opts, force_process=force_process)
        elif opts.graph_build_mode == 'dynamic':
            return SATBenchDataset_InMemory_v3(
                data_dir, splits, sample_size, use_contrastive_learning, opts, force_process=force_process)
        else:
            raise ValueError(
                "Please specify graph_build_mode for LCG_CCG graph.")
    else:
        # Default to v2 for standard LCG (it is backward compatible)
        return SATBenchDataset_InMemory_v2(
            data_dir, splits, sample_size, use_contrastive_learning, opts, force_process=force_process)


def get_dataloader(data_dir, splits, sample_size, opts, mode, use_contrastive_learning=False):

    if mode == 'train':
        batch_size = opts.batch_size // len(
            splits) if opts.data_fetching == 'parallel' else opts.batch_size
        dataset = _get_dataset_instance(
            data_dir, splits, sample_size, use_contrastive_learning, opts)
        shuffle = True

    elif mode == 'valid_batch':
        batch_size = opts.batch_size // len(
            splits) if opts.data_fetching == 'parallel' else opts.batch_size
        dataset = _get_dataset_instance(
            data_dir, splits, sample_size, use_contrastive_learning, opts)
        shuffle = False

    else:
        batch_size = 1
        otps_eval = copy.deepcopy(opts)
        otps_eval.data_fetching = 'sequential'
        dataset = _get_dataset_instance(
            data_dir, splits, sample_size, use_contrastive_learning, otps_eval)
        shuffle = False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=opts.num_workers,
    ), len(dataset)
