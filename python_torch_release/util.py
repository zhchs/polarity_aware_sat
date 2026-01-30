import subprocess
import threading
import inspect
import random
import tempfile
import numpy as np
import os
import uuid
import time
import dill as pickle
from datetime import datetime, timedelta, timezone
import fcntl

# misc


def flip(p=0.5): return random.random() < p


def is_small_enough(max_n_nodes, n_vars, n_clauses):
    return (2 * n_vars + n_clauses) < max_n_nodes

# system


def get_caller_linenum(offset=1):
    frame = inspect.stack()[1 + offset]
    return os.path.basename(frame[0].f_code.co_filename), frame.lineno


def get_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')


def get_hostname(expensive=False):
    # Warning: memory usage spike from the fork, may not be worth constantly re-calling
    if expensive:
        return subprocess.check_output(['hostname']).strip().decode('ascii')
    else:
        return "<unknown>"


def timeit(f, *args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

# mariadb


# numpy
def np_top_k_idxs(arr, k):
    assert (k <= np.size(arr))
    return arr.argsort()[-k:][::-1]


def sample_unnormalized(qs):
    if np.isnan(qs).any():
        raise Exception("sample_unnormalized found nans: %s" % str(qs))
    qs = np.array(qs)
    ps = qs / np.sum(qs)
    idx = np.random.choice(len(ps), size=1, p=ps)[0]
    return idx, ps[idx]


def sample_unnormalized_dict(kvs):
    qs = np.zeros(len(kvs))
    keys = list(kvs.keys())
    for i, key in enumerate(keys):
        qs[i] = kvs[key]

    idx, p = sample_unnormalized(qs)
    return keys[idx], p


def np_placeholder():
    return np.zeros(0)


def gd_scratch_bcname(gd_id): return "v2gdscratch%d" % gd_id
def gd_tfr_bcname(gd_id): return "v2gdtfrs%d" % gd_id


def gd_scratch_dir(gd_id):
    return "/dev/shm/scratch"  # RAM


def _read_last_line(fp) -> str:
    fp.seek(0)
    lines = fp.readlines()
    if not lines:
        return ''
    return lines[-1].decode('utf-8', errors='ignore').rstrip('\n')


def gen_id(hours=8, cmd="test"):
    ckpt_dir = checkpoint_super_root_dir()
    os.makedirs(ckpt_dir, exist_ok=True)
    info_path = os.path.join(ckpt_dir, "train_id_info.tsv")

    with open(info_path, 'a+b') as fp:
        fcntl.flock(fp, fcntl.LOCK_EX)

        last_line = _read_last_line(fp)
        last_id = 0
        if last_line:
            parts = last_line.split('\t', 1)
            try:
                last_id = int(parts[0].strip())
            except Exception:
                last_id = 0

        new_id = last_id + 1

        tz = timezone(timedelta(hours=hours))
        ts = datetime.now(tz=tz).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        line = f"{new_id}\t{ts}\t{cmd}\n"
        fp.write(line.encode('utf-8'))
        fp.flush()
        os.fsync(fp.fileno())
    return new_id


def checkpoint_super_root_dir(): return "../checkpoints/"
def checkpoint_root_dir(): return "../checkpoints/checkpoints_v5"
def checkpoint_base_dir(train_id): return "train_id%d" % train_id
def checkpoint_dir(train_id): return os.path.join(
    checkpoint_root_dir(), checkpoint_base_dir(train_id=train_id))


def train_config_dir(file_name): return f"../configs/train/{file_name}"


def eval_root_dir(): return "../checkpoints/eval_v5"
def eval_base_dir(eval_id): return "eval_id%d" % eval_id


def eval_dir(eval_id): return os.path.join(
    eval_root_dir(), eval_base_dir(eval_id=eval_id))
