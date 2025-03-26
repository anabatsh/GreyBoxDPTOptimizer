import os
import sys
import torch
import argparse
from functools import partial
from tqdm.contrib.concurrent import process_map

root_path = '../'
sys.path.insert(0, root_path)

from scripts.run_solver import load_results
 

def get_unique(sequence):
    seen = []
    indexes = []
    for i, x in enumerate(sequence):
        new = True
        for j, x_seen in enumerate(seen):
            if torch.equal(x, x_seen):
                new = False
                break
        if new:
            seen.append(x)
            indexes.append(i)

    indexes = torch.tensor(indexes)
    return indexes


def remove_duplicates(read_path, read_dir, save_dir):
    save_path = read_path.replace(read_dir, save_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = load_results(read_path)
    indexes = get_unique(results['y_list'])
    results['y_list'] = results['y_list'][indexes]
    results['x_list'] = results['x_list'][indexes]
    torch.save(results, save_path)


def move(read_path, read_dir, save_dir):
    save_path = read_path.replace(read_dir, save_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = load_results(read_path)
    torch.save(results, save_path)


def get_all_files(read_dir):
    paths = []
    for root, dirs, files in os.walk(read_dir):
        for name in files:
            full_path = os.path.join(root, name)
            if 'test' in full_path:
                paths.append(full_path)
    return paths


if __name__ == '__main__':

    read_dir = "../../trajectories/normal_gurobi"
    save_dir = "../../trajectories/normal"

    read_paths = get_all_files(read_dir)
    print(f"Total files found: {len(read_paths)}")

    # func = partial(remove_duplicates, read_dir=read_dir, save_dir=save_dir)
    func = partial(move, read_dir=read_dir, save_dir=save_dir)
    process_map(func, read_paths, max_workers=30, chunksize=8)