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
    counts = []
    for i, x in enumerate(sequence):
        new = True
        for j, x_seen in enumerate(seen):
            if torch.equal(x, x_seen):
                counts[j] += 1
                new = False
                break
        if new:
            seen.append(x)
            indexes.append(i)
            counts.append(1)

    indexes = torch.tensor(indexes)
    counts = torch.tensor(counts)
    return indexes, counts


def get_all_files(read_dir):
    paths = []
    for root, dirs, files in os.walk(read_dir):
        for name in files:
            full_path = os.path.join(root, name)
            if '/test/' in full_path:
                paths.append(full_path)
    return paths


def remove_duplicates(read_path, read_dir, save_dir):
    save_path = read_path.replace(read_dir, save_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = load_results(read_path)
    indexes, counts = get_unique(results['y_list'])
    results['y_list'] = results['y_list'][indexes]
    results['x_list'] = results['x_list'][indexes]
    # results['indexs'] = indexes
    # results['counts'] = counts
    torch.save(results, save_path)


# def custom_remove(read_path, read_dir, save_dir):
#     results = load_results(read_path)
#     if 'time' in results:
#         results.pop('time')
#     if 'counts' in results:
#         results.pop('counts')
#     if 'indexs' in results:
#         results.pop('indexs')
#     torch.save(results, read_path)


def move(read_path, read_dir, save_dir):
    save_path = read_path.replace(read_dir, save_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = load_results(read_path)
    # if 'time' in results:
    #     results.pop('time')
    torch.save(results, save_path)


if __name__ == '__main__':

    read_dir = "../results/normal_gurobi"
    save_dir = "../results/normal"

    read_paths = get_all_files(read_dir)
    print(f"Total files found: {len(read_paths)}")

    # func = partial(remove_duplicates, read_dir=read_dir, save_dir=save_dir)
    # func = partial(custom_remove, read_dir=read_dir, save_dir=save_dir)
    func = partial(move, read_dir=read_dir, save_dir=save_dir)
    process_map(func, read_paths, max_workers=30, chunksize=8)