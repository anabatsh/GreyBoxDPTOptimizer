import os
import sys
import argparse
# import numpy as np
from natsort import natsorted

root_path = '../'
sys.path.insert(0, root_path)

from train_dpt import DPTSolver
from notebooks.utils import *


def get_checkpoint(read_dir, name):
    read_dir = os.path.join(read_dir, name, "checkpoints")
    checkpoint = natsorted(os.listdir(read_dir))[-1]
    checkpoint = os.path.join(read_dir, checkpoint)
    return checkpoint


def run_model(model, read_dir, problem, suffix='test', budget=100):
    problem_path = os.path.join(read_dir, problem, suffix)
    problems = load_problem_set(problem_path)
    dataset = OnlineDataset(problems)
    collate_fn = partial(custom_collate_fn, problem_class=Problem)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1000,
        # num_workers=1,
        # pin_memory=True,
        # shuffle=False,
        collate_fn=collate_fn
    )
    tester = L.Trainer(logger=False, precision=model.config['precision'])
    logs = {}
    for warmup, do_sample in ((1, True),): #((0, False), (0, True), (50, False), (50, True)):
        model.config['do_sample'] = do_sample
        model.config['warmup_steps'] = warmup
        model.config['online_steps'] = int(budget - warmup)

        with torch.inference_mode():
            tester.test(model=model, dataloaders=dataloader)

        warmup_mode = 'warmup' if warmup > 0 else 'no warmup'
        sample_mode = 'sample' if do_sample else 'argmax'
        y_list = model.trajectory.cpu().numpy()
        logs[f'({warmup_mode}) ({sample_mode})'] = {
            'm_list': np.arange(budget) + 1,
            'y_list (mean)': y_list,
            'y_list (std)': np.zeros_like(y_list)
        }
    return logs


if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description='Load configuration file.')
    parser.add_argument('--data_dir', type=str, default="/mnt/data/normal")
    parser.add_argument('--checkpoint_dir', type=str, default="../../logs/DPT_3")
    parser.add_argument('--checkpoint', type=str, default="point_next_ce")
    parser.add_argument('--save_dir', type=str, default="../../results/normal")
    parser.add_argument('--budget', type=int, default=1000)

    # Parse arguments
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = get_checkpoint(args.checkpoint_dir, args.checkpoint)
    model = DPTSolver.load_from_checkpoint(checkpoint).to(device)
    problem_list = natsorted(os.listdir(args.data_dir))

    # run the model
    model_results = defaultdict(dict)
    for problem in problem_list:
        model_results[problem] = run_model(model, args.data_dir, problem, budget=5)#args.budget)

    # save the results
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'{args.checkpoint}.npy')
    np.save(save_path, model_results)