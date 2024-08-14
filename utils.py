import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from prettytable import PrettyTable


def show_results(read_dir, solvers=[]):
    key_list = ('time', 'x_best', 'y_best', 'm_list', 'y_list')
    solver_results = {}
    for solver in solvers:#os.listdir(read_dir):
        solver_dir = os.path.join(read_dir, solver)
        if os.path.isdir(solver_dir):
            solver_results[solver] = {key: [] for key in key_list}
            for seed in os.listdir(solver_dir):
                seed_dir = os.path.join(solver_dir, seed)
                if os.path.isdir(seed_dir):
                    with open(os.path.join(seed_dir, 'results.json')) as f:
                        r = json.load(f)
                        for key in key_list:
                            solver_results[solver][key].append(r[key])

    save_path_img = os.path.join(read_dir, 'results.png')
    save_path_txt = os.path.join(read_dir, 'results.txt')
    f = open(save_path_txt, 'w')
    # if os.path.exists(save_path_txt):
    #     os.remove(save_path_txt)

    plt.figure(figsize=(8, 4))#, facecolor='black')
    # ax = plt.axes()
    # ax.set_facecolor('black')
    # ax.spines['bottom'].set_color('white')
    # ax.spines['top'].set_color('white') 
    # ax.spines['right'].set_color('white')
    # ax.spines['left'].set_color('white')
    # ax.xaxis.label.set_color('white')
    # ax.yaxis.label.set_color('white')
    # ax.tick_params(colors='white', which='both')

    plt.title('Optimization Process')#, color='white')
    plt.ylabel('Target Value')
    plt.xlabel('Iteration')

    cmap = cm.get_cmap('jet')
    c_list = np.linspace(0.1, 0.9, len(solver_results))[::-1]

    tb = PrettyTable()
    tb.field_names = ["Solver", "y best", "y mean", "y std", "time mean"]

    for (solver, results), c in zip(solver_results.items(), c_list):
        color = cmap(c)
        y_best = np.min(results['y_best'])
        y_mean = np.max(results['y_best'])
        y_std = np.std(results['y_best'])
        time_mean = np.mean(results['time'])
        f.write(f'{solver:<10}: best = {y_best: .5f} mean = {y_mean: .5f} std = {y_std: .5f} time = {time_mean:.3f}\n')
        
        tb.add_row([solver, f'{y_best: .5f}', f'{y_mean: .5f}', f'{y_std: .5f}', f'{time_mean:.3f}'])

        m_list_full = sum(results['m_list'], [])
        m_intr = np.linspace(np.min(m_list_full), np.max(m_list_full), 10).astype(np.int32)
        y_intr = [np.interp(m_intr, m_list, y_list) for (m_list, y_list) in zip(results['m_list'], results['y_list'])]

        # for (m_list, y_list) in zip(results['m_list'], results['y_list']):
        #     plt.plot(m_list, y_list, '--', c=color)
        plt.plot(m_intr, np.mean(y_intr, 0), label=solver, c=color)
        plt.fill_between(m_intr, np.min(y_intr, 0), np.max(y_intr, 0), alpha=0.2, color=color)

    # lgd = plt.legend(facecolor='black', labelcolor='white', loc='center left', bbox_to_anchor=(1, 0.5))
    lgd = plt.legend(facecolor='white', labelcolor='black', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(save_path_img, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.show()

    print(tb, file=f)
    f.close()