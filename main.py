
""" 
Main script.
"""

import os
import models 
import argparse
import numpy as np
from datetime import datetime as dt


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--task', required=True, type=str, help='rain or temp')
    ap.add_argument('-c', '--config', required=True, type=str, help='Path to configuration file')
    ap.add_argument('-r', '--root', required=True, type=str, help='Path to directory where data is saved')
    ap.add_argument('-n', '--num-runs', default=10, type=int, help='Number of runs of the model')
    ap.add_argument('-o', '--output', default=dt.now().strftime("%Y-%m-%d_%H-%M"), type=str, help='Path to output directory')
    ap.add_argument('-l', '--load', type=str, help='Path to directory from which best_model.ckpt should be loaded')
    args = vars(ap.parse_args())

    loss_list, mape_list = [], []

    # Initialize model
    for i in range(args['num_runs']):
        print(f"\n[INFO] NOW STARTING RUN {i+1}\n")
        trainer = models.Trainer(args)
        best_loss, best_mape = trainer.train(run_id=i)
        loss_list.append(best_loss)
        mape_list.append(best_mape)

    with open(os.path.join('outputs', args['task'], 'mse_run_stats.txt'), 'w') as f:
        f.write(" ".join([str(l) for l in loss_list])) 
        f.write(f"\nMean: {np.mean(loss_list)}")
        f.write(f"\nStdev: {np.std(loss_list)}")

    with open(os.path.join('outputs', args['task'], 'mape_run_stats.txt'), 'w') as f:
        f.write(" ".join([str(l) for l in mape_list])) 
        f.write(f"\nMean: {np.mean(mape_list)}")
        f.write(f"\nStdev: {np.std(mape_list)}")