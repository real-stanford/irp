import os
import pathlib

import numpy as np
import pandas as pd
import scipy.stats as st
import wandb

def get_checkpoint_df(checkpoint_dir):
    all_checkpoint_paths = sorted(pathlib.Path(checkpoint_dir).glob('*.ckpt'))
    rows = list()
    for path in all_checkpoint_paths:
        fname = path.stem
        row = dict()
        split_result = fname.split('-')
        if len(split_result) != 2:
            continue
        for item in split_result:
            key, value = item.split('=')
            row[key] = float(value)
        row['path'] = str(path.absolute())
        rows.append(row)
    checkpoint_df = pd.DataFrame(rows)
    return checkpoint_df


def get_best_checkpoint(output_dir):
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    checkpoint_df = get_checkpoint_df(checkpoint_dir)
    checkpoint_path = checkpoint_df.loc[checkpoint_df.val_loss.idxmin()].path
    return checkpoint_path


def get_error_plots_log(key, errors):
    k_min_error = np.minimum.accumulate(errors, axis=-1)
    early_stop_error = errors.copy()
    for i in range(len(early_stop_error)):
        curr_min = early_stop_error[i,0]
        for j in range(len(early_stop_error[i])):
            this = early_stop_error[i,j]
            if curr_min < 0.02:
                early_stop_error[i,j] = curr_min
            else:
                curr_min = min(curr_min, this)

    rope_log_df = pd.DataFrame({
        'step': np.arange(errors.shape[-1]),
        'error': np.mean(errors, axis=0)*100,
        'error_sem': st.sem(errors, axis=0)*100,
        'median_error': np.median(errors, axis=0)*100,
        'k_min_error': np.mean(k_min_error, axis=0)*100,
        'early_stop_error': np.mean(early_stop_error, axis=0)*100,
        'early_stop_error_sem': st.sem(early_stop_error, axis=0)*100
    })

    table = wandb.Table(dataframe=rope_log_df)
    log = {
        key+'/table': table,
        key+'/error': wandb.plot.line(
            table, 'step', 'error', title="error(cm)"),
        key+'/median_error': wandb.plot.line(
            table, 'step', 'median_error', title="median error(cm)"),
        key+'/k_min_error': wandb.plot.line(
            table, 'step', 'k_min_error', title="k-min error(cm)"),
        key+'/early_stop_error': wandb.plot.line(
            table, 'step', 'early_stop_error', title="early stop error(cm)")
    }
    return log

def get_raw_runs_df(api, path):
    runs_iter = api.runs(path=path)
    rows = list()
    for run in runs_iter:
        row = {
            'name': run.name,
            'id': run.id,
            'tags': run.tags,
            'state': run.state,
            'created_at': pd.Timestamp(run.created_at),
            'heartbeat_at': pd.Timestamp(run.heartbeat_at),
            'notes': run.notes,
            'summary': run.summary._json_dict,
            'config': run.config,
            'run': run
        }
        rows.append(row)
    runs_df = pd.DataFrame(rows)
    return runs_df

def get_row_tag(
        row, 
        keys=['Dress', 'Jumpsuit', 'Skirt', 'Top', 'Trousers', 'Tshirt']):
    tags = set(row.tags)
    for key in keys:
        if key in tags:
            return key
    return None

def get_row_output_dir(row):
    return row['config']['output_dir']
