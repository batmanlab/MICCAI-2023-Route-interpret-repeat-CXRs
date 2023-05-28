import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))


def config():
    parser = argparse.ArgumentParser(description='Get important concepts masks')
    parser.add_argument('--output', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                        help='path to checkpoints')

    parser.add_argument('--dataset', type=str, default="stanford_cxr", help='dataset name')
    parser.add_argument('--root', type=str, default="BB/lr_0.01_epochs_5_loss_CE", help='root')
    parser.add_argument('--disease', type=str, default="edema", help='disease')
    parser.add_argument('--arch', type=str, default="densenet121", help='Arch')
    parser.add_argument('--csv_file', type=str, default="train_results.csv", help='csv file')
    parser.add_argument('--seed', type=int, default="0", help='seed')
    parser.add_argument('--pos_correct', type=int, default="100", help='pos_correct')
    parser.add_argument('--neg_correct', type=int, default="200", help='neg_correct')
    parser.add_argument('--pos_incorrect', type=int, default="200", help='pos_incorrect')
    parser.add_argument('--neg_incorrect', type=int, default="500", help='neg_incorrect')

    return parser.parse_args()


def prepared_df_for_SSL(args):
    output = Path(f"{args.output}/{args.dataset}/{args.root}/{args.arch}/{args.disease}")
    print(output)
    tot_samples = args.pos_correct + args.neg_correct + args.pos_incorrect + args.neg_incorrect
    df = pd.read_csv(output / args.csv_file)
    print(f"Whole df size: {df.shape}")
    print("*********** Creating positive datasets ***********")
    df_pos_correct = df[(df["GT"] == 1) & (df["pred_w_mimic_model"] == 1)].sample(
        args.pos_correct, random_state=args.seed
    )
    ones = np.ones(df_pos_correct.shape[0])
    df_neg_correct = df[(df["GT"] == 0) & (df["pred_w_mimic_model"] == 0)].sample(
        args.neg_correct, random_state=args.seed
    )
    zeros = np.zeros(df_neg_correct.shape[0])
    df_correct_frames = [df_pos_correct, df_neg_correct]
    df_correct = pd.concat(df_correct_frames)
    df_correct["target"] = np.concatenate((ones, zeros), axis=0)
    print(f"Correct sizes, pos: {df_pos_correct.shape}, neg: {df_neg_correct.shape}, tot: {df_correct.shape}")
    df_correct.to_csv(output / f"correct_tot_{tot_samples}.csv")

    print("*********** Creating negative datasets ***********")
    df_pos_incorrect = df[(df["GT"] == 1) & (df["pred_w_mimic_model"] == 0)].sample(
        args.pos_incorrect, random_state=args.seed
    )
    df_neg_incorrect = df[(df["GT"] == 0) & (df["pred_w_mimic_model"] == 1)].sample(
        args.neg_incorrect, random_state=args.seed
    )
    df_incorrect_frames = [df_pos_incorrect, df_neg_incorrect]
    df_incorrect = pd.concat(df_incorrect_frames)
    df_incorrect["target"] = np.full(df_incorrect.shape[0], -1)
    print(f"Incorrect sizes, pos: {df_pos_incorrect.shape}, neg: {df_neg_incorrect.shape}, tot: {df_incorrect.shape}")
    df_incorrect.to_csv(output / f"incorrect_tot_{tot_samples}.csv")

    df_master_frames = [df_correct, df_incorrect]
    df_master = pd.concat(df_master_frames)
    df_master.to_csv(output / f"master_tot_{tot_samples}.csv")

    print(f"Files save to: {output}")


def main():
    args = config()
    prepared_df_for_SSL(args)


if __name__ == '__main__':
    main()
