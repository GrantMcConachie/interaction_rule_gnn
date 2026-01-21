"""
Training script for models
"""

import os
import yaml
import argparse

import torch


def main(args):
    # run main training script
    model = train(args)

    # save results
    save_results(args)


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', type=str, help='Model to be trained.', required=True
    )
    parser.add_argument(
        '-c', '--config', type=str, help='Path to config file.', required=True
    )
    parser.add_argument(
        '-d', '--dataset', type=str, help='Path to dataset.', required=True
    )
    parser.add_argument(
        '-sp', '--save_path', type=str, help='model save path.', required=True
    )
    parser.add_argument(
        '-lp', '--log_path', type=str, help='model log path.', required=True
    )
    args = parser.parse_args()

    # main script logic
    main(args)
