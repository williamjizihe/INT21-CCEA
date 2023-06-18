import argparse
import random

import numpy as np

def distance(pos1, pos2):
    return np.sqrt(np.sum(np.square(np.array(pos1) - np.array(pos2))))

def set_rand_gen(seed):
    random_number_generator = random.Random()
    random_number_generator.seed(42)
    return random_number_generator

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--output_dir', type=str, default='')
    args.add_argument('--room_size', type=int, default=10)
    args.add_argument('--problem_size', type=int, default=5)
    args.add_argument('--grid_scale', type=int, default=200)
    args.add_argument('--alpha', type=float, default=0.2)
    args.add_argument('--seed', type=int, default=42)
    return args.parse_args()

def statistics(results_list):
    return np.mean(results_list), np.std(results_list)