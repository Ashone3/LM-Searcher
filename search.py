import os
import re
import json
import time
import random
import argparse

from decimal import Decimal
from openai import OpenAI

from utils import generate_random_cell, sample_new_cell

# -----------------------------
# Argument parser configuration
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='history', help="Directory to save search results.")
parser.add_argument('--chat_model', type=str, default='path-to-the-checkpoint', help="LLM model used for sampling new cells.")
parser.add_argument('--trial_num', type=int, default=192, help="Number of search trials to run.")
args = parser.parse_args()
print(args)

# -----------------------------
# Define the search space here
# (Customize according to your task)
# -----------------------------
search_space = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5] # Search space with 5^10 solutions

performance_history = []
trial_dict = {}

# -----------------------------
# Create output directory if it doesn’t exist
# -----------------------------
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

num_iters = 0
for iteration in range(num_iters, args.trial_num):
    # Control number of previous trials referenced by the model
    if iteration <= 200:
        output_num = iteration
    else:
        output_num = 200

    # First few trials are random
    if iteration <= 4:
        cell = generate_random_cell(search_space, trial_dict)
    # Later trials sample based on history
    else:
        cell = sample_new_cell(trial_dict, output_num, args.chat_model)
    
    # -----------------------------
    # Here the "reward function" is defined.
    # Replace this with your custom evaluation metric.
    # -----------------------------
    val_acc = random.uniform(0, 100)

    # Record results for the current trial
    trial_dict[f"Trial{iteration+1}"] = {}
    trial_dict[f"Trial{iteration+1}"]["cell"] = cell
    trial_dict[f"Trial{iteration+1}"]["prediction"] = val_acc

    # Save all historical results to file
    with open('{}/historical_results.json'.format(args.output_dir), 'w') as f:
        json.dump(trial_dict, f)