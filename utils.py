import random

import numpy as np
from openai import OpenAI

def generate_random_cell(search_space, historical_cell_list=None):
    while True:
        cell = ''
        for option_number in search_space:
            cell += str(random.choice(list(range(option_number))))
        if historical_cell_list == None:
            break
        elif cell not in historical_cell_list:
            break
    return cell

def get_trials_with_top_scores(result_dict, context_num=5):
    sorted_dict = {k: v for k, v in sorted(result_dict.items(), key=lambda item: float(item[1]["prediction"]), reverse=True)}
    top_dict = dict(list(sorted_dict.items())[0:context_num])
    return top_dict

def search_cell_llama(sampled_dict, random_cell_list, chat_model):
    reward = [float(item["prediction"]) for _, item in sampled_dict.items()]
    # Find min and max values
    min_pred = min(reward)
    max_pred = max(reward)

    # Normalize predictions
    for _, item in sampled_dict.items():
        item["normalized_reward"] = (float(item["prediction"]) - min_pred) / (max_pred - min_pred)

    prompt = ""
    for _, trial_dict in sampled_dict.items():
        prompt += trial_dict["cell"]
        prompt += '|' + str(round(trial_dict["normalized_reward"], 4)) + ','
    for random_cell in random_cell_list:
        prompt += random_cell
        prompt += ';'

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    prompts = [{"role": "system", "content": prompt}]
    completion = client.chat.completions.create(
        n=1,
        model=chat_model,
        messages=prompts,
        max_tokens=100,
        logprobs=False,
    )
    response_setting = completion.choices[0].message.content
    cell_setting = response_setting
    print("LLM's output: ", cell_setting)

    return cell_setting

def sample_new_cell(result_dict, context_num, chat_model):
    while True:
        random_trial_list = []
        for _ in range(10):
            random_trial = generate_random_cell([trial_dict["cell"] for trial_dict in result_dict.values()])
            random_trial_list.append(random_trial)
        selected_trials = get_trials_with_top_scores(result_dict, context_num)
        # selected_trials = get_trials_randomly(result_dict, candidate_num)
        sampled_cell = search_cell_llama(selected_trials, random_trial_list, chat_model)
        if sampled_cell in random_trial_list:
            return sampled_cell
        else:
            continue