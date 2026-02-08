import csv
import json
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
import os
import time
import sys
import argparse
from utils.hallu_utils import *


### to evaluate your method, implement and run generate_answer function!

def extract_answer_after_think(text):
    """Extract answer after </think> tag if present, otherwise return original text."""
    if isinstance(text, str) and "</think>" in text:
        # Find the last occurrence of </think> and extract everything after it
        parts = text.split("</think>")
        return parts[-1].strip()
    return text


def safe_percentage(numerator, denominator, decimal_places=4):
    """Safely calculate percentage, returning 0 if denominator is 0."""
    if denominator == 0:
        return 0.0
    return round(100 * numerator / denominator, decimal_places)


# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate hallucination benchmark results')
parser.add_argument('input_json', type=str, help='Input JSON file name (in current directory)')
parser.add_argument('--load_json', action='store_true', help='Load existing evaluation results if available')
args = parser.parse_args()

root_dir = "."
input_file_name = args.input_json
# Create results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(__file__), "..", "results/Hallusion")
os.makedirs(results_dir, exist_ok=True)

# Extract base name without extension for output files
base_name = os.path.splitext(os.path.basename(input_file_name))[0]
save_json_path_vd = os.path.join(results_dir, f"{base_name}_output_vd_model.json")
save_json_path_vs = os.path.join(results_dir, f"{base_name}_output_vs_model.json")
load_json = args.load_json
model_output_entry = "model_prediction"
model_correctness_entry = "gpt4v_output_gpt_check"


def generate_answer(data, model_output_entry):

    ## TODO
    ## implement this section with yout model!
    ## your_function(img_filename, question) -> "0" (No), "1" (Yes), "2" (Uncertain)
    # for r in data:
        # r[model_output_entry] = your_function(r["filename"], r["question"])

    return data


if __name__ == "__main__":

    data_vd = []
    data_vs = []
    
    # Check if input file exists
    if not os.path.exists(input_file_name):
        print(f"Error: Input file '{input_file_name}' not found!")
        sys.exit(1)
    
    with open(input_file_name) as json_file:
        datas = json.load(json_file)

    # Extract answers after </think> tag if present
    for data in datas:
        if model_output_entry in data:
            data[model_output_entry] = extract_answer_after_think(data[model_output_entry])

    for data in tqdm(datas):
        if data['category'] == 'VD':
            data_vd.append(data)
        if data['category'] == 'VS':
            data_vs.append(data)

    data_vd = evaluate_by_chatgpt(data_vd, model_output_entry, model_correctness_entry, load_json=load_json, save_json_path=save_json_path_vd)
    data_vd = check_same_by_chatgpt(data_vd, model_output_entry, load_json=load_json, save_json_path=save_json_path_vd)
    #time.sleep(60) #
    try:
        data_vs = evaluate_by_chatgpt(data_vs, model_output_entry, model_correctness_entry, load_json=load_json, save_json_path=save_json_path_vs)
        data_vs = check_same_by_chatgpt(data_vs, model_output_entry, load_json=load_json, save_json_path=save_json_path_vs)
    except:
        time.sleep(60)
        data_vs = evaluate_by_chatgpt(data_vs, model_output_entry, model_correctness_entry, load_json=load_json, save_json_path=save_json_path_vs)
        data_vs = check_same_by_chatgpt(data_vs, model_output_entry, load_json=load_json, save_json_path=save_json_path_vs)
        
    print("##### GPT Evaluate #####")

    data_vd = assign_correctness(data_vd, correctness_entry=model_correctness_entry)
    data_vs = assign_correctness(data_vs, correctness_entry=model_correctness_entry)
    data = data_vd + data_vs

    all_data = get_eval_all(data, model_correctness_entry)
    all_vd = get_eval_all(data_vd, model_correctness_entry)
    all_vs = get_eval_all(data_vs, model_correctness_entry)

    table1 = [["per question", "Total"], 
              ["VD", safe_percentage(all_vd["correct"], all_vd["total"])], 
              ["VS", safe_percentage(all_vs["correct"], all_vs["total"])], 
              ["Overall", safe_percentage(all_data["correct"], all_data["total"])]]
    tab1 = PrettyTable(table1[0])
    tab1.add_rows(table1[1:])


    q_acc_gpt = safe_percentage(all_data["correct"], all_data["total"])

    all_data = get_eval_pair_all(data, model_correctness_entry)
    easy = get_eval_pair_easy(data)
    hard = get_eval_pair_hard(data)
    all_vd = get_eval_pair_all(data_vd, model_correctness_entry)
    easy_vd = get_eval_pair_easy(data_vd)
    hard_vd = get_eval_pair_hard(data_vd)
    all_vs = get_eval_pair_all(data_vs, model_correctness_entry)
    easy_vs = get_eval_pair_easy(data_vs)
    hard_vs = get_eval_pair_hard(data_vs)
    # question pair level
    table3 = [["per question pair", "Easy", "Hard", "Total"], 
              ["VD", safe_percentage(easy_vd["correct"], easy_vd["total"]), safe_percentage(hard_vd["correct"], hard_vd["total"]), safe_percentage(all_vd["correct"], all_vd["total"])], 
              ["VS", safe_percentage(easy_vs["correct"], easy_vs["total"]), safe_percentage(hard_vs["correct"], hard_vs["total"]), safe_percentage(all_vs["correct"], all_vs["total"])], 
              ["Overall", safe_percentage(easy["correct"], easy["total"]), safe_percentage(hard["correct"], hard["total"]), safe_percentage(all_data["correct"], all_data["total"])]]
    tab3 = PrettyTable(table3[0])
    tab3.add_rows(table3[1:])
    print(tab3)

    fig_all = get_eval_fig(data)
    fig_vd = get_eval_fig(data_vd)
    fig_vs = get_eval_fig(data_vs)

    # image level 
    table2 = [["per figure", "Correct", "Wrong", "Score"], 
              ["VD", safe_percentage(fig_vd["correct"], fig_vd["total"]), safe_percentage(fig_vd["inconsistent"], fig_vd["total"]) + safe_percentage(fig_vd["wrong"], fig_vd["total"]), round(fig_vd["score"], 4) if fig_vd["total"] > 0 else 0.0], 
              ["VS", safe_percentage(fig_vs["correct"], fig_vs["total"]), safe_percentage(fig_vs["inconsistent"], fig_vs["total"]) + safe_percentage(fig_vs["wrong"], fig_vs["total"]), round(fig_vs["score"], 4) if fig_vs["total"] > 0 else 0.0], 
              ["Overall", safe_percentage(fig_all["correct"], fig_all["total"]), safe_percentage(fig_all["inconsistent"], fig_all["total"]) + safe_percentage(fig_all["wrong"], fig_all["total"]), round(fig_all["score"], 4) if fig_all["total"] > 0 else 0.0]]
    tab2 = PrettyTable(table2[0])
    tab2.add_rows(table2[1:])

    pair_acc_gpt = safe_percentage(all_data["correct"], all_data["total"])
    figure_acc_gpt = safe_percentage(fig_all["correct"], fig_all["total"])
    easy_acc_gpt = safe_percentage(easy["correct"], easy["total"])
    hard_acc_gpt = safe_percentage(hard["correct"], hard["total"])



    print("##### Question Stats #####")
    print("Easy Questions: " + str(easy_vd["total_q"]) + "(Visual Dependent) + " + str(easy_vs["total_q"]) + "(Visual Supplement)")
    print("Hard Questions: " + str(hard_vd["total_q"]) + "(Visual Dependent) + " + str(hard_vs["total_q"]) + "(Visual Supplement)")
    print("Total Questions: " + str(all_data["total_q"]))


    print("##### Figure Stats #####")
    print("Visual Dependent Figures: " + str(fig_vd["total"]))
    print("Visual Supplement Figures: " + str(fig_vs["total"]))
    print("Total Figures: " + str(fig_all["total"]))

    print("##### Leaderboard Stats #####")

    table = [["", "Acc per question pair (qAcc)", "Acc per figure (fAcc)", "Acc per easy question (easy aAcc)", "Acc per hard question (hard aAcc)", "Acc per question (aAcc)"], 
              ["GPT Eval", pair_acc_gpt, figure_acc_gpt, easy_acc_gpt, hard_acc_gpt, q_acc_gpt]]
    leaderboard = PrettyTable(table[0])
    leaderboard.add_rows(table[1:])
    print(leaderboard)


    stats = yes_ratio_stats(data)
    
    total_errors = all_data["LH_cg"] + all_data["VI_cg"] + all_data["Mix_cg"]

    table = [["", "Yes/No Bias (Pct Diff)", "Yes/No Bias (FP Ratio)", "Consistency Test (correct)", "Consistency Test (inconsistent)", "Consistency Test (wrong)", "LH", "VI", "Mixed"], 
              ["GPT Eval", stats["diff"], stats["fp"], safe_percentage(fig_all["correct"], fig_all["total"]), safe_percentage(fig_all["inconsistent"], fig_all["total"]), safe_percentage(fig_all["wrong"], fig_all["total"]), safe_percentage(all_data["LH_cg"], total_errors), safe_percentage(all_data["VI_cg"], total_errors), safe_percentage(all_data["Mix_cg"], total_errors)]]
    test = PrettyTable(table[0])
    test.add_rows(table[1:])
    print(test)
    
    # Save summary results to file
    summary_path = os.path.join(results_dir, f"{base_name}_summary.json")
    summary_results = {
        "input_file": input_file_name,
        "leaderboard": {
            "qAcc": pair_acc_gpt,
            "fAcc": figure_acc_gpt,
            "easy_aAcc": easy_acc_gpt,
            "hard_aAcc": hard_acc_gpt,
            "aAcc": q_acc_gpt
        },
        "bias_and_consistency": {
            "yes_no_bias_pct_diff": stats["diff"],
            "yes_no_bias_fp_ratio": stats["fp"],
            "consistency_correct": safe_percentage(fig_all["correct"], fig_all["total"]),
            "consistency_inconsistent": safe_percentage(fig_all["inconsistent"], fig_all["total"]),
            "consistency_wrong": safe_percentage(fig_all["wrong"], fig_all["total"]),
            "LH_percentage": safe_percentage(all_data["LH_cg"], total_errors),
            "VI_percentage": safe_percentage(all_data["VI_cg"], total_errors),
            "Mixed_percentage": safe_percentage(all_data["Mix_cg"], total_errors)
        },
        "detailed_stats": {
            "total_questions": all_data["total_q"],
            "easy_questions": {"VD": easy_vd["total_q"], "VS": easy_vs["total_q"]},
            "hard_questions": {"VD": hard_vd["total_q"], "VS": hard_vs["total_q"]},
            "total_figures": fig_all["total"],
            "VD_figures": fig_vd["total"],
            "VS_figures": fig_vs["total"]
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"\n##### Results saved to: #####")
    print(f"VD output: {save_json_path_vd}")
    print(f"VS output: {save_json_path_vs}")
    print(f"Summary: {summary_path}")