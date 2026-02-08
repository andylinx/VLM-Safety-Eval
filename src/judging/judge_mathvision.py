import fire
import re
from tqdm import tqdm
import time
import json
import pandas as pd
from utils.mathvision_utils import timestamp, save_jsonl, load_jsonl, find_math_answer, is_equal, is_number
import os

def load_json(file_path):
    """Load JSON file and return DataFrame"""
    return pd.read_json(file_path)

def save_json(df, file_path):
    """Save DataFrame to JSON file"""
    df.to_json(file_path, orient='records', indent=2)


def evaluate(answer_file, regen_answer=False):
    # Load JSON file instead of CSV
    df = load_json(answer_file)
    
    for idx in tqdm(df.index, desc='gen_correct'):
        row = df.loc[idx]
        
        # Extract ground truth answer from the row itself (no need for separate reference file)
        gt_answer = str(row['answer'])
        
        # Handle options if they exist
        options_str = str(row.get('options', ''))
        if options_str and options_str != 'nan' and len(options_str.strip()) > 0:
            # Parse options from the formatted string "(A) option1\n(B) option2\n..."
            options_lines = [line.strip() for line in options_str.split('\n') if line.strip()]
            options = []
            for line in options_lines:
                if line.startswith('(') and ')' in line:
                    option_text = line.split(')', 1)[1].strip()
                    options.append(option_text)
            
            # Get ground truth value if it's a letter answer
            if len(gt_answer) == 1 and gt_answer.upper() in 'ABCDE':
                letter_idx = ord(gt_answer.upper()) - ord('A')
                if 0 <= letter_idx < len(options):
                    gt_answer_value = options[letter_idx]
                else:
                    gt_answer_value = ''
            else:
                gt_answer_value = ''
        else:
            gt_answer_value = ''

        # Extract model answer from model_response column
        if pd.isna(row.get('model_answer')) or regen_answer:
            model_response = str(row.get('model_response', ''))
            print(f"Processing index {idx} with model response: {model_response}")
            if "</think>" not in model_response:
                print(f"Warning: '</think>' tag not found in model response for index {idx}")
                print(f"Model response: {model_response}")
                print("=========================")
                continue
            
            if model_response == 'nan':
                model_response = ''
                
            model_answer = model_response.strip()
            
            # Parse answer from model response
            for c in 'ABCDE':
                if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                    model_answer = c
            
            if is_number(model_answer.split('is ')[-1].rstrip('.')):
                model_answer = model_answer.split('is ')[-1].rstrip('.')
            
            if 'oxed{' not in model_answer:
                for flag in ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']:
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
                    flag = flag.replace('the', 'The')
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split('\n')[0].split('. ')[0]
            elif model_answer.count('oxed{') > 1:
                model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]
                
            model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').rstrip('.').lstrip(':').strip()
            df.loc[idx, 'model_answer'] = model_answer
        else:
            model_answer = str(row['model_answer'])
        
        # Determine correctness
        is_correct = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
        df.loc[idx, 'correct'] = is_correct
    
    # Save updated JSON
    save_json(df, answer_file)


def math_level_subject_acc(answer_file):
    print(answer_file)
    df = load_json(answer_file)
    
    results_dict = {}
    for idx in tqdm(df.index, desc='math_level_subject_acc'):
        row = df.loc[idx]
        correct = row['correct']
        subject = str(row['subject'])
        level = str(row['level'])
        
        for key in [
            '-all', 
            f'-level{level}', 
            f'{subject}', 
            f'{subject}_level{level}', 
            f'-level{level}_{subject}'
            ]:
            if key not in results_dict:
                results_dict[key] = [0,0]
            results_dict[key][0] += 1 if correct else 0
            results_dict[key][1] += 1

    for key in results_dict.keys():
        if results_dict[key][1] == 0:
            results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}=0'
        else:
            results_dict[key] = f'{results_dict[key][0]}/{results_dict[key][1]}={round(results_dict[key][0]/ max(results_dict[key][1], 1)*100, 2)}%'

    results_dict = {key: results_dict[key] for key in sorted(results_dict.keys())}
    print(os.path.basename(answer_file), ':\t', results_dict['-all'])
    json.dump(results_dict, open(answer_file.replace('.json', '_result.json'), 'w'), indent=4, ensure_ascii=False)



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Judge MathVision responses from JSON files')
    parser.add_argument('input', type=str, help='Specific JSON file to process')
    parser.add_argument('--regen', action='store_true', help='Regenerate model answers even if they exist')
    
    args = parser.parse_args()
    

    if os.path.exists(args.input) and args.input.endswith('.json'):
        try:
            print(f"Processing {args.input}...")
            evaluate(args.input, args.regen)
            math_level_subject_acc(args.input)
        except Exception as e:
            print(f"Error processing {args.input}: {e}")
    else:
        print(f"File {args.input} not found or not a JSON file")
    