#!/usr/bin/env python3
"""
Judge script for MMVP Benchmark evaluation using GPT-4 via Azure.
Follows the official MMVP evaluation methodology.
"""

import json
import argparse
import os
import sys
import time
import re
from tqdm import tqdm
from prettytable import PrettyTable

# Import from local models module
from models.azure_openai_client import AzureOpenAIClient


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


def get_yes_no_answer(azure_client, question, max_retries=5):
    """
    Query Azure OpenAI GPT-4 to get a yes/no answer for evaluation.
    
    Args:
        azure_client: AzureOpenAIClient instance
        question: The evaluation question to ask
        max_retries: Maximum number of retries on failure
    
    Returns:
        "yes", "no", or "Could not determine yes or no."
    """
    system_prompt = "You are a helpful and precise assistant for checking the quality of the answer. Please answer in only yes or no."
    
    for attempt in range(max_retries):
        try:
            response = azure_client.client.chat.completions.create(
                model=azure_client.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.2,
                max_tokens=10,
                timeout=30
            )
            answer = response.choices[0].message.content.strip()
            
            yes_no_regex = re.compile(r"^(yes|no)$", re.IGNORECASE)
            
            if yes_no_regex.match(answer):
                return answer.lower()
            else:
                # Try to extract yes/no from a longer response
                if "yes" in answer.lower() and "no" not in answer.lower():
                    return "yes"
                elif "no" in answer.lower() and "yes" not in answer.lower():
                    return "no"
                else:
                    return "Could not determine yes or no."
                    
        except Exception as e:
            print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)  # Wait before retry
            else:
                return "Could not determine yes or no."
    
    return "Could not determine yes or no."


def evaluate_mmvp_responses(input_file, azure_client, load_json=False, save_json_path=None):
    """
    Evaluate MMVP responses using GPT-4.
    
    MMVP evaluation methodology:
    - Questions come in pairs (2 questions per image)
    - Both questions in a pair must be correct for the pair to be correct
    - Final accuracy is based on the number of correct pairs
    
    Args:
        input_file: Path to the JSON file with model responses
        azure_client: AzureOpenAIClient instance for GPT-4 evaluation
        load_json: Whether to load existing evaluation results
        save_json_path: Path to save intermediate results
    
    Returns:
        List of evaluated data with GPT-4 judgments
    """
    # Load responses
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        if "responses" in data:
            responses = data["responses"]
        else:
            responses = [data]
    else:
        responses = data
    
    print(f"Loaded {len(responses)} responses from {input_file}")
    
    # Check if we should load existing evaluations
    if load_json and save_json_path and os.path.exists(save_json_path):
        print(f"Loading existing evaluations from {save_json_path}")
        with open(save_json_path, 'r') as f:
            evaluated_data = json.load(f)
        return evaluated_data
    
    # Extract answers after </think> tag if present
    for item in responses:
        if "response" in item:
            item["response"] = extract_answer_after_think(item["response"])
    
    # Evaluate each response with GPT-4
    print("Evaluating responses with GPT-4...")
    evaluated_responses = []
    
    for item in tqdm(responses, desc="Evaluating"):
        question = item.get("prompt", "")
        correct_answer = item.get("answer", "")
        model_response = item.get("response", "")
        
        # Construct evaluation question
        eval_question = (
            f"Given the following question {question}, "
            f"the correct answer is {correct_answer}. "
            f"Does the following answer correctly answers the question, you should note that if the choice is (a) yes, then both (a) and yes are acceptable answers."
            f"answer:{model_response}?"
        )
        
        # Get GPT-4 judgment
        gpt_grade = get_yes_no_answer(azure_client, eval_question)
        
        # Add evaluation result
        item["gpt4_judgment"] = gpt_grade
        item["is_correct"] = (gpt_grade == "yes")
        evaluated_responses.append(item)
        
        # Progressive save every 10 items
        if len(evaluated_responses) % 10 == 0 and save_json_path:
            with open(save_json_path, 'w') as f:
                json.dump(evaluated_responses, f, indent=2)
    
    # Final save
    if save_json_path:
        with open(save_json_path, 'w') as f:
            json.dump(evaluated_responses, f, indent=2)
        print(f"Evaluation results saved to {save_json_path}")
    
    return evaluated_responses


def calculate_mmvp_accuracy(evaluated_responses):
    """
    Calculate MMVP accuracy based on question pairs.
    
    In MMVP:
    - Questions come in pairs (indexed by question_id)
    - question_id increments by 1 for each question
    - Pairs are formed by consecutive questions: (1,2), (3,4), (5,6), etc.
    - Both questions in a pair must be correct for the pair to be correct
    
    Args:
        evaluated_responses: List of evaluated responses with GPT-4 judgments
    
    Returns:
        Dictionary with accuracy statistics
    """
    # Group responses by pairs
    pairs = {}
    for item in evaluated_responses:
        question_id = item.get("question_id", 0)
        # Determine pair index: (question_id - 1) // 2
        pair_idx = (question_id - 1) // 2
        
        if pair_idx not in pairs:
            pairs[pair_idx] = []
        pairs[pair_idx].append(item)
    
    # Count correct pairs
    num_correct_pairs = 0
    num_total_pairs = len(pairs)
    
    individual_correct = 0
    individual_total = len(evaluated_responses)
    
    pair_details = []
    
    for pair_idx in sorted(pairs.keys()):
        pair_items = pairs[pair_idx]
        
        # Count individual correct answers
        pair_correct_count = sum(1 for item in pair_items if item.get("is_correct", False))
        individual_correct += pair_correct_count
        
        # Check if all items in pair are correct
        all_correct = len(pair_items) == 2 and all(item.get("is_correct", False) for item in pair_items)
        
        if all_correct:
            num_correct_pairs += 1
        
        pair_details.append({
            "pair_idx": pair_idx,
            "question_ids": [item.get("question_id") for item in pair_items],
            "all_correct": all_correct,
            "correct_count": pair_correct_count,
            "total_count": len(pair_items)
        })
    
    # Calculate accuracies
    pair_accuracy = safe_percentage(num_correct_pairs, num_total_pairs, 4)
    individual_accuracy = safe_percentage(individual_correct, individual_total, 4)
    
    return {
        "pair_accuracy": pair_accuracy,
        "individual_accuracy": individual_accuracy,
        "num_correct_pairs": num_correct_pairs,
        "num_total_pairs": num_total_pairs,
        "individual_correct": individual_correct,
        "individual_total": individual_total,
        "pair_details": pair_details
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate MMVP benchmark results using GPT-4')
    parser.add_argument('input_json', type=str, help='Input JSON file name (path to model responses)')
    parser.add_argument('--load_json', action='store_true', 
                       help='Load existing evaluation results if available')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_json):
        print(f"Error: Input file '{args.input_json}' not found!")
        sys.exit(1)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results/MMVP")
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract base name without extension for output files
    base_name = os.path.splitext(os.path.basename(args.input_json))[0]
    save_json_path = os.path.join(results_dir, f"{base_name}_evaluated.json")
    summary_path = os.path.join(results_dir, f"{base_name}_summary.json")
    
    # Initialize Azure client
    print("Initializing Azure OpenAI client...")
    try:
        azure_client = AzureOpenAIClient()
        print("Successfully initialized Azure OpenAI client")
    except Exception as e:
        print(f"Error initializing Azure client: {e}")
        sys.exit(1)
    
    # Evaluate responses
    print(f"\nEvaluating responses from: {args.input_json}")
    evaluated_responses = evaluate_mmvp_responses(
        input_file=args.input_json,
        azure_client=azure_client,
        load_json=args.load_json,
        save_json_path=save_json_path
    )
    
    # Calculate accuracy
    print("\n##### Calculating MMVP Accuracy #####")
    stats = calculate_mmvp_accuracy(evaluated_responses)
    
    # Display results
    print("\n##### MMVP Evaluation Results #####")
    print(f"Total question pairs: {stats['num_total_pairs']}")
    print(f"Correct pairs (both questions correct): {stats['num_correct_pairs']}")
    print(f"Pair accuracy: {stats['pair_accuracy']:.2f}%")
    print()
    print(f"Total individual questions: {stats['individual_total']}")
    print(f"Correct individual questions: {stats['individual_correct']}")
    print(f"Individual question accuracy: {stats['individual_accuracy']:.2f}%")
    
    # Create summary table
    table = [
        ["Metric", "Value"],
        ["Pair Accuracy (Official Metric)", f"{stats['pair_accuracy']:.2f}%"],
        ["Individual Question Accuracy", f"{stats['individual_accuracy']:.2f}%"],
        ["Correct Pairs", f"{stats['num_correct_pairs']}/{stats['num_total_pairs']}"],
        ["Correct Questions", f"{stats['individual_correct']}/{stats['individual_total']}"]
    ]
    
    result_table = PrettyTable(table[0])
    result_table.add_rows(table[1:])
    print("\n##### Summary #####")
    print(result_table)
    
    # Save summary results
    summary_results = {
        "input_file": args.input_json,
        "evaluation_method": "GPT-4 via Azure",
        "official_metric": {
            "pair_accuracy": stats["pair_accuracy"],
            "correct_pairs": stats["num_correct_pairs"],
            "total_pairs": stats["num_total_pairs"]
        },
        "additional_metrics": {
            "individual_accuracy": stats["individual_accuracy"],
            "correct_questions": stats["individual_correct"],
            "total_questions": stats["individual_total"]
        },
        "pair_details": stats["pair_details"]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"\n##### Results saved to: #####")
    print(f"Evaluated responses: {save_json_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
