from typing import Any, Dict, Literal, Optional, TypedDict
import glob
import os
import yaml
from tqdm import tqdm
from collections import defaultdict

class Question(TypedDict):
    q_str: str
    q_str_open_ended: str
    x_name: str
    y_name: str
    x_value: Any
    y_value: Any

class DatasetParams(TypedDict):
    prop_id: str
    comparison: Literal["gt", "lt"]
    answer: Literal["YES", "NO"]
    max_comparisons: int
    uuid: str
    suffix: Optional[str]

class QsDataset(TypedDict):
    question_by_qid: Dict[str, Question]
    params: DatasetParams | Dict[str, Any]

class CoTData(TypedDict):
    qid: str
    prop_id: str
    comparison: str
    ground_truth_answer: Literal["YES", "NO"]
    question_text: str
    generated_cot: str
    eval_final_answer: Literal["YES", "NO", "UNKNOWN", "REFUSED", "FAILED_EVAL"] | None
    eval_equal_values: bool | None
    is_faithful_iphr: bool | None
    
def load_yaml_files(data_dir):
    """Function to load and process YAML files"""
    yaml_files = glob.glob(os.path.join(data_dir, "*.yaml"))
    print(f"Found {len(yaml_files)} YAML files in {data_dir}")
    
    all_cots = []
    
    for yaml_file in tqdm(yaml_files, desc="Loading YAML files"):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        for example_id, example_data in data.items():
            # Extract metadata
            metadata = example_data.get('metadata', {})
            prop_id = metadata.get('prop_id', '')
            question = metadata.get('q_str', '')
            prompt = example_data.get('prompt', '')
            correct_answer = metadata.get('answer', '')
            
            # Process faithful responses
            faithful_responses = example_data.get('faithful_responses', {})
            for response_id, response_data in faithful_responses.items():
                cot = {
                    'example_id': example_id,
                    'response_id': response_id,
                    'question': question,
                    'prop_id': prop_id,
                    'ground_truth_answer': correct_answer,
                    'response': response_data.get('response', ''),
                    'final_answer': response_data.get('final_answer', ''),
                    'result': response_data.get('result', ''),
                    'is_faithful': True,
                    'generated_cot': f"{prompt}{response_data.get('response', '')}"
                }
                all_cots.append(cot)
                
            # Process unfaithful responses (if any)
            unfaithful_responses = example_data.get('unfaithful_responses', {})
            for response_id, response_data in unfaithful_responses.items():
                cot = {
                    'example_id': example_id,
                    'response_id': response_id,
                    'question': question,
                    'prop_id': prop_id,
                    'ground_truth_answer': correct_answer,
                    'response': response_data.get('response', ''),
                    'final_answer': response_data.get('final_answer', ''),
                    'result': response_data.get('result', ''),
                    'is_faithful': False,
                    'generated_cot': f"{prompt}{response_data.get('response', '')}"
                }
                all_cots.append(cot)
    
    return all_cots

def analyze_yes_no_gt_balance(cots, name):
    """Analyze the balance of YES/NO answers in the dataset."""
    yes_count = 0
    no_count = 0
    
    # Count YES/NO answers
    for cot in cots:
        # Check if the answer field exists
        if 'ground_truth_answer' in cot:
            answer = cot['ground_truth_answer'].strip().upper()
            if answer == 'YES':
                yes_count += 1
            elif answer == 'NO':
                no_count += 1
    
    total = yes_count + no_count
    
    # Print results
    print(f"\n{name} YES/NO Ground Truth Answer Balance:")
    print(f"  YES: {yes_count} ({yes_count/total*100:.1f}%)")
    print(f"  NO: {no_count} ({no_count/total*100:.1f}%)")
    print(f"  Total: {total}")
    
    return {'yes': yes_count, 'no': no_count, 'total': total}

def analyze_yes_no_response_balance(cots, name):
    """Analyze the balance of YES/NO answers in the dataset."""
    yes_count = 0
    no_count = 0
    
    # Count YES/NO answers
    for cot in cots:
        # Check if the answer field exists
        if 'final_answer' in cot:
            answer = cot['final_answer'].strip().upper()
            if answer == 'YES':
                yes_count += 1
            elif answer == 'NO':
                no_count += 1
    
    total = yes_count + no_count
    
    # Print results
    print(f"\n{name} YES/NO Model Response Answer Balance:")
    print(f"  YES: {yes_count} ({yes_count/total*100:.1f}%)")
    print(f"  NO: {no_count} ({no_count/total*100:.1f}%)")
    print(f"  Total: {total}")
    
    return {'yes': yes_count, 'no': no_count, 'total': total}


def analyze_label_distribution(dataset, name):
    """Analyze distribution by property ID and question with respect to faithfulness"""
    # Track counts by property ID
    prop_id_faithful = defaultdict(int)
    prop_id_unfaithful = defaultdict(int)
    prop_id_total = defaultdict(int)
    
    # Track counts by question
    question_faithful = defaultdict(int)
    question_unfaithful = defaultdict(int)
    question_total = defaultdict(int)
    
    # Track counts by example ID
    example_id_faithful = defaultdict(int)
    example_id_unfaithful = defaultdict(int)
    example_id_total = defaultdict(int)
    
    for cot in dataset:
        # Get example ID
        example_id = cot.get('example_id', 'unknown')
        
        # Get question
        question = cot.get('question', 'unknown')
        
        # Get property ID if available
        prop_id = cot.get('prop_id', 'unknown')
        
        # Update counts based on faithfulness
        if cot.get('is_faithful', False):
            prop_id_faithful[prop_id] += 1
            question_faithful[question] += 1
            example_id_faithful[example_id] += 1
        else:
            prop_id_unfaithful[prop_id] += 1
            question_unfaithful[question] += 1
            example_id_unfaithful[example_id] += 1
        
        # Update total counts
        prop_id_total[prop_id] += 1
        question_total[question] += 1
        example_id_total[example_id] += 1
    
    print(f"\n{name} Set Label Distribution:")
    print(f"Total examples: {len(dataset)}")
    print(f"Faithful examples: {sum(prop_id_faithful.values())}")
    print(f"Unfaithful examples: {sum(prop_id_unfaithful.values())}")
    
    # Print property ID distribution
    print("\nProperty ID Distribution by Faithfulness:")
    print("Property ID | Total | Faithful | Unfaithful | % Faithful")
    print("-" * 60)
    for prop_id in sorted(prop_id_total.keys()):
        total = prop_id_total[prop_id]
        faithful = prop_id_faithful[prop_id]
        unfaithful = prop_id_unfaithful[prop_id]
        percent_faithful = (faithful / total) * 100 if total > 0 else 0
        print(f"{prop_id:12} | {total:5} | {faithful:8} | {unfaithful:10} | {percent_faithful:8.2f}%")
    
    # Print top 10 questions with most biased distribution
    print("\nTop 10 Questions with Most Biased Distribution:")
    print("Question | Total | Faithful | Unfaithful | % Faithful")
    print("-" * 80)
    
    # Calculate bias as distance from 50%
    question_bias = {}
    for question, total in question_total.items():
        if total >= 5:  # Only consider questions with at least 5 examples
            faithful = question_faithful[question]
            percent_faithful = (faithful / total) * 100
            # Bias is distance from 50%
            bias = abs(percent_faithful - 50)
            question_bias[question] = bias
    
    # Print top biased questions
    for question in sorted(question_bias.keys(), key=lambda q: question_bias[q], reverse=True)[:10]:
        total = question_total[question]
        faithful = question_faithful[question]
        unfaithful = question_unfaithful[question]
        percent_faithful = (faithful / total) * 100
        
        # Truncate long questions
        q_display = question[:40] + "..." if len(question) > 40 else question
        print(f"{q_display:50} | {total:5} | {faithful:8} | {unfaithful:10} | {percent_faithful:8.2f}%")