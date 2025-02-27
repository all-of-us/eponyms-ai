import json
import pandas as pd
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Optional
from pathlib import Path

from resources.constants import data_path


def extract_final_answer(response: Optional[str]) -> Optional[str]:
    """
    Extract Yes/No from the final answer section of the response.
    Returns None if the response is empty or invalid.
    """
    if not response:
        return None

    # Extract text between <final_answer> tags
    match = re.search(r'<final_answer>(.*)', response, re.DOTALL)
    if not match:
        return None

    final_answer = match.group(1).strip()

    # Check if the answer contains Yes or No
    if "Yes" in final_answer:
        return "Yes"
    elif "No" in final_answer:
        return "No"
    else:
        return None


def calculate_completeness_score(entry: Dict) -> int:
    """
    Calculate a completeness score for an entry based on filled fields.
    Higher score means more complete data.
    """
    score = 0

    # Check AI responses
    if entry['openai_response']:
        score += 1
    if entry['claude_response']:
        score += 1
    if entry['gemini_response']:
        score += 1

    # Check extracted answers
    if extract_final_answer(entry['openai_response']):
        score += 1
    if extract_final_answer(entry['claude_response']):
        score += 1
    if extract_final_answer(entry['gemini_response']):
        score += 1

    return score


def process_jsonl_files(input_dir, output_path, batch_type=None, file_prefix="minimal"):
    """
    Process all JSONL files matching the pattern and extract relevant information.
    Handles duplicates by keeping the most complete version of each entry.

    Args:
        input_dir: Directory containing the JSONL files
        output_path: Path to save the output CSV
        batch_type: Type of batch processing ('openai', 'claude', or None for regular)
        file_prefix: Prefix for input files to process
    """
    input_path = Path(input_dir)

    # Use dictionary to track entries by disease name
    entries_by_disease = defaultdict(list)

    # First pass: collect all entries grouped by disease
    for filename in input_path.glob(f"{file_prefix}*"):
        with open(filename, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                if batch_type == 'openai':
                    custom_id = entry['custom_id']
                    content = entry['response']['body']['choices'][0]['message']['content']
                    entries_by_disease[custom_id].append(content)
                elif batch_type == 'claude':
                    custom_id = entry['custom_id']
                    if 'result' in entry:
                        content = entry['result']['message']['content'][0]['text']
                    else:
                        print(f"Unexpected entry format: {line}")
                    entries_by_disease[custom_id].append(content)
                else:
                    disease = entry['disease']
                    entries_by_disease[disease].append(entry)

    # Process entries, handling duplicates
    data = []

    for disease, entries in entries_by_disease.items():
        # If multiple entries exist for this disease, select the most complete one
        if len(entries) > 1:
            # Sort entries by completeness score and take the most complete one
            most_complete_entry = max(entries, key=calculate_completeness_score)
        else:
            most_complete_entry = entries[0]

        # Process Presidio response
        presidio_entities = json.loads(most_complete_entry['presidio_response'])
        presidio_names = []
        presidio_scores = []
        if presidio_entities:
            presidio_names = [entity['name'].strip() for entity in presidio_entities]
            presidio_scores = [str(entity['score']) for entity in presidio_entities]

        # Extract answers from the most complete entry
        data.append({
            'disease': disease,
            'openai_answer': extract_final_answer(most_complete_entry['openai_response']),
            'claude_answer': extract_final_answer(most_complete_entry['claude_response']),
            'gemini_answer': extract_final_answer(most_complete_entry['gemini_response']),
            'presidio_entities': ';'.join(presidio_names),
            'presidio_scores': ';'.join(presidio_scores)
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save the main results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description='Process AI model responses from JSONL files.')

    parser.add_argument('-i','--input', help='Directory containing the input JSONL files')
    parser.add_argument('-o','--output', help='Path to save the output CSV file')
    parser.add_argument('-b','--batch-type', choices=['openai', 'claude', 'none'], help='Type of batch processing (openai, claude, or none)')
    parser.add_argument('-p','--prefix', help='Prefix for input files to process')
    parser.add_argument('-s','--stats', action='store_true', help='Print summary statistics after processing')

    args = parser.parse_args()

    # Convert 'none' to None for batch_type
    batch_type = None if args.batch_type == 'none' else args.batch_type
    # Process files
    df = process_jsonl_files(
        input_dir=data_path / 'eponyms_input' / args.input,
        output_path=data_path / 'eponyms_output' / args.output,
        batch_type=batch_type,
        file_prefix=args.prefix
    )

    # Print summary statistics if requested
    if args.stats:
        print("\nSummary Statistics:")
        print(f"Total processed entries: {len(df)}")
        print("\nResponse distributions:")
        for model in ['openai', 'claude', 'gemini']:
            print(f"\n{model.capitalize()} responses:")
            print(df[f'{model}_answer'].value_counts(dropna=False))

        # Print duplicate handling statistics
        print("\nDuplicate handling statistics:")
        initial_diseases = df['disease'].value_counts()
        print(f"Number of unique diseases: {len(initial_diseases)}")
        print(f"Number of diseases with multiple entries: {sum(initial_diseases > 1)}")


if __name__ == "__main__":
    main()