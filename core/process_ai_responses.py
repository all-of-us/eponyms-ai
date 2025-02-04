import json
import pandas as pd
import re
from collections import defaultdict
from typing import Dict, List, Optional

from resources.constants import data_path


def extract_final_answer(response: Optional[str]) -> Optional[str]:
    """
    Extract Yes/No from the final answer section of the response.
    Returns None if the response is empty or invalid.
    """
    if not response:
        return None

    # Extract text between <final_answer> tags
    match = re.search(r'<final_answer>(.*?)</final_answer>', response, re.DOTALL)
    if not match:
        return None

    final_answer = match.group(1).strip()

    # Check if the answer contains Yes or No
    if "Yes." in final_answer:
        return "Yes"
    elif "No." in final_answer:
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

    # Check Presidio response
    if entry['presidio_response']:
        presidio_entities = json.loads(entry['presidio_response'])
        if presidio_entities:
            score += 1

    return score


def process_jsonl_files(file_pattern):
    """
    Process all JSONL files matching the pattern and extract relevant information.
    Handles duplicates by keeping the most complete version of each entry.
    """
    # Use dictionary to track entries by disease name
    entries_by_disease = defaultdict(list)
    missing_gemini = []

    # First pass: collect all entries grouped by disease
    for filename in file_pattern.glob("minimal_desc_out*"):
        with open(filename, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
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

        # Check for missing Gemini response in the selected entry
        if not most_complete_entry['gemini_response']:
            missing_gemini.append(disease)

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
    df.to_csv(data_path / 'eponyms_output' / 'processed_responses_1.csv', index=False)

    # Save missing Gemini responses
    if missing_gemini:
        with open(data_path / 'eponyms_output' / 'missing_gemini_responses.txt', 'w') as f:
            for disease in missing_gemini:
                f.write(f"{disease}\n")
        print(f"Found {len(missing_gemini)} missing Gemini responses")

    return df


if __name__ == "__main__":
    # Process all matching files in the data directory
    file_pattern = data_path / 'eponyms_output'
    df = process_jsonl_files(file_pattern)

    # Print summary statistics
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
