import csv
import json
import pandas as pd
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Optional
from pathlib import Path



def extract_final_answer(response: Optional[str]) -> Optional[str]:
    """
    Extract Yes/No from the final answer section of the response.
    Returns None if the response is empty or invalid.
    """
    if not response:
        return None

    # Exclude thinking tokens
    response = re.sub(r'<think>(.*)</think>', '', response, re.DOTALL)

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


def extract_eponym(text):
    """
    Extract the eponym (if any) from a numbered list analysis of medical terms.

    Args:
        text (str): Text containing the numbered list analysis.

    Returns:
        str or None: The identified eponym, or None if no eponym is found.
    """
    # Check if there's a conclusion statement about no eponyms
    if not text or "no eponyms" in text.lower():
        return None
    
    # Exclude thinking tokens
    text = re.sub(r'<think>(.*)</think>', '', text, re.DOTALL)

    eponyms = []

    # Regular expression to find numbered items
    pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|\n\nUpon|\Z)'

    # Find all items in the numbered list
    items = re.findall(pattern, text, re.DOTALL)

    # Process each item to look for eponym mentions
    for item in items:
        # Look for phrases that indicate an eponym
        if "derived from" in item.lower() and "person's name" in item.lower() and "not" not in item.lower():
            # Extract the term being described (usually at the beginning)
            term_match = re.match(r'([^:-]+)[:-]', item)
            if term_match:
                eponyms.append(term_match.group(1).strip())

        # Alternative pattern: directly states it's an eponym
        if "eponym" in item.lower() and "not" not in item.lower():
            term_match = re.match(r'([^:-]+)[:-]', item)
            if term_match:
                eponyms.append(term_match.group(1).strip())

    # Check if there's a positive identification elsewhere in the text
    positive_match = re.search(r'(\w+) is (\w+) eponym', text)
    if positive_match:
        eponyms.append(positive_match.group(1).strip())

    return "|".join(eponyms).replace("\n", " ")


def calculate_completeness_score(entry: Dict) -> int:
    """
    Calculate a completeness score for an entry based on filled fields.
    Higher score means more complete data.
    """
    score = 0

    # Check AI responses
    if entry.get('response'):
        score += 1
    if entry.get('openai_response'):
        score += 1
    if entry.get('claude_response'):
        score += 1
    if entry.get('gemini_response'):
        score += 1

    # Check extracted answers
    if extract_final_answer(entry.get('response')):
        score += 1
    if extract_final_answer(entry.get('openai_response')):
        score += 1
    if extract_final_answer(entry.get('claude_response')):
        score += 1
    if extract_final_answer(entry.get('gemini_response')):
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
    custom_ids = {}
    print(input_path)
    for filename in sorted(input_path.glob(f"{file_prefix}*.jsonl"), key=lambda x: x.name):
        print(f"Processing {filename}")
        i = 0
        with open(filename, 'r') as f:
            for line in f:
                i += 1
                entry = json.loads(line.strip())
                if batch_type == 'openai':
                    custom_id = entry['custom_id']
                    content = entry['response']['body']['choices'][0]['message']['content']
                    entries_by_disease[custom_id].append({"disease": custom_id, "openai_response": content})
                elif batch_type == 'claude':
                    custom_id = entry['custom_id']
                    if 'result' in entry:
                        content = entry['result']['message']['content'][0]['text']
                    else:
                        print(f"Unexpected entry format: {line}")
                    entries_by_disease[custom_id].append({"disease": custom_id, "claude_response": content})
                elif batch_type == 'gemini':
                    custom_id = entry['request']['custom_id']
                    if 'result' in entry:
                        content = entry['result']['message']['content'][0]['text']
                    else:
                        print(f"Unexpected entry format: {line}")
                    entries_by_disease[custom_id].append({"disease": custom_id, "claude_response": content})
                else:
                    disease = entry['disease']
                    custom_id = disease
                    # if filename.name == "desc_out_2.jsonl":
                    #     custom_id = f"{str(i)}{re.sub(r'[^a-zA-Z0-9_-]', '', disease.strip())}"[:63]
                    #     custom_ids[disease] = custom_id
                    # else:
                    #     custom_id = custom_ids[disease]
                    entries_by_disease[custom_id].append(entry)

    # Process entries, handling duplicates
    data = []

    for disease, entries in entries_by_disease.items():
        # If multiple entries exist for this disease, select the most complete one
        if len(entries) > 1:
            # Sort entries by completeness score and take the most complete one
            most_complete_entry = max(entries, key=calculate_completeness_score)
        else:
            most_complete_entry = entries[0]

        presidio_names = []
        presidio_scores = []
        # Process Presidio response
        if batch_type == 'none':
            presidio_entities = json.loads(most_complete_entry['presidio_response'])
            if presidio_entities:
                presidio_names = [entity['name'].strip() for entity in presidio_entities]
                presidio_scores = [str(entity['score']) for entity in presidio_entities]

        # Extract answers from the most complete entry
        data.append({
            'disease': disease,
            'q30b8_answer': extract_final_answer(most_complete_entry.get('response',"")),
            'q30b8_eponym': extract_eponym(most_complete_entry.get('response',"")),
            'openai_answer': extract_final_answer(most_complete_entry.get('openai_response',"")),
            'openai_eponym': extract_eponym(most_complete_entry.get('openai_response',"")),
            'claude_answer': extract_final_answer(most_complete_entry.get('claude_response',"")),
            'claude_eponym': extract_eponym(most_complete_entry.get('claude_response',"")),
            'gemini_answer': extract_final_answer(most_complete_entry.get('gemini_response',"")),
            'gemini_eponym': extract_eponym(most_complete_entry.get('gemini_response',"")),
            'presidio_entities': ';'.join(presidio_names),
            'presidio_scores': ';'.join(presidio_scores)
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save the main results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, quoting=csv.QUOTE_MINIMAL, sep=',', index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description='Process AI model responses from JSONL files.')

    parser.add_argument('-i','--input', help='Directory containing the input JSONL files')
    parser.add_argument('-o','--output', help='Path to save the output CSV file')
    parser.add_argument('-b','--batch-type', choices=['openai', 'claude', 'gemini', 'none'], help='Type of batch processing (openai, claude, gemini, or none)')
    parser.add_argument('-p','--prefix', help='Prefix for input files to process')
    parser.add_argument('-s','--stats', action='store_true', help='Print summary statistics after processing')

    args = parser.parse_args()

    # Convert 'none' to None for batch_type
    batch_type = None if args.batch_type == 'none' else args.batch_type
    # Process files
    df = process_jsonl_files(
        input_dir=args.input,
        output_path=args.output,
        batch_type=batch_type,
        file_prefix=args.prefix
    )

    # Print summary statistics if requested
    if args.stats:
        print("\nSummary Statistics:")
        print(f"Total processed entries: {len(df)}")
        print("\nResponse distributions:")
        for model in ['q30b8', 'openai', 'claude', 'gemini']:
            print(f"\n{model.capitalize()} responses:")
            print(df[f'{model}_answer'].value_counts(dropna=False))

        # Print duplicate handling statistics
        print("\nDuplicate handling statistics:")
        initial_diseases = df['disease'].value_counts()
        print(f"Number of unique diseases: {len(initial_diseases)}")
        print(f"Number of diseases with multiple entries: {sum(initial_diseases > 1)}")


if __name__ == "__main__":
    main()