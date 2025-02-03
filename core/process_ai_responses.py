import json
import pandas as pd
import re

from resources.constants import data_path


def extract_final_answer(response):
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


def process_jsonl_files(file_pattern):
    """
    Process all JSONL files matching the pattern and extract relevant information.
    """
    data = []
    missing_gemini = []

    # Process each file matching the pattern
    for filename in file_pattern.glob("minimal_desc_out*"):
        with open(filename, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())

                # Extract basic information
                disease = entry['disease']

                # Process AI responses
                openai_answer = extract_final_answer(entry['openai_response'])
                claude_answer = extract_final_answer(entry['claude_response'])
                gemini_answer = extract_final_answer(entry['gemini_response'])

                # Check for missing Gemini responses
                if not entry['gemini_response']:
                    missing_gemini.append(disease)

                # Process Presidio response
                presidio_entities = json.loads(entry['presidio_response'])
                presidio_names = []
                presidio_scores = []
                if presidio_entities:
                    presidio_names = [entity['name'].strip() for entity in presidio_entities]
                    presidio_scores = [str(entity['score']) for entity in presidio_entities]

                data.append({
                    'disease': disease,
                    'openai_answer': openai_answer,
                    'claude_answer': claude_answer,
                    'gemini_answer': gemini_answer,
                    'presidio_entities': ';'.join(presidio_names),
                    'presidio_scores': ';'.join(presidio_scores)
                })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save the main results
    df.to_csv(data_path / 'eponyms_output' / 'processed_responses.csv', index=False)

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
