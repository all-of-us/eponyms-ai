# coding=utf-8

import csv
from pathlib import Path

import pandas as pd
from openai import OpenAI
import google.generativeai as genai

from resources.constants import data_path, ANNOTATED_COLS


def openai_api_key(path):
    """
    Get the OpenAI API key from the given path
    :param path:
    :return:
    """
    key_path = Path(path).expanduser()
    return key_path.read_text().strip()


def openai_client():
    """
    Create an OpenAI client
    :return:
    """
    return OpenAI(api_key=openai_api_key())


def claude_client():
    """
    Create an OpenAI client
    :return:
    """
    return OpenAI(api_key=openai_api_key())


def read_guideline(guideline_path):
    """
    Read the guideline from the given path
    :param guideline_path:
    :return:
    """
    return guideline_path.read_text()


def main(input_path, output_path):
    """
    Main function
    :param input_path:
    :param output_path:
    :return:
    """
    oai_client = openai_client("path/to/oai_key")
    genai.configure(api_key="path/to/gemini_key")
    gemini_model = genai.GenerativeModel("gemini-1.5-flash",
                                         system_instruction="You are a helpful clinical assistant.")

    eponyms_apos_df = pd.read_csv(eponymous_apos_path,
                                  delimiter=',',
                                  header=0,
                                  names=ANNOTATED_COLS,
                                  dtype=str)

    with output_path.open('w+') as f:
        w = csv.DictWriter(f, ["disease", "response"])
        w.writeheader()
        for row in eponyms_apos_df.itertuples():

            prompt = (
                f"Is any word in '{row.concept_name}' an eponym? "
                f"Respond with Yes or No and if Yes, provide the name(s) of the person/people. "
                f"Be STRICT and only say Yes if the eponym is literally named after a person, and "
                f"not initials, derivations, locations, associations, proteins, mythology. "
                f"Keep your response short, to a maximum of 2 sentences if Yes and just 'No.' if No. "
                f"Also have the final answer Yes or No at the end of the response. "
            )

            response = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful clinical assistant."},
                    {"role": "user",
                     "content": prompt}
                ]
            )

            message = response.choices[0].message.content

            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.1,
                )
            )

            output_dict = {
                "disease": row.concept_name,
                "oai_response": message.replace('\n', ' '),
                "gemini_response": response
            }

            w.writerow(output_dict)


if __name__ == "__main__":
    eponymous_apos_path = data_path / "eponym" / "eponymous_diseases_apostrophe.csv"
    output_path = data_path / "open_ai_annotations_apos.csv"
    main(eponymous_apos_path, output_path)
