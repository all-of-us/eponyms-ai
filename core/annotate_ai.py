# coding=utf-8

import json
import logging
from pathlib import Path
import argparse

import pandas as pd
from google.genai import types
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
# import google.generativeai as genai
from google import genai
from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerationConfig
from vertexai.preview.generative_models import GenerativeModel
from presidio_analyzer import AnalyzerEngine

from resources.constants import data_path, ANNOTATED_COLS, DEFAULT_COLS


def file_contents(path):
    """
    Get the OpenAI API key from the given path
    :param path:
    :return:
    """
    key_path = Path(path).expanduser()
    return key_path.read_text().strip()


def read_guideline(guideline_path):
    """
    Read the guideline from the given path
    :param guideline_path:
    :return:
    """
    return guideline_path.read_text()


def main(input_path, output_path, endpoint_path, azure_api_key_path, openai_key_path, gemini_sa_path, gemini_key_path, anthropic_key_path):
    """
    Main function
    :param input_path:
    :param output_path:
    :param endpoint_path:
    :param azure_api_key_path:
    :param openai_key_path:
    :param gemini_key_path:
    :return:
    """
    # oai_client = AzureOpenAI(
    #     azure_endpoint=file_contents(endpoint_path),
    #     api_key=file_contents(azure_api_key_path),
    #     api_version="2024-10-21"
    # )

    guideline = read_guideline(data_path / "identify_eponym_prompt.md")
    system_prompt, user_prompt = guideline.split('-----')

    oai_client = OpenAI(api_key=file_contents(openai_key_path))

    analyzer = AnalyzerEngine()

    PROJECT_ID = ""
    REGION = ""
    # vertexai.init(project=PROJECT_ID, location=REGION, credentials=vertexai.Credentials.from_service_account_file(gemini_key_path))
    # aiplatform.init(project=PROJECT_ID, location=REGION, credentials=service_account.Credentials.from_service_account_file(gemini_key_path))

    credentials = service_account.Credentials.from_service_account_file(gemini_sa_path)
    gemini_client = genai.Client(api_key=file_contents(gemini_key_path), http_options={'api_version':'v1alpha'})
    # scoped_credentials = credentials.with_scopes(
    #     ['https://www.googleapis.com/auth/cloud-platform'])

    claude_client = Anthropic(api_key=file_contents(anthropic_key_path))

    # gemini_model = GenerativeModel("gemini-2.0-flash-thinking-exp", system_instruction=system_prompt)

    # gemini_model = gemini_client.models.("gemini-2.0-flash", system_instruction=system_prompt)

    eponyms_apos_df = pd.read_csv(input_path,
                                  delimiter=',',
                                  header=0,
                                  names=DEFAULT_COLS,
                                  dtype=str)

    with output_path.open('a+') as f:
        # w = csv.DictWriter(f, ["disease", "openai_response", "gemini_response", "claude_response"])
        # w.writeheader()
        for row in eponyms_apos_df[5600:].itertuples():

            openai_response = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(concept_name=row.concept_name)}
                ]
            )

            logging.log(logging.INFO, f"Fetching response from OpenAI for {row.concept_name}")
            openai_message = openai_response.choices[0].message.content
            # openai_message = ''

            try:
                logging.log(logging.INFO, f"Fetching response from Gemini for {row.concept_name}")
                # gemini_response = gemini_model.generate_content(
                #     user_prompt.format(concept_name=row.concept_name),
                #     generation_config=GenerationConfig(
                #         max_output_tokens=100,
                #         temperature=0.1,
                #     )
                # )
                gemini_response = gemini_client.models.generate_content(
                    model='gemini-2.0-flash-thinking-exp',
                    contents=user_prompt.format(concept_name=row.concept_name),
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(include_thoughts=False),
                        system_instruction=system_prompt,
                        temperature=0.3,
                    ),
                ).text
                # gemini_response = ''
            except Exception as e:
                logging.log(logging.INFO, f"500 Error while fetching response from Gemini for {row.concept_name}")
                gemini_response = ''

            logging.log(logging.INFO, f"Fetching response from Claude for {row.concept_name}")
            claude_message = claude_client.messages.create(
                max_tokens=512,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt.format(concept_name=row.concept_name)
                    }
                ],
                model="claude-3-5-haiku-latest",
            )
            # claude_message = ''

            presidio_results = analyzer.analyze(text=row.concept_name, entities=["PERSON"], language='en')

            output_dict = {
                "disease": row.concept_name,
                "openai_response": openai_message,
                "gemini_response": gemini_response,
                "claude_response": json.loads(claude_message.model_dump_json())['content'][0]['text'],
                "presidio_response": f'[{",".join([json.dumps({"name": row.concept_name[r.start:r.end+1],"score": r.score}) for r in presidio_results if r.entity_type == "PERSON"])}]'
            }

            f.write(json.dumps(output_dict) + '\n')


# Accept arguments from the command line
def get_parser():
    """
    Get the parser
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=Path, help="The name of the input file")
    parser.add_argument("-o", "--output_file", type=Path, help="The name of the output file")
    parser.add_argument("-e", "--azure_endpoint_path", type=Path, help="The path to the azure endpoint file")
    parser.add_argument("-k", "--azure_key_path", type=Path, help="The path to the azure key file")
    parser.add_argument("-a", "--openai_key_path", type=Path, help="The path to the openai key file")
    parser.add_argument("-g", "--gemini_sa_path", type=Path, help="The path to the gemini SA json file")
    parser.add_argument("-m", "--gemini_key_path", type=Path, help="The path to the gemini key file")
    parser.add_argument("-c", "--anthropic_key_path", type=Path, help="The path to the anthropic key file")

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = get_parser()

    args = parser.parse_args()

    input_path = data_path / "eponyms_input" / args.input_file
    output_path = data_path / "eponyms_output" / args.output_file
    main(input_path, output_path, args.azure_endpoint_path, args.azure_key_path, args.openai_key_path,
         args.gemini_sa_path, args.gemini_key_path, args.anthropic_key_path)
