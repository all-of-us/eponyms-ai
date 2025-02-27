# coding=utf-8

import json
import logging
import re
import time
from pathlib import Path
import argparse

import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from google.genai import types
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from google.cloud import aiplatform_v1beta1
from google.protobuf import json_format

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


def run_openai_batch_request(oai_client, df, system_prompt, user_prompt, concept_prompt):
    with open(data_path / "batch" / "openai_batch_input.jsonl", "w+") as f:
        for row in df.itertuples():
            custom_string = re.sub(r'[^a-zA-Z0-9_-]', '', row.tokenized_concept_name.strip())
            request = {"custom_id": f"{str(row.Index)}{custom_string}"[:63],
                       "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini",
                         "messages": [{"role": "system",
                                       "content": system_prompt},
                                      {"role": "user",
                                       "content": user_prompt},
                                      {"role": "user",
                                       "content": concept_prompt.format(concept_name=row.concept_name)}],
                         "max_tokens": 1000}
                }
            f.write(json.dumps(request) + '\n')
    batch_input_file = oai_client.files.create(
        file=open(data_path / "batch" / "openai_batch_input.jsonl", "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    oai_client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )


def run_gemini_batch_request(gemini_client, df, bucket_id, system_prompt, user_prompt, concept_prompt):
    with open(data_path / "batch" / "gemini_batch_input.jsonl", "w+") as f:
        for row in df.itertuples():
            request = {"request": {"contents":
              [{"role": "system", "parts": [{"text": system_prompt},{"text": user_prompt}]},
               {"role": "user", "parts": [{"text": concept_prompt.format(concept_name=row.concept_name)}]}
              ]}
            }
            f.write(json.dumps(request) + '\n')

    storage_uri = f"gs://{bucket_id}/input/gemini_batch_input.jsonl"
    output_uri = f"gs://{bucket_id}/output/gemini_batch_output.jsonl"

    job = gemini_client.batches.create(
        model="gemini-2.0-flash-001",
        src=storage_uri,
        config=CreateBatchJobConfig(dest=output_uri),
    )


def run_claude_batch_request(claude_client, df, system_prompt, user_prompt, concept_prompt):
    requests = []
    for row in df.itertuples():
        requests.append(
            Request(
                custom_id=f"{str(row.Index)}{re.sub(r'[^a-zA-Z0-9_-]', '', row.tokenized_concept_name.strip())}"[:63],
                params=MessageCreateParamsNonStreaming(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=512,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    messages=[{
                        "role": "user",
                        "content": concept_prompt.format(concept_name=row.concept_name)
                    }]
                )
            )
        )

    message_batch = claude_client.messages.batches.create(
        requests=requests
    )


def main(input_path, output_path, endpoint_path, azure_api_key_path, openai_key_path,
         gemini_sa_path, gemini_key_path, anthropic_key_path, bucket_id):
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
    system_prompt, concept_prompt, user_prompt = guideline.split('-----')

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
                                  names=ANNOTATED_COLS,
                                  dtype=str)

    run_openai_batch_request(oai_client, eponyms_apos_df, system_prompt, user_prompt, concept_prompt)
    run_gemini_batch_request(gemini_client, eponyms_apos_df, bucket_id, system_prompt, user_prompt, concept_prompt)
    run_claude_batch_request(claude_client, eponyms_apos_df, system_prompt, user_prompt, concept_prompt)

    with output_path.open('a+') as f:
        # w = csv.DictWriter(f, ["disease", "openai_response", "gemini_response", "claude_response"])
        # w.writeheader()
        for row in eponyms_apos_df.itertuples():
            # time.sleep(0.2)

            openai_response = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "user", "content": concept_prompt.format(concept_name=row.concept_name)},
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
                    contents=concept_prompt.format(concept_name=row.concept_name) + "\n" + user_prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(include_thoughts=False),
                        system_instruction=system_prompt,
                        temperature=0.3,
                    ),
                ).text
                # gemini_response = ''
            except Exception as e:
                logging.log(logging.INFO, e)
                # time.sleep(1)
                gemini_response = ''

            logging.log(logging.INFO, f"Fetching response from Claude for {row.concept_name}")
            claude_message = claude_client.messages.create(
                max_tokens=512,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": concept_prompt.format(concept_name=row.concept_name)},
                    {"role": "user", "content": user_prompt}
                ],
                model="claude-3-5-haiku-latest",
            )
            claude_message = json.loads(claude_message.model_dump_json())['content'][0]['text']
            # claude_message = ''

            presidio_results = analyzer.analyze(text=row.concept_name, entities=["PERSON"], language='en')

            presidio_names_scores = [json.dumps({"name": row.concept_name[r.start:r.end+1],"score": r.score})
                                     for r in presidio_results if r.entity_type == "PERSON"]

            presidio_response = f'[{",".join(presidio_names_scores)}]'

            output_dict = {
                "disease": row.concept_name,
                "openai_response": openai_message,
                "gemini_response": gemini_response,
                "claude_response": claude_message,
                "presidio_response": presidio_response
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
    parser.add_argument("-b", "--bucket_id", type=Path, help="The google bucket ID")

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = get_parser()

    args = parser.parse_args()

    input_path = data_path / "eponyms_input" / args.input_file
    output_path = data_path / "eponyms_output" / args.output_file
    main(input_path, output_path, args.azure_endpoint_path, args.azure_key_path, args.openai_key_path,
         args.gemini_sa_path, args.gemini_key_path, args.anthropic_key_path, args.bucket_id)
