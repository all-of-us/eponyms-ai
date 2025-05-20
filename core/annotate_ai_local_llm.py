# from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
from pathlib import Path
from time import time
from mlx_lm import load, generate
import pandas as pd

from resources.annotate_ai_utils import read_guideline
from resources.constants import ANNOTATED_COLS, data_path


def call_local_llm(model, tokenizer, prompt):
    # messages = [
    #     {"role": "user", "content": prompt}
    # ]
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # # conduct text completion
    # generated_ids = model.generate(
    #     **model_inputs,
    #     max_new_tokens=32768
    # )
    # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # # parsing thinking content
    # try:
    #     # rindex finding 151668 (</think>)
    #     index = len(output_ids) - output_ids[::-1].index(151668)
    # except ValueError:
    #     index = 0

    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    # content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # print("thinking content:", thinking_content)
    # print("content:", content)

    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

    response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=10240)

    return response


# Accept arguments from the command line
def get_local_llm_parser():
    """
    Get the parser
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=Path, help="The name of the input file")
    parser.add_argument("-o", "--output_path", type=Path, help="The name of the output file")

    return parser


def main():
    parser = get_local_llm_parser()
    args = parser.parse_args()

    guideline = read_guideline(data_path / "identify_eponym_prompt.md")
    system_prompt, concept_prompt, user_prompt = guideline.split('-----')
    
    eponyms_apos_df = pd.read_csv(args.input_path,
                                  delimiter=',',
                                  header=0,
                                  names=ANNOTATED_COLS,
                                  dtype=str)
    

    model_name = "lmstudio-community/Qwen3-30B-A3B-MLX-4bit"

    model, tokenizer = load(model_name)
    
    with args.output_path.open('w+') as f:
        for row in eponyms_apos_df.itertuples():
            start_time = time()
            prompt = system_prompt + "\n" + concept_prompt.format(concept_name=row.concept_name) + "\n" + user_prompt
            response = call_local_llm(model, tokenizer, prompt)
            end_time = time()
            row_dict = {
                "disease": row.concept_name,
                "response": response,
                "time_taken": end_time - start_time
            }
            f.write(json.dumps(row_dict) + '\n')


if __name__ == "__main__":
    main()
