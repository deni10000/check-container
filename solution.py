"""Qwen3-14B — с Beam Search"""
import signal

TOTAL_TIME_LIMIT = 60 * 30 - 30

def timeout_handler(signum, frame):
    raise RuntimeError("Time limit exceeded")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(TOTAL_TIME_LIMIT)

import json
import pickle
# import os
#
# os.environ["VLLM_DISABLE_FLASHINFER"] = "1"
# os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
#
# os.environ["VLLM_TORCH_COMPILE"] = "0"
# os.environ["VLLM_USE_V1"] = "0"
# import unsloth

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from razdel import tokenize
import pymorphy3


MODEL_DIR = "./weights"
MAX_NEW_TOKENS = 1024 + 256
# NUM_BEAMS = 2


GENERAL_PROMPT = ["""
You are a professional translator. Your task is to translate the text from Russian to Abkhaz.
IMPORTANT: The response must contain only the translation without any comments or clarifications.
Example:
Text: The 2016 European Under-18 Tennis Championship takes place in Klosters, Switzerland
Translation: 18 шықәса зхымҵыц рыбжьара аԥкьаҭмпыл азы Европа Ачемпионат 2016 швеицариатәи ақалақь Клиустер аҿы имҩаԥысит

Dictionary:
""", "Text to translate:"]


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=True)

    with open("ru-ab.json", "r", encoding='utf-8') as f:
        dct = json.load(f)

    morph = pymorphy3.MorphAnalyzer()

    # print(rows)
    llm = LLM(
        model=MODEL_DIR,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        seed=0,
        max_model_len=MAX_NEW_TOKENS * 2
    )

    print(1)

    prompts = []
    for row in rows:
        words = tokenize(row['src'])
        add_words = {}
        for word in words:
            word = word.text
            if not word.isalpha():
                continue
            parsed = morph.parse(word)
            x = parsed[0].normal_form
            if x in dct:
                add_words[x] = dct[x]

        arr = []
        for x in add_words:
            arr.append(f'{x} is {", ".join(add_words[x])}')

        prompt = '\n'.join([GENERAL_PROMPT[0]] + arr + [GENERAL_PROMPT[1], row["src"]])

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompts.append(text)

    # prompts = [
    #     tokenizer.apply_chat_template(
    #         [{"role": "user", "content": GENERAL_PROMPT + row["src"]}],
    #         tokenize=False,
    #         add_generation_prompt=True,
    #         enable_thinking=False,
    #     )
    #     for row in rows
    # ]

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        top_k=-1,
    )

    outputs = llm.generate(prompts, sampling_params=sampling)

    results = [
        {
            'rid': row['rid'],
            'translation': out.outputs[0].text.strip(),
        }
        for row, out in zip(rows, outputs)
    ]

    # print(results)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()