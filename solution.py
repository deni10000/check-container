"""Qwen3-14B — с Beam Search"""
# import signal
#
# TOTAL_TIME_LIMIT = 60 * 5 - 20
#
# def timeout_handler(signum, frame):
#     raise RuntimeError("Time limit exceeded")
#
# signal.signal(signal.SIGALRM, timeout_handler)
# signal.alarm(TOTAL_TIME_LIMIT)

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


MODEL_DIR = "./weights"
MAX_NEW_TOKENS = 1024 + 512
# NUM_BEAMS = 2


GENERAL_PROMPT = """
Ты профессиональный переводчик. Переведи текст с английского на русский.
В ответе должен быть ТОЛЬКО перевод, без комментариев, объяснений и преамбул.

Пример:
English: The new AI technology has completely revolutionized the way we work, making complex tasks much simpler.
Russian: Новая технология искусственного интеллекта полностью изменила подход к работе, сделав сложные задачи значительно проще.

Текст для перевода:
"""


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=True)


    print(rows)
    llm = LLM(
        model=MODEL_DIR,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        seed=0,
        max_model_len=MAX_NEW_TOKENS * 2
    )

    print(1)

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": GENERAL_PROMPT + row["src"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]

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

    print(results)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()