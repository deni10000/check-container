"""Qwen3-14B — с Beam Search"""
import signal

TOTAL_TIME_LIMIT = 60 * 5 - 20

def timeout_handler(signum, frame):
    raise RuntimeError("Time limit exceeded")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(TOTAL_TIME_LIMIT)



import json
import pickle

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_DIR = "./weights"
MAX_NEW_TOKENS = 1024 + 512
NUM_BEAMS = 2


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
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype="bfloat16",
        device_map="auto",
    )

    print(1)
    results = []
    for row in rows:
        messages = [{"role": "user", "content": GENERAL_PROMPT + row["src"]}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=True,
            do_sample=False,
            temperature=None,
        )

        generated_ids = outputs[0][input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        results.append({
            'rid': row['rid'],
            'translation': response.strip(),
        })

        print(results[-1])

        del inputs, outputs

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()