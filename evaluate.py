
import os
import re
import torch
import pandas as pd

from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
from qwen_vl_utils import process_vision_info


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = (
    "vishal98m/"
    "qwen2.5-vl-3b-finetuned-medical-vqa-rad"
)

RESULT_FILE = "qwen_vqa_451_results.csv"


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading Medical VQA model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    use_fast=False,
    min_pixels=256 * 28 * 28,
    max_pixels=768 * 28 * 28
)

model.generation_config.max_length = None
model.eval()

print("Model loaded!")


# =========================================================
# LOAD DATASET
# =========================================================

print("Loading VQA-RAD...")

dataset = load_dataset(
    "flaviagiammarino/vqa-rad"
)

print(dataset)


# =========================================================
# NORMALIZE ANSWER
# =========================================================

def normalize_answer(answer):

    answer = str(answer).lower().strip()

    answer = re.sub(
        r"[^\w\s]",
        "",
        answer
    )

    answer = re.sub(
        r"\s+",
        " ",
        answer
    )

    return answer


# =========================================================
# TOKEN F1
# =========================================================

def token_f1(prediction, ground_truth):

    pred_tokens = normalize_answer(
        prediction
    ).split()

    gt_tokens = normalize_answer(
        ground_truth
    ).split()

    if not pred_tokens or not gt_tokens:
        return (
            1.0
            if pred_tokens == gt_tokens
            else 0.0
        )

    pred_count = {}

    for token in pred_tokens:

        pred_count[token] = (
            pred_count.get(token, 0) + 1
        )

    overlap = 0

    for token in gt_tokens:

        if pred_count.get(token, 0) > 0:

            overlap += 1
            pred_count[token] -= 1

    if overlap == 0:
        return 0.0

    precision = (
        overlap / len(pred_tokens)
    )

    recall = (
        overlap / len(gt_tokens)
    )

    return (
        2 * precision * recall
        / (precision + recall)
    )


# =========================================================
# PREDICT
# =========================================================

def predict(image, question):

    prompt = (
        f"{question}\n"
        "Provide only the short direct answer in 1-5 words. "
        "Do not explain or add any extra text."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = (
        process_vision_info(messages)
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids
        )
    ]

    prediction = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    return prediction


# =========================================================
# EVALUATION
# =========================================================

print("\nStarting evaluation...")

results = []

for i in tqdm(
    range(len(dataset["test"])),
    desc="Evaluating VQA-RAD"
):

    sample = dataset["test"][i]

    ground_truth = sample["answer"]

    prediction = predict(
        sample["image"],
        sample["question"]
    )

    gt_norm = normalize_answer(
        ground_truth
    )

    pred_norm = normalize_answer(
        prediction
    )

    exact_match = (
        gt_norm == pred_norm
    )

    f1 = token_f1(
        prediction,
        ground_truth
    )

    if gt_norm in ["yes", "no"]:
        question_type = "Yes/No"
    else:
        question_type = "Non-Yes/No"

    results.append({

        "id": i + 1,

        "question":
            sample["question"],

        "ground_truth":
            ground_truth,

        "prediction":
            prediction,

        "type":
            question_type,

        "exact_match":
            exact_match,

        "token_f1":
            f1
    })


# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(results)

df.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# METRICS
# =========================================================

overall_em = (
    df["exact_match"].mean() * 100
)

overall_f1 = (
    df["token_f1"].mean() * 100
)

yes_no = df[
    df["type"] == "Yes/No"
]

non_yes_no = df[
    df["type"] == "Non-Yes/No"
]

yes_no_em = (
    yes_no["exact_match"].mean() * 100
)

non_yes_no_em = (
    non_yes_no["exact_match"].mean() * 100
)

non_yes_no_f1 = (
    non_yes_no["token_f1"].mean() * 100
)


# =========================================================
# RESULT
# =========================================================

print("\n========================================")
print("        MEDICAL VQA EVALUATION")
print("========================================")

print(
    f"Total samples: {len(df)}"
)

print("\nOverall:")
print(
    f"  Exact Match : {overall_em:.2f}%"
)

print(
    f"  Token F1    : {overall_f1:.2f}%"
)

print("\nYes/No:")
print(
    f"  Samples     : {len(yes_no)}"
)

print(
    f"  Exact Match : {yes_no_em:.2f}%"
)

print("\nNon-Yes/No:")
print(
    f"  Samples     : {len(non_yes_no)}"
)

print(
    f"  Exact Match : {non_yes_no_em:.2f}%"
)

print(
    f"  Token F1    : {non_yes_no_f1:.2f}%"
)

print("\nSaved:")
print(RESULT_FILE)

print("========================================")
