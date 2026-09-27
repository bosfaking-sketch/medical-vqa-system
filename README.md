
# Medical VQA System

## 1. Introduction

This project implements a Medical Visual Question Answering (Medical VQA) system.

The system receives:

- A medical image
- A natural-language question

and generates an answer using a pretrained Medical VQA model.

Pipeline:

Medical Image + Question
        ↓
Qwen2.5-VL Medical VQA
        ↓
Generated Answer


## 2. Model

Model:

vishal98m/qwen2.5-vl-3b-finetuned-medical-vqa-rad

The model is based on Qwen2.5-VL-3B and is fine-tuned for Medical VQA.

The model is used for inference only in this project.


## 3. Dataset

Dataset:

VQA-RAD

Dataset structure:

- image
- question
- answer

Dataset split used in this project:

- Train: 1793 samples
- Test: 451 samples


## 4. Evaluation

The model was evaluated on all 451 test samples.

Results:

| Metric | Result |
|---|---:|
| Overall Exact Match | 51.88% |
| Overall Token F1 | 56.51% |
| Yes/No Exact Match | 73.71% |
| Non-Yes/No Exact Match | 24.50% |
| Non-Yes/No Token F1 | 34.94% |

Inference time in the interactive demo was approximately 1-2 seconds per question on a Tesla T4 GPU.


## 5. Baseline Comparison

A baseline experiment was also performed using BLIP on the same 50 test samples.

| Model | Exact Match | Token F1 |
|---|---:|---:|
| BLIP | 34.00% | 34.00% |
| Qwen2.5-VL | 60.00% | 66.64% |

Qwen2.5-VL was therefore used as the main model for the final demo.


## 6. Application

The project includes a Gradio interface.

Users can:

1. Upload a medical image.
2. Enter a question.
3. Click the Ask button.
4. Receive the generated answer.
5. View inference time.


## 7. Project Structure

medical-vqa/

├── Medical_VQA.ipynb
├── app.py
├── requirements.txt
├── README.md
├── qwen_vqa_451_results.csv
└── blip_vqa_50_results.csv


## 8. Run

Install dependencies:

```bash
pip install -r requirements.txt
