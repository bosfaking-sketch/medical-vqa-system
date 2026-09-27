# Medical VQA System

## 1. Introduction

This project implements a Medical Visual Question Answering (Medical VQA) system.

The system receives:

- A medical image
- A natural-language question

and generates an answer using a pretrained Medical VQA model.

### Pipeline

Medical Image + Question
â†“
Qwen2.5-VL Medical VQA
â†“
Generated Answer


## 2. Model

Model:

`vishal98m/qwen2.5-vl-3b-finetuned-medical-vqa-rad`

The model is based on Qwen2.5-VL-3B and is fine-tuned for Medical VQA on VQA-RAD.

The model is used for inference in this project.


## 3. Dataset

Dataset:

**VQA-RAD**

Dataset structure:

- `image`
- `question`
- `answer`

Dataset split used in this project:

- Train: 1793 samples
- Test: 451 samples


## 4. Evaluation

The model was evaluated on all 451 test samples.

### Qwen2.5-VL Results

| Metric | Result |
|---|---:|
| Overall Exact Match | 51.88% |
| Overall Token F1 | 56.51% |
| Yes/No Exact Match | 73.71% |
| Non-Yes/No Exact Match | 24.50% |
| Non-Yes/No Token F1 | 34.94% |

The interactive Gradio demo was tested on a Tesla T4 GPU, with observed inference times of approximately 1-2.5 seconds per question.


## 5. Baseline Comparison

A baseline experiment was performed using BLIP on the same 50 test samples used for the Qwen2.5-VL comparison.

| Model | Exact Match | Token F1 |
|---|---:|---:|
| BLIP | 34.00% | 34.00% |
| Qwen2.5-VL | 60.00% | 66.64% |

Based on this experiment, Qwen2.5-VL was selected as the main model for the final demonstration.


## 6. Application

The project includes a Gradio web interface.

Users can:

1. Upload a medical image.
2. Enter a natural-language question.
3. Click the Ask button.
4. Receive the generated answer.
5. View the inference time.

The application is designed as an educational and research demonstration.


## 7. Project Structure

```text
medical-vqa/
â”œâ”€â”€ Medical_VQA.ipynb
â”œâ”€â”€ app.py
â”œâ”€â”€ evaluate.py
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
â”œâ”€â”€ qwen_vqa_451_results.csv
â”œâ”€â”€ qwen_vqa_50_results.csv
â”œâ”€â”€ blip_vqa_50_results.csv
â””â”€â”€ test_medical_image.png
```


## 8. Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

The application uses Gradio to provide a web interface.


## 9. Hardware

The experiments in this project were performed using:

- Google Colab
- NVIDIA Tesla T4 GPU
- 14.56 GB GPU memory
- CUDA-enabled PyTorch environment

The Qwen2.5-VL-3B model was tested on the Tesla T4 GPU.

The Windows development machine used for project management does not have an NVIDIA CUDA GPU, so model inference was performed in Google Colab.


## 10. Limitations

This system has several limitations:

- The model may generate incorrect answers for some medical images.
- Performance is lower on non-Yes/No questions than on Yes/No questions in the tested VQA-RAD evaluation.
- Exact Match may classify semantically similar answers as incorrect.
- Some predictions may contain a correct general concept but miss specific medical details.
- The system is intended for educational and research demonstration only.
- Generated answers must not be used for clinical diagnosis or medical decision-making.
