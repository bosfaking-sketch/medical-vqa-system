
import time
import torch
import gradio as gr

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
print("Device:", DEVICE)


# =========================================================
# PREDICT
# =========================================================

def predict(image, question):

    if image is None:
        return "Please upload a medical image."

    if not question or not question.strip():
        return "Please enter a question."

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

    image_inputs, video_inputs = process_vision_info(
        messages
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
# GRADIO UI
# =========================================================

def medical_vqa_ui(image, question):

    if image is None:
        return "Please upload a medical image.", ""

    if not question or not question.strip():
        return "Please enter a question.", ""

    try:

        start = time.time()

        answer = predict(
            image,
            question
        )

        elapsed = time.time() - start

        return answer, f"{elapsed:.2f} seconds"

    except Exception as e:

        return (
            f"Error: {str(e)}",
            ""
        )


with gr.Blocks(
    title="Medical VQA System"
) as demo:

    gr.Markdown(
        """
        # 🩺 Medical Visual Question Answering

        Upload a medical image and ask a question about it.
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Medical Image"
            )

            question_input = gr.Textbox(
                label="Question",
                placeholder="Example: Is there evidence of a pleural effusion?",
                lines=3
            )

            with gr.Row():

                ask_button = gr.Button(
                    "🔍 Ask",
                    variant="primary"
                )

                clear_button = gr.ClearButton(
                    components=[
                        image_input,
                        question_input
                    ],
                    value="🗑️ Clear"
                )

        with gr.Column():

            answer_output = gr.Textbox(
                label="Model Answer",
                lines=4,
                interactive=False
            )

            time_output = gr.Textbox(
                label="Inference Time",
                interactive=False
            )

    ask_button.click(
        fn=medical_vqa_ui,
        inputs=[
            image_input,
            question_input
        ],
        outputs=[
            answer_output,
            time_output
        ]
    )

    gr.Markdown(
        """
        ---
        ⚠️ **Educational / Research Demonstration Only**

        This system is not intended for clinical diagnosis or
        medical decision-making.
        """
    )


if __name__ == "__main__":

    demo.launch(
        share=True,
        debug=False
    )
