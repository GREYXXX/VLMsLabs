import os

if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.patches as patches
import fitz  # PyMuPDF
from typing import Optional, Any, List, Union
from io import BytesIO

from tqdm import tqdm
import argparse
import joblib


def inference(image, prompt, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda") if torch.cuda.is_available() else inputs.to("mps")

    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=24000)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text


def visualize_layout(image, ocr_output_text, page_num=None):
    """
    Visualize the layout detection results on the original image using PIL
    """
    ocr_data = json.loads(ocr_output_text[0])
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)

    # Define colors for different categories
    colors = {
        "Section-header": "#FF0000",  # red
        "List-item": "#0000FF",  # blue
        "Text": "#00FF00",  # green
        "Title": "#800080",  # purple
        "Table": "#FFA500",  # orange
        "Picture": "#00FFFF",  # cyan
        "Caption": "#FFFF00",  # yellow
        "Footnote": "#FFC0CB",  # pink
        "Formula": "#A52A2A",  # brown
        "Page-header": "#808080",  # gray
        "Page-footer": "#808000",  # olive
    }

    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
    except:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
            )
            font_small = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10
            )
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

    # Draw bounding boxes
    for i, item in enumerate(ocr_data):
        bbox = item["bbox"]
        category = item["category"]
        text = item.get("text", "")

        x1, y1, x2, y2 = bbox
        color = colors.get(category, "#000000")  # default to black
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{category} ({i+1})"

        bbox_text = draw.textbbox((0, 0), label, font=font_small)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]

        # Draw white background for text
        label_bg = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]
        draw.rectangle(label_bg, fill="white", outline=color, width=1)
        draw.text((x1 + 2, y1 - text_height - 2), label, fill=color, font=font_small)

    output_dir = "output_image"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if page_num is not None:
        output_path = os.path.join(output_dir, f"layout_page_{page_num:03d}.jpg")
    else:
        output_path = os.path.join(output_dir, "layout_visualization.jpg")

    draw_img.save(output_path)
    print(f"Saved: {output_path}")


def pdf_to_images(pdf_path):
    """
    Convert PDF pages to PIL Image objects in memory
    """
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_data = pix.tobytes("ppm")

        # Convert to PIL Image directly in memory
        img = Image.open(BytesIO(img_data))
        images.append((img, page_num))

    doc.close()
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DotsOCR inference on an image or PDF."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input image or PDF file.",
    )
    args = parser.parse_args()
    input_path = args.input_path

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model_path = "./weights/DotsOCR"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="sdpa",  # change to "flash_attention_2" if flash attention 2 is installed
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    prompt_mode = "prompt_layout_all_en"
    prompt = dict_promptmode_to_prompt[prompt_mode]
    output_texts = []
    if input_path.lower().endswith(".pdf"):
        images = pdf_to_images(input_path)

        for img, page_num in tqdm(
            images, total=len(images), desc="Processing PDF pages"
        ):
            print(f"Processing page {page_num + 1}...")
            output_text = inference(img, prompt, model, processor)
            output_texts.append(output_text)
            visualize_layout(img, output_text, page_num + 1)

            ocr_data = json.loads(output_text[0])
            extracted_text = "\n\n".join(
                [item["text"] for item in ocr_data if "text" in item]
            )
            print(extracted_text)
    else:
        # Single image processing
        output_text = inference(input_path, prompt, model, processor)
        output_texts.append(output_text)
        visualize_layout(input_path, output_text)

        ocr_data = json.loads(output_text[0])
        extracted_text = "\n\n".join(
            [item["text"] for item in ocr_data if "text" in item]
        )
        print(extracted_text)

    joblib.dump(output_texts, "ocr_outputs.pkl")  # Save all outputs to a file
