import json
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login


# ---------------- LOGIN ----------------

login(token=os.environ["HF_TOKEN"])


# ---------------- PROMPT ----------------

def build_prompt(batch):

    sentences = "\n".join(
        [f"Sentence {i+1}: {s}" for i, s in enumerate(batch)]
    )

    return f"""
You are a biomedical named entity recognition system. Extract biomedical entities from the sentences.

Return ONLY valid JSON.

Format:
[
  {{
    "entity": "text",
    "labels": ["T047"]
  }}
]

Rules:
- Do NOT explain anything
- Do NOT add markdown
- Do NOT add text before or after JSON
- Only extract entities present in the text
- Use UMLS semantic type T-codes
- Assign the most relevant labels
- If no entities exist, return []

EXAMPLES:

INPUT: Metformin treatment improved insulin sensitivity.
OUTPUT:
[
  {{ "entity": "Metformin", "labels": ["T121"] }},
  {{ "entity": "insulin sensitivity", "labels": ["T033", "T042"] }}
]

INPUT: High expression of BRCA1 protein in breast cancer cells.
OUTPUT:
[
  {{ "entity": "BRCA1 protein", "labels": ["T116"] }},
  {{ "entity": "breast cancer", "labels": ["T191"] }},
  {{ "entity": "cells", "labels": ["T025"] }}
]

INPUT: The liver biopsy revealed significant chronic inflammation.
OUTPUT:
[
  {{ "entity": "liver", "labels": ["T023"] }},
  {{ "entity": "biopsy", "labels": ["T060"] }},
  {{ "entity": "inflammation", "labels": ["T047"] }}
]

Sentences:
{sentences}
"""


# ---------------- MAIN ----------------

def main():

    # 🔥 CHANGE TO 70B MODEL (with 4-bit quantization for RTX 6000s)
    model_name = "meta-llama/Llama-3.1-70B-Instruct"  # or Llama-3-70B-Instruct
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with 4-bit quantization...")
    
    # 🔥 CRITICAL: 4-bit quantization config for 2x RTX 6000s (48GB total VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,  # <-- USE 4-BIT!
        device_map="auto",  # Automatically splits across both GPUs
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    model.eval()

    print("Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    # ---------------- LOAD DATA ----------------

    with open("./Segmenting/test.json", "r", encoding="utf-8") as f:
        data = json.load(f)[:3]

    # ---------------- PREPROCESS ----------------

    processed = []

    for sample in data:

        pmid = sample["pmid"]

        title = ""
        abstract = ""

        for passage in sample["passages"]:

            if passage["type"] == "title":
                title = passage["text"][0]

            elif passage["type"] == "abstract":
                abstract = passage["text"][0]

        combined_text = title + ". " + abstract

        processed.append({
            "pmid": pmid,
            "text": combined_text
        })

    # ---------------- INFERENCE ----------------

    all_outputs = []

    for item in processed:

        pmid = item["pmid"]

        print(f"\n========== PMID: {pmid} ==========\n")

        prompt = build_prompt([item["text"]])

        # CHAT FORMAT FOR LLAMA INSTRUCT
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a biomedical named entity recognition system. "
                    "You output ONLY valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # TOKENIZATION
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True
        )

        # move tensors to GPU
        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

        print("Input tokens:", inputs["input_ids"].shape[1])

        # ---------------- GENERATION ----------------

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # remove prompt tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        print("\nRAW OUTPUT:\n")
        print(response[:100])

        print("\n===================================\n")

        all_outputs.append({
            "pmid": pmid,
            "raw_output": response
        })

    # ---------------- SAVE ----------------

    out_dir = "/home/migre/NER_PROJECT/llm_outputs"

    os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(
        out_dir,
        "llama70B.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            all_outputs,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nSaved output to:\n{output_path}")


# ---------------- RUN ----------------

if __name__ == "__main__":
    main()