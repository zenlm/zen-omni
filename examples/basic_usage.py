"""
Basic usage example for zen-omni multimodal model
"""
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
import torch


def main():
    # Load model and processor
    model_name = "zenlm/zen-omni"
    print(f"Loading {model_name}...")

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)

    # Example: Text-only generation
    print("\n--- Text Generation ---")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "What is the meaning of life?"}]}
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    response = processor.decode(output[0], skip_special_tokens=True)
    print(f"Response: {response}")

    # Example: Text + Image (requires PIL and an image file)
    print("\n--- Vision-Language Example ---")
    print("To use with images:")
    print("""
    from PIL import Image
    image = Image.open("path/to/image.jpg")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What's in this image?"}
        ]
    }]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=256)
    """)

    # Example: Thinking mode for complex reasoning
    print("\n--- Thinking Mode (Extended Reasoning) ---")
    print("For complex reasoning tasks, use thinking mode:")
    print("""
    output = model.generate(
        **inputs,
        thinker_do_sample=True,
        thinker_temperature=0.7,
        thinker_max_new_tokens=32768,  # Extended thinking
        max_new_tokens=512,
    )
    """)


if __name__ == "__main__":
    main()
