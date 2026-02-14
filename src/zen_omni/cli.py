"""
Zen Omni CLI - Command Line Interface for Translation and Dubbing
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Zen Omni - Hypermodal Language Model for Translation + Audio Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate speech
  zen-omni translate audio.wav --lang en --output translated.wav
  
  # Dub a video
  zen-omni dub video.mp4 --lang en --output dubbed.mp4
  
  # Interactive chat
  zen-omni chat
  
For more info: https://zenlm.org
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Translate command
    translate_parser = subparsers.add_parser("translate", help="Translate speech to another language")
    translate_parser.add_argument("input", type=str, help="Input audio file")
    translate_parser.add_argument("--lang", "-l", type=str, required=True, help="Target language code")
    translate_parser.add_argument("--output", "-o", type=str, help="Output audio file")
    translate_parser.add_argument("--model", type=str, default="zenlm/zen-omni", help="Model to use")
    translate_parser.add_argument("--text-only", action="store_true", help="Output text only, no audio")
    
    # Dub command
    dub_parser = subparsers.add_parser("dub", help="Dub a video with translation and lip sync")
    dub_parser.add_argument("input", type=str, help="Input video file")
    dub_parser.add_argument("--lang", "-l", type=str, required=True, help="Target language code")
    dub_parser.add_argument("--output", "-o", type=str, help="Output video file")
    dub_parser.add_argument("--model", type=str, default="zenlm/zen-omni-30b-instruct", help="Model to use")
    dub_parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with Zen Omni")
    chat_parser.add_argument("--model", type=str, default="zenlm/zen-omni", help="Model to use")
    
    # Caption command
    caption_parser = subparsers.add_parser("caption", help="Generate captions for images/videos")
    caption_parser.add_argument("input", type=str, help="Input image or video file")
    caption_parser.add_argument("--output", "-o", type=str, help="Output text file")
    caption_parser.add_argument("--model", type=str, default="zenlm/zen-omni-30b-captioner", help="Model to use")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "translate":
        cmd_translate(args)
    elif args.command == "dub":
        cmd_dub(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "caption":
        cmd_caption(args)


def cmd_translate(args):
    """Handle translate command."""
    from .translator import ZenOmniTranslator
    import soundfile as sf
    
    print(f"Translating {args.input} to {args.lang}...")
    
    translator = ZenOmniTranslator(args.model)
    
    if args.text_only:
        text = translator.translate_speech(
            args.input,
            target_lang=args.lang,
            return_audio=False,
        )
        print(f"\nTranslation:\n{text}")
    else:
        text, audio = translator.translate_speech(
            args.input,
            target_lang=args.lang,
            return_audio=True,
        )
        
        output_path = args.output or f"{Path(args.input).stem}_{args.lang}.wav"
        sf.write(output_path, audio, 24000)
        
        print(f"\nTranslation: {text}")
        print(f"Audio saved to: {output_path}")


def cmd_dub(args):
    """Handle dub command."""
    from .pipeline import ZenDubbingPipeline
    
    print(f"Dubbing {args.input} to {args.lang}...")
    
    pipeline = ZenDubbingPipeline(translator_model=args.model)
    
    output_path = args.output or f"{Path(args.input).stem}_{args.lang}_dubbed.mp4"
    
    pipeline.dub(
        args.input,
        target_lang=args.lang,
        output_path=output_path,
        fps=args.fps,
    )


def cmd_chat(args):
    """Handle chat command."""
    from .translator import ZenOmniTranslator
    
    print("Starting Zen Omni chat...")
    print("Type 'exit' to quit.\n")
    
    translator = ZenOmniTranslator(args.model)
    translator.load()
    
    messages = [
        {"role": "system", "content": "You are Zen, a helpful AI assistant created by Hanzo AI."}
    ]
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input:
                continue
            
            messages.append({"role": "user", "content": user_input})
            
            inputs = translator.processor.apply_chat_template(
                messages, return_tensors="pt"
            ).to(translator.model.device)
            
            outputs = translator.model.generate(**inputs, max_new_tokens=512)
            response = translator.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant response
            response = response.split("assistant")[-1].strip()
            
            print(f"Zen: {response}\n")
            messages.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


def cmd_caption(args):
    """Handle caption command."""
    from .translator import ZenOmniTranslator
    from PIL import Image
    
    print(f"Generating caption for {args.input}...")
    
    translator = ZenOmniTranslator(args.model)
    translator.load()
    
    # Load image
    image = Image.open(args.input)
    
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image in detail."}
    ]}]
    
    inputs = translator.processor(messages, return_tensors="pt").to(translator.model.device)
    outputs = translator.model.generate(**inputs, max_new_tokens=512)
    caption = translator.processor.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nCaption:\n{caption}")
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(caption)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
