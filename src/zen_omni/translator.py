"""
Zen Omni Translator - Speech-to-Speech Translation with Voice Preservation

Uses Qwen3-Omni-30B-A3B as the backbone for multimodal translation.
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple
from pathlib import Path


class ZenOmniTranslator:
    """
    Hypermodal translator using Zen Omni (Qwen3-Omni-30B-A3B).
    
    Supports:
    - Speech-to-speech translation (19 input / 10 output languages)
    - Text translation (119 languages)
    - Voice preservation during translation
    - Streaming inference for real-time applications
    """
    
    SPEECH_INPUT_LANGUAGES = [
        "en", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru",
        "ar", "hi", "th", "vi", "id", "ms", "tr", "pl", "nl"
    ]
    
    SPEECH_OUTPUT_LANGUAGES = [
        "en", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru"
    ]
    
    def __init__(
        self,
        model_path: str = "zenlm/zen-omni",
        device: Optional[str] = None,
        torch_dtype: str = "auto",
        use_flash_attn: bool = True,
    ):
        """
        Initialize the Zen Omni translator.
        
        Args:
            model_path: HuggingFace model path or local path
            device: Device to use (auto-detected if None)
            torch_dtype: Data type for model weights
            use_flash_attn: Whether to use Flash Attention 2
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.use_flash_attn = use_flash_attn
        
        self.model = None
        self.processor = None
        self._loaded = False
        
    def load(self):
        """Load the model and processor."""
        if self._loaded:
            return
            
        try:
            from transformers import Qwen3OmniModel, Qwen3OmniProcessor
        except ImportError:
            # Fallback for older transformers versions
            from transformers import AutoModelForCausalLM, AutoProcessor
            Qwen3OmniModel = AutoModelForCausalLM
            Qwen3OmniProcessor = AutoProcessor
        
        print(f"Loading Zen Omni from {self.model_path}...")
        
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        if self.use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = Qwen3OmniModel.from_pretrained(
            self.model_path,
            **model_kwargs
        )
        
        self.processor = Qwen3OmniProcessor.from_pretrained(self.model_path)
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self._loaded = True
        print("Zen Omni loaded successfully!")
        
    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Translate text between languages.
        
        Args:
            text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Translated text
        """
        self.load()
        
        messages = [
            {"role": "system", "content": f"You are a professional translator. Translate from {source_lang} to {target_lang} accurately."},
            {"role": "user", "content": f"Translate the following text to {target_lang}:\n\n{text}"}
        ]
        
        inputs = self.processor.apply_chat_template(
            messages, 
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the translation from the response
        return self._extract_translation(response)
    
    def translate_speech(
        self,
        audio: Union[np.ndarray, str, Path],
        target_lang: str,
        sample_rate: int = 16000,
        preserve_prosody: bool = True,
        return_audio: bool = True,
    ) -> Union[Tuple[str, np.ndarray], str]:
        """
        Translate speech to another language with optional speech output.
        
        Args:
            audio: Audio array or path to audio file
            target_lang: Target language code
            sample_rate: Audio sample rate
            preserve_prosody: Whether to preserve speaking style
            return_audio: Whether to return synthesized audio
            
        Returns:
            Tuple of (translated_text, audio_array) if return_audio=True
            Otherwise just translated_text
        """
        self.load()
        
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            import librosa
            audio, sample_rate = librosa.load(str(audio), sr=sample_rate)
        
        # Build multimodal message
        prosody_instruction = ""
        if preserve_prosody:
            prosody_instruction = " Preserve the original speaker's emotion, pace, and intonation."
        
        output_instruction = ""
        if return_audio:
            if target_lang not in self.SPEECH_OUTPUT_LANGUAGES:
                raise ValueError(
                    f"Speech output not supported for {target_lang}. "
                    f"Supported: {self.SPEECH_OUTPUT_LANGUAGES}"
                )
            output_instruction = f" Speak your translation in {target_lang}."
        
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio, "sample_rate": sample_rate},
                {"type": "text", "text": f"Translate this speech to {target_lang}.{prosody_instruction}{output_instruction}"}
            ]}
        ]
        
        inputs = self.processor(messages, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            return_audio=return_audio,
        )
        
        translated_text = self.processor.decode(outputs.text[0], skip_special_tokens=True)
        
        if return_audio and hasattr(outputs, 'audio') and outputs.audio is not None:
            return translated_text, outputs.audio[0]
        
        return translated_text
    
    def translate_video_audio(
        self,
        video_path: Union[str, Path],
        target_lang: str,
        preserve_voice: bool = True,
    ) -> Tuple[str, np.ndarray]:
        """
        Extract and translate audio from video.
        
        Args:
            video_path: Path to video file
            target_lang: Target language for translation
            preserve_voice: Preserve original voice characteristics
            
        Returns:
            Tuple of (translated_text, translated_audio)
        """
        import subprocess
        import tempfile
        
        # Extract audio from video
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            tmp_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        try:
            result = self.translate_speech(
                tmp_path,
                target_lang=target_lang,
                preserve_prosody=preserve_voice,
                return_audio=True,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        
        return result
    
    def _extract_translation(self, response: str) -> str:
        """Extract the translation from model response."""
        # Handle common response patterns
        if "Translation:" in response:
            return response.split("Translation:")[-1].strip()
        if "翻译：" in response:
            return response.split("翻译：")[-1].strip()
        
        # Return the last paragraph as the translation
        paragraphs = response.strip().split("\n\n")
        return paragraphs[-1].strip()
    
    def stream_translate_speech(
        self,
        audio_stream,
        target_lang: str,
        chunk_duration: float = 2.0,
    ):
        """
        Stream speech translation for real-time applications.
        
        Args:
            audio_stream: Iterator yielding audio chunks
            target_lang: Target language
            chunk_duration: Duration of each chunk in seconds
            
        Yields:
            Translated audio chunks
        """
        self.load()
        
        buffer = []
        for chunk in audio_stream:
            buffer.append(chunk)
            
            # Process when we have enough audio
            combined = np.concatenate(buffer)
            if len(combined) / 16000 >= chunk_duration:
                _, translated_audio = self.translate_speech(
                    combined,
                    target_lang=target_lang,
                    return_audio=True,
                )
                yield translated_audio
                buffer = []
        
        # Process remaining audio
        if buffer:
            combined = np.concatenate(buffer)
            _, translated_audio = self.translate_speech(
                combined,
                target_lang=target_lang,
                return_audio=True,
            )
            yield translated_audio
