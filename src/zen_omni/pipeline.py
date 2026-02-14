"""
Zen Dubbing Pipeline - End-to-End Video Dubbing with Translation and Lip Sync

Integrates:
- zen-omni: Speech translation with voice preservation
- zen-dub: Neural lip synchronization
"""

import os
import tempfile
import subprocess
from typing import Optional, Union
from pathlib import Path
import numpy as np

from .translator import ZenOmniTranslator


class ZenDubbingPipeline:
    """
    End-to-end video dubbing pipeline combining zen-omni translation
    and zen-dub lip synchronization.
    
    Example:
        pipeline = ZenDubbingPipeline()
        dubbed_video = pipeline.dub(
            "input.mp4",
            target_lang="en",
            output_path="output.mp4"
        )
    """
    
    def __init__(
        self,
        translator_model: str = "zenlm/zen-omni-30b-instruct",
        zen_dub_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the dubbing pipeline.
        
        Args:
            translator_model: HuggingFace model path for zen-omni
            zen_dub_path: Path to zen-dub installation (auto-detected if None)
            device: Device for inference
        """
        self.translator = ZenOmniTranslator(translator_model, device=device)
        self.zen_dub_path = zen_dub_path or self._find_zen_dub()
        self._validate_dependencies()
        
    def _find_zen_dub(self) -> str:
        """Auto-detect zen-dub installation."""
        possible_paths = [
            "../zen-dub",
            "../../zen-dub",
            os.path.expanduser("~/work/zen/zen-dub"),
            "/opt/zen-dub",
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "scripts/inference.py")):
                return os.path.abspath(path)
        
        raise RuntimeError(
            "zen-dub not found. Please install it or specify zen_dub_path."
        )
    
    def _validate_dependencies(self):
        """Validate required dependencies are available."""
        # Check ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
    
    def dub(
        self,
        video_path: Union[str, Path],
        target_lang: str,
        output_path: Optional[Union[str, Path]] = None,
        preserve_voice: bool = True,
        fps: int = 30,
        batch_size: int = 8,
    ) -> Path:
        """
        Dub a video to a target language with lip synchronization.
        
        Args:
            video_path: Path to input video
            target_lang: Target language code (e.g., "en", "ja", "zh")
            output_path: Path for output video (auto-generated if None)
            preserve_voice: Preserve original speaker characteristics
            fps: Output video frame rate
            batch_size: Batch size for lip sync inference
            
        Returns:
            Path to dubbed video
        """
        video_path = Path(video_path)
        
        if output_path is None:
            output_path = video_path.with_stem(f"{video_path.stem}_{target_lang}_dubbed")
        output_path = Path(output_path)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Step 1: Extract audio
            print("Step 1/4: Extracting audio from video...")
            audio_path = tmpdir / "source_audio.wav"
            self._extract_audio(video_path, audio_path)
            
            # Step 2: Translate speech
            print(f"Step 2/4: Translating speech to {target_lang}...")
            translated_text, translated_audio = self.translator.translate_speech(
                audio_path,
                target_lang=target_lang,
                preserve_prosody=preserve_voice,
                return_audio=True,
            )
            
            # Save translated audio
            translated_audio_path = tmpdir / "translated_audio.wav"
            self._save_audio(translated_audio, translated_audio_path)
            
            # Step 3: Generate lip-synced video
            print("Step 3/4: Generating lip-synced video...")
            lip_synced_path = tmpdir / "lip_synced.mp4"
            self._run_lip_sync(
                video_path,
                translated_audio_path,
                lip_synced_path,
                fps=fps,
                batch_size=batch_size,
            )
            
            # Step 4: Finalize output
            print("Step 4/4: Finalizing output...")
            self._finalize_video(lip_synced_path, output_path)
        
        print(f"Dubbed video saved to: {output_path}")
        return output_path
    
    def _extract_audio(self, video_path: Path, output_path: Path):
        """Extract audio from video using ffmpeg."""
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _save_audio(self, audio: np.ndarray, path: Path, sample_rate: int = 24000):
        """Save audio array to file."""
        import soundfile as sf
        sf.write(str(path), audio, sample_rate)
    
    def _run_lip_sync(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        fps: int,
        batch_size: int,
    ):
        """Run zen-dub lip synchronization."""
        import yaml
        
        # Create inference config
        config = {
            "task_0": {
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "result_name": output_path.name,
            }
        }
        
        config_path = output_path.parent / "inference_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Run zen-dub inference
        cmd = [
            "python", os.path.join(self.zen_dub_path, "scripts/inference.py"),
            "--inference_config", str(config_path),
            "--result_dir", str(output_path.parent),
            "--fps", str(fps),
            "--batch_size", str(batch_size),
            "--use_float16",
            "--version", "v15",
        ]
        
        subprocess.run(cmd, check=True, cwd=self.zen_dub_path)
    
    def _finalize_video(self, source_path: Path, output_path: Path):
        """Copy final video to output location."""
        import shutil
        shutil.copy(source_path, output_path)
    
    def stream_dub(
        self,
        video_stream,
        target_lang: str,
        fps: int = 30,
    ):
        """
        Stream video dubbing for real-time applications.
        
        Args:
            video_stream: Iterator yielding (frame, audio_chunk) tuples
            target_lang: Target language
            fps: Frame rate
            
        Yields:
            Dubbed (frame, audio) tuples
        """
        # Buffer for audio chunks
        audio_buffer = []
        frame_buffer = []
        
        for frame, audio_chunk in video_stream:
            audio_buffer.append(audio_chunk)
            frame_buffer.append(frame)
            
            # Process when we have enough data (e.g., 2 seconds)
            combined_audio = np.concatenate(audio_buffer)
            if len(combined_audio) / 16000 >= 2.0:
                # Translate audio
                _, translated_audio = self.translator.translate_speech(
                    combined_audio,
                    target_lang=target_lang,
                    return_audio=True,
                )
                
                # TODO: Run real-time lip sync on buffered frames
                # For now, yield frames with translated audio
                for i, frame in enumerate(frame_buffer):
                    audio_slice = translated_audio[
                        int(i * len(translated_audio) / len(frame_buffer)):
                        int((i + 1) * len(translated_audio) / len(frame_buffer))
                    ]
                    yield frame, audio_slice
                
                audio_buffer = []
                frame_buffer = []
        
        # Process remaining data
        if audio_buffer:
            combined_audio = np.concatenate(audio_buffer)
            _, translated_audio = self.translator.translate_speech(
                combined_audio,
                target_lang=target_lang,
                return_audio=True,
            )
            
            for i, frame in enumerate(frame_buffer):
                audio_slice = translated_audio[
                    int(i * len(translated_audio) / len(frame_buffer)):
                    int((i + 1) * len(translated_audio) / len(frame_buffer))
                ]
                yield frame, audio_slice


class HanzoOrchestrationLayer:
    """
    Real-time orchestration layer for streaming video dubbing.
    
    Handles:
    - WebSocket connections for real-time video/audio streams
    - Load balancing across multiple GPU workers
    - Caching for frequently used avatars
    - Latency optimization
    """
    
    def __init__(
        self,
        translator_model: str = "zenlm/zen-omni-30b-instruct",
        num_workers: int = 1,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the orchestration layer.
        
        Args:
            translator_model: Model for translation
            num_workers: Number of parallel workers
            cache_dir: Directory for caching preprocessed data
        """
        self.translator_model = translator_model
        self.num_workers = num_workers
        self.cache_dir = cache_dir or tempfile.gettempdir()
        
        self._workers = []
        self._avatar_cache = {}
    
    async def start(self, host: str = "0.0.0.0", port: int = 8765):
        """
        Start the WebSocket server for real-time streaming.
        
        Args:
            host: Server host
            port: Server port
        """
        import asyncio
        import websockets
        
        async def handler(websocket, path):
            """Handle incoming WebSocket connections."""
            config = await websocket.recv()
            config = json.loads(config)
            
            target_lang = config.get("target_lang", "en")
            avatar_id = config.get("avatar_id")
            
            pipeline = ZenDubbingPipeline(self.translator_model)
            
            async for message in websocket:
                if isinstance(message, bytes):
                    # Audio/video frame
                    # Process and send back dubbed frame
                    pass
        
        server = await websockets.serve(handler, host, port)
        print(f"Orchestration layer started at ws://{host}:{port}")
        await server.wait_closed()
    
    def preprocess_avatar(self, avatar_id: str, video_path: str):
        """
        Preprocess an avatar for faster real-time inference.
        
        Args:
            avatar_id: Unique identifier for the avatar
            video_path: Path to avatar reference video
        """
        # Extract face landmarks and latents for the avatar
        # Cache for future use
        cache_path = os.path.join(self.cache_dir, f"avatar_{avatar_id}")
        
        # Run zen-dub preprocessing
        # Store in cache
        self._avatar_cache[avatar_id] = cache_path
        
        return cache_path
    
    def get_status(self) -> dict:
        """Get current orchestration layer status."""
        return {
            "workers": self.num_workers,
            "active_connections": len(self._workers),
            "cached_avatars": list(self._avatar_cache.keys()),
        }
