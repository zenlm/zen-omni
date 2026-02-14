"""
Zen Omni - Hypermodal Language Model for Translation + Audio Generation

Part of the Zen LM family by Hanzo AI
https://zenlm.org
"""

__version__ = "0.1.0"
__author__ = "Hanzo AI"

from .translator import ZenOmniTranslator
from .pipeline import ZenDubbingPipeline

__all__ = ["ZenOmniTranslator", "ZenDubbingPipeline"]
