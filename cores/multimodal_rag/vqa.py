"""
multimodal_rag/vqa.py

Visual Question Answering and Image Captioning.

Provides:
- VQAProvider: Answer questions about images
- CaptioningProvider: Generate image descriptions
- BLIP2VQA: BLIP-2 based VQA
- LLaVAVQA: LLaVA based VQA
- ImageGrounding: Locate objects in images

Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import io

import numpy as np

from .providers import ImageData, VQAResult, CaptionResult

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class VQAConfig:
    """VQA configuration."""
    model_name: str = "Salesforce/blip2-opt-2.7b"
    device: str = "auto"
    max_answer_length: int = 100
    temperature: float = 0.7
    num_beams: int = 5
    do_sample: bool = False


@dataclass
class CaptioningConfig:
    """Captioning configuration."""
    model_name: str = "Salesforce/blip-image-captioning-base"
    device: str = "auto"
    max_length: int = 75
    min_length: int = 5
    num_beams: int = 3


# ============================================================================
# Base VQA Provider
# ============================================================================


class BaseVQAProvider(ABC):
    """Base class for VQA providers."""
    
    def __init__(self, config: VQAConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._device = None
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model."""
        pass
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._is_loaded:
            self._load_model()
    
    @abstractmethod
    async def answer(
        self,
        image: ImageData,
        question: str,
    ) -> VQAResult:
        """Answer question about image."""
        pass
    
    def unload(self) -> None:
        """Unload model."""
        if self._model:
            del self._model
            self._model = None
        if self._processor:
            del self._processor
            self._processor = None
        
        self._is_loaded = False
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ============================================================================
# BLIP-2 VQA Provider
# ============================================================================


class BLIP2VQAProvider(BaseVQAProvider):
    """
    BLIP-2 based Visual Question Answering.
    
    Uses OPT or FlanT5 as language model backbone.
    """
    
    def _load_model(self) -> None:
        """Load BLIP-2 model."""
        try:
            import torch
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device
            
            logger.info(f"Loading BLIP-2 model: {self.config.model_name}")
            
            self._processor = Blip2Processor.from_pretrained(self.config.model_name)
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )
            self._model.to(self._device)
            self._model.eval()
            
            self._is_loaded = True
            logger.info("BLIP-2 model loaded")
            
        except ImportError as e:
            raise RuntimeError(f"Install transformers: {e}")
        except Exception as e:
            logger.error(f"Failed to load BLIP-2: {e}")
            raise
    
    async def answer(
        self,
        image: ImageData,
        question: str,
    ) -> VQAResult:
        """Answer question using BLIP-2."""
        self._ensure_loaded()
        start_time = time.perf_counter()
        
        import torch
        from PIL import Image
        
        # Load image
        if not image.raw_bytes:
            raise ValueError("Image data required")
        
        pil_image = Image.open(io.BytesIO(image.raw_bytes)).convert("RGB")
        
        # Prepare prompt
        prompt = f"Question: {question} Answer:"
        
        # Process
        inputs = self._processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_answer_length,
                num_beams=self.config.num_beams,
                temperature=self.config.temperature if self.config.do_sample else None,
                do_sample=self.config.do_sample,
            )
        
        # Decode
        answer = self._processor.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up
        answer = answer.strip()
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return VQAResult(
            question=question,
            answer=answer,
            image_id=image.id,
            confidence=0.8,  # BLIP-2 doesn't provide confidence
            model_used=self.config.model_name,
            time_ms=elapsed_ms,
        )


# ============================================================================
# LLaVA VQA Provider
# ============================================================================


class LLaVAVQAProvider(BaseVQAProvider):
    """
    LLaVA (Large Language and Vision Assistant) based VQA.
    
    Supports conversational VQA with chat-like interface.
    """
    
    def _load_model(self) -> None:
        """Load LLaVA model."""
        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device
            
            logger.info(f"Loading LLaVA model: {self.config.model_name}")
            
            self._processor = AutoProcessor.from_pretrained(self.config.model_name)
            self._model = LlavaForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )
            self._model.to(self._device)
            self._model.eval()
            
            self._is_loaded = True
            logger.info("LLaVA model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load LLaVA: {e}")
            raise
    
    async def answer(
        self,
        image: ImageData,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> VQAResult:
        """Answer question using LLaVA."""
        self._ensure_loaded()
        start_time = time.perf_counter()
        
        import torch
        from PIL import Image
        
        # Load image
        if not image.raw_bytes:
            raise ValueError("Image data required")
        
        pil_image = Image.open(io.BytesIO(image.raw_bytes)).convert("RGB")
        
        # Build conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        
        # Apply chat template
        prompt = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        
        # Process
        inputs = self._processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_answer_length,
                do_sample=self.config.do_sample,
            )
        
        # Decode
        answer = self._processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        if "assistant" in answer.lower():
            answer = answer.split("assistant")[-1].strip()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return VQAResult(
            question=question,
            answer=answer,
            image_id=image.id,
            model_used=self.config.model_name,
            time_ms=elapsed_ms,
        )


# ============================================================================
# Captioning Provider
# ============================================================================


class BaseCaptioningProvider(ABC):
    """Base class for captioning providers."""
    
    def __init__(self, config: CaptioningConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._device = None
        self._is_loaded = False
    
    @abstractmethod
    def _load_model(self) -> None:
        pass
    
    def _ensure_loaded(self) -> None:
        if not self._is_loaded:
            self._load_model()
    
    @abstractmethod
    async def caption(
        self,
        image: ImageData,
        style: str = "descriptive",
    ) -> CaptionResult:
        """Generate caption for image."""
        pass


class BLIPCaptioningProvider(BaseCaptioningProvider):
    """
    BLIP based image captioning.
    """
    
    def _load_model(self) -> None:
        """Load BLIP captioning model."""
        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device
            
            logger.info(f"Loading BLIP captioning: {self.config.model_name}")
            
            self._processor = BlipProcessor.from_pretrained(self.config.model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.config.model_name
            )
            self._model.to(self._device)
            self._model.eval()
            
            self._is_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load BLIP captioning: {e}")
            raise
    
    async def caption(
        self,
        image: ImageData,
        style: str = "descriptive",
        prompt: Optional[str] = None,
    ) -> CaptionResult:
        """Generate caption using BLIP."""
        self._ensure_loaded()
        start_time = time.perf_counter()
        
        import torch
        from PIL import Image
        
        # Load image
        if not image.raw_bytes:
            raise ValueError("Image data required")
        
        pil_image = Image.open(io.BytesIO(image.raw_bytes)).convert("RGB")
        
        # Style prompts
        style_prompts = {
            "descriptive": "a photograph of",
            "detailed": "a detailed image showing",
            "concise": "",
            "technical": "a technical image of",
        }
        
        text_prompt = prompt or style_prompts.get(style, "")
        
        # Process
        if text_prompt:
            inputs = self._processor(
                images=pil_image,
                text=text_prompt,
                return_tensors="pt",
            )
        else:
            inputs = self._processor(
                images=pil_image,
                return_tensors="pt",
            )
        
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=self.config.max_length,
                min_length=self.config.min_length,
                num_beams=self.config.num_beams,
            )
        
        # Decode
        caption = self._processor.decode(outputs[0], skip_special_tokens=True)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return CaptionResult(
            image_id=image.id,
            caption=caption.strip(),
            style=style,
            confidence=0.85,
            model_used=self.config.model_name,
            time_ms=elapsed_ms,
        )


# ============================================================================
# Unified VQA/Captioning Provider
# ============================================================================


class UnifiedVQAProvider:
    """
    Unified VQA and Captioning provider.
    
    Orchestrates multiple models for best results.
    """
    
    def __init__(
        self,
        vqa_config: Optional[VQAConfig] = None,
        caption_config: Optional[CaptioningConfig] = None,
    ):
        self.vqa_config = vqa_config or VQAConfig()
        self.caption_config = caption_config or CaptioningConfig()
        
        self._vqa_provider: Optional[BaseVQAProvider] = None
        self._caption_provider: Optional[BaseCaptioningProvider] = None
    
    def _get_vqa_provider(self) -> BaseVQAProvider:
        """Get or create VQA provider."""
        if not self._vqa_provider:
            if "blip2" in self.vqa_config.model_name.lower():
                self._vqa_provider = BLIP2VQAProvider(self.vqa_config)
            elif "llava" in self.vqa_config.model_name.lower():
                self._vqa_provider = LLaVAVQAProvider(self.vqa_config)
            else:
                # Default to BLIP-2
                self._vqa_provider = BLIP2VQAProvider(self.vqa_config)
        
        return self._vqa_provider
    
    def _get_caption_provider(self) -> BaseCaptioningProvider:
        """Get or create captioning provider."""
        if not self._caption_provider:
            self._caption_provider = BLIPCaptioningProvider(self.caption_config)
        
        return self._caption_provider
    
    async def answer_question(
        self,
        image: ImageData,
        question: str,
    ) -> VQAResult:
        """Answer question about image."""
        provider = self._get_vqa_provider()
        return await provider.answer(image, question)
    
    async def answer_multiple_questions(
        self,
        image: ImageData,
        questions: List[str],
    ) -> List[VQAResult]:
        """Answer multiple questions about same image."""
        results = []
        for question in questions:
            result = await self.answer_question(image, question)
            results.append(result)
        return results
    
    async def generate_caption(
        self,
        image: ImageData,
        style: str = "descriptive",
    ) -> CaptionResult:
        """Generate image caption."""
        provider = self._get_caption_provider()
        return await provider.caption(image, style)
    
    async def describe_image(
        self,
        image: ImageData,
        aspects: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive image description.
        
        Generates caption and answers common questions.
        """
        aspects = aspects or ["content", "objects", "colors", "scene"]
        
        # Generate caption
        caption_result = await self.generate_caption(image, "detailed")
        
        # Answer aspect questions
        aspect_questions = {
            "content": "What is the main subject of this image?",
            "objects": "What objects can you see in this image?",
            "colors": "What are the dominant colors in this image?",
            "scene": "What type of scene or setting is this?",
            "mood": "What mood or atmosphere does this image convey?",
            "text": "Is there any text visible in this image?",
        }
        
        answers = {}
        for aspect in aspects:
            if aspect in aspect_questions:
                result = await self.answer_question(image, aspect_questions[aspect])
                answers[aspect] = result.answer
        
        return {
            "image_id": image.id,
            "caption": caption_result.caption,
            "aspects": answers,
            "model_used": self.vqa_config.model_name,
        }
    
    async def visual_qa_rag(
        self,
        image: ImageData,
        question: str,
        context: Optional[str] = None,
    ) -> VQAResult:
        """
        VQA with retrieved context.
        
        Enhances VQA with relevant text context.
        """
        # Build enhanced question
        if context:
            enhanced_question = f"""Context: {context}

Based on the image and context above, please answer: {question}"""
        else:
            enhanced_question = question
        
        return await self.answer_question(image, enhanced_question)
    
    def unload(self) -> None:
        """Unload all models."""
        if self._vqa_provider:
            self._vqa_provider.unload()
        if self._caption_provider:
            self._caption_provider.unload()
    
    def health_check(self) -> Dict[str, Any]:
        """Check VQA health."""
        return {
            "vqa_model": self.vqa_config.model_name,
            "caption_model": self.caption_config.model_name,
            "vqa_loaded": self._vqa_provider.is_loaded if self._vqa_provider else False,
            "caption_loaded": self._caption_provider._is_loaded if self._caption_provider else False,
        }
