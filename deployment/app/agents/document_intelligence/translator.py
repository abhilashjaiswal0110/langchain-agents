"""LLM-based translation service for the Document Intelligence Agent.

This module provides translation capabilities using the same LLM
(GPT-4/Claude) as the main agent, avoiding additional API dependencies.

Following Enterprise Development Standards:
- Software Architect: Modular translation service
- Software Engineer: Type-safe with language validation
"""

import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


# Supported language codes and names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "vi": "Vietnamese",
    "th": "Thai",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "id": "Indonesian",
}


class LLMTranslator:
    """Translation service using LLM (GPT-4/Claude).

    Provides high-quality translation without requiring additional
    translation API subscriptions (Google Translate, DeepL, etc.).
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        """Initialize the translator.

        Args:
            llm: Language model to use. If None, creates one from config.
        """
        self._llm = llm
        self._default_target = os.getenv("DEFAULT_TARGET_LANGUAGE", "en")

        self._translation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional translator with expertise in multiple languages.

Your task is to translate text accurately while:
1. Preserving the original meaning and tone
2. Maintaining formatting (paragraphs, lists, etc.)
3. Keeping technical terms consistent
4. Adapting idioms appropriately for the target language

IMPORTANT: Output ONLY the translated text, nothing else. No explanations, no notes."""),
            ("human", """Translate the following text from {source_language} to {target_language}:

---
{text}
---

Translation:"""),
        ])

        self._detection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a language identification expert.
Identify the language of the given text and respond with ONLY the ISO 639-1 language code (e.g., 'en', 'es', 'fr', 'de', 'zh', 'ja').
If uncertain, respond with your best guess."""),
            ("human", "Identify the language of this text:\n\n{text}"),
        ])

    def _get_llm(self) -> BaseChatModel:
        """Get or create the LLM instance.

        Returns:
            BaseChatModel instance
        """
        if self._llm is not None:
            return self._llm

        # Create LLM from environment
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        elif os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.3)
        else:
            msg = "No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
            raise RuntimeError(msg)

        return self._llm

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str = "auto",
    ) -> dict[str, Any]:
        """Translate text to target language.

        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'es', 'fr', 'de')
            source_language: Source language code or 'auto' for detection

        Returns:
            Dict with translated text and metadata
        """
        if not text.strip():
            return {
                "success": False,
                "error": "Empty text provided",
                "original": text,
                "translated": "",
            }

        # Detect source language if needed
        if source_language == "auto":
            source_language = self.detect_language(text)

        # Validate languages
        target_name = self._get_language_name(target_language)
        source_name = self._get_language_name(source_language)

        # Skip translation if same language
        if source_language == target_language:
            return {
                "success": True,
                "original": text,
                "translated": text,
                "source_language": source_language,
                "source_language_name": source_name,
                "target_language": target_language,
                "target_language_name": target_name,
                "note": "Source and target language are the same",
            }

        try:
            llm = self._get_llm()
            chain = self._translation_prompt | llm | StrOutputParser()

            translated = chain.invoke({
                "text": text,
                "target_language": target_name,
                "source_language": source_name,
            })

            return {
                "success": True,
                "original": text,
                "translated": translated.strip(),
                "source_language": source_language,
                "source_language_name": source_name,
                "target_language": target_language,
                "target_language_name": target_name,
            }

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "original": text,
                "translated": "",
                "source_language": source_language,
                "target_language": target_language,
            }

    def detect_language(self, text: str) -> str:
        """Detect the language of text.

        Args:
            text: Text to analyze

        Returns:
            ISO 639-1 language code
        """
        # First try langdetect (faster, no API call)
        try:
            from langdetect import detect
            return detect(text[:1000])  # Sample first 1000 chars
        except ImportError:
            pass
        except Exception:
            pass

        # Fall back to LLM detection
        try:
            llm = self._get_llm()
            chain = self._detection_prompt | llm | StrOutputParser()

            result = chain.invoke({"text": text[:500]})
            code = result.strip().lower()[:2]

            # Validate code
            if code in SUPPORTED_LANGUAGES:
                return code
            return "en"  # Default to English if unknown

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en"

    def _get_language_name(self, code: str) -> str:
        """Get the full language name from code.

        Args:
            code: ISO 639-1 language code

        Returns:
            Full language name
        """
        return SUPPORTED_LANGUAGES.get(code.lower(), code.capitalize())

    def get_supported_languages(self) -> dict[str, str]:
        """Get dictionary of supported languages.

        Returns:
            Dict mapping language codes to names
        """
        return SUPPORTED_LANGUAGES.copy()


# Global instance
_translator_instance: LLMTranslator | None = None


def get_translator(llm: BaseChatModel | None = None) -> LLMTranslator:
    """Get the global translator instance.

    Args:
        llm: Optional LLM to use

    Returns:
        LLMTranslator instance
    """
    global _translator_instance
    if _translator_instance is None or llm is not None:
        _translator_instance = LLMTranslator(llm)
    return _translator_instance
