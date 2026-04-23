"""
Module core — Logique métier du processeur de factures.

Exporte les classes publiques pour simplifier les imports :
    from app.core import OCREngine, Extractor
"""
from app.core.ocr_engine import OCREngine
from app.core.extractor import Extractor

__all__ = ["OCREngine", "Extractor"]
