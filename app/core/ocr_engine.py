"""
Moteur OCR basé sur PaddleOCR — Pattern Singleton thread-safe.

Responsabilités :
    - Charger PaddleOCR UNE SEULE FOIS en mémoire (économie de RAM).
    - Exposer une interface propre pour lire du texte depuis une image.
    - Désactiver les logs internes de PaddleOCR par défaut (terminal propre).
    - Déléguer la gestion des erreurs aux aspects (AOP), zéro try/except ici.
"""
import os
import logging
import threading
from pathlib import Path
from typing import List, Tuple

from paddleocr import PaddleOCR

from app.core.aspect import handle_exceptions
from app.core.exceptions import (
    OCREngineError,
    ImageNotFoundError,
    UnsupportedImageFormatError,
)

# ── Silence total sur les logs de PaddleOCR et ses dépendances ──────────────
os.environ["FLAGS_minloglevel"] = "3"          # PaddlePaddle C++ logs
os.environ["FLAGS_use_mkldnn"] = "0"           # Désactive oneDNN (compatibilité CPU)
logging.getLogger("ppocr").setLevel(logging.CRITICAL)

# Formats de fichiers supportés par PaddleOCR (images + pdf)
_SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".pdf"}


class OCREngine:
    """
    Singleton thread-safe pour le moteur PaddleOCR.

    Utilisation :
        engine = OCREngine()          # Première instanciation → charge le modèle
        engine2 = OCREngine()         # Même instance, zéro allocation supplémentaire
        assert engine is engine2      # True

        results = engine.read("facture.png")
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Double-checked locking pour garantir une seule instance, même en multi-thread."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, lang: str = "fr", use_gpu: bool = False):
        # Guard : ne pas ré-initialiser si le modèle est déjà chargé
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._ocr = PaddleOCR(
            use_angle_cls=True,    # Détection de l'orientation du texte
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,        # Désactive les logs de PaddleOCR
        )
        self._initialized = True

    # ── Méthodes publiques ──────────────────────────────────────────────────

    @handle_exceptions(FileNotFoundError, raise_as=ImageNotFoundError)
    @handle_exceptions(ValueError, raise_as=UnsupportedImageFormatError)
    def read(self, image_path: str) -> List[Tuple[str, float]]:
        """
        Lit le texte contenu dans une image.

        Paramètres
        ----------
        image_path : str
            Chemin absolu ou relatif vers l'image de la facture.

        Retour
        ------
        List[Tuple[str, float]]
            Liste de tuples (texte_détecté, score_de_confiance).

        Lève
        ----
        ImageNotFoundError
            Si le fichier n'existe pas.
        UnsupportedImageFormatError
            Si le format de l'image n'est pas supporté.
        OCREngineError
            Pour toute autre erreur liée au moteur OCR.
        """
        path = Path(image_path)
        self._validate_image(path)
        return self._run_ocr(path)

    @handle_exceptions(Exception, raise_as=OCREngineError)
    def read_batch(self, image_paths: List[str]) -> dict:
        """
        Traite un lot d'images en séquence.

        Retour
        ------
        dict
            Dictionnaire {chemin_image: résultats | erreur}.
        """
        results = {}
        for img_path in image_paths:
            try:
                results[img_path] = self.read(img_path)
            except (ImageNotFoundError, UnsupportedImageFormatError, OCREngineError) as exc:
                results[img_path] = {"error": str(exc)}
        return results

    # ── Méthodes privées ────────────────────────────────────────────────────

    @staticmethod
    def _validate_image(path: Path) -> None:
        """Valide l'existence et le format du fichier image."""
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        if path.suffix.lower() not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Format '{path.suffix}' non supporté. "
                f"Formats acceptés : {', '.join(sorted(_SUPPORTED_FORMATS))}"
            )

    @handle_exceptions(Exception, raise_as=OCREngineError)
    def _run_ocr(self, path: Path) -> List[Tuple[str, float]]:
        """Exécute PaddleOCR (images ou PDF multi-pages) et retourne les résultats nettoyés."""
        raw_results = self._ocr.ocr(str(path), cls=True)

        if not raw_results:
            return []

        extracted = []
        for page in raw_results:
            if not page:
                continue
            for line_info in page:
                if line_info and len(line_info) > 1 and line_info[1]:
                    extracted.append((line_info[1][0], round(line_info[1][1], 4)))
        
        return extracted

