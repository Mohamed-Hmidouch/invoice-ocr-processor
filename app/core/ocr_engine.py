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
from typing import Dict, List, Optional, Tuple

# ── Flags Paddle AVANT tout import de paddle/paddleocr ───────────────────────
os.environ.setdefault("FLAGS_minloglevel", "3")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_DISABLE_MKLDNN", "1")

from paddleocr import PaddleOCR
import paddle

# ── Désactiver oneDNN via l'API Python (seule méthode fiable en Paddle 3.x) ──
try:
    paddle.set_flags({"FLAGS_use_mkldnn": False})
except Exception:
    pass

from app.core.aspect import handle_exceptions
from app.core.exceptions import (
    OCREngineError,
    ImageNotFoundError,
    UnsupportedImageFormatError,
)

logging.getLogger("ppocr").setLevel(logging.CRITICAL)
logging.getLogger("paddleocr").setLevel(logging.CRITICAL)

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

    def __init__(self, lang: str = "fr"):
        # Guard : ne pas ré-initialiser si le modèle est déjà chargé
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,  # Désactive le modèle doc_ori (oneDNN crash)
            use_doc_unwarping=False,              # Désactive UVDoc (oneDNN crash)
            use_textline_orientation=False,       # Désactive textline_ori (oneDNN crash)
        )
        self._initialized = True

    # ── Méthodes publiques ──────────────────────────────────────────────────

    @handle_exceptions(FileNotFoundError, raise_as=ImageNotFoundError)
    @handle_exceptions(ValueError, raise_as=UnsupportedImageFormatError)
    def read(self, image_path: str) -> Tuple[List[dict], Optional[Dict[str, int]]]:
        """
        Lit le texte contenu dans une image.

        Paramètres
        ----------
        image_path : str
            Chemin absolu ou relatif vers l'image de la facture.

        Retour
        ------
        Tuple[List[dict], Optional[Dict[str, int]]]
            (résultats OCR, {"width": W, "height": H} de l'image originale)

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
        extracted, ocr_image_size = self._run_ocr(path)
        # Si _run_ocr n'a pas pu déterminer la taille (format 2.x),
        # fallback vers _get_image_size (PIL/fitz)
        if ocr_image_size is None:
            ocr_image_size = self._get_image_size(path)
        return extracted, ocr_image_size

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
    def _get_image_size(path: Path) -> Optional[Dict[str, int]]:
        """Récupère les dimensions (width, height) de l'image/PDF originale en pixels."""
        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                # PaddleOCR convertit les PDF en images en interne.
                # On utilise fitz (PyMuPDF) pour lire la taille de la page à 72 DPI
                # puis on calcule la taille en pixels au DPI que PaddleOCR utilise (par défaut 300).
                try:
                    import fitz
                    doc = fitz.open(str(path))
                    page = doc[0]
                    # Empiriquement, PaddleOCR 3.x rend les PDFs à 2.0x le 72 DPI standard.
                    # Vérifié: 595.28pt * 2.0 = 1191px, ce qui correspond aux bbox observées.
                    zoom = 2.0
                    width = int(page.rect.width * zoom)
                    height = int(page.rect.height * zoom)
                    doc.close()
                    return {"width": width, "height": height}
                except ImportError:
                    # Fallback: essayer pdf2image
                    try:
                        from pdf2image import convert_from_path
                        images = convert_from_path(str(path), first_page=1, last_page=1, dpi=200)
                        if images:
                            return {"width": images[0].width, "height": images[0].height}
                    except ImportError:
                        pass
                    return None
            else:
                # Image standard → PIL
                from PIL import Image
                with Image.open(str(path)) as img:
                    return {"width": img.width, "height": img.height}
        except Exception as exc:
            logging.getLogger(__name__).warning("Impossible de lire la taille de %s : %s", path, exc)
            return None

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
    def _run_ocr(self, path: Path) -> Tuple[List[dict], Optional[Dict[str, int]]]:
        """Exécute PaddleOCR (images ou PDF multi-pages) et retourne les résultats nettoyés.
        Compatible PaddleOCR 2.x (liste de listes) et 3.x (objets OCRResult).
        Retourne un tuple (dicts contenant text/confidence/bbox, image_size ou None).
        """
        raw_results = self._ocr.ocr(str(path))

        if not raw_results:
            return [], None

        extracted = []
        ocr_image_size = None
        
        def _sanitize_bbox(b):
            if b is None: return None
            try:
                if hasattr(b, 'tolist'): b = b.tolist()
                return [[float(pt[0]), float(pt[1])] for pt in b]
            except Exception:
                return None

        for page in raw_results:
            if not page:
                continue

            # ── Format PaddleOCR 3.x : objet dict-like (OCRResult) ou attribut direct ──
            # OCRResult hérite de dict, donc rec_texts est une clé dict, pas un attribut Python.
            rec_texts = None
            rec_scores = None
            dt_polys = None

            if hasattr(page, 'rec_texts'):
                rec_texts = page.rec_texts
                rec_scores = page.rec_scores
                dt_polys = page.dt_polys if hasattr(page, 'dt_polys') else None
            elif isinstance(page, dict) and 'rec_texts' in page:
                rec_texts = page['rec_texts']
                rec_scores = page['rec_scores']
                dt_polys = page.get('dt_polys')

            # ── Extraire la taille de l'image interne (PaddleOCR 3.x) ──
            # doc_preprocessor_res['output_img'] contient l'image numpy (H, W, C)
            # qui correspond EXACTEMENT à l'espace de coordonnées des bbox.
            if ocr_image_size is None:
                try:
                    dpr = None
                    if hasattr(page, 'doc_preprocessor_res'):
                        dpr = page.doc_preprocessor_res
                    elif isinstance(page, dict) and 'doc_preprocessor_res' in page:
                        dpr = page['doc_preprocessor_res']
                    
                    if dpr and isinstance(dpr, dict) and 'output_img' in dpr:
                        img_array = dpr['output_img']
                        if hasattr(img_array, 'shape') and len(img_array.shape) >= 2:
                            h, w = img_array.shape[:2]
                            ocr_image_size = {"width": int(w), "height": int(h)}
                except Exception:
                    pass

            if rec_texts is not None and rec_scores is not None:
                for idx, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                    if text and text.strip():
                        bbox = dt_polys[idx] if dt_polys is not None and idx < len(dt_polys) else None
                        extracted.append({
                            "text": text.strip(),
                            "confidence": round(float(score), 4),
                            "bbox": _sanitize_bbox(bbox)
                        })
                continue

            # ── Format PaddleOCR 2.x : liste de [bbox, (texte, score)] ──
            if isinstance(page, list):
                for line_info in page:
                    try:
                        bbox = line_info[0]
                        text = line_info[1][0]
                        score = line_info[1][1]
                        if text and text.strip():
                            extracted.append({
                                "text": text.strip(),
                                "confidence": round(float(score), 4),
                                "bbox": _sanitize_bbox(bbox)
                            })
                    except (IndexError, TypeError):
                        continue

        return extracted, ocr_image_size

