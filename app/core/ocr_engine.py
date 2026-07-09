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
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

    @staticmethod
    def get_pdf_page_count(path: str) -> int:
        """Retourne le nombre de pages d'un PDF (1-based indexing pour l'appelant)."""
        pdf_path = Path(path)
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Le fichier n'est pas un PDF : {path}")
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            count = len(doc)
            doc.close()
            return count
        except ImportError as exc:
            raise OCREngineError("PyMuPDF (fitz) requis pour lire les PDF multi-pages.") from exc

    @handle_exceptions(FileNotFoundError, raise_as=ImageNotFoundError)
    @handle_exceptions(ValueError, raise_as=UnsupportedImageFormatError)
    def read(
        self,
        image_path: str,
        pages: Optional[List[int]] = None,
    ) -> Tuple[List[dict], Optional[Union[Dict[str, int], Dict[str, Any]]]]:
        """
        Lit le texte contenu dans une image ou un PDF.

        Paramètres
        ----------
        image_path : str
            Chemin absolu ou relatif vers l'image de la facture.
        pages : Optional[List[int]]
            Pages 1-based à OCRiser (PDF uniquement). None = toutes les pages.

        Retour
        ------
        Tuple[List[dict], Optional[dict]]
            (résultats OCR, métadonnées de taille d'image)

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

        if path.suffix.lower() == ".pdf" and pages is not None:
            extracted, ocr_image_size = self._run_ocr_selected_pages(path, pages)
        else:
            extracted, ocr_image_size = self._run_ocr(path)
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
    def _get_image_size(path: Path, page_number: int = 1) -> Optional[Dict[str, int]]:
        """Récupère les dimensions (width, height) de l'image/PDF en pixels."""
        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                # PaddleOCR convertit les PDF en images en interne.
                # On utilise fitz (PyMuPDF) pour lire la taille de la page à 72 DPI
                # puis on calcule la taille en pixels au DPI que PaddleOCR utilise (par défaut 300).
                try:
                    import fitz
                    doc = fitz.open(str(path))
                    page_index = max(0, page_number - 1)
                    if page_index >= len(doc):
                        doc.close()
                        return None
                    page = doc[page_index]
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

    @staticmethod
    def _sanitize_bbox(b) -> Optional[List[List[float]]]:
        if b is None:
            return None
        try:
            if hasattr(b, "tolist"):
                b = b.tolist()
            return [[float(pt[0]), float(pt[1])] for pt in b]
        except Exception:
            return None

    def _extract_page_results(
        self,
        page_result,
        page_number: Optional[int] = None,
    ) -> Tuple[List[dict], Optional[Dict[str, int]]]:
        """Parse un résultat PaddleOCR (une page) en lignes structurées."""
        if not page_result:
            return [], None

        extracted: List[dict] = []
        ocr_image_size = None

        rec_texts = None
        rec_scores = None
        dt_polys = None

        if hasattr(page_result, "rec_texts"):
            rec_texts = page_result.rec_texts
            rec_scores = page_result.rec_scores
            dt_polys = page_result.dt_polys if hasattr(page_result, "dt_polys") else None
        elif isinstance(page_result, dict) and "rec_texts" in page_result:
            rec_texts = page_result["rec_texts"]
            rec_scores = page_result["rec_scores"]
            dt_polys = page_result.get("dt_polys")

        try:
            dpr = None
            if hasattr(page_result, "doc_preprocessor_res"):
                dpr = page_result.doc_preprocessor_res
            elif isinstance(page_result, dict) and "doc_preprocessor_res" in page_result:
                dpr = page_result["doc_preprocessor_res"]

            if dpr and isinstance(dpr, dict) and "output_img" in dpr:
                img_array = dpr["output_img"]
                if hasattr(img_array, "shape") and len(img_array.shape) >= 2:
                    h, w = img_array.shape[:2]
                    ocr_image_size = {"width": int(w), "height": int(h)}
        except Exception:
            pass

        if rec_texts is not None and rec_scores is not None:
            for idx, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                if text and text.strip():
                    bbox = dt_polys[idx] if dt_polys is not None and idx < len(dt_polys) else None
                    line = {
                        "text": text.strip(),
                        "confidence": round(float(score), 4),
                        "bbox": self._sanitize_bbox(bbox),
                    }
                    if page_number is not None:
                        line["page"] = page_number
                    extracted.append(line)
            return extracted, ocr_image_size

        if isinstance(page_result, list):
            for line_info in page_result:
                try:
                    bbox = line_info[0]
                    text = line_info[1][0]
                    score = line_info[1][1]
                    if text and text.strip():
                        line = {
                            "text": text.strip(),
                            "confidence": round(float(score), 4),
                            "bbox": self._sanitize_bbox(bbox),
                        }
                        if page_number is not None:
                            line["page"] = page_number
                        extracted.append(line)
                except (IndexError, TypeError):
                    continue

        return extracted, ocr_image_size

    @handle_exceptions(Exception, raise_as=OCREngineError)
    def _run_ocr(self, path: Path) -> Tuple[List[dict], Optional[Dict[str, int]]]:
        """Exécute PaddleOCR (images ou PDF multi-pages) et retourne les résultats nettoyés."""
        raw_results = self._ocr.ocr(str(path))

        if not raw_results:
            return [], None

        extracted: List[dict] = []
        ocr_image_size = None

        for page in raw_results:
            page_lines, page_size = self._extract_page_results(page)
            extracted.extend(page_lines)
            if ocr_image_size is None and page_size is not None:
                ocr_image_size = page_size

        return extracted, ocr_image_size

    @handle_exceptions(Exception, raise_as=OCREngineError)
    def _run_ocr_selected_pages(
        self,
        path: Path,
        pages: List[int],
    ) -> Tuple[List[dict], Dict[str, Any]]:
        """OCRise uniquement les pages sélectionnées d'un PDF via PyMuPDF."""
        try:
            import fitz
        except ImportError as exc:
            raise OCREngineError("PyMuPDF (fitz) requis pour l'OCR sélectif de pages.") from exc

        doc = fitz.open(str(path))
        total_pages = len(doc)
        zoom = 2.0
        matrix = fitz.Matrix(zoom, zoom)

        extracted: List[dict] = []
        image_sizes: Dict[str, Dict[str, int]] = {}

        try:
            for page_number in pages:
                page_index = page_number - 1
                if page_index < 0 or page_index >= total_pages:
                    raise ValueError(
                        f"Page {page_number} invalide (document de {total_pages} page(s))."
                    )

                page = doc[page_index]
                pixmap = page.get_pixmap(matrix=matrix)
                width = pixmap.width
                height = pixmap.height
                image_sizes[str(page_number)] = {"width": width, "height": height}

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    pixmap.save(tmp_path)

                try:
                    raw_results = self._ocr.ocr(tmp_path)
                    if raw_results:
                        for page_result in raw_results:
                            page_lines, page_size = self._extract_page_results(
                                page_result, page_number=page_number
                            )
                            extracted.extend(page_lines)
                            if page_size is not None:
                                image_sizes[str(page_number)] = page_size
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
        finally:
            doc.close()

        primary_page = pages[0]
        primary_size = image_sizes.get(str(primary_page), {"width": 0, "height": 0})

        return extracted, {
            "width": primary_size["width"],
            "height": primary_size["height"],
            "image_sizes": image_sizes,
            "primary_page": primary_page,
        }

