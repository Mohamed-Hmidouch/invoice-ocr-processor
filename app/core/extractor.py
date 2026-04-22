"""
Extracteur de données structurées à partir du texte brut OCR.

Responsabilités :
    - Nettoyer le texte brut issu de PaddleOCR.
    - Extraire les champs clés d'une facture via des expressions régulières.
    - Construire et retourner un objet Invoice peuplé.
    - Déléguer la gestion d'erreurs aux aspects AOP (zéro try/except ici).

Design :
    - Les regex sont compilées au chargement du module (performance).
    - Stockées dans un dict → ajouter un champ = ajouter une entrée (Open/Closed).
    - Chaque méthode d'extraction est statique et pure (Single Responsibility).
"""
import re
from decimal import Decimal, InvalidOperation
from datetime import date
from typing import List, Tuple, Optional

from app.core.aspect import handle_exceptions
from app.core.exceptions import ExtractionError
from app.models.invoice import Invoice, InvoiceItem


# ═══════════════════════════════════════════════════════════════════════════════
# REGEX COMPILÉES — Chargées UNE SEULE FOIS au import du module
# ═══════════════════════════════════════════════════════════════════════════════

_PATTERNS = {
    "invoice_number": re.compile(
        r"(?:facture|invoice|fact|n[°o])\s*[:#]?\s*([A-Z0-9][\w\-/]{2,})",
        re.IGNORECASE,
    ),
    "date": re.compile(
        r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})"
    ),
    "total_ttc": re.compile(
        r"(?:total\s*(?:ttc|t\.t\.c|général|general|à\s*payer|net))"
        r"\s*[:#]?\s*([\d\s]+[.,]\d{2})",
        re.IGNORECASE,
    ),
    "total_ht": re.compile(
        r"(?:total\s*(?:ht|h\.t|hors\s*taxe))"
        r"\s*[:#]?\s*([\d\s]+[.,]\d{2})",
        re.IGNORECASE,
    ),
    "tva": re.compile(
        r"(?:tva|t\.v\.a|taxe)"
        r"\s*[:#]?\s*([\d\s]+[.,]\d{2})",
        re.IGNORECASE,
    ),
    "tax_id": re.compile(
        r"(?:ice|siret|siren|tva\s*intra|n[°o]\s*(?:identification|id))"
        r"\s*[:#]?\s*([\dA-Z]{8,})",
        re.IGNORECASE,
    ),
}

_ITEM_LINE_PATTERN = re.compile(
    r"^(.{5,50}?)\s+"             # Description (5-50 chars)
    r"(\d+(?:[.,]\d+)?)\s+"       # Quantité
    r"([\d\s]+[.,]\d{2})\s+"      # Prix unitaire
    r"([\d\s]+[.,]\d{2})$",       # Prix total ligne
    re.MULTILINE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE EXTRACTEUR
# ═══════════════════════════════════════════════════════════════════════════════

class Extractor:
    """
    Extrait les données structurées d'une facture à partir des résultats OCR.

    Utilisation :
        extractor = Extractor()
        invoice = extractor.extract(ocr_results)
        print(invoice.invoice_number, invoice.total_amount_incl_tax)
    """

    # ── API publique ────────────────────────────────────────────────────────

    @handle_exceptions(Exception, raise_as=ExtractionError)
    def extract(self, ocr_results: List[Tuple[str, float]]) -> Invoice:
        """
        Pipeline complet : résultats OCR bruts → objet Invoice structuré.

        Paramètres
        ----------
        ocr_results : List[Tuple[str, float]]
            Sortie de OCREngine.read() → [(texte_détecté, score_confiance), ...].

        Retour
        ------
        Invoice
            Objet facture avec tous les champs peuplés (ou None si non détectés).

        Lève
        ----
        ExtractionError
            En cas d'erreur durant le parsing (via AOP, jamais de try/except ici).
        """
        raw_text = self._build_raw_text(ocr_results)
        avg_confidence = self._compute_confidence(ocr_results)

        return Invoice(
            invoice_number=self._extract_invoice_number(raw_text),
            date=self._extract_date(raw_text),
            supplier_name=self._guess_supplier_name(ocr_results),
            supplier_tax_id=self._extract_tax_id(raw_text),
            total_amount_excl_tax=self._extract_amount(raw_text, "total_ht"),
            tax_amount=self._extract_amount(raw_text, "tva"),
            total_amount_incl_tax=self._extract_amount(raw_text, "total_ttc"),
            items=self._extract_items(raw_text),
            confidence_score=avg_confidence,
        )

    # ── Construction du texte ───────────────────────────────────────────────

    @staticmethod
    def _build_raw_text(ocr_results: List[Tuple[str, float]]) -> str:
        """Concatène tous les blocs OCR en un seul texte navigable par les regex."""
        return "\n".join(text for text, _ in ocr_results)

    @staticmethod
    def _compute_confidence(ocr_results: List[Tuple[str, float]]) -> float:
        """Calcule le score de confiance moyen de l'ensemble des résultats OCR."""
        if not ocr_results:
            return 0.0
        scores = [score for _, score in ocr_results]
        return round(sum(scores) / len(scores), 4)

    # ── Extraction des champs individuels ───────────────────────────────────

    @staticmethod
    def _extract_invoice_number(text: str) -> Optional[str]:
        """Extrait le numéro de facture (ex: F-2024-001, INV/2024/042)."""
        match = _PATTERNS["invoice_number"].search(text)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_date(text: str) -> Optional[date]:
        """
        Extrait la date au format JJ/MM/AAAA (ou variantes avec - ou .).

        Gère les années sur 2 chiffres (ex: 24 → 2024).
        """
        match = _PATTERNS["date"].search(text)
        if not match:
            return None

        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))

        # Normalisation : année sur 2 chiffres → 4 chiffres
        if year < 100:
            year += 2000

        # Validation via le constructeur date (plutôt qu'un if/else fragile)
        try:
            return date(year, month, day)
        except ValueError:
            return None

    @staticmethod
    def _extract_tax_id(text: str) -> Optional[str]:
        """Extrait l'identifiant fiscal (ICE, SIRET, SIREN, TVA intra, etc.)."""
        match = _PATTERNS["tax_id"].search(text)
        return match.group(1).strip() if match else None

    # ── Extraction des montants ─────────────────────────────────────────────

    @staticmethod
    def _parse_decimal(raw_value: str) -> Optional[Decimal]:
        """
        Convertit une chaîne brute en Decimal de manière sûre.

        Gère les formats français : "1 200,50" → Decimal("1200.50").
        """
        cleaned = raw_value.replace(" ", "").replace(",", ".")
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None

    def _extract_amount(self, text: str, pattern_key: str) -> Optional[Decimal]:
        """
        Extrait un montant monétaire selon la clé de pattern fournie.

        Paramètres
        ----------
        text : str
            Texte brut OCR.
        pattern_key : str
            Clé dans _PATTERNS ("total_ht", "tva", "total_ttc").
        """
        match = _PATTERNS[pattern_key].search(text)
        if not match:
            return None
        return self._parse_decimal(match.group(1))

    # ── Extraction des lignes d'articles ────────────────────────────────────

    def _extract_items(self, text: str) -> List[InvoiceItem]:
        """
        Extrait les lignes d'articles (description, qté, PU, total).

        Le pattern attend un format tabulaire classique :
            "Câble HDMI 2m    3    15,00    45,00"
        """
        items = []

        for match in _ITEM_LINE_PATTERN.finditer(text):
            description, raw_qty, raw_unit_price, raw_total = match.groups()

            quantity = float(raw_qty.replace(",", "."))
            unit_price = self._parse_decimal(raw_unit_price)
            total_price = self._parse_decimal(raw_total)

            # On ne crée l'item que si les montants sont valides
            if unit_price is not None and total_price is not None:
                items.append(InvoiceItem(
                    description=description.strip(),
                    quantity=quantity,
                    unit_price=unit_price,
                    total_price=total_price,
                ))

        return items

    # ── Heuristiques ────────────────────────────────────────────────────────

    @staticmethod
    def _guess_supplier_name(ocr_results: List[Tuple[str, float]]) -> Optional[str]:
        """
        Heuristique : le nom du fournisseur est souvent dans les premières
        lignes en haut de la facture.

        On prend la première ligne non-numérique de plus de 3 caractères
        parmi les 5 premiers résultats OCR.
        """
        for text, _ in ocr_results[:5]:
            cleaned = text.strip()
            if len(cleaned) > 3 and not cleaned.replace(" ", "").isdigit():
                return cleaned
        return None
