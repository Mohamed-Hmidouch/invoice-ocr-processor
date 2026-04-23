"""
Extracteur Agentique (LLM-based) de données structurées.

Responsabilités :
    - Recevoir le texte brut de l'OCR.
    - Contacter un LLM (e.g., Gemini) avec un "Master Prompt" stict pour 
      l'extraction contextuelle et la validation mathématique.
    - Convertir le JSON renvoyé par le LLM en un objet de domaine `Invoice`.
"""
import os
import json
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import List, Tuple

import google.generativeai as genai

from app.core.aspect import handle_exceptions
from app.core.exceptions import ExtractionError
from app.models.invoice import Invoice, InvoiceItem

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT INVOICE AGENT
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """
**System Role:**
You are an elite Global Document Intelligence Agent. Your mission is to extract highly accurate, structured data from raw, noisy OCR text of any invoice type (Standard, Commercial, Freight, International, Proforma).

**Core Directives & Logic:**
1. **Adaptive Extraction:** Scan the document for standard billing details AND international trade details (Consignee, Buyer, Port, Transport, Incoterms).
2. **OMIT MISSING DATA (STRICT RULE):** If a specific piece of information is NOT explicitly found or confidently deduced from the text, DO NOT include its key in the JSON output. Never output `null` values; simply omit the key entirely.
3. **Math & Logic Validation:** Cross-check `Subtotal + Tax = Grand Total`. If numbers are misread by OCR (e.g., '0' instead of 'O'), use math to deduce the correct value.
4. **Data Formatting:**
   - Dates: Strict `YYYY-MM-DD` format.
   - Amounts: Strict float format (e.g., `1500.50`). Strip all spaces, letters, and currency symbols.
   - Currency: Deduce the 3-letter ISO code (e.g., MAD, XOF, EUR, USD).

**Target JSON Schema (Use keys ONLY if data is present):**
{
  "invoice_number": "string",
  "date": "YYYY-MM-DD",
  "supplier_name": "string (Exporter / Seller / Biller)",
  "supplier_tax_id": "string (ICE, VAT, SIRET, RC)",
  "destinataire": "string (Consignee / Entity receiving goods)",
  "importateur": "string (Buyer / Acheteur)",
  "port": "string (Port of loading/discharge)",
  "moyen_transport": "string (Transport method / Vessel / Carrier)",
  "incoterm": "string (e.g., FOB, CIF, EXW)",
  "amounts": {
    "total_excl_tax": "float (HT / Subtotal)",
    "tax_amount": "float (TVA / Tax)",
    "total_incl_tax": "float (TTC / Grand Total)"
  },
  "currency": "string",
  "items": [
    {
      "description": "string",
      "quantity": "float",
      "unit_price": "float",
      "total_price": "float"
    }
  ],
  "confidence_score": "float (0.0 to 1.0)"
}

**Output format:** Return pure JSON only. Do not use markdown formatting blocks (no ```json). do it for any fuckin,g type of facture
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE EXTRACTOR (AGENTIC)
# ═══════════════════════════════════════════════════════════════════════════════

class Extractor:
    """
    Agentic Extractor propulsé par LLM (Gemini).
    """

    def __init__(self):
        """Initialise le modèle LLM à partir de la clé API présente dans l'environnement."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ExtractionError("La clé GEMINI_API_KEY n'est pas définie dans l'environnement.")
        
        genai.configure(api_key=api_key)
        
        # Configuration stricte pour outputer systématiquement du JSON
        generative_config = genai.types.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json"
        )
        
        self.model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=_SYSTEM_PROMPT,
            generation_config=generative_config
        )

    # ── API publique ────────────────────────────────────────────────────────

    @handle_exceptions(Exception, raise_as=ExtractionError)
    def extract(self, ocr_results: List[Tuple[str, float]]) -> Invoice:
        """
        Passe le texte OCR à l'Agent et mappe le JSON retourné vers l'objet Invoice.
        """
        raw_text = self._build_raw_text(ocr_results)
        
        prompt_context = (
            f"Analyze this raw OCR text stream and extract the required fields as specified:\n\n{raw_text}"
        )
        
        # Appel LLM (Synchrone)
        response = self.model.generate_content(prompt_context)
        llm_json = self._parse_json(response.text)
        
        return self._map_to_invoice(llm_json)

    # ── Méthodes privées ────────────────────────────────────────────────────

    @staticmethod
    def _build_raw_text(ocr_results: List[Tuple[str, float]]) -> str:
        """Concatène tous les blocs OCR en un seul texte."""
        return "\n".join(text for text, _ in ocr_results)

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Nettoie le texte (au cas où il resterait des balises markdown) et charge le JSON."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())

    @staticmethod
    def _parse_decimal(value) -> Decimal:
        """Gère les valeurs qui viennent du JSON."""
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None

    def _map_to_invoice(self, data: dict) -> Invoice:
        """Convertit le dictionnaire JSON LLM vers l'objet Invoice natif du projet."""
        
        # Parse la Date
        parsed_date = None
        raw_date = data.get("date")
        if raw_date:
            try:
                parsed_date = date.fromisoformat(raw_date)
            except ValueError:
                pass
                
        # Construit les items
        items = []
        for it in data.get("items", []):
            items.append(InvoiceItem(
                description=it.get("description", ""),
                quantity=float(it.get("quantity", 0.0) or 0.0),
                unit_price=self._parse_decimal(it.get("unit_price")) or Decimal('0.0'),
                total_price=self._parse_decimal(it.get("total_price")) or Decimal('0.0')
            ))
            
        # Extrait les montants du sous-objet
        amounts = data.get("amounts", {})
        
        return Invoice(
            invoice_number=data.get("invoice_number"),
            date=parsed_date,
            supplier_name=data.get("supplier_name"),
            supplier_tax_id=data.get("supplier_tax_id"),
            destinataire=data.get("destinataire"),
            importateur=data.get("importateur"),
            port=data.get("port"),
            moyen_transport=data.get("moyen_transport"),
            incoterm=data.get("incoterm"),
            total_amount_excl_tax=self._parse_decimal(amounts.get("total_excl_tax")),
            tax_amount=self._parse_decimal(amounts.get("tax_amount")),
            total_amount_incl_tax=self._parse_decimal(amounts.get("total_incl_tax")),
            currency=data.get("currency"),
            items=items,
            confidence_score=float(data.get("confidence_score") or 0.0),
        )
