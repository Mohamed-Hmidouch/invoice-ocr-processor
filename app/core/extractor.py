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
import time
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import List, Tuple

from openai import OpenAI

from app.core.aspect import handle_exceptions
from app.core.exceptions import ExtractionError
from app.models.invoice import Invoice, InvoiceItem

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT INVOICE AGENT
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """
Tu es un expert en extraction de données OCR. Ta mission est de lire le texte brut suivant et de le convertir STRICTEMENT selon le format JSON ci-dessous.

RÈGLES STRICTES :
1. RESPECT DU SCHÉMA : Tu dois utiliser EXACTEMENT les clés fournies dans l'exemple JSON. Ne modifie pas les noms des clés principales.
2. DONNÉES MANQUANTES : Si une information n'est pas présente dans le texte, mets la valeur à null. N'invente rien.
3. LE CHAMP EXTRA_DATA (IMPORTANT) : Si tu trouves des informations importantes dans la facture (comme les coordonnées bancaires, clauses, conditions de paiement, numéros de RC/ICE, incoterms, adresses complètes, contacts) qui n'ont pas de place dans les clés principales, tu DOIS les ajouter sous forme de paires clé/valeur à l'intérieur de l'objet extra_data.
4. GESTION DES TABLEAUX : Reconstruis intelligemment les lignes de facturation. Aligne correctement les descriptions, quantités, prix unitaires et montants, même si l'OCR les a décalés.
5. CONFIDENCE SCORE : Mets toujours 1.0 si l'extraction s'est bien passée.
6. FORMAT DE SORTIE : Renvoie UNIQUEMENT un objet JSON valide, sans aucun texte autour, sans balises Markdown.
7. REFERENCES LIGNES OCR : Chaque ligne de texte brut commence par un identifiant `[ID]`. Dans ta réponse JSON, ajoute une clé `ocr_line_references` qui associe le nom exact de chaque clé extraite à un tableau contenant l'identifiant (ID) ou les identifiants des lignes d'où tu as tiré la valeur. N'inclus PAS les `[ID]` dans les valeurs extraites elles-mêmes. Par exemple: `"ocr_line_references": {"supplier_name": [0], "total_amount_incl_tax": [45, 46]}`. Ne renvoie AUCUNE coordonnée (x, y), seulement l'ID de la ligne.

FORMAT JSON STRICT À RESPECTER :
{
  "invoice_number": "string | null",
  "date": "YYYY-MM-DD | null",
  "supplier_name": "string | null",
  "supplier_tax_id": "string | null",
  "destinataire": "string | null",
  "importateur": "string | null",
  "port": "string | null",
  "moyen_transport": "string | null",
  "incoterm": "string | null",
  "total_amount_excl_tax": "number | null",
  "tax_amount": "number | null",
  "total_amount_incl_tax": "number | null",
  "items": [
    {
      "description": "string",
      "quantity": "number | null",
      "unit_price": "number | null",
      "total_price": "number | null"
    }
  ],
  "currency": "string | null",
  "confidence_score": 1.0,
  "extra_data": {
    "cle_dynamique": "toute donnée utile non couverte par les clés principales (banque, ICE, RC, IBAN, SWIFT, adresse, conditions paiement, mentions légales, etc.)"
  },
  "ocr_line_references": {
    "nom_du_champ_principal_ou_extra": [0]
  }
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE EXTRACTOR (AGENTIC)
# ═══════════════════════════════════════════════════════════════════════════════

class Extractor:
    """
    Agentic Extractor propulsé par LLM (Gemini via API Google).
    """

    def __init__(self):
        """Initialise le client OpenAI pour pointer vers l'API Gemini de Google."""
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key
        )
        self.model_name = "gemini-2.5-flash"

    # ── API publique ────────────────────────────────────────────────────────

    @handle_exceptions(Exception, raise_as=ExtractionError)
    def extract(self, ocr_results: List[dict]) -> Invoice:
        """
        Passe le texte OCR à l'Agent et mappe le JSON retourné vers l'objet Invoice.
        """
        raw_text = self._build_raw_text(ocr_results)
        
        prompt_context = (
            f"Analyze this raw OCR text stream and extract the required fields as specified:\n\n{raw_text}"
        )
        
        # Appel LLM (Synchrone) avec retry
        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_context}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                break
            except Exception as e:
                if "503" in str(e) and attempt < max_retries - 1:
                    print(f"[DEBUG] L'API Gemini est surchargée (503). Nouvelle tentative dans {2 ** attempt}s... (Essai {attempt + 1}/{max_retries})")
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        llm_json = self._parse_json(response.choices[0].message.content)
        
        return self._map_to_invoice(llm_json)

    # ── Méthodes privées ────────────────────────────────────────────────────

    @staticmethod
    def _build_raw_text(ocr_results: List[dict]) -> str:
        """Concatène tous les blocs OCR en un seul texte avec leur ID de ligne."""
        return "\n".join(f"[{i}] {res['text']}" for i, res in enumerate(ocr_results))

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
        """Convertit le dictionnaire JSON LLM (schéma fixe) vers l'objet Invoice."""

        # ── Date ───────────────────────────────────────────────────────────
        parsed_date = None
        raw_date = data.get("date")
        if raw_date:
            try:
                parsed_date = date.fromisoformat(str(raw_date))
            except ValueError:
                pass

        # ── Line items ─────────────────────────────────────────────────────
        items = []
        for it in (data.get("items") or []):
            items.append(InvoiceItem(
                description=it.get("description", ""),
                quantity=float(it.get("quantity") or 0.0),
                unit_price=self._parse_decimal(it.get("unit_price")) or Decimal("0.0"),
                total_price=self._parse_decimal(it.get("total_price")) or Decimal("0.0"),
            ))

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
            total_amount_excl_tax=self._parse_decimal(data.get("total_amount_excl_tax")),
            tax_amount=self._parse_decimal(data.get("tax_amount")),
            total_amount_incl_tax=self._parse_decimal(data.get("total_amount_incl_tax")),
            currency=data.get("currency"),
            items=items,
            confidence_score=float(data.get("confidence_score") or 0.0),
            extra_data=data.get("extra_data") or {},
            ocr_line_references=data.get("ocr_line_references") or {},
        )
