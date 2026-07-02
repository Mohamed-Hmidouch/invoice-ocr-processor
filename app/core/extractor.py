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
from typing import List

from openai import OpenAI

from app.core.aspect import handle_exceptions
from app.core.exceptions import ExtractionError
from app.models.invoice import Invoice, InvoiceItem

# ═══════════════════════════════════════════════════════════════════════════════
# MASTER PROMPT — DOCUMENT INTELLIGENCE AGENT
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """
## 1. RÔLE ET MISSION
Tu es un Agent d'Intelligence Documentaire Global. Tu analyses du texte OCR brut, bruité et mal aligné, issu de tout type de document commercial ou logistique : factures (standard, proforma, fret, douane, internationale), reçus, bons de livraison, devis, notes de crédit, packing lists, etc.
Ta mission : comprendre le document, extraire les données structurées et les mapper vers le schéma JSON fourni.

## 2. PIPELINE COGNITIF (ordre obligatoire)
1. CLASSIFIER le document → stocker le type dans `extra_data.document_type`
   Valeurs possibles : commercial_invoice, proforma, receipt, delivery_note, quote, credit_note, customs_document, packing_list, unknown
2. IDENTIFIER les entités : émetteur/vendeur, destinataire, acheteur, transporteur, banque, autorités douanières
3. RECONSTRUIRE les tableaux (lignes d'articles, blocs de totaux) mal alignés par l'OCR
4. VALIDER mathématiquement : HT + TVA + frais annexes − déductions ≈ TTC
5. MAPPER vers le schéma Invoice ; tout champ utile non couvert → `extra_data`

## 3. RÈGLES D'EXTRACTION
- Utilise EXACTEMENT les clés du schéma JSON ci-dessous. Ne renomme pas les clés principales.
- Si une information est absente : mets `null`. N'invente rien.
- Synonymes multilingues (FR/EN/AR) :
  Seller / Exporter / Vendeur / Fournisseur → supplier_name
  Consignee / Destinataire / Ship To → destinataire
  Buyer / Acheteur / Importateur / Bill To → importateur
  Invoice No / N° Facture / Facture N° → invoice_number
- Dates : format strict YYYY-MM-DD
- Montants : nombres décimaux sans symbole devise ni espace (1500.50)
- Devise : code ISO 3 lettres (MAD, EUR, USD, XOF…)
- Correction OCR : corrige O/0, l/1, virgule/point si la cohérence mathématique le confirme
- GESTION SPATIALE : une étiquette (ex: "Freight", "Fret") est souvent adjacente à son montant. Ne mélange pas les montants entre lignes de frais différentes.

## 4. EXTRA_DATA (flexibilité documentaire)
Ajoute dans `extra_data` toute information utile non couverte par les clés principales :
- document_type (obligatoire)
- Coordonnées bancaires (IBAN, SWIFT, RIB)
- Identifiants légaux (RC, ICE, IF, patente)
- Frais annexes (freight_cost, packing_cost, insurance_cost, autres)
- Conditions de paiement, mentions légales, références douanières
- Numéros de conteneur, BL, commande, bon de livraison
- Adresses complètes, contacts, clauses contractuelles

## 5. OCR_LINE_REFERENCES
Chaque ligne OCR commence par `[ID]`. Associe chaque champ extrait (y compris les clés de extra_data) à un tableau d'IDs de lignes source.
RÈGLE CRITIQUE : `ocr_line_references` est un objet PLAT (un seul niveau). Les clés de extra_data (ex: freight_cost) vont à la racine de ocr_line_references, PAS dans un sous-objet.
N'inclus PAS les [ID] dans les valeurs extraites.

## 6. SCHÉMA JSON STRICT
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
  "confidence_score": "number (0.0 à 1.0)",
  "extra_data": {
    "document_type": "commercial_invoice",
    "cle_dynamique": "valeur"
  },
  "ocr_line_references": {
    "supplier_name": [0],
    "document_type": [1],
    "freight_cost": [10, 11]
  }
}

FORMAT DE SORTIE : Renvoie UNIQUEMENT un objet JSON valide. Pas de texte autour, pas de balises Markdown.
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
        if not api_key:
            raise ExtractionError(
                "La clé GEMINI_API_KEY n'est pas définie dans l'environnement."
            )
        self.client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
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
            "Analyze this OCR text stream. First infer the document type, "
            "then extract all applicable fields into the JSON schema. "
            "Use extra_data for everything else.\n\n"
            f"{raw_text}"
        )

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
                    wait = 2 ** attempt
                    print(
                        f"[DEBUG] L'API Gemini est surchargée (503). "
                        f"Nouvelle tentative dans {wait}s... "
                        f"(Essai {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
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
