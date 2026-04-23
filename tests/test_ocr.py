"""
Tests unitaires pour composants centraux de l'Invoice Processor.

Approche:
- Tests isolés via l'utilisation intensive des Mocks.
- Aucun appel réseau réel au LLM (Gemini).
- Zéro chargement lourd de PaddleOCR durant les tests.
"""
import pytest
from unittest.mock import patch, MagicMock
from decimal import Decimal
from datetime import date

from app.models.invoice import Invoice, InvoiceItem
from app.core.exceptions import ExtractionError
from app.core.extractor import Extractor

# ═══════════════════════════════════════════════════════════════════════════════
# TESTS AGENTIC EXTRACTOR (GEMINI)
# ═══════════════════════════════════════════════════════════════════════════════

def test_extractor_missing_api_key(monkeypatch):
    """Vérifie que l'extracteur crash net au démarrage si la clé API manque."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ExtractionError, match="GEMINI_API_KEY n'est pas définie"):
        Extractor()

@patch("app.core.extractor.genai.GenerativeModel")
def test_agentic_extractor_success_mapping(mock_model_class, monkeypatch):
    """
    Simule une réponse LLM parfaite incluant des données de Freight/Douane
    afin de vérifier le parsing vers l'objet `Invoice`.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_for_test")
    
    # Prépare le Mock
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance
    
    # Construit la fausse réponse de l'Agent LLM (avec markdown simulé)
    mock_response = MagicMock()
    mock_response.text = '''```json
    {
        "invoice_number": "INV-100",
        "date": "2024-01-15",
        "supplier_name": "TECH CORP",
        "supplier_tax_id": "ICE123",
        "destinataire": "CLIENT RECEIVER",
        "incoterm": "FOB",
        "amounts": {
            "total_excl_tax": 1000.0,
            "tax_amount": 200.0,
            "total_incl_tax": 1200.0
        },
        "currency": "MAD",
        "items": [
            {
                "description": "Laptop Pro",
                "quantity": 2,
                "unit_price": 500.0,
                "total_price": 1000.0
            }
        ],
        "confidence_score": 0.99
    }
    ```'''
    mock_instance.generate_content.return_value = mock_response

    # Exécute
    extractor = Extractor()
    dummy_ocr_text_stream = [("Invoice TECH CORP INV-100", 0.9)]
    invoice = extractor.extract(dummy_ocr_text_stream)

    # ── Assertions Majeures ──
    assert isinstance(invoice, Invoice)
    assert invoice.invoice_number == "INV-100"
    assert invoice.date == date(2024, 1, 15)
    assert invoice.supplier_name == "TECH CORP"
    
    # Champs Internationaux
    assert invoice.destinataire == "CLIENT RECEIVER"
    assert invoice.incoterm == "FOB"
    assert invoice.port is None  # Omitted by LLM -> mapped to None
    
    # Typage métier strict
    assert invoice.total_amount_incl_tax == Decimal("1200.0")
    assert invoice.currency == "MAD"
    
    # Items
    assert len(invoice.items) == 1
    assert isinstance(invoice.items[0], InvoiceItem)
    assert invoice.items[0].description == "Laptop Pro"

@patch("app.core.extractor.genai.GenerativeModel")
def test_agentic_extractor_tolerate_missing_fields(mock_model_class, monkeypatch):
    """
    Test la résilience si l'Agent se plie à la règle de : 
    "NEUTRALISER (omettre) LES CLÉS MANQUANTES".
    """
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_for_test")
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance
    
    # JSON incomplet
    mock_response = MagicMock()
    mock_response.text = '{"supplier_name": "MINIMALIST VENDOR"}'
    mock_instance.generate_content.return_value = mock_response

    extractor = Extractor()
    invoice = extractor.extract([("MINIMALIST VENDOR", 0.9)])

    # Tout doit être None sauf le fournisseur
    assert invoice.supplier_name == "MINIMALIST VENDOR"
    assert invoice.invoice_number is None
    assert invoice.total_amount_incl_tax is None
    assert invoice.destinataire is None
    assert invoice.port is None
    assert len(invoice.items) == 0
