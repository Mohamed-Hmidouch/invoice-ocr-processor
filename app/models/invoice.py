"""
Data Class pour structurer une facture.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from decimal import Decimal
from datetime import date

@dataclass
class InvoiceItem:
    """Représente une ligne (un article) sur la facture."""
    description: str
    quantity: float
    unit_price: Decimal
    total_price: Decimal
    tax_rate: Optional[Decimal] = None


@dataclass
class Invoice:
    """
    Représente les données extraites d'une facture.
    
    L'utilisation de `Optional` est primordiale car l'extraction (OCR + RegEx) 
    n'est pas garantie à 100%. Certains champs pourraient être introuvables.
    """
    invoice_number: Optional[str] = None
    date: Optional[date] = None
    
    # Fournisseur
    supplier_name: Optional[str] = None
    supplier_tax_id: Optional[str] = None  # SIRET, ICE, N° TVA, etc.
    
    # Montants (L'utilisation de Decimal est recommandée pour les devises)
    total_amount_excl_tax: Optional[Decimal] = None  # HT
    tax_amount: Optional[Decimal] = None             # TVA
    total_amount_incl_tax: Optional[Decimal] = None  # TTC
    
    # Lignes de la facture avec une liste vide par défaut (bonne pratique dataclasses)
    items: List[InvoiceItem] = field(default_factory=list)
    
    # Score de confiance global de l'extraction (pour juger s'il faut une revue humaine)
    confidence_score: float = 0.0
    
    def is_valid(self) -> bool:
        """
        Vérifie la validité basique des données obligatoires selon les règles métiers.
        """
        return bool(self.invoice_number and self.total_amount_incl_tax and self.supplier_name)
