"""
Module models — Structures de données du projet.

Exporte les data classes publiques :
    from app.models import Invoice, InvoiceItem
"""
from app.models.invoice import Invoice, InvoiceItem

__all__ = ["Invoice", "InvoiceItem"]
