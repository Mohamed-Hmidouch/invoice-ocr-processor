"""
Package app — Processeur OCR de factures.

Façade publique du package. Centralise les exports pour
que main.py puisse importer tout depuis un seul endroit :

    from app import OCREngine, Extractor, FileManager, DatabaseManager, Invoice

Principe : le consommateur (main.py) ne connaît PAS la structure
interne des sous-modules. Si on réorganise les fichiers demain,
main.py ne change pas.
"""
from app.core import OCREngine, Extractor
from app.models import Invoice, InvoiceItem
from app.utils import FileManager
from app.database import DatabaseManager
from app.core.exceptions import InvoiceProcessorError

__all__ = [
    "OCREngine",
    "Extractor",
    "FileManager",
    "DatabaseManager",
    "Invoice",
    "InvoiceItem",
    "InvoiceProcessorError",
]