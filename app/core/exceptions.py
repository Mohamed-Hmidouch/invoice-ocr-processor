"""
Exceptions métier personnalisées pour le processeur de factures.

Hiérarchie claire : toutes les exceptions héritent d'une base commune,
ce qui permet un filtrage granulaire ou global selon le besoin.
"""


class InvoiceProcessorError(Exception):
    """Exception racine du projet. Toutes les erreurs métier en héritent."""

    def __init__(self, message: str = "", detail: str = ""):
        self.detail = detail
        super().__init__(message)


class OCREngineError(InvoiceProcessorError):
    """Erreur liée à l'initialisation ou à l'exécution du moteur OCR."""


class ImageNotFoundError(OCREngineError):
    """Le fichier image spécifié est introuvable ou illisible."""


class UnsupportedImageFormatError(OCREngineError):
    """Le format de l'image n'est pas supporté par le moteur OCR."""


class ExtractionError(InvoiceProcessorError):
    """Erreur survenue lors de l'extraction des données (RegEx / parsing)."""


class FileManagerError(InvoiceProcessorError):
    """Erreur liée à la gestion des fichiers (lecture, écriture, déplacement)."""

class SecurityValidationError(InvoiceProcessorError):
    """Exception levée pour des raisons de sécurité (trop gros, mauvais MIME, Path Traversal)."""


class DatabaseError(InvoiceProcessorError):
    """Erreur liée à la persistance PostgreSQL (connexion, insertion, transaction)."""
