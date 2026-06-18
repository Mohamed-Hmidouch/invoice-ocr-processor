"""
Gestion des dossiers d'entrée / sortie pour le pipeline de factures.

Responsabilités :
    - Créer automatiquement l'arborescence nécessaire au démarrage.
    - Lister les fichiers images en attente de traitement.
    - Déplacer les fichiers traités vers un dossier d'archivage.
    - Exporter les résultats d'extraction au format JSON.
    - Déléguer la gestion d'erreurs aux aspects AOP (zéro try/except ici).

Design :
    - Les attributs internes sont encapsulés (préfixe _) et exposés via @property.
    - L'encodeur JSON custom gère nativement Decimal et date (pas de hack).
    - Les extensions supportées sont importées de ocr_engine (DRY).
"""
import json
import shutil
import logging
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import List

from app.core.aspect import handle_exceptions
from app.core.exceptions import FileManagerError
from app.core.ocr_engine import _SUPPORTED_FORMATS
from app.models.invoice import Invoice
from app.utils.security import FileValidator


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODEUR JSON CUSTOM
# ═══════════════════════════════════════════════════════════════════════════════

class _InvoiceJSONEncoder(json.JSONEncoder):
    """
    Encodeur JSON personnalisé pour sérialiser les types non-natifs.

    Gère :
        - Decimal  → float  (précision financière préservée à l'affichage)
        - date     → str    (format ISO 8601 : "2024-03-15")
        - datetime → str    (format ISO 8601 : "2024-03-15T14:30:00")
    """

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE FILE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class FileManager:
    """
    Gestionnaire de fichiers pour le pipeline OCR.

    Structure des dossiers créée automatiquement :
        input_dir/              ← Factures à traiter (images)
        output_dir/
            ├── json/           ← Résultats d'extraction (JSON)
            └── processed/      ← Images traitées (archivage)

    Utilisation :
        fm = FileManager()
        for image in fm.get_pending_files():
            ...
            fm.save_result(invoice, image.name)
            fm.move_to_processed(image)
    """

    def __init__(
        self,
        input_dir: str = "data/input",
        output_dir: str = "data/output",
    ):
        self._input_dir = Path(input_dir)
        self._output_dir = Path(output_dir)
        self._json_dir = self._output_dir / "json"
        self._processed_dir = self._output_dir / "processed"

        self._ensure_directories()

    # ── Propriétés publiques (lecture seule) ─────────────────────────────────

    @property
    def input_dir(self) -> Path:
        """Chemin du dossier d'entrée."""
        return self._input_dir

    @property
    def output_dir(self) -> Path:
        """Chemin du dossier de sortie."""
        return self._output_dir

    # ── API publique ────────────────────────────────────────────────────────

    @handle_exceptions(Exception, raise_as=FileManagerError)
    def get_pending_files(self) -> List[Path]:
        """
        Liste les fichiers images en attente dans le dossier d'entrée après validation Security.
        """
        valid_files = []
        for path in self._input_dir.iterdir():
            if path.is_file() and path.suffix.lower() in _SUPPORTED_FORMATS:
                try:
                    safe_path = FileValidator.validate_and_sanitize(path)
                    valid_files.append(safe_path)
                except Exception as e:
                    logging.warning(f"SECURITY ALERT - Fichier bloqué ({path}): {e}")
        return sorted(valid_files)

    @handle_exceptions(Exception, raise_as=FileManagerError)
    def move_to_processed(self, file_path: Path) -> Path:
        """
        Déplace un fichier traité vers le dossier d'archivage « processed ».

        Paramètres
        ----------
        file_path : Path
            Chemin du fichier image à archiver.

        Retour
        ------
        Path
            Nouveau chemin du fichier déplacé.
        """
        destination = self._processed_dir / file_path.name
        shutil.move(str(file_path), str(destination))
        return destination

    @handle_exceptions(Exception, raise_as=FileManagerError)
    def save_result(self, invoice: Invoice, source_filename: str) -> Path:
        """
        Exporte une facture extraite au format JSON.

        Le fichier JSON porte le même nom que l'image source
        (ex: facture_001.png → facture_001.json).

        Paramètres
        ----------
        invoice : Invoice
            L'objet facture à sérialiser.
        source_filename : str
            Nom du fichier source (utilisé pour nommer le JSON).

        Retour
        ------
        Path
            Chemin du fichier JSON créé.
        """
        stem = Path(source_filename).stem
        json_path = self._json_dir / f"{stem}.json"

        output = asdict(invoice)

        json_path.write_text(
            json.dumps(output, cls=_InvoiceJSONEncoder, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return json_path

    # ── Méthodes privées ────────────────────────────────────────────────────

    def _ensure_directories(self) -> None:
        """Crée l'arborescence nécessaire si elle n'existe pas."""
        for directory in (self._input_dir, self._json_dir, self._processed_dir):
            directory.mkdir(parents=True, exist_ok=True)

