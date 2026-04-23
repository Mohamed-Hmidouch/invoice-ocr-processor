import os
import shutil
from pathlib import Path
import filetype
from werkzeug.utils import secure_filename

from app.core.exceptions import SecurityValidationError

class FileValidator:
    """
    Couche de sécurité métier (Security Architect) pour les fichiers en entrée.
    
    Responsabilités :
    1. MAX_SIZE : Protège contre les attaques DoS et Out-of-Memory.
    2. REAL MIME TYPE : Protège contre l'obfuscation d'extensions (Malware masquerading).
    3. SANITIZATION : Protège contre le Path Traversal (LFI) via le renommage sécurisé.
    """
    
    MAX_FILE_SIZE_MB = 10
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # Types MIME acceptés par notre OCR
    ALLOWED_MIME_TYPES = {
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/bmp",
        "image/webp",
        "application/pdf"
    }

    @classmethod
    def validate_and_sanitize(cls, file_path: Path) -> Path:
        """
        Valide la sûreté d'un fichier et nettoie son nom de tout caractère malveillant.
        
        Retourne :
            Path: Le nouveau chemin vers le fichier sécurisé.
            
        Lève :
            SecurityValidationError: Si le fichier est dangereux ou invalide.
        """
        # 1. Validation de la Taille Finale
        if not file_path.exists():
            raise SecurityValidationError(f"Fichier fantôme introuvable : {file_path}")
            
        file_size = os.path.getsize(file_path)
        if file_size > cls.MAX_FILE_SIZE_BYTES:
            # Défense contre DoS / Zip-Bombs basique
            file_path.unlink()  # On le supprime direct pour protéger le serveur
            raise SecurityValidationError(
                f"Fichier trop lourd ({file_size / 1024 / 1024:.2f} MB). Max autorisé : {cls.MAX_FILE_SIZE_MB} MB."
            )

        # 2. Validation du Type Réel (Magic Bytes)
        # On ne fait *JAMAIS* confiance à l'extension du fichier (.pdf, .jpg)
        kind = filetype.guess(str(file_path))
        if kind is None or kind.mime not in cls.ALLOWED_MIME_TYPES:
            file_path.unlink() 
            raise SecurityValidationError(
                f"Type MIME non supporté ou malveillant. Attendu : {cls.ALLOWED_MIME_TYPES}"
            )

        # 3. Sanitization du Nom de fichier (Path Traversal / LFI)
        # werkzeug va virer les '../', '/', les espaces compliqués et les caractères spéciaux
        safe_name = secure_filename(file_path.name)
        
        # S'assurer qu'on garde l'extension réelle (et non celle falsifiée du nom)
        real_extension = f".{kind.extension}"
        if not safe_name.lower().endswith(real_extension):
            safe_name = f"{os.path.splitext(safe_name)[0]}{real_extension}"
            
        # Si le nom a été nettoyé/modifié, on renomme le fichier physiquement
        safe_path = file_path.with_name(safe_name)
        if file_path != safe_path:
            # On vérifie qu'on n'écrase pas un autre fichier existant
            if safe_path.exists():
                safe_path = file_path.with_name(f"secure_{os.urandom(4).hex()}_{safe_name}")
            shutil.move(str(file_path), str(safe_path))
            
        return safe_path
