"""
Script de migration : Ajouter image_size aux factures existantes dans la base de données.

Pour chaque facture qui a un fichier source PDF/image mais pas de image_size
dans ocr_data, ce script recalcule les dimensions et met à jour le JSONB.

Usage:
    python -m scripts.migrate_image_sizes
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json, RealDictCursor

load_dotenv()

UPLOADS_DIR = Path("data/uploads")


def get_image_size(filepath: Path) -> dict | None:
    """Récupère les dimensions de l'image/PDF."""
    try:
        suffix = filepath.suffix.lower()
        if suffix == ".pdf":
            try:
                import fitz
                doc = fitz.open(str(filepath))
                page = doc[0]
                zoom = 2.0  # PaddleOCR 3.x empirical zoom
                width = int(page.rect.width * zoom)
                height = int(page.rect.height * zoom)
                doc.close()
                return {"width": width, "height": height}
            except ImportError:
                return None
        else:
            from PIL import Image
            with Image.open(str(filepath)) as img:
                return {"width": img.width, "height": img.height}
    except Exception as e:
        print(f"  ⚠ Erreur: {e}")
        return None


def migrate():
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "invoice_db"),
        user=os.getenv("POSTGRES_USER", "invoice_user"),
        password=os.getenv("POSTGRES_PASSWORD", "invoice_pass"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )
    conn.autocommit = False
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        cursor.execute(
            "SELECT id, source_filename, ocr_data FROM invoices ORDER BY id"
        )
        invoices = cursor.fetchall()
        
        updated = 0
        skipped = 0
        
        for inv in invoices:
            inv_id = inv["id"]
            filename = inv.get("source_filename")
            ocr_data = inv.get("ocr_data") or {}

            # Skip si image_size existe déjà
            if ocr_data.get("image_size"):
                print(f"  ✓ Invoice #{inv_id}: image_size déjà présent, skip")
                skipped += 1
                continue

            if not filename:
                print(f"  ⚠ Invoice #{inv_id}: pas de source_filename, skip")
                skipped += 1
                continue

            filepath = UPLOADS_DIR / filename
            if not filepath.exists():
                print(f"  ⚠ Invoice #{inv_id}: fichier '{filename}' introuvable, skip")
                skipped += 1
                continue

            image_size = get_image_size(filepath)
            if not image_size:
                print(f"  ⚠ Invoice #{inv_id}: impossible de lire les dimensions, skip")
                skipped += 1
                continue

            # Mettre à jour ocr_data avec image_size
            ocr_data["image_size"] = image_size
            cursor.execute(
                "UPDATE invoices SET ocr_data = %s WHERE id = %s",
                (Json(ocr_data), inv_id)
            )
            print(f"  ✅ Invoice #{inv_id}: image_size={image_size}")
            updated += 1

        conn.commit()
        print(f"\n{'='*50}")
        print(f"Migration terminée: {updated} mises à jour, {skipped} ignorées")

    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    print("🔄 Migration: Ajout de image_size aux factures existantes...\n")
    migrate()
