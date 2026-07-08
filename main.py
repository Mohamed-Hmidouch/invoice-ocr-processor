"""
Point d'entrée du pipeline de traitement de factures.

Ce fichier reste LÉGER : il orchestre, il ne contient aucune logique métier.
La gestion globale des erreurs est centralisée ici — un seul try/except
au sommet de la pile d'appels, conformément au principe AOP.
"""
import os
# ── Doit être défini AVANT tout import PaddlePaddle / PaddleOCR ─────────────
os.environ["FLAGS_use_mkldnn"] = "0"          # Désactive oneDNN (bug pir::ArrayAttribute)
os.environ["FLAGS_minloglevel"] = "3"         # Supprime les logs C++ de Paddle
os.environ["FLAGS_enable_pir_api"] = "0"      # Désactive le nouveau PIR executor (cause du crash oneDNN)
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"     # Double sécurité pour désactiver oneDNN
os.environ["FLAGS_new_executor_micro_batching"] = "False"  # Désactive le micro-batching PIR
os.environ["DNNL_MAX_CPU_ISA"] = "VANILLA"    # Force oneDNN en mode basique (bypass ArrayAttribute bug)
os.environ["ONEDNN_MAX_CPU_ISA"] = "VANILLA"  # Alias récent de DNNL_MAX_CPU_ISA

import sys
from dotenv import load_dotenv

# Charge les variables d'environnement depuis .env (ex: NVIDIA_API_KEY)
load_dotenv()

from app import OCREngine, Extractor, FileManager, DatabaseManager, InvoiceProcessorError


def main() -> None:
    """
    Pipeline principal :
        1. Lister les factures en attente
        2. Lire chaque image via OCR (Singleton → une seule instance en RAM)
        3. Extraire les données structurées via LLM (Gemini)
        4. Persister en PostgreSQL et archiver l'image
    """
    file_manager = FileManager()
    ocr_engine = OCREngine()
    extractor = Extractor()
    db_manager = DatabaseManager()

    # ── Connexion à PostgreSQL ──────────────────────────────────────────────
    db_manager.connect()

    pending = file_manager.get_pending_files()

    if not pending:
        print("[INFO] Aucune facture en attente dans le dossier d'entrée.")
        db_manager.close()
        return

    print(f"[INFO] {len(pending)} facture(s) trouvee(s). Traitement en cours...\n")

    success_count = 0
    error_count = 0

    for image_path in pending:
        print(f"  > {image_path.name}", end=" ")

        try:
            ocr_results = ocr_engine.read(str(image_path))
            invoice = extractor.extract(ocr_results)

            # Persistance en PostgreSQL (transactionnelle)
            invoice_id = db_manager.save_invoice(invoice, image_path.name)
            file_manager.move_to_processed(image_path)

            status = "[OK]" if invoice.is_valid() else "[WARN] donnees incompletes"
            print(f"-> {status}  [DB id={invoice_id}]")
            success_count += 1

        except InvoiceProcessorError as exc:
            print(f"-> [FAIL] {exc}")
            error_count += 1

    # ── Fermeture propre de la connexion ────────────────────────────────────
    db_manager.close()

    # ── Résumé final ────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"  Résultat : {success_count} réussie(s), {error_count} erreur(s)")
    print(f"  Stockage : PostgreSQL ({os.getenv('POSTGRES_DB', 'invoice_db')})")


if __name__ == "__main__":
    # ── Gestion GLOBALE des erreurs (dernier filet de sécurité) ─────────────
    try:
        main()
    except InvoiceProcessorError as exc:
        print(f"\n[FATAL] Erreur fatale : {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[INFO] Traitement interrompu par l'utilisateur.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[FATAL] Erreur inattendue : {exc}", file=sys.stderr)
        sys.exit(1)
