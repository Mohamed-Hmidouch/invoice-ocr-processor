"""
Point d'entrée du pipeline de traitement de factures.

Ce fichier reste LÉGER : il orchestre, il ne contient aucune logique métier.
La gestion globale des erreurs est centralisée ici — un seul try/except
au sommet de la pile d'appels, conformément au principe AOP.
"""
import sys

from app import OCREngine, Extractor, FileManager, InvoiceProcessorError


def main() -> None:
    """
    Pipeline principal :
        1. Lister les factures en attente
        2. Lire chaque image via OCR (Singleton → une seule instance en RAM)
        3. Extraire les données structurées (RegEx)
        4. Sauvegarder en JSON et archiver l'image
    """
    file_manager = FileManager()
    ocr_engine = OCREngine()
    extractor = Extractor()

    pending = file_manager.get_pending_files()

    if not pending:
        print("[INFO] Aucune facture en attente dans le dossier d'entrée.")
        return

    print(f"[INFO] {len(pending)} facture(s) trouvee(s). Traitement en cours...\n")

    success_count = 0
    error_count = 0

    for image_path in pending:
        print(f"  > {image_path.name}", end=" ")

        try:
            ocr_results = ocr_engine.read(str(image_path))
            invoice = extractor.extract(ocr_results)

            json_path = file_manager.save_result(invoice, image_path.name)
            file_manager.move_to_processed(image_path)

            status = "[OK]" if invoice.is_valid() else "[WARN] donnees incompletes"
            print(f"-> {status}  [{json_path.name}]")
            success_count += 1

        except InvoiceProcessorError as exc:
            print(f"-> [FAIL] {exc}")
            error_count += 1

    # ── Résumé final ────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"  Résultat : {success_count} réussie(s), {error_count} erreur(s)")
    print(f"  Sortie   : {file_manager.output_dir}/")


if __name__ == "__main__":
    # ── Gestion GLOBALE des erreurs (dernier filet de sécurité) ─────────
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

