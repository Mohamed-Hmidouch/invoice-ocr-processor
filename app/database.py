"""
Module de persistance PostgreSQL pour le pipeline de factures.

Responsabilités :
    - Gérer la connexion à la base PostgreSQL (via psycopg2).
    - Insérer les factures extraites (Invoice + InvoiceItems) de façon
      transactionnelle (ACID).
    - Sérialiser `extra_data` en JSONB de manière sécurisée.

Sécurité :
    - Toutes les requêtes utilisent des parameterized queries (%s).
    - Aucune interpolation de chaîne (f-string) dans les requêtes SQL.
    - Le dict `extra_data` est converti via psycopg2.extras.Json
      (échappement automatique, protection contre l'injection).
"""
import os
import logging
from decimal import Decimal

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from app.core.aspect import handle_exceptions
from app.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """
    Gestionnaire de persistance PostgreSQL.

    Encapsule le cycle de vie de la connexion et fournit une API
    transactionnelle pour insérer les factures extraites.

    Utilisation :
        db = DatabaseManager()
        db.connect()
        db.save_invoice(invoice, "facture_001.pdf")
        db.close()
    """

    # ── Requête INSERT facture ──────────────────────────────────────────────
    _INSERT_INVOICE = """
        INSERT INTO invoices (
            invoice_number, invoice_date, supplier_name, supplier_tax_id,
            destinataire, importateur, port, moyen_transport, incoterm,
            total_amount_excl_tax, tax_amount, total_amount_incl_tax,
            currency, confidence_score, extra_data, ocr_data, source_filename
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id;
    """

    # ── Requête INSERT ligne de facturation ─────────────────────────────────
    _INSERT_ITEM = """
        INSERT INTO invoice_items (
            invoice_id, description, quantity, unit_price, total_price, tax_rate
        ) VALUES (%s, %s, %s, %s, %s, %s);
    """

    # ── Requêtes SELECT ─────────────────────────────────────────────────────
    _SELECT_ALL_INVOICES = """
        SELECT id, invoice_number, invoice_date, supplier_name, supplier_tax_id,
               destinataire, importateur, port, moyen_transport, incoterm,
               total_amount_excl_tax, tax_amount, total_amount_incl_tax,
               currency, confidence_score, extra_data, ocr_data, source_filename, created_at
        FROM invoices
        ORDER BY created_at DESC;
    """

    _SELECT_INVOICE_BY_ID = """
        SELECT id, invoice_number, invoice_date, supplier_name, supplier_tax_id,
               destinataire, importateur, port, moyen_transport, incoterm,
               total_amount_excl_tax, tax_amount, total_amount_incl_tax,
               currency, confidence_score, extra_data, ocr_data, source_filename, created_at
        FROM invoices
        WHERE id = %s;
    """

    _SELECT_ITEMS_BY_INVOICE_ID = """
        SELECT id, description, quantity, unit_price, total_price, tax_rate
        FROM invoice_items
        WHERE invoice_id = %s
        ORDER BY id;
    """

    def __init__(self):
        """Prépare les paramètres de connexion depuis les variables d'environnement."""
        self._conn = None
        self._db_config = {
            "dbname":   os.getenv("POSTGRES_DB",       "invoice_db"),
            "user":     os.getenv("POSTGRES_USER",     "invoice_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "invoice_pass"),
            "host":     os.getenv("POSTGRES_HOST",     "localhost"),
            "port":     os.getenv("POSTGRES_PORT",     "5432"),
        }

    # ── Cycle de vie ────────────────────────────────────────────────────────

    @handle_exceptions(Exception, raise_as=DatabaseError)
    def connect(self) -> None:
        """Établit la connexion à PostgreSQL."""
        if self._conn and not self._conn.closed:
            return

        self._conn = psycopg2.connect(**self._db_config)
        # Autocommit désactivé → on gère les transactions manuellement
        self._conn.autocommit = False
        logger.info("Connexion PostgreSQL établie (%s@%s:%s/%s)",
                     self._db_config["user"],
                     self._db_config["host"],
                     self._db_config["port"],
                     self._db_config["dbname"])

    @handle_exceptions(Exception, raise_as=DatabaseError)
    def close(self) -> None:
        """Ferme proprement la connexion."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("Connexion PostgreSQL fermée.")

    # ── API publique : WRITE ────────────────────────────────────────────────

    @handle_exceptions(Exception, raise_as=DatabaseError)
    def save_invoice(self, invoice, source_filename: str) -> int:
        """
        Persiste une facture et ses lignes dans PostgreSQL.

        L'opération est atomique : si l'insertion d'un item échoue,
        toute la transaction est annulée (ROLLBACK).

        Paramètres
        ----------
        invoice : Invoice
            L'objet facture à persister.
        source_filename : str
            Nom du fichier source (pour traçabilité).

        Retour
        ------
        int
            L'ID de la facture insérée.
        """
        ocr_data = {
            "ocr_line_references": invoice.ocr_line_references,
            "ocr_lines": getattr(invoice, "ocr_lines", []),
            "image_size": getattr(invoice, "ocr_image_size", None),
        }
        
        cursor = self._conn.cursor()

        try:
            # ── 1. INSERT facture ───────────────────────────────────────────
            cursor.execute(self._INSERT_INVOICE, (
                invoice.invoice_number,
                invoice.date,
                invoice.supplier_name,
                invoice.supplier_tax_id,
                invoice.destinataire,
                invoice.importateur,
                invoice.port,
                invoice.moyen_transport,
                invoice.incoterm,
                self._decimal_to_float(invoice.total_amount_excl_tax),
                self._decimal_to_float(invoice.tax_amount),
                self._decimal_to_float(invoice.total_amount_incl_tax),
                invoice.currency,
                invoice.confidence_score,
                Json(invoice.extra_data),       # Conversion sécurisée dict → JSONB
                Json(ocr_data),                 # Données de matching OCR
                source_filename,
            ))

            invoice_id = cursor.fetchone()[0]

            # ── 2. INSERT lignes de facturation ─────────────────────────────
            for item in invoice.items:
                cursor.execute(self._INSERT_ITEM, (
                    invoice_id,
                    item.description,
                    item.quantity,
                    self._decimal_to_float(item.unit_price),
                    self._decimal_to_float(item.total_price),
                    self._decimal_to_float(item.tax_rate) if item.tax_rate else None,
                ))

            # ── 3. COMMIT atomique ──────────────────────────────────────────
            self._conn.commit()
            logger.info("Facture #%d insérée (%s, %d items)",
                        invoice_id, source_filename, len(invoice.items))
            return invoice_id

        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    @handle_exceptions(Exception, raise_as=DatabaseError)
    def save_invoice_from_dict(self, data: dict) -> int:
        """
        Persiste une facture depuis un dictionnaire brut (utilisé par l'API REST).

        Accepte le même format JSON que celui généré par le LLM Gemini.

        Paramètres
        ----------
        data : dict
            Dictionnaire contenant les champs de la facture.

        Retour
        ------
        int
            L'ID de la facture insérée.
        """
        cursor = self._conn.cursor()

        try:
            cursor.execute(self._INSERT_INVOICE, (
                data.get("invoice_number"),
                data.get("date"),
                data.get("supplier_name"),
                data.get("supplier_tax_id"),
                data.get("destinataire"),
                data.get("importateur"),
                data.get("port"),
                data.get("moyen_transport"),
                data.get("incoterm"),
                data.get("total_amount_excl_tax"),
                data.get("tax_amount"),
                data.get("total_amount_incl_tax"),
                data.get("currency"),
                float(data.get("confidence_score") or 0.0),
                Json(data.get("extra_data") or {}),
                Json({
                    "ocr_line_references": data.get("ocr_line_references") or {},
                    "ocr_lines": data.get("ocr_lines") or []
                }),
                data.get("source_filename", "api_upload"),
            ))

            invoice_id = cursor.fetchone()[0]

            # Insert items si présents
            for item in (data.get("items") or []):
                cursor.execute(self._INSERT_ITEM, (
                    invoice_id,
                    item.get("description", ""),
                    item.get("quantity"),
                    item.get("unit_price"),
                    item.get("total_price"),
                    item.get("tax_rate"),
                ))

            self._conn.commit()
            logger.info("Facture #%d insérée via API (%d items)",
                        invoice_id, len(data.get("items") or []))
            return invoice_id

        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    # ── API publique : READ ─────────────────────────────────────────────────

    @handle_exceptions(Exception, raise_as=DatabaseError)
    def get_all_invoices(self) -> list:
        """
        Récupère toutes les factures avec leurs items.

        Retour
        ------
        list[dict]
            Liste de dictionnaires, chaque facture contient ses `items`.
        """
        cursor = self._conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute(self._SELECT_ALL_INVOICES)
            invoices = [dict(row) for row in cursor.fetchall()]

            # Attacher les items à chaque facture
            for inv in invoices:
                inv["items"] = self._fetch_items(inv["id"])
                # Convertir les types non-JSON-serializable
                self._serialize_row(inv)

            return invoices
        finally:
            cursor.close()

    @handle_exceptions(Exception, raise_as=DatabaseError)
    def get_invoice_by_id(self, invoice_id: int) -> dict | None:
        """
        Récupère une facture par son ID, avec ses items.

        Paramètres
        ----------
        invoice_id : int
            L'identifiant de la facture.

        Retour
        ------
        dict | None
            La facture avec ses items, ou None si introuvable.
        """
        cursor = self._conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute(self._SELECT_INVOICE_BY_ID, (invoice_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            invoice = dict(row)
            invoice["items"] = self._fetch_items(invoice_id)
            self._serialize_row(invoice)

            return invoice
        finally:
            cursor.close()

    # ── Méthodes privées ────────────────────────────────────────────────────

    def _fetch_items(self, invoice_id: int) -> list:
        """Récupère les lignes de facturation pour une facture donnée."""
        cursor = self._conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(self._SELECT_ITEMS_BY_INVOICE_ID, (invoice_id,))
            items = []
            for row in cursor.fetchall():
                item = dict(row)
                # Convertir Decimal → float pour la sérialisation JSON
                for key in ("unit_price", "total_price", "tax_rate"):
                    if item.get(key) is not None:
                        item[key] = float(item[key])
                items.append(item)
            return items
        finally:
            cursor.close()

    @staticmethod
    def _serialize_row(row: dict) -> None:
        """Convertit les types PostgreSQL en types JSON-serializable (in-place)."""
        from datetime import date, datetime
        for key, value in row.items():
            if isinstance(value, Decimal):
                row[key] = float(value)
            elif isinstance(value, (date, datetime)):
                row[key] = value.isoformat()

    @staticmethod
    def _decimal_to_float(value) -> float:
        """Convertit Decimal → float pour psycopg2 (NUMERIC accepte float)."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        return value
