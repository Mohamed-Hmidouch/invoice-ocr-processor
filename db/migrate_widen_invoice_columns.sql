-- =============================================================================
-- Migration — Élargir incoterm et currency (VARCHAR 10 → 255)
--
-- À exécuter UNE FOIS sur une base déjà initialisée avec l'ancien schéma.
-- Les nouveaux déploiements n'en ont pas besoin (voir db/init.sql).
--
-- Usage (depuis la machine hôte) :
--   docker exec -i invoice-db psql -U invoice_user -d invoice_db < db/migrate_widen_invoice_columns.sql
--
-- Ou en une ligne :
--   docker exec invoice-db psql -U invoice_user -d invoice_db -c "
--     ALTER TABLE invoices ALTER COLUMN incoterm TYPE VARCHAR(255);
--     ALTER TABLE invoices ALTER COLUMN currency TYPE VARCHAR(255);
--   "
-- =============================================================================

ALTER TABLE invoices ALTER COLUMN incoterm TYPE VARCHAR(255);
ALTER TABLE invoices ALTER COLUMN currency TYPE VARCHAR(255);
