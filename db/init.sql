-- =============================================================================
-- Schema d'initialisation — Invoice Processor
--
-- Exécuté automatiquement au premier lancement du conteneur PostgreSQL.
-- Les contraintes CHECK sur extra_data garantissent l'intégrité des données
-- dynamiques sans sacrifier la flexibilité du JSONB.
-- =============================================================================

-- ─── Table principale des factures ───────────────────────────────────────────

CREATE TABLE IF NOT EXISTS invoices (
    id                    SERIAL       PRIMARY KEY,

    -- Champs fixes (colonnes dédiées, indexables, typées)
    invoice_number        VARCHAR(100),
    invoice_date          DATE,
    supplier_name         VARCHAR(255),
    supplier_tax_id       VARCHAR(100),
    destinataire          VARCHAR(255),
    importateur           VARCHAR(255),
    port                  VARCHAR(255),
    moyen_transport       VARCHAR(255),
    incoterm              VARCHAR(10),
    total_amount_excl_tax NUMERIC(15, 2),
    tax_amount            NUMERIC(15, 2),
    total_amount_incl_tax NUMERIC(15, 2),
    currency              VARCHAR(10),
    confidence_score      REAL         DEFAULT 0.0,

    -- Données dynamiques (JSONB sécurisé)
    extra_data            JSONB        NOT NULL DEFAULT '{}',
    ocr_data              JSONB,

    -- Métadonnées
    source_filename       VARCHAR(255) NOT NULL,
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    confirmed_by_user_id  INTEGER,
    confirmed_at          TIMESTAMPTZ,

    -- ── Contraintes de sécurité sur extra_data ──────────────────────────────
    -- 1. Doit être un objet JSON (pas un array, pas un scalar)
    CONSTRAINT extra_data_is_object CHECK (jsonb_typeof(extra_data) = 'object'),
    -- 2. Taille maximale de 64 Ko (protection contre les abus)
    CONSTRAINT extra_data_max_size  CHECK (length(extra_data::text) < 65536)
);


-- ─── Table des lignes de facturation (relation 1:N) ─────────────────────────

CREATE TABLE IF NOT EXISTS invoice_items (
    id          SERIAL       PRIMARY KEY,
    invoice_id  INTEGER      NOT NULL REFERENCES invoices(id) ON DELETE CASCADE,
    description TEXT         NOT NULL,
    quantity    REAL,
    unit_price  NUMERIC(15, 2),
    total_price NUMERIC(15, 2),
    tax_rate    NUMERIC(5, 2)
);


-- ─── Index pour les requêtes fréquentes ─────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_invoices_number   ON invoices(invoice_number);
CREATE INDEX IF NOT EXISTS idx_invoices_supplier  ON invoices(supplier_name);
CREATE INDEX IF NOT EXISTS idx_invoices_date      ON invoices(invoice_date);
CREATE INDEX IF NOT EXISTS idx_invoices_created   ON invoices(created_at);

-- Index GIN sur extra_data : permet les recherches rapides dans le JSONB
-- Ex: SELECT * FROM invoices WHERE extra_data @> '{"iban": "FR76..."}';
CREATE INDEX IF NOT EXISTS idx_invoices_extra_gin ON invoices USING GIN (extra_data);


-- ─── Authentification (JWT) ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id              SERIAL       PRIMARY KEY,
    username        VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL
);

-- Utilisateur admin par défaut (mot de passe : admin) — à changer en production
INSERT INTO users (username, hashed_password)
VALUES (
    'admin',
    '$2b$12$hA1IP592ROzqi4ntfqaRluVVX3CTJfj7bSw6ybUrIUNYws7h0mdzK'
)
ON CONFLICT (username) DO NOTHING;

ALTER TABLE invoices
    DROP CONSTRAINT IF EXISTS invoices_confirmed_by_user_id_fkey;
ALTER TABLE invoices
    ADD CONSTRAINT invoices_confirmed_by_user_id_fkey
    FOREIGN KEY (confirmed_by_user_id) REFERENCES users(id);
