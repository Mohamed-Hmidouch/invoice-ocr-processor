# Invoice Processor — Guide de déploiement (Backend)

Tutoriel pour lancer l'API + la base de données PostgreSQL avec Docker, **sans le code source**.

---

## Contenu du package

```
invoice-backend-deploy/
├── DEPLOY.md              ← ce guide
├── docker-compose.yml     ← lance DB + API
├── .env.example           ← modèle de configuration
└── db/
    └── init.sql           ← schéma PostgreSQL (tables, user admin)
```

**Image Docker (API)** : `mohamedhmidouch/ocr-api:1.0.0`  
Hébergée sur Docker Hub — téléchargée automatiquement par `docker compose pull`.

---

## Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé (Windows, Mac ou Linux)
- Docker Compose v2 (inclus avec Docker Desktop)
- Une clé API Google Gemini : https://aistudio.google.com/app/apikey

> Si le repo Docker est **privé**, chaque membre doit d'abord exécuter `docker login` avec un compte autorisé.

---

## Déploiement en 5 minutes

### 1. Préparer la configuration

```bash
cp .env.example .env
```

Ouvrir `.env` et remplir **obligatoirement** :

| Variable | Description | Exemple |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Clé API Gemini (extraction LLM) | `AIzaSy...` |
| `JWT_SECRET_KEY` | Secret pour les tokens JWT | Générer avec : `openssl rand -hex 32` |
| `POSTGRES_PASSWORD` | Mot de passe PostgreSQL | `invoice_pass` (choisir un mot de passe fort) |

Les autres variables ont des valeurs par défaut dans `.env.example`.

### 2. Dossiers de données — pas besoin de les créer manuellement

> **La DB ≠ les fichiers.** PostgreSQL stocke les **données structurées** (numéro de facture, montants, lignes, résultat OCR en JSON).  
> Les **fichiers bruts** (PDF, images uploadées) sont stockés sur le disque dans `data/uploads/`.

| Emplacement | Contenu |
|-------------|---------|
| **PostgreSQL** (`invoice-db`) | Métadonnées facture, items, `extra_data`, `ocr_data`, nom du fichier source |
| **`data/uploads/`** | Fichiers PDF/images uploadés via `POST /invoices/upload` (servis par `GET /files/{filename}`) |
| **`data/output/`** | Sorties du pipeline CLI batch (peu utilisé en mode API Docker) |

Ces dossiers sont **créés automatiquement** au démarrage (Docker + l'application).  
Tu n'as **rien à faire** ici — cette étape peut être ignorée.

### 3. Télécharger l'image et démarrer

```bash
docker compose pull
docker compose up -d
```

Au **premier lancement**, le téléchargement de l'image peut prendre plusieurs minutes (PaddleOCR + dépendances).

### 4. Vérifier que tout fonctionne

```bash
docker compose ps
```

Les deux services doivent être `Up` et `healthy` :
- `invoice-db` — PostgreSQL
- `invoice-api` — API FastAPI

Test rapide :

```bash
curl http://localhost:8000/health
```

Réponse attendue :

```json
{"status":"ok"}
```

---

## Documentation API (Swagger)

Une fois l'API démarrée, ouvrir dans le navigateur :

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | **Swagger UI** — tester les endpoints interactivement |
| http://localhost:8000/redoc | Documentation alternative (ReDoc) |
| http://localhost:8000/openapi.json | Schéma OpenAPI (import Postman) |

---

## Authentification

Un utilisateur **admin** est créé automatiquement au premier démarrage de la base :

| Champ | Valeur |
|-------|--------|
| Username | `admin` |
| Password | `admin` |

> **Production** : changer ce mot de passe après le premier login.

### Obtenir un token JWT

**Swagger** : endpoint `POST /token` → cliquer *Try it out* → username `admin`, password `admin`.

**curl** :

```bash
curl -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin"
```

Réponse :

```json
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}
```

Utiliser ce token pour les endpoints protégés : bouton **Authorize** dans Swagger, ou header :

```
Authorization: Bearer eyJ...
```

---

## Endpoints principaux

| Méthode | Route | Auth | Description |
|---------|-------|------|-------------|
| GET | `/health` | Non | Vérification de santé |
| POST | `/token` | Non | Login → JWT |
| GET | `/users/me` | Oui | Infos utilisateur connecté |
| GET | `/invoices` | Non | Lister toutes les factures |
| GET | `/invoices/{id}` | Non | Détail d'une facture |
| POST | `/invoices` | Non | Créer une facture (JSON) |
| POST | `/invoices/upload` | Non | Upload PDF/image → OCR + LLM |
| PATCH | `/invoices/{id}` | Oui | Modifier / confirmer une facture |
| GET | `/files/{filename}` | Non | Télécharger un fichier uploadé |

### Exemple : upload d'une facture

Dans Swagger → `POST /invoices/upload` → choisir un fichier PDF ou image.

Ou avec curl :

```bash
curl -X POST http://localhost:8000/invoices/upload \
  -F "file=@/chemin/vers/facture.pdf"
```

---

## Import Postman (optionnel)

1. Ouvrir Postman → **Import**
2. Coller l'URL : `http://localhost:8000/openapi.json`
3. Postman génère la collection automatiquement
4. Créer une variable d'environnement `base_url` = `http://localhost:8000`
5. Appeler `POST /token` pour récupérer le JWT, puis l'utiliser dans les requêtes protégées

---

## Commandes utiles

```bash
# Voir les logs en direct
docker compose logs -f invoice-api

# Voir les logs de la base
docker compose logs -f invoice-db

# Arrêter (données conservées)
docker compose down

# Arrêter ET supprimer toutes les données (reset complet)
docker compose down -v

# Redémarrer après modification du .env
docker compose down
docker compose up -d

# Mettre à jour vers une nouvelle version de l'image
docker compose pull
docker compose up -d
```

---

## Dépannage

### L'API ne démarre pas — `JWT_SECRET_KEY est requis`

Le fichier `.env` est absent ou incomplet. Vérifier que `JWT_SECRET_KEY` est bien défini.

### L'API ne démarre pas — `POSTGRES_PASSWORD est requis`

Même cause : remplir `POSTGRES_PASSWORD` dans `.env`.

### `connection refused` sur PostgreSQL

Attendre que `invoice-db` soit `healthy` :

```bash
docker compose ps
docker compose logs invoice-db
```

### Port 8000 déjà utilisé

Modifier dans `docker-compose.yml` :

```yaml
ports:
  - "8080:8000"   # API accessible sur http://localhost:8080
```

### La base est vide / tables manquantes

Le script `db/init.sql` ne s'exécute qu'**au tout premier démarrage** de PostgreSQL.  
Si la base existait déjà sans schéma :

```bash
docker compose down -v    # ⚠️ supprime toutes les données
docker compose up -d      # recrée la DB avec init.sql
```

### Upload échoue — erreur Gemini

Vérifier que `GEMINI_API_KEY` dans `.env` est valide et que le compte Google AI Studio a du quota.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Machine hôte                                   │
│                                                 │
│  docker compose                                 │
│  ┌──────────────┐      ┌───────────────────┐   │
│  │  invoice-db  │◄─────│   invoice-api     │   │
│  │  PostgreSQL  │      │   FastAPI + OCR   │   │
│  │  :5432       │      │   :8000           │   │
│  └──────────────┘      └───────────────────┘   │
│         ▲                       ▲               │
│    db/init.sql            .env (secrets)        │
│    volume pgdata          data/uploads          │
└─────────────────────────────────────────────────┘
         ▲
    Navigateur / Postman
    http://localhost:8000/docs
```

---
