"""
API REST pour le processeur de factures.

Endpoints :
    POST   /invoices         → Créer une facture depuis un JSON structuré
    POST   /invoices/upload  → Uploader un PDF/image → OCR + LLM → DB
    GET    /invoices          → Récupérer toutes les factures
    GET    /invoices/{id}     → Récupérer une facture par son ID

Framework : FastAPI (validation Pydantic, docs Swagger automatiques).
Sécurité  : Validation stricte des entrées via Pydantic, CORS configurable.
"""
import os

# ── Paddle / oneDNN flags — avant tout import PaddleOCR ─────────────────────
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_minloglevel", "3")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_DISABLE_MKLDNN", "1")
os.environ.setdefault("DNNL_MAX_CPU_ISA", "VANILLA")
os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "VANILLA")

from dotenv import load_dotenv

load_dotenv()

import shutil
import uuid
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

# ── Logs clairs et homogènes pour toute l'application ────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("invoice_api")

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from app.database import DatabaseManager
from app.core import OCREngine, Extractor
from app.core.exceptions import InvoiceProcessorError
from app.core.auth import Token, create_access_token, verify_password, get_current_user


# Formats de fichiers acceptés par l'endpoint d'upload
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".pdf"}

# Dossier pour stocker les uploads persistants
UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_pages(pages: str) -> List[int]:
    """Parse le paramètre pages (JSON array ou liste séparée par virgules, 1-based)."""
    pages = pages.strip()
    if not pages:
        raise HTTPException(status_code=400, detail="Le paramètre pages ne peut pas être vide.")

    try:
        if pages.startswith("["):
            parsed = json.loads(pages)
            if not isinstance(parsed, list):
                raise ValueError("Le JSON pages doit être un tableau.")
            page_list = [int(p) for p in parsed]
        else:
            page_list = [int(p.strip()) for p in pages.split(",") if p.strip()]
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Format pages invalide. Utilisez '[1,3,5]' ou '1,3,5'.",
        ) from exc

    if not page_list:
        raise HTTPException(status_code=400, detail="Au moins une page doit être sélectionnée.")

    if any(p < 1 for p in page_list):
        raise HTTPException(status_code=400, detail="Les numéros de page doivent être >= 1.")

    return sorted(set(page_list))


def _invoice_to_storage_dict(invoice) -> dict:
    """Sérialise une facture extraite pour stockage JSON (page_extractions)."""
    return {
        "invoice_number": invoice.invoice_number,
        "date": invoice.date.isoformat() if invoice.date else None,
        "supplier_name": invoice.supplier_name,
        "supplier_tax_id": invoice.supplier_tax_id,
        "destinataire": invoice.destinataire,
        "importateur": invoice.importateur,
        "port": invoice.port,
        "moyen_transport": invoice.moyen_transport,
        "incoterm": invoice.incoterm,
        "total_amount_excl_tax": float(invoice.total_amount_excl_tax) if invoice.total_amount_excl_tax is not None else None,
        "tax_amount": float(invoice.tax_amount) if invoice.tax_amount is not None else None,
        "total_amount_incl_tax": float(invoice.total_amount_incl_tax) if invoice.total_amount_incl_tax is not None else None,
        "currency": invoice.currency,
        "confidence_score": invoice.confidence_score,
        "items": [
            {
                "description": item.description,
                "quantity": float(item.quantity) if item.quantity is not None else None,
                "unit_price": float(item.unit_price) if item.unit_price is not None else None,
                "total_price": float(item.total_price) if item.total_price is not None else None,
                "tax_rate": float(item.tax_rate) if item.tax_rate is not None else None,
            }
            for item in invoice.items
        ],
        "extra_data": invoice.extra_data,
        "ocr_line_references": invoice.ocr_line_references,
    }


def _remap_ocr_line_references(refs: dict, local_to_global: List[int]) -> dict:
    """Convertit les indices OCR locaux (par page) en indices globaux."""

    def remap_value(value):
        if isinstance(value, list):
            if not value:
                return value
            if isinstance(value[0], dict):
                return [{k: remap_value(v) for k, v in row.items()} for row in value]
            remapped = []
            for idx in value:
                if isinstance(idx, int) and 0 <= idx < len(local_to_global):
                    remapped.append(local_to_global[idx])
            return remapped
        return value

    return {k: remap_value(v) for k, v in refs.items()}


def _extract_per_page(ocr_results: list, pages_list: List[int], extractor) -> dict:
    """Exécute l'extraction LLM séparément pour chaque page sélectionnée."""
    page_extractions = {}

    for page_num in pages_list:
        page_lines = [line for line in ocr_results if line.get("page") == page_num]
        if not page_lines:
            continue

        local_to_global = [
            i for i, line in enumerate(ocr_results) if line.get("page") == page_num
        ]
        page_invoice = extractor.extract(page_lines)
        storage = _invoice_to_storage_dict(page_invoice)
        storage["ocr_line_references"] = _remap_ocr_line_references(
            page_invoice.ocr_line_references,
            local_to_global,
        )
        page_extractions[str(page_num)] = storage

    return page_extractions


def _ensure_page_extractions(invoice: dict) -> dict:
    """
    Génère page_extractions depuis ocr_lines déjà en base si manquant.
    Évite de re-uploader un fichier dont le chemin est déjà persisté.
    """
    ocr_data = invoice.get("ocr_data") or {}
    selected_pages = ocr_data.get("selected_pages")
    if not selected_pages or len(selected_pages) <= 1:
        return invoice
    if ocr_data.get("page_extractions"):
        return invoice

    ocr_lines = ocr_data.get("ocr_lines") or []
    if not ocr_lines or not any(line.get("page") for line in ocr_lines):
        return invoice

    try:
        page_extractions = _extract_per_page(ocr_lines, selected_pages, extractor)
        if not page_extractions:
            return invoice
        db_manager.update_ocr_data(invoice["id"], {"page_extractions": page_extractions})
        invoice["ocr_data"] = {**ocr_data, "page_extractions": page_extractions}
        logger.info(
            "page_extractions générées pour facture #%s (%d pages)",
            invoice["id"],
            len(page_extractions),
        )
    except InvoiceProcessorError as exc:
        logger.warning(
            "Impossible de générer page_extractions pour facture #%s : %s",
            invoice.get("id"),
            exc,
        )
    return invoice



# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS (Validation des entrées / sorties)
# ═══════════════════════════════════════════════════════════════════════════════

class InvoiceItemCreate(BaseModel):
    """Schéma d'une ligne de facturation (entrée)."""
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    tax_rate: Optional[float] = None


class InvoiceCreate(BaseModel):
    """
    Schéma de création d'une facture (entrée POST).

    Correspond exactement au JSON généré par Gemini, ce qui permet
    au front d'envoyer le même format sans transformation.
    """
    invoice_number: Optional[str] = None
    date: Optional[str] = Field(None, description="Format YYYY-MM-DD")
    supplier_name: Optional[str] = None
    supplier_tax_id: Optional[str] = None
    destinataire: Optional[str] = None
    importateur: Optional[str] = None
    port: Optional[str] = None
    moyen_transport: Optional[str] = None
    incoterm: Optional[str] = None
    total_amount_excl_tax: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount_incl_tax: Optional[float] = None
    currency: Optional[str] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    items: List[InvoiceItemCreate] = Field(default_factory=list)
    extra_data: dict = Field(default_factory=dict)
    source_filename: str = Field(default="api_upload")

class InvoiceUpdate(BaseModel):
    """Schéma de mise à jour/confirmation d'une facture."""
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    supplier_name: Optional[str] = None
    supplier_tax_id: Optional[str] = None
    destinataire: Optional[str] = None
    importateur: Optional[str] = None
    port: Optional[str] = None
    moyen_transport: Optional[str] = None
    incoterm: Optional[str] = None
    total_amount_excl_tax: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount_incl_tax: Optional[float] = None
    currency: Optional[str] = None
    extra_data: Optional[dict] = None

class InvoiceItemResponse(BaseModel):
    """Schéma d'une ligne de facturation (sortie)."""
    id: int
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    tax_rate: Optional[float] = None


class InvoiceResponse(BaseModel):
    """Schéma complet d'une facture (sortie GET)."""
    id: int
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    supplier_name: Optional[str] = None
    supplier_tax_id: Optional[str] = None
    destinataire: Optional[str] = None
    importateur: Optional[str] = None
    port: Optional[str] = None
    moyen_transport: Optional[str] = None
    incoterm: Optional[str] = None
    total_amount_excl_tax: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount_incl_tax: Optional[float] = None
    currency: Optional[str] = None
    confidence_score: Optional[float] = None
    extra_data: dict = Field(default_factory=dict)
    ocr_data: Optional[dict] = Field(default_factory=dict)
    source_filename: Optional[str] = None
    created_at: Optional[str] = None
    confirmed_by_user_id: Optional[int] = None
    confirmed_at: Optional[str] = None
    items: List[InvoiceItemResponse] = Field(default_factory=list)


class InvoiceCreateResponse(BaseModel):
    """Réponse après création d'une facture."""
    id: int
    message: str


class UploadInvoiceResponse(InvoiceResponse):
    """Réponse après upload : inclut un indicateur de doublon."""
    already_exists: bool = False
    message: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# LIFESPAN (Connexion DB gérée au démarrage / arrêt du serveur)
# ═══════════════════════════════════════════════════════════════════════════════

db_manager = DatabaseManager()
ocr_engine = OCREngine()
extractor = Extractor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de la connexion DB avec le serveur."""
    db_manager.connect()
    yield
    db_manager.close()


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION FASTAPI
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Invoice Processor API",
    description="API REST pour la gestion des factures extraites par OCR + Gemini AI.",
    version="1.0.2",
    lifespan=lifespan,
)

# ── CORS (origines autorisées chargées depuis l'environnement) ──────────────
# CORS_ORIGINS = liste d'URL séparées par des virgules (ex:
# "https://app.exemple.com,http://localhost:3000"). Défaut restrictif : vide.
_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS active pour les origines : %s", _cors_origins)
else:
    logger.info("CORS desactive (aucune origine dans CORS_ORIGINS).")


@app.get("/health", summary="Verification de sante", tags=["monitoring"])
def health_check():
    """Endpoint léger pour le healthcheck Docker/compose."""
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS AUTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = db_manager.get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": str(user["id"]), "username": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", summary="Get current user info")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS FACTURES
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/invoices",
    response_model=InvoiceCreateResponse,
    status_code=201,
    summary="Créer une facture",
    description="Reçoit un JSON de facture (même format que Gemini) et le persiste en PostgreSQL.",
)
def create_invoice(invoice: InvoiceCreate):
    """
    POST /invoices

    Le frontend envoie le JSON structuré, l'API le valide via Pydantic
    puis le persiste dans PostgreSQL.
    """
    try:
        invoice_id = db_manager.save_invoice_from_dict(invoice.model_dump())
        return InvoiceCreateResponse(
            id=invoice_id,
            message=f"Facture créée avec succès (id={invoice_id})"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'insertion : {exc}")


@app.get(
    "/invoices",
    response_model=List[InvoiceResponse],
    summary="Lister toutes les factures",
    description="Récupère toutes les factures avec leurs lignes de facturation et extra_data.",
)
def get_all_invoices():
    """
    GET /invoices

    Retourne toutes les factures triées par date de création (récentes en premier).
    """
    try:
        invoices = db_manager.get_all_invoices()
        return invoices
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération : {exc}")


@app.get(
    "/invoices/{invoice_id}",
    response_model=InvoiceResponse,
    summary="Récupérer une facture par ID",
    description="Récupère une facture spécifique avec ses items et extra_data.",
)
def get_invoice_by_id(invoice_id: int):
    """
    GET /invoices/{invoice_id}

    Retourne la facture correspondante ou une erreur 404 si introuvable.
    """
    try:
        invoice = db_manager.get_invoice_by_id(invoice_id)
        if invoice is None:
            raise HTTPException(
                status_code=404,
                detail=f"Facture avec id={invoice_id} introuvable."
            )
        invoice = _ensure_page_extractions(invoice)
        return invoice
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération : {exc}")

@app.patch(
    "/invoices/{invoice_id}",
    response_model=InvoiceResponse,
    summary="Mettre à jour et/ou confirmer une facture",
)
def update_invoice(
    invoice_id: int, 
    updates: InvoiceUpdate, 
    current_user: dict = Depends(get_current_user)
):
    """
    PATCH /invoices/{invoice_id}
    Permet de modifier les champs de la facture et enregistre l'utilisateur
    qui a confirmé l'opération via le token JWT fourni.
    """
    existing = db_manager.get_invoice_by_id(invoice_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Facture avec id={invoice_id} introuvable.")
    
    update_data = updates.model_dump(exclude_unset=True, exclude_none=False)
    
    updated = db_manager.update_invoice(
        invoice_id=invoice_id, 
        data=update_data, 
        user_id=current_user["user_id"]
    )
    return updated

@app.get("/files/{filename}", summary="Récupérer un fichier source")
def get_file(filename: str):
    """
    Sert les fichiers PDF ou images uploadés.
    """
    file_path = UPLOADS_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(str(file_path))

@app.post(
    "/invoices/upload",
    response_model=UploadInvoiceResponse,
    summary="Uploader et traiter une facture",
    description="Reçoit un fichier PDF ou image, exécute OCR + extraction LLM, persiste en PostgreSQL.",
)
def upload_invoice(
    file: UploadFile = File(...),
    pages: Optional[str] = Form(None),
):
    """
    POST /invoices/upload

    Le frontend envoie un fichier PDF ou image via multipart/form-data.
    Optionnel : `pages` pour limiter l'OCR à certaines pages PDF (ex. "[1,3]" ou "1,3").
    Le backend fait : validation → OCR → extraction LLM → persistance DB.
    Retourne la facture structurée complète.
    """
    # ── 1. Validation du format ─────────────────────────────────────────────
    original_filename = file.filename or "upload"
    file_ext = Path(original_filename).suffix.lower()

    if file_ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format '{file_ext}' non supporté. "
                   f"Formats acceptés : {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )

    # ── 2. Détection doublon (même nom de fichier déjà traité) ────────────
    existing_invoice = db_manager.get_invoice_by_original_filename(original_filename)
    if existing_invoice is not None:
        logger.info(
            "Upload ignoré — facture déjà existante (id=%s, fichier=%s)",
            existing_invoice["id"],
            original_filename,
        )
        return UploadInvoiceResponse(
            **existing_invoice,
            already_exists=True,
            message=f"La facture « {original_filename} » a déjà été uploadée et traitée.",
        )

    # ── 3. Sauvegarde persistante du fichier ────────────────────────────────
    try:
        unique_id = str(uuid.uuid4())[:8]
        saved_filename = f"{unique_id}_{original_filename.replace(' ', '_')}"
        file_path = UPLOADS_DIR / saved_filename

        file.file.seek(0)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = file_path.stat().st_size
        print(f"[DEBUG] Uploaded file saved to {file_path}. Size: {file_size} bytes.")
        
        if file_size == 0:
            # Cleanup et erreur
            file_path.unlink()
            raise HTTPException(status_code=400, detail="Le fichier uploadé est vide (0 octet).")

        pages_list: Optional[List[int]] = None
        total_pages: Optional[int] = None

        if pages is not None:
            pages_list = _parse_pages(pages)
            if file_ext != ".pdf":
                raise HTTPException(
                    status_code=400,
                    detail="Le paramètre pages ne s'applique qu'aux fichiers PDF.",
                )
            total_pages = ocr_engine.get_pdf_page_count(str(file_path))
            invalid = [p for p in pages_list if p > total_pages]
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Page(s) invalide(s) : {invalid}. "
                        f"Le document contient {total_pages} page(s)."
                    ),
                )
        elif file_ext == ".pdf":
            total_pages = ocr_engine.get_pdf_page_count(str(file_path))

        # ── 4. Pipeline OCR → Extraction LLM ───────────────────────────────
        print(f"[DEBUG] Starting OCR on {file_path}, pages={pages_list}")
        ocr_results, image_size = ocr_engine.read(str(file_path), pages=pages_list)
        print(f"[DEBUG] OCR complete, found {len(ocr_results)} text boxes, image_size={image_size}")

        page_extractions = None
        if pages_list and len(pages_list) > 1:
            page_extractions = _extract_per_page(ocr_results, pages_list, extractor)
            first_lines = [line for line in ocr_results if line.get("page") == pages_list[0]]
            invoice = extractor.extract(first_lines)
            local_to_global = [
                i for i, line in enumerate(ocr_results) if line.get("page") == pages_list[0]
            ]
            invoice.ocr_line_references = _remap_ocr_line_references(
                invoice.ocr_line_references,
                local_to_global,
            )
        else:
            invoice = extractor.extract(ocr_results)

        # Attacher les résultats OCR à la facture pour persistance
        invoice.ocr_lines = ocr_results
        invoice.ocr_image_size = image_size
        if pages_list is not None:
            invoice.selected_pages = pages_list
        if total_pages is not None:
            invoice.total_pages = total_pages
        if isinstance(image_size, dict) and "image_sizes" in image_size:
            invoice.ocr_image_sizes = image_size["image_sizes"]
        if page_extractions:
            invoice.page_extractions = page_extractions
        print(f"[DEBUG] Extraction complete")

        # ── 5. Persistance en PostgreSQL ────────────────────────────────────
        invoice_id = db_manager.save_invoice(invoice, saved_filename)

        # ── 6. Récupérer la facture complète depuis la DB pour la réponse ──
        saved_invoice = db_manager.get_invoice_by_id(invoice_id)
        return UploadInvoiceResponse(
            **saved_invoice,
            already_exists=False,
            message="Facture traitée avec succès.",
        )

    except InvoiceProcessorError as exc:
        raise HTTPException(status_code=422, detail=f"Erreur de traitement : {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur inattendue : {exc}")
