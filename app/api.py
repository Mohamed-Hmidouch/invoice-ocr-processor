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
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.database import DatabaseManager
from app.core import OCREngine, Extractor
from app.core.exceptions import InvoiceProcessorError


# Formats de fichiers acceptés par l'endpoint d'upload
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".pdf"}

# Dossier pour stocker les uploads persistants
UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


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
    items: List[InvoiceItemResponse] = Field(default_factory=list)


class InvoiceCreateResponse(BaseModel):
    """Réponse après création d'une facture."""
    id: int
    message: str


# ═══════════════════════════════════════════════════════════════════════════════
# LIFESPAN (Connexion DB gérée au démarrage / arrêt du serveur)
# ═══════════════════════════════════════════════════════════════════════════════
from dotenv import load_dotenv
load_dotenv()

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
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (permet au frontend de communiquer avec l'API) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # En prod : remplacer par le domaine du front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
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
        return invoice
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération : {exc}")

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
    response_model=InvoiceResponse,
    status_code=201,
    summary="Uploader et traiter une facture",
    description="Reçoit un fichier PDF ou image, exécute OCR + extraction LLM, persiste en PostgreSQL.",
)
def upload_invoice(file: UploadFile = File(...)):
    """
    POST /invoices/upload

    Le frontend envoie un fichier PDF ou image via multipart/form-data.
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

    # ── 2. Sauvegarde persistante du fichier ────────────────────────────────
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

        # ── 3. Pipeline OCR → Extraction LLM ───────────────────────────────
        print(f"[DEBUG] Starting OCR on {file_path}")
        ocr_results, image_size = ocr_engine.read(str(file_path))
        print(f"[DEBUG] OCR complete, found {len(ocr_results)} text boxes, image_size={image_size}")
        
        invoice = extractor.extract(ocr_results)
        # Attacher les résultats OCR à la facture pour persistance
        invoice.ocr_lines = ocr_results
        invoice.ocr_image_size = image_size
        print(f"[DEBUG] Extraction complete")

        # ── 4. Persistance en PostgreSQL ────────────────────────────────────
        invoice_id = db_manager.save_invoice(invoice, saved_filename)

        # ── 5. Récupérer la facture complète depuis la DB pour la réponse ──
        saved_invoice = db_manager.get_invoice_by_id(invoice_id)
        return saved_invoice

    except InvoiceProcessorError as exc:
        raise HTTPException(status_code=422, detail=f"Erreur de traitement : {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur inattendue : {exc}")
