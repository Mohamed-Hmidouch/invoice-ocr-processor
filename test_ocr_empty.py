from app.core.ocr_engine import OCREngine
from pathlib import Path

engine = OCREngine()
Path("empty.pdf").touch()
try:
    engine.read("empty.pdf")
except Exception as e:
    print("Empty PDF Error:", repr(e))

Path("empty.png").touch()
try:
    engine.read("empty.png")
except Exception as e:
    print("Empty PNG Error:", repr(e))
