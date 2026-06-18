from app.core.ocr_engine import OCREngine
from pathlib import Path
engine = OCREngine()
print("Initialized")
# Try passing a dummy pdf
dummy = Path("dummy.pdf")
if not dummy.exists():
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(dummy))
    c.drawString(100, 100, "Hello World")
    c.save()
print("Testing PDF...")
try:
    res = engine.read("dummy.pdf")
    print(res)
except Exception as e:
    import traceback
    traceback.print_exc()

# Try passing a dummy png
dummy_img = Path("dummy.png")
if not dummy_img.exists():
    from PIL import Image
    img = Image.new('RGB', (100, 30), color = (73, 109, 137))
    img.save(str(dummy_img))
print("Testing PNG...")
try:
    res = engine.read("dummy.png")
    print(res)
except Exception as e:
    import traceback
    traceback.print_exc()
