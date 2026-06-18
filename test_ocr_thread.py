from app.core.ocr_engine import OCREngine
from pathlib import Path
import threading

engine = OCREngine()
print("Initialized in main thread")

dummy_img = Path("dummy.png")
if not dummy_img.exists():
    from PIL import Image
    img = Image.new('RGB', (100, 30), color = (73, 109, 137))
    img.save(str(dummy_img))

def run_ocr():
    try:
        print("Testing PNG in thread...")
        res = engine.read("dummy.png")
        print("Success in thread:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()

t = threading.Thread(target=run_ocr)
t.start()
t.join()
