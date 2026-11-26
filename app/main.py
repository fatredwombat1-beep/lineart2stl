import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from converter import ConversionError, Settings, convert

DATA = Path(os.environ.get("DATA_DIR", "/data"))
KEEP_HOURS = float(os.environ.get("KEEP_HOURS", "24"))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "25"))
WORKERS = int(os.environ.get("MAX_CONCURRENT", "2"))

DATA.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="lineart2stl")
_slots = threading.BoundedSemaphore(WORKERS)
INDEX = (Path(__file__).parent / "static" / "index.html").read_text()
ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _cleanup():
    cutoff = time.time() - KEEP_HOURS * 3600
    for d in DATA.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/convert")
def api_convert(
    file: UploadFile = File(...),
    mode: str = Form("lineart"),
    width_mm: float = Form(200.0),
    thickness_mm: float = Form(2.4),
    thicken_mm: float = Form(0.4),
    min_speck_mm2: float = Form(1.0),
    bridge_mm: float = Form(1.0),
    frame_mm: float = Form(4.0),
    invert: bool = Form(False),
    crop: bool = Form(True),
    absorb_border: bool = Form(True),
    detail: int = Form(5),
    background: int = Form(3),
    style: str = Form("fine"),
):
    _cleanup()
    data = file.file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB:g} MB.")
    s = Settings(mode=mode, width_mm=width_mm, thickness_mm=thickness_mm, thicken_mm=thicken_mm,
                 min_speck_mm2=min_speck_mm2, bridge_mm=bridge_mm, frame_mm=frame_mm,
                 invert=invert, crop=crop, absorb_border=absorb_border, detail=detail,
                 background=background, style=style)
    if not _slots.acquire(timeout=120):
        raise HTTPException(503, "Server busy - try again shortly.")
    try:
        result = convert(data, s)
    except ConversionError as e:
        raise HTTPException(422, str(e))
    finally:
        _slots.release()

    job = uuid.uuid4().hex
    out = DATA / job
    out.mkdir()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(file.filename or "image").stem)[:60] or "image"
    (out / f"{stem}.stl").write_bytes(result["stl"])
    (out / "preview.png").write_bytes(result["preview"])
    (out / "stats.json").write_text(json.dumps(result["stats"]))
    return JSONResponse({
        "id": job,
        "stl_url": f"/files/{job}/{stem}.stl",
        "preview_url": f"/files/{job}/preview.png",
        "stats": result["stats"],
    })


@app.get("/files/{job}/{name}")
def files(job: str, name: str):
    if not ID_RE.match(job) or "/" in name or name.startswith("."):
        raise HTTPException(404)
    p = DATA / job / name
    if not p.is_file():
        raise HTTPException(404, "File expired or not found.")
    if name.endswith(".stl"):
        return FileResponse(p, media_type="model/stl", filename=name)
    return FileResponse(p)
