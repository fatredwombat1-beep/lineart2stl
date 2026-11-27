# lineart2stl — notes for Claude Code

Web app in a container: upload a JPEG/PNG and download a **single-piece, base-less STL outline** for 3D printing
(Bambu P1S, 0.4 mm nozzle). It runs on Matt's Unraid server (container `lineart2stl`, host port 8090, template
`unraid/lineart2stl.xml`). See README.md for user-facing details.

## Layout
- `app/converter.py`: core pipeline. Image → mask → shapely polygons (mm) → thicken → optional frame (absorbs a
  border already drawn in the image) → drop specks → bridge islands with a minimum spanning tree → extrude
  (manifold3d) → STL + preview PNG. Bridges are drawn red in the preview.
- `app/lineart_model.py`: **AI photo mode**. ONNX "Informative Drawings" generator (ControlNet `lineart`
  annotator weights `sk_model.pth` = "fine", `sk_model2.pth` = "bold"). Output = dark lines on white, so the
  line strength is `1 - y`. Thresholded with Otsu (+ per-style `THRESH_OFFSET`), clamped to 0.25–0.6. Background thinning uses MediaPipe person
  segmentation. Strokes get a minimum width of about 0.5 mm.
- `app/portrait.py`: "Classic" photo mode (no network): MediaPipe face mesh draws the features, people get an
  outline from segmentation, DoG edges fill in the rest. Also the fallback when the models are missing.
- `app/main.py`: FastAPI app (`/`, `/api/convert`, `/files/{id}/{name}`, `/health`). `app/static/index.html`: UI.
- `tools/export_lineart.py`: runs in Docker build stage 1. Downloads the weights from Hugging Face, loads them
  with `strict=True` and exports ONNX (opset 17, dynamic H/W). The final image only has onnxruntime.
- Python 3.12 is required: mediapipe 0.10.21 is the last release that bundles the legacy solutions models.
  mediapipe pulls in GUI OpenCV and JAX; the Dockerfile swaps in headless OpenCV and removes JAX.

## Deploying changes on Unraid
```bash
cd /mnt/user/appdata/lineart2stl-src && git pull && docker build -t lineart2stl:latest .
docker rm -f lineart2stl   # then re-add from the my-lineart2stl template in the Docker tab, or from a shell:
cd /usr/local/emhttp && eval "$(php -r '$docroot="/usr/local/emhttp"; require_once "$docroot/webGui/include/Wrappers.php";
  require_once "$docroot/plugins/dynamix.docker.manager/include/DockerClient.php"; require_once "$docroot/plugins/dynamix.docker.manager/include/Helpers.php";
  $var=parse_ini_file("state/var.ini"); $subnet=DockerUtil::network(DockerUtil::custom());
  echo xmlToCommand("/boot/config/plugins/dockerMan/templates-user/my-lineart2stl.xml")[0];')" && docker start lineart2stl
```
Gotcha: clicking Edit → Apply with no changes did **not** move the container to the new image. Check with
`docker ps --filter name=lineart2stl --format '{{.Image}}'`.

## Testing without the web UI
The ONNX models live inside the image at `/app/models/`. To run the pipeline on a test photo:
```bash
docker run --rm -v "$PWD/test:/test" -w /app lineart2stl:latest python -c "
from converter import convert, Settings
r = convert(open('/test/photo.jpg','rb').read(), Settings(mode='photo', style='bold', detail=7, background=5))
open('/test/out.stl','wb').write(r['stl']); open('/test/preview.png','wb').write(r['preview']); print(r['stats'])"
```
Matt's test photo is a full-length group shot of four people (faces about 18 mm wide at 200 mm print width).
**Do not commit personal photos to this repo.**

## Done 2026-11-27 (Matt's feedback round 1)
1. **False moustache**: not the threshold. `_mask_to_geometry` filled holes whose contour touched the outline
   (invalid polygon → `make_valid` filled it), so nostril/mouth outlines became solid. Pieces are now built as
   outline minus the union of holes. Blob thinning in face regions was tried and dropped (no gain once fixed).
2. **Threshold**: Otsu on the real weights is ~0.41 (fine) / ~0.43 (bold) at detail 5–9; the clamp never
   triggers. `THRESH_OFFSET` lowers fine by 0.06 (fuller glasses/eyes/mouths, fewer islands).
3. **Background**: default 5. Long near-vertical lines off the people (Hough on the off-person skeleton,
   ≤8° from vertical, ≥ L/14 long) are removed; brick courses and the ground stay. Fine style draws little
   brick even at 5; bold keeps more.
4. **Islands**: photo mode closes gaps under `merge_mm` (0.6 mm, in the UI) before bridging: 209 → 86 islands
   on the test photo. 1.0 mm starts filling eyes. Bridge width stays 1 mm.
5. **Thin joins**: caused by joins, not stroke width. After simplify, the outline is shrunk by 0.3 mm and the
   resulting pieces are bridged together, so `thin_check` is now "ok". These show as small red marks, and the
   `bridges` stat counts them. Bridge endpoints use the simplified shapes: conversion 14 s → 7 s.

## Conventions
- Matt prefers concise answers and complete, ready-to-use files.
- Commit messages: imperative summary plus a short body.
