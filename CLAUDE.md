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
  line strength is `1 - y`. Thresholded with Otsu clamped to 0.25–0.6. Background thinning uses MediaPipe person
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
docker rm -f lineart2stl   # then re-add from the my-lineart2stl template in the Docker tab
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

## Open tasks (from Matt's feedback, 2026-11-27)
1. **False moustache** on the groom in AI style (bold, detail 7): the network's shading under the nose survives
   the threshold. Idea: use a higher threshold, or remove thick, low-contrast blobs, inside face regions (face
   boxes from MediaPipe face detection, as in `portrait._faces`), while keeping eyes, glasses and the mouth line.
   Compare the fine and bold styles too.
2. **Threshold tuning** for the real weights. The Otsu clamp (0.25–0.6) was chosen without the real model. Check it
   on the test photo at detail 5–9.
3. **Background:** Matt *likes some background* for context (brick wall, ground). Keep it, but remove stray long
   straight verticals (door edge, downpipe) if they look like noise. Raise the UI default for *Background lines*
   from 3 to about 5.
4. Many small islands lead to many red bridges on faces. Consider merging very close islands first, or a lower
   default bridge width (0.8 mm).
5. The preview shows a "some joins are under 0.6 mm" warning. Check whether the min-width step in
   `lineart_model.py` should be stronger.

## Conventions
- Matt prefers concise answers and complete, ready-to-use files.
- Commit messages: imperative summary plus a short body.
