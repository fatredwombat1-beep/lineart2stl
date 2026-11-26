# lineart2stl

A small web app: drop in a JPEG/PNG of line art and download a **single-piece STL outline**. It prints flat with no base and no floating parts.

What it does to the image:
1. Converts it to black and white (Otsu threshold) and traces the lines into vectors.
2. Thickens the lines (default +0.4 mm) so a 0.4 mm nozzle can print them.
3. Adds an optional solid frame, and merges any border already drawn in the image into that frame.
4. Removes specks below a size threshold (default 1 mm²).
5. Joins every floating piece (eyes, logos and so on) to the main body with short bridges (minimum spanning tree, default 1 mm wide).
6. Extrudes the result into a watertight STL and makes a preview, with the bridges shown in red.

**Photo mode** turns a photo into a colouring-page style drawing before the steps above:
- **Faces** are found with MediaPipe, and the jawline, brows, eyes, irises, nose and lips are drawn as clean lines from the face landmarks.
- **People** get a bold outline from person segmentation, plus some interior detail (clothing folds and hands), controlled by *Detail on people*.
- **The background** is reduced to a few long lines (*Background lines*, 0 = none), so the scene is hinted at rather than traced.

Photo mode works best on photos with a few people who are reasonably close to the camera. Very small faces get simplified features.

## Run on Unraid (shows in the Docker tab)

1. In the Unraid terminal, get the code, build the image and install the template:
   ```bash
   cd /mnt/user/appdata
   git clone https://github.com/fatredwombat1-beep/lineart2stl.git lineart2stl-src
   cd lineart2stl-src
   docker build -t lineart2stl:latest .
   cp unraid/lineart2stl.xml /boot/config/plugins/dockerMan/templates-user/my-lineart2stl.xml
   ```
2. In the web GUI go to **Docker → Add Container**, pick **lineart2stl** from the *Template*
   dropdown (under User templates), check the port and path, then click **Apply**.
3. Click the container icon → **WebUI**, or open `http://<server-ip>:8085`.

The container then works like any other in the Docker tab: start and stop, logs, Edit, autostart and the WebUI link.
Because the image is built locally, Unraid's update check shows "not available". That's expected.

**Updating:**
```bash
cd /mnt/user/appdata/lineart2stl-src && git pull && docker build -t lineart2stl:latest .
```
Then click the container → **Edit** → **Apply** (no changes needed) to recreate it on the new image.

### Alternative: Docker Compose
Unraid doesn't come with Compose. Install the **Compose Manager Plus** plugin from Community Apps first,
then run `docker compose up -d --build` in this folder. Compose containers show in the plugin's own section of the Docker tab.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `KEEP_HOURS` | 24 | Generated files are deleted after this many hours |
| `MAX_UPLOAD_MB` | 25 | Upload size limit |
| `MAX_CONCURRENT` | 2 | Conversions allowed at the same time (each one uses about 1 CPU core and a few hundred MB of RAM) |

The container runs as `99:100` (nobody:users), which matches Unraid appdata permissions. It only needs to write to `/data`.

## Tips

- If the preview warns that some joins are under 0.6 mm, increase **Thicken lines by** to 0.6–0.8.
- If a noisy image gives "too many pieces", raise **Remove specks smaller than**.
- For stiffer prints, use 3–4 mm thickness. PETG flexes better than ABS for thin lattices.
- It works behind a reverse proxy, including on a sub-path, because all URLs are relative.

## API

`POST /api/convert` (multipart form): `file` plus any of `mode`, `width_mm`, `thickness_mm`,
`thicken_mm`, `min_speck_mm2`, `bridge_mm`, `frame_mm`, `invert`, `crop`, `absorb_border`, `detail`.
Returns JSON with `stl_url`, `preview_url` and `stats`.

```bash
curl -F file=@drawing.jpg -F width_mm=150 http://server:8085/api/convert
```
