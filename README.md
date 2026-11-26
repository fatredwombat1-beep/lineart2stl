# lineart2stl

A small web app: drop in a JPEG/PNG of line art and download a **single-piece STL outline**. It prints flat with no base and no floating parts.

What it does to the image:
1. Converts it to black and white (Otsu threshold) and traces the lines into vectors.
2. Thickens the lines (default +0.4 mm) so a 0.4 mm nozzle can print them.
3. Adds an optional solid frame, and merges any border already drawn in the image into that frame.
4. Removes specks below a size threshold (default 1 mm²).
5. Joins every floating piece (eyes, logos and so on) to the main body with short bridges (minimum spanning tree, default 1 mm wide).
6. Extrudes the result into a watertight STL and makes a preview, with the bridges shown in red.

"Photo" mode is experimental. It uses edge detection, so results depend heavily on the photo. Clean, high-contrast line drawings work best.

## Run on Unraid

1. Copy this folder to the server, e.g. `/mnt/user/appdata/lineart2stl-src/`.
2. In the Unraid terminal:
   ```bash
   cd /mnt/user/appdata/lineart2stl-src
   docker compose up -d --build
   ```
   (Compose is built into Unraid 7. On Unraid 6 use the *Compose Manager* plugin, or
   `docker build -t lineart2stl . && docker run -d --name lineart2stl -p 8085:8080 -v /mnt/user/appdata/lineart2stl:/data --restart unless-stopped lineart2stl`.)
3. Open `http://<server-ip>:8085`.

To update after editing the code, run `docker compose up -d --build` again.

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
