# Seam Carving

Content-aware image resizing using seam carving, with a local web interface for side-by-side comparison against standard bilinear resize.

Seam carving removes connected pixel paths ("seams") of least visual importance instead of uniformly scaling the image. The result preserves faces, edges, and salient content far better than a standard crop or resize.

Quick reference: https://en.wikipedia.org/wiki/Seam_carving

---

## Technologies

### Backend — C++ core (`src/`)

| Technology | Role |
|---|---|
| **C++17** | Core algorithm implementation |
| **OpenCV 4.x** | Image I/O (`cv::imread` / `cv::imwrite`), convolution (`cv::filter2D`), colour conversion, Canny edge detection, bilinear resize (`cv::resize`) |

The C++ binary is compiled to `build/seam_carving` and exposes a command-line interface. It does all heavy computation: energy map calculation, dynamic-programming seam finding, iterative seam removal, and standard resize.

### Web server — Python Flask (`web/app.py`)

| Technology | Role |
|---|---|
| **Python 3.9+** | Runtime |
| **Flask 2.3+** | HTTP server, routing, template rendering |
| **Werkzeug 2.3+** | Flask dependency (WSGI, file handling) |
| **Pillow 10+** | Image dimension detection fallback (when cv2 is not in the Flask venv) |
| **subprocess** | Launches the C++ binary as a child process |

Flask's only job is to accept HTTP requests, validate inputs, invoke the C++ binary via `subprocess.run()`, and stream the result file back to the browser.

### Frontend — Vanilla JS/HTML/CSS (`web/templates/index.html`)

No frameworks, no build step, no dependencies. A single self-contained HTML file served by Flask's `render_template()`.

| Feature | Implementation |
|---|---|
| Drag-and-drop upload | `dragover` / `drop` events on a styled `<div>` |
| Instant image preview | `FileReader.readAsDataURL()` — no network round-trip |
| Side-by-side comparison | CSS flexbox with three columns |
| Parallel processing | `Promise.all([fetch("/process"), fetch("/process_normal")])` |
| Download | Programmatic anchor click on a `blob:` URL |

---

## Prerequisites

### macOS

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenCV (includes pkg-config metadata used by the Makefile)
brew install opencv

# Install Python dependencies
pip3 install -r requirements_web.txt
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get install -y libopencv-dev python3-pip
pip3 install -r requirements_web.txt
```

### Verify OpenCV is found by the build system

```bash
make check-opencv
# Expected output:
#   OPENCV_CFLAGS: -I/opt/homebrew/include/opencv4   (path will vary)
#   OPENCV_LIBS:   -L/opt/homebrew/lib -lopencv_core ...
#   OpenCV: OK
```

---

## Build

```bash
cd /path/to/seam-carving
make
```

This compiles `src/main.cpp` and `src/seam_carving.cpp` into `build/seam_carving` with `-std=c++17 -O2`.

To clean the build:

```bash
make clean
```

---

## Running the web app

```bash
make run
```

This builds the C++ binary (if not already built) and starts Flask on `http://localhost:5000` with threading enabled.

> **macOS note:** Port 5000 may be occupied by AirPlay Receiver. Disable it in System Settings → General → AirDrop & Handoff, or kill the process first:
> ```bash
> lsof -ti :5000 | xargs kill -9
> ```

Open your browser at `http://localhost:5000`.

---

## Using the web interface

1. **Upload** — Drag and drop an image onto the upload zone, or click to browse. Supported formats: JPG, PNG, BMP, TIFF, WebP (max 20 MB).

2. **Configure** — Choose your options in the controls panel:

   | Control | Options | Notes |
   |---|---|---|
   | **Energy Algorithm** | Sobel, Prewitt, Laplacian, Roberts, Canny, Forward Energy | Sobel is a good general-purpose default; Forward Energy often produces fewer artefacts on gradual colour gradients |
   | **Seam Direction** | Vertical, Horizontal, Both | Vertical removes seams → reduces width; Horizontal reduces height; Both runs two passes |
   | **Resize Mode** | Scale %, Pixels | Scale is a percentage of the original dimension; Pixels lets you enter an exact target |

3. **Process** — Click "Process Image". The page fires both the seam-carving and normal-resize requests simultaneously and shows a spinner while they run.

4. **Compare** — Three panels appear side-by-side: Original, Seam Carving (green label), and Normal Resize (orange label), each with pixel dimensions.

5. **Download** — Click "Download Seam Carved" or "Download Normal Resize" to save the result as a PNG.

---

## CLI usage

The C++ binary can be used directly without the web interface:

```
./build/seam_carving --input <path> --output <path>
                     [--mode seam|normal]    default: seam
                     [--energy s|p|l|r|c|f]  default: s
                     [--direction v|h]        default: v
                     (--scale 0.05-0.99 | --width <px> | --height <px>)
```

### Examples

```bash
# Reduce width to 60% using Sobel energy
./build/seam_carving --input photo.jpg --output result.png --scale 0.6

# Reduce height to 400 px using Forward Energy
./build/seam_carving --input photo.jpg --output result.png \
  --energy f --direction h --height 400

# Reduce width to 800 px using Canny energy
./build/seam_carving --input photo.jpg --output result.png \
  --energy c --direction v --width 800

# Standard bilinear resize to 50% (for comparison)
./build/seam_carving --input photo.jpg --output result_normal.png \
  --mode normal --scale 0.5
```

### Flags

| Flag | Values | Description |
|---|---|---|
| `--input` | file path | Input image (any OpenCV-supported format) |
| `--output` | file path | Output image (written as PNG) |
| `--mode` | `seam` \| `normal` | `seam` = content-aware seam carving (default); `normal` = bilinear resize |
| `--energy` | `s` \| `p` \| `l` \| `r` \| `c` \| `f` | Energy algorithm (see below). Ignored when `--mode normal`. |
| `--direction` | `v` \| `h` | `v` removes vertical seams (reduces width); `h` removes horizontal seams (reduces height) |
| `--scale` | 0.05–0.99 | Target as a fraction of the original dimension |
| `--width` | positive int | Target width in pixels (use with `--direction v`) |
| `--height` | positive int | Target height in pixels (use with `--direction h`) |

Exactly one of `--scale`, `--width`, `--height` must be given.

---

## Energy algorithms

The energy algorithm determines how each pixel's "importance" is measured. High-energy pixels sit near strong edges or textures; the algorithm avoids removing them.

| Code | Algorithm | Description |
|---|---|---|
| `s` | **Sobel** | 3×3 gradient operator (centre-weighted). Fast, robust, good for most images. |
| `p` | **Prewitt** | Like Sobel but with uniform kernel weights. Slightly noisier. |
| `l` | **Laplacian** | Second-derivative (isotropic). Detects edges as zero-crossings; sensitive to noise. |
| `r` | **Roberts** | Diagonal 2×2 gradient. Oldest edge detector; fast but misses near-axis edges. |
| `c` | **Canny** | Multi-stage detector with hysteresis thresholding. Produces thin, clean edge maps. |
| `f` | **Forward Energy** | Rubinstein, Shamir & Avidan (2008). Measures the new gradient introduced *by* removing a seam, not the existing gradient. Often reduces visible artefacts on smooth colour gradients. |

---

## Project structure

```
seam-carving/
├── src/
│   ├── seam_carving.h       Public API: enums, structs, function declarations
│   ├── seam_carving.cpp     Core algorithm: energy functions, DP seam finder,
│   │                         seam removal loop, runSeamCarving, runNormalResize
│   └── main.cpp             CLI entry point: argument parsing, validation, dispatch
├── web/
│   ├── app.py               Flask server: upload, process, process_normal, health routes
│   └── templates/
│       └── index.html       Single-page frontend (HTML + CSS + vanilla JS)
├── build/
│   └── seam_carving         Compiled binary (created by `make`)
├── Makefile                 Build rules + OpenCV auto-detection
├── requirements_web.txt     Python dependencies: flask, werkzeug, Pillow
└── README.md                This file
```

---

## Web API reference

### `POST /upload`

Upload an image to start a session.

**Request:** `multipart/form-data` with field `image` (file).

**Response `200`:**
```json
{ "session_id": "a3f9...", "width": 1920, "height": 1080 }
```

**Error responses:** `400` (missing/unsupported file), `413` (file > 20 MB).

---

### `POST /process`

Run seam-carving resize for a previously uploaded image.

**Request body (JSON):**

```json
{
  "session_id": "a3f9...",
  "energy": "s",
  "direction": "v",
  "scale": 0.5
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | string | yes | From `/upload` |
| `energy` | string | no | `s`/`p`/`l`/`r`/`c`/`f` (default `"s"`) |
| `direction` | string | no | `"v"`, `"h"`, or `"both"` (default `"v"`) |
| `scale` | float | one of | Fraction in [0.05, 0.99]. For single direction. |
| `width` | int | one of | Target width px. With `direction="v"`. |
| `height` | int | one of | Target height px. With `direction="h"`. |
| `scale_v` + `scale_h` | float | one of | Independent scales. With `direction="both"`. |
| `width` + `height` | int | one of | Independent px targets. With `direction="both"`. |

**Response `200`:** PNG binary stream with headers:
- `Content-Type: image/png`
- `X-Result-Width: <int>`
- `X-Result-Height: <int>`

**Error responses:** `400` (bad params), `404` (session not found), `500` (binary error), `504` (timeout).

---

### `POST /process_normal`

Same as `/process` but runs bilinear resize (`--mode normal`). Same request/response format.

---

### `GET /health`

```json
{ "status": "ok", "binary": "/path/to/build/seam_carving", "binary_ok": true }
```

---

## Troubleshooting

**`Binary: missing — run make` badge**
The C++ binary hasn't been compiled yet. Run `make` from the project root.

**`OpenCV: FAILED`**
OpenCV isn't installed or pkg-config can't find it. Run `brew install opencv` (macOS) or `sudo apt-get install libopencv-dev` (Linux).

**`Address already in use` on port 5000 (macOS)**
AirPlay Receiver uses port 5000. Either disable it in System Settings → General → AirDrop & Handoff, or kill the process: `lsof -ti :5000 | xargs kill -9`.

**Processing takes a long time**
This is expected for large images: removing N seams requires N iterations of energy computation + DP. A 1920-wide image reduced by 50% requires 960 seam removals. Use `--energy s` (Sobel) for the fastest energy computation, or consider downscaling the image before uploading.

**Both results show the same image**
Make sure Flask is running with threading (`--with-threads` flag in `make run`). Without threading, the second parallel request queues behind the first.

---

## Acknowledgements

Original Python implementation by [gabrielkunz](https://github.com/gabrielkunz/seam-carving), drawing on work from:
- https://github.com/andrewdcampbell/seam-carving
- https://github.com/axu2/improved-seam-carving
- https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

Forward energy algorithm: Rubinstein, M., Shamir, A., & Avidan, S. (2008). *Improved Seam Carving for Video Retargeting*. ACM SIGGRAPH.

## License

[MIT](https://choosealicense.com/licenses/mit/)
