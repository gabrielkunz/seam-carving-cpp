"""
app.py — Flask web server for the Seam Carving tool.

Architecture overview:
  Browser  ──POST /upload──►  app.py  saves file, returns {session_id, w, h}
  Browser  ──POST /process──► app.py  calls C++ binary via subprocess, streams PNG back
  Browser  ──POST /process_normal──► same, but with --mode normal (bilinear resize)

The C++ binary (../../build/seam_carving) does all heavy lifting.
Flask's only jobs are: receive HTTP requests, validate inputs, build the
subprocess command, and stream the result file back to the browser.

Session model:
  Each upload generates a UUID-based subdirectory under UPLOAD_DIR.
  Files live there for the lifetime of the server process (no cleanup scheduled).
  The browser keeps the session_id and sends it with every /process request.
"""

import os
import uuid
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template

app = Flask(__name__)

# Path to the compiled C++ binary, relative to this file's location.
# BASE_DIR is web/, so BINARY_PATH is <project_root>/build/seam_carving.
BASE_DIR    = Path(__file__).parent
BINARY_PATH = BASE_DIR.parent / "build" / "seam_carving"

# Temporary directory for uploaded images and processed results.
# Uses the OS temp dir (e.g. /var/folders/... on macOS) so it survives
# reboots gracefully without polluting the project directory.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "seam_carving_web"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# File extensions accepted at /upload.  We rely on the C++ binary (OpenCV)
# for actual format support — this list is a fast pre-flight check.
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Hard limit on uploaded file size.  Checked before saving to disk to avoid
# filling up the temp directory with huge uploads.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# Single-character codes understood by the C++ binary's --energy flag.
VALID_ENERGY_CODES = {"s", "p", "l", "r", "c", "f"}

# Direction values understood by the binary's --direction flag, plus "both"
# which is handled entirely in Python (two sequential binary calls).
VALID_DIRECTIONS = {"v", "h", "both"}


def _allowed(filename: str) -> bool:
    """Return True if the filename has an accepted image extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _session_dir(session_id: str) -> Path:
    """
    Return (and create if necessary) the per-session working directory.

    All files for a single upload+process cycle live here:
      input.<ext>          — the original upload
      result.png           — seam-carved output
      result_normal.png    — normal-resize output
      intermediate.png     — temp file used when direction='both'
    """
    d = UPLOAD_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _image_dimensions(path: Path):
    """
    Read pixel dimensions (width, height) from an image file.

    Tries OpenCV first (faster, already a dependency of the C++ binary's
    environment), then falls back to Pillow.  Returns (0, 0) if both fail
    so callers get a graceful zero rather than an exception.
    """
    # Attempt 1: OpenCV (cv2 may not be installed in the Flask venv)
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            # cv2 shape is (height, width, channels) — note reversed order
            return img.shape[1], img.shape[0]
    except Exception:
        pass

    # Attempt 2: Pillow (listed in requirements_web.txt)
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            return im.size  # Pillow returns (width, height) directly
    except Exception:
        pass

    return 0, 0


def _run_binary(input_path: Path, output_path: Path,
                energy: str, direction: str,
                scale: float = None, width: int = None, height: int = None,
                mode: str = "seam"):
    """
    Invoke the C++ binary as a subprocess and wait for it to finish.

    Builds a command list from the provided parameters, then calls
    subprocess.run() with capture_output=True so stderr (progress messages
    and timing) is collected but not printed to the Flask log.

    Parameters:
      input_path  — Source image file (must exist).
      output_path — Where the binary should write its PNG result.
      energy      — Single-letter energy code: s|p|l|r|c|f.
      direction   — "v" (reduce width) or "h" (reduce height).
      scale       — Fractional target size, e.g. 0.5.  Mutually exclusive
                    with width and height.
      width       — Target width in pixels (direction="v" only).
      height      — Target height in pixels (direction="h" only).
      mode        — "seam" for seam carving, "normal" for bilinear resize.

    Raises RuntimeError if the binary exits with a non-zero status code,
    including the first 2 000 characters of stderr as the error message.
    """
    cmd = [
        str(BINARY_PATH),
        "--input",     str(input_path),
        "--output",    str(output_path),
        "--mode",      mode,
        "--energy",    energy,
        "--direction", direction,
    ]

    # Exactly one of scale / width / height must be appended.
    # The caller is responsible for passing only one non-None value.
    if scale is not None:
        cmd += ["--scale", str(scale)]
    elif width is not None:
        cmd += ["--width", str(width)]
    elif height is not None:
        cmd += ["--height", str(height)]

    # timeout=300 s (5 min) — generous for large images; prevents the server
    # from hanging indefinitely if the binary gets stuck.
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:2000] or "C++ binary exited with non-zero status")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page frontend (web/templates/index.html)."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accept a multipart/form-data upload and save the image to a session dir.

    Expected form field: "image" (file).

    Validation steps:
      1. Field "image" must be present.
      2. Filename extension must be in ALLOWED_EXTENSIONS.
      3. File size must be ≤ MAX_UPLOAD_BYTES (checked without loading into RAM
         by seeking to the end of the stream).

    On success, returns JSON:
      { "session_id": "<hex uuid>", "width": <int>, "height": <int> }

    The browser stores session_id and sends it with every subsequent /process
    call so we can locate the saved file without re-uploading.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if not file.filename or not _allowed(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Check size before saving — seek to end to get byte count without reading
    file.seek(0, 2)   # 0 bytes from end
    size = file.tell()
    file.seek(0)      # reset for subsequent read/save
    if size > MAX_UPLOAD_BYTES:
        return jsonify({"error": "File too large (max 20 MB)"}), 413

    # Create a UUID-based session directory and save the file
    session_id = uuid.uuid4().hex
    sdir = _session_dir(session_id)
    suffix = Path(file.filename).suffix.lower()
    input_path = sdir / f"input{suffix}"   # keep original extension for OpenCV
    file.save(str(input_path))

    # Return dimensions so the UI can show them immediately
    w, h = _image_dimensions(input_path)
    return jsonify({"session_id": session_id, "width": w, "height": h}), 200


@app.route("/process", methods=["POST"])
def process():
    """
    Run seam-carving resize and stream the result PNG back to the browser.

    Expected JSON body fields:
      session_id  — UUID from /upload (required).
      energy      — Energy code: s|p|l|r|c|f  (default "s").
      direction   — "v", "h", or "both"        (default "v").

      For direction "v" or "h" — exactly one of:
        scale   — float in [0.05, 0.99]
        width   — positive int (used when direction="v")
        height  — positive int (used when direction="h")

      For direction "both" — one of:
        scale_v + scale_h   — independent scale fractions for each axis
        width   + height    — pixel targets for both axes

    The "both" direction is handled entirely in Python: the binary is called
    twice, first to reduce width (direction=v), then to reduce height
    (direction=h) using the first call's output as input.

    On success: streams a PNG file with custom response headers:
      X-Result-Width, X-Result-Height — dimensions of the output image.

    On error: returns JSON { "error": "...", "detail": "..." }.
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Expected JSON body"}), 400

    # --- Session lookup ---

    session_id = data.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    sdir = _session_dir(session_id)
    # Find the uploaded file regardless of its extension
    candidates = [f for f in sdir.iterdir()
                  if f.stem == "input" and f.suffix in ALLOWED_EXTENSIONS]
    if not candidates:
        return jsonify({"error": "Session not found or no uploaded image"}), 404
    input_path = candidates[0]

    # --- Parameter validation ---

    energy = data.get("energy", "s")
    if energy not in VALID_ENERGY_CODES:
        return jsonify({"error": f"Invalid energy code: {energy}"}), 400

    direction = data.get("direction", "v")
    if direction not in VALID_DIRECTIONS:
        return jsonify({"error": f"Invalid direction: {direction}"}), 400

    # scale_v / scale_h are used for "both" direction.
    # scale_v is also reused for single-direction scale to avoid extra variables.
    scale_v = scale_h = None
    target_w = target_h = None

    raw_scale   = data.get("scale")
    raw_scale_v = data.get("scale_v")
    raw_scale_h = data.get("scale_h")
    raw_width   = data.get("width")
    raw_height  = data.get("height")

    try:
        if direction == "both":
            # "both" requires two independent resize specs (one per axis)
            if raw_scale_v is not None and raw_scale_h is not None:
                scale_v = float(raw_scale_v)
                scale_h = float(raw_scale_h)
                if not (0.05 <= scale_v < 1.0) or not (0.05 <= scale_h < 1.0):
                    raise ValueError("scale values must be in [0.05, 0.99]")
            elif raw_width is not None and raw_height is not None:
                target_w = int(raw_width)
                target_h = int(raw_height)
                if target_w < 1 or target_h < 1:
                    raise ValueError("width and height must be positive integers")
            else:
                return jsonify({"error": "For direction 'both' supply scale_v+scale_h or width+height"}), 400
        else:
            # Single axis: exactly one of scale / width / height
            count = sum(x is not None for x in [raw_scale, raw_width, raw_height])
            if count != 1:
                return jsonify({"error": "Exactly one of scale, width, height must be provided"}), 400
            if raw_scale is not None:
                scale_v = float(raw_scale)
                if not (0.05 <= scale_v < 1.0):
                    raise ValueError("scale must be in [0.05, 0.99]")
            elif raw_width is not None:
                target_w = int(raw_width)
                if target_w < 1:
                    raise ValueError("width must be a positive integer")
            else:
                target_h = int(raw_height)
                if target_h < 1:
                    raise ValueError("height must be a positive integer")
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    if not BINARY_PATH.exists():
        return jsonify({"error": "C++ binary not found. Run 'make' first."}), 500

    output_path = sdir / "result.png"

    # --- Invoke binary ---

    try:
        if direction == "both":
            # Two-pass: first reduce width (v), then reduce height (h).
            # intermediate.png is a throw-away file between the two passes.
            intermediate = sdir / "intermediate.png"
            _run_binary(input_path, intermediate, energy, "v",
                        scale=scale_v, width=target_w)
            _run_binary(intermediate, output_path, energy, "h",
                        scale=scale_h, height=target_h)
        elif direction == "v":
            _run_binary(input_path, output_path, energy, "v",
                        scale=scale_v, width=target_w)
        else:  # direction == "h"
            # Note: single-direction horizontal uses scale_v (not scale_h),
            # because scale_h is only populated for the "both" path.
            _run_binary(input_path, output_path, energy, "h",
                        scale=scale_v, height=target_h)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timed out (5 min limit)"}), 504
    except FileNotFoundError:
        return jsonify({"error": "C++ binary not found. Run 'make' first."}), 500
    except RuntimeError as e:
        return jsonify({"error": "Processing failed", "detail": str(e)}), 500

    if not output_path.exists():
        return jsonify({"error": "Binary succeeded but output file not created"}), 500

    # Read output dimensions so the browser can display them without decoding the image
    rw, rh = _image_dimensions(output_path)

    response = send_file(
        str(output_path),
        mimetype="image/png",
        as_attachment=False,
        download_name="seam_carved_result.png",
    )
    # Custom headers carry the output dimensions alongside the image binary.
    # Access-Control-Expose-Headers is required for the browser's fetch() to
    # see non-standard headers when the page and server share the same origin.
    response.headers["X-Result-Width"]  = str(rw)
    response.headers["X-Result-Height"] = str(rh)
    response.headers["Access-Control-Expose-Headers"] = "X-Result-Width, X-Result-Height"
    return response


@app.route("/process_normal", methods=["POST"])
def process_normal():
    """
    Run standard bilinear resize and stream the result PNG back.

    Accepts exactly the same JSON body as /process.  The only difference is
    that _run_binary() is called with mode="normal", which passes --mode normal
    to the C++ binary.  The binary then calls cv::resize (INTER_LINEAR) instead
    of the seam-carving loop.

    This route exists so the frontend can fire /process and /process_normal in
    parallel (Promise.all) and display both results side by side for comparison.

    The energy parameter is forwarded to the binary but has no effect in normal
    mode — the C++ binary ignores it when --mode normal is given.

    Error handling and response format are identical to /process.
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Expected JSON body"}), 400

    session_id = data.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    sdir = _session_dir(session_id)
    candidates = [f for f in sdir.iterdir()
                  if f.stem == "input" and f.suffix in ALLOWED_EXTENSIONS]
    if not candidates:
        return jsonify({"error": "Session not found or no uploaded image"}), 404
    input_path = candidates[0]

    energy    = data.get("energy", "s")   # forwarded but ignored by normal-mode binary
    direction = data.get("direction", "v")
    if direction not in VALID_DIRECTIONS:
        return jsonify({"error": f"Invalid direction: {direction}"}), 400

    raw_scale   = data.get("scale")
    raw_scale_v = data.get("scale_v")
    raw_scale_h = data.get("scale_h")
    raw_width   = data.get("width")
    raw_height  = data.get("height")

    scale_v = scale_h = None
    target_w = target_h = None

    try:
        if direction == "both":
            if raw_scale_v is not None and raw_scale_h is not None:
                scale_v = float(raw_scale_v)
                scale_h = float(raw_scale_h)
            elif raw_width is not None and raw_height is not None:
                target_w = int(raw_width)
                target_h = int(raw_height)
            else:
                return jsonify({"error": "For direction 'both' supply scale_v+scale_h or width+height"}), 400
        else:
            count = sum(x is not None for x in [raw_scale, raw_width, raw_height])
            if count != 1:
                return jsonify({"error": "Exactly one of scale, width, height must be provided"}), 400
            if raw_scale is not None:
                scale_v = float(raw_scale)
            elif raw_width is not None:
                target_w = int(raw_width)
            else:
                target_h = int(raw_height)
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    if not BINARY_PATH.exists():
        return jsonify({"error": "C++ binary not found. Run 'make' first."}), 500

    # Separate output file so it doesn't overwrite the seam-carving result
    # if both requests complete around the same time.
    output_path = sdir / "result_normal.png"

    try:
        if direction == "both":
            intermediate = sdir / "intermediate_normal.png"
            _run_binary(input_path, intermediate, energy, "v",
                        scale=scale_v, width=target_w, mode="normal")
            _run_binary(intermediate, output_path, energy, "h",
                        scale=scale_h, height=target_h, mode="normal")
        elif direction == "v":
            _run_binary(input_path, output_path, energy, "v",
                        scale=scale_v, width=target_w, mode="normal")
        else:
            _run_binary(input_path, output_path, energy, "h",
                        scale=scale_v, height=target_h, mode="normal")
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Processing timed out"}), 504
    except FileNotFoundError:
        return jsonify({"error": "C++ binary not found. Run 'make' first."}), 500
    except RuntimeError as e:
        return jsonify({"error": "Processing failed", "detail": str(e)}), 500

    if not output_path.exists():
        return jsonify({"error": "Binary succeeded but output file not created"}), 500

    rw, rh = _image_dimensions(output_path)
    response = send_file(
        str(output_path),
        mimetype="image/png",
        as_attachment=False,
        download_name="normal_resized_result.png",
    )
    response.headers["X-Result-Width"]  = str(rw)
    response.headers["X-Result-Height"] = str(rh)
    response.headers["Access-Control-Expose-Headers"] = "X-Result-Width, X-Result-Height"
    return response


@app.route("/health")
def health():
    """
    Liveness + readiness probe used by the frontend badge in the page header.

    Returns JSON: { "status": "ok", "binary": "<path>", "binary_ok": <bool> }

    binary_ok is True only if the C++ binary exists AND is executable.
    The frontend shows a green "Binary: OK" badge when binary_ok is True, or a
    red warning prompting the user to run 'make' otherwise.
    """
    binary_ok = BINARY_PATH.exists() and os.access(str(BINARY_PATH), os.X_OK)
    return jsonify({
        "status":    "ok",
        "binary":    str(BINARY_PATH),
        "binary_ok": binary_ok,
    })


if __name__ == "__main__":
    # threaded=True is required so that /process and /process_normal can run
    # concurrently.  The browser fires both requests simultaneously via
    # Promise.all(); without threading, the second request would queue behind
    # the first (which blocks waiting for the C++ subprocess to finish),
    # effectively doubling total processing time.
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
