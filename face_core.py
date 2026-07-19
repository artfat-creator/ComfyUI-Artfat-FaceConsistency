"""Artfat Face Consistency — shared core.

Face-identity scoring with OpenCV's built-in YuNet (detector) + SFace
(recognizer). No insightface / onnx-runtime / dlib, no GPU — pure cv2 + numpy on
CPU, so it never fights the sampler for VRAM.

Both ComfyUI nodes (Batch / Sort) and the standalone face_consistency_sort.py
script build on this module, so scoring stays identical everywhere.

Public API:
    get_models()                      -> (detector, recognizer)   (cached)
    comfy_to_bgr(image_tensor)        -> [bgr_uint8, ...]          (ComfyUI IMAGE -> cv2)
    bgr_to_comfy(bgr)                 -> torch [1,H,W,C] 0..1 RGB  (cv2 -> ComfyUI IMAGE)
    embed_bgr(bgr)                    -> (embedding|None, n_faces)
    cosine(a, b)                      -> float
    build_reference(bgr_list)         -> (centroid|None, n_used, self_consistency)
    load_folder_bgr(folder)           -> [(path, bgr), ...]
    verdict_for(sim, keep, reject)    -> "keep"|"borderline"|"reject"|"noface"
    contact_sheet(rows, cols, thumb)  -> bgr canvas
"""
import os
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    raise ImportError("cv2 not found. ComfyUI's python must have opencv (it ships with it).")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_HERE = Path(__file__).resolve().parent


def _comfy_models_dir():
    """Portable path to ComfyUI's models/ dir (works on any install)."""
    try:
        import folder_paths
        return Path(folder_paths.models_dir)
    except Exception:
        # pack lives at .../custom_nodes/<pack>/ -> app/models is two up
        return _HERE.parent.parent / "models"


_MODELS = _comfy_models_dir()

# yunet/sface ship inside this pack; also look in ComfyUI's models/face_check_models.
_MODEL_DIRS = [
    _HERE / "models",
    _MODELS / "face_check_models",
]

# SFace: cosine >= 0.363 => same identity (OpenCV's documented same-id threshold).
SAME_ID = 0.363

_DETECTOR = None
_RECOGNIZER = None


def _find_model(name):
    for d in _MODEL_DIRS:
        p = d / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"{name} not found in {[str(d) for d in _MODEL_DIRS]}. "
        "Place yunet.onnx + sface.onnx in the pack's models/ folder."
    )


def get_models():
    """Load detector + recognizer once, reuse across calls."""
    global _DETECTOR, _RECOGNIZER
    if _DETECTOR is None:
        _DETECTOR = cv2.FaceDetectorYN_create(_find_model("yunet.onnx"), "", (320, 320), 0.6, 0.3, 5000)
    if _RECOGNIZER is None:
        _RECOGNIZER = cv2.FaceRecognizerSF_create(_find_model("sface.onnx"), "")
    return _DETECTOR, _RECOGNIZER


# ---- ComfyUI IMAGE <-> cv2 BGR -------------------------------------------------

def comfy_to_bgr(image_tensor):
    """ComfyUI IMAGE (torch [B,H,W,C] float 0..1 RGB) -> list of BGR uint8."""
    arr = image_tensor.detach().cpu().numpy() if hasattr(image_tensor, "detach") else np.asarray(image_tensor)
    if arr.ndim == 3:
        arr = arr[None, ...]
    out = []
    for i in range(arr.shape[0]):
        rgb = (np.clip(arr[i], 0.0, 1.0) * 255.0).astype(np.uint8)
        out.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return out


def bgr_to_comfy(bgr):
    """cv2 BGR uint8 -> ComfyUI IMAGE (torch [1,H,W,C] float 0..1 RGB)."""
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb)[None, ...]


# ---- IO ------------------------------------------------------------------------

def imread_unicode(path):
    """cv2.imread chokes on non-ASCII / spaced Windows paths; decode via numpy."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, bgr):
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, bgr)
    if ok:
        buf.tofile(str(path))
    return ok


def list_images(folder):
    p = Path(folder)
    if not p.is_dir():
        return []
    return sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXTS)


def load_folder_bgr(folder):
    out = []
    for p in list_images(folder):
        img = imread_unicode(p)
        if img is not None:
            out.append((p, img))
    return out


# ---- embedding / scoring -------------------------------------------------------

def _largest_face(detector, img):
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    idx = int(np.argmax(faces[:, 2] * faces[:, 3]))  # cols 2,3 = box w,h -> max area
    return faces[idx]


# ---- YOLO face detection (optional, robust on full-body / small faces) ---------

# where face-detection .pt models may live (portable — relative to ComfyUI models/)
_YOLO_DIRS = [
    _MODELS / "ultralytics" / "bbox",
    _MODELS / "ultralytics",
    _MODELS / "upscale_models",
    _HERE / "models",
]
_YOLO_CACHE = {}


def list_yolo_models():
    """Return {display_name: full_path} of available face .pt models."""
    out = {}
    for d in _YOLO_DIRS:
        if not d.is_dir():
            continue
        for p in d.glob("*.pt"):
            n = p.name.lower()
            if "face" in n or "yolo" in n:
                out.setdefault(p.name, str(p))
    return out


def resolve_yolo(name):
    m = list_yolo_models()
    if name in m:
        return m[name]
    # name may already be a full path
    return name if Path(name).exists() else None


def get_yolo(model_path):
    if model_path not in _YOLO_CACHE:
        from ultralytics import YOLO
        _YOLO_CACHE[model_path] = YOLO(model_path)
    return _YOLO_CACHE[model_path]


def _yolo_largest_box(bgr, model_path, conf):
    path = resolve_yolo(model_path)
    if path is None:
        return None
    res = get_yolo(path)(bgr, verbose=False, conf=conf)
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        return None
    xyxy = res[0].boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return xyxy[int(areas.argmax())]


def embed_bgr(bgr, detector="yunet", yolo_model="face_yolov8m.pt", yolo_conf=0.5, pad=0.3, min_face_frac=0.0):
    """Return (embedding, n_faces). embedding is None when no face / error.

    detector: "yunet" (fast cv2, default) or "yolo" (robust — finds small faces
    in full-body shots, then crops+re-aligns so SFace gets a big clean face).
    pad: expand the detected box by this fraction before embedding.
    min_face_frac: ignore faces smaller than this fraction of the frame area.
    """
    if bgr is None:
        return None, 0
    _detector, recognizer = get_models()
    H, W = bgr.shape[:2]

    if detector == "yolo":
        box = _yolo_largest_box(bgr, yolo_model, yolo_conf)
        if box is None:
            return None, 0
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if (bw * bh) / float(W * H) < min_face_frac:
            return None, 0
        x1 = max(0, int(x1 - pad * bw)); y1 = max(0, int(y1 - pad * bh))
        x2 = min(W, int(x2 + pad * bw)); y2 = min(H, int(y2 + pad * bh))
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None, 0
        # face is big in the crop now -> YuNet aligns it cleanly
        face = _largest_face(_detector, crop)
        try:
            if face is not None:
                aligned = recognizer.alignCrop(crop, face)
            else:
                aligned = cv2.resize(crop, (112, 112))  # fallback: no landmarks
            return np.asarray(recognizer.feature(aligned), dtype=np.float32).flatten(), 1
        except Exception:
            return None, 0

    # --- default: YuNet on the full image ---
    face = _largest_face(_detector, bgr)
    if face is None:
        return None, 0
    if min_face_frac > 0:
        if (face[2] * face[3]) / float(W * H) < min_face_frac:
            return None, 0
    try:
        aligned = recognizer.alignCrop(bgr, face)
        return np.asarray(recognizer.feature(aligned), dtype=np.float32).flatten(), 1
    except Exception:
        return None, 0


def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def build_reference(bgr_list, **det_kwargs):
    """Average embeddings of the reference images into one centroid.

    Returns (centroid|None, n_used, self_consistency_mean). More refs = a more
    stable anchor (dataset-centroid > single image > self-medoid). det_kwargs are
    forwarded to embed_bgr (detector/yolo_model/yolo_conf/pad/min_face_frac).
    """
    embs = []
    for bgr in bgr_list:
        e, n = embed_bgr(bgr, **det_kwargs)
        if e is not None:
            embs.append(e)
    if not embs:
        return None, 0, None
    self_cons = None
    if len(embs) > 1:
        pair = [cosine(embs[i], embs[j]) for i in range(len(embs)) for j in range(i + 1, len(embs))]
        self_cons = float(np.mean(pair)) if pair else None
    return np.mean(embs, axis=0), len(embs), self_cons


def medoid_reference(embs_by_key):
    """Fallback anchor when no ref given: the face most similar to all others."""
    items = [(k, e) for k, (e, n) in embs_by_key.items() if e is not None]
    if not items:
        return None, None
    best_k, best = None, -1.0
    for k, e in items:
        sims = [cosine(e, e2) for k2, e2 in items if k2 is not k]
        m = float(np.mean(sims)) if sims else 1.0
        if m > best:
            best_k, best = k, m
    return best_k, best


def verdict_for(sim, keep, reject):
    if sim is None:
        return "noface"
    return "keep" if sim >= keep else ("reject" if sim < reject else "borderline")


# ---- contact sheet -------------------------------------------------------------

_COLOURS = {"keep": (80, 200, 80), "borderline": (0, 200, 230),
            "reject": (60, 60, 220), "noface": (150, 150, 150)}  # BGR


def contact_sheet(rows, cols=6, thumb=256, pad=8):
    """rows = list of (bgr_image, sim|None, verdict, name). Labelled grid, one page."""
    cell = thumb + pad * 2
    label_h = 34
    n = max(1, len(rows))
    r = (n + cols - 1) // cols
    canvas = np.full(((cell + label_h) * r, cell * cols, 3), 30, np.uint8)
    for i, (img, sim, verdict, name) in enumerate(rows):
        gy, gx = divmod(i, cols)
        x0, y0 = gx * cell, gy * (cell + label_h)
        col = _COLOURS.get(verdict, (150, 150, 150))
        if img is not None:
            h, w = img.shape[:2]
            s = thumb / max(h, w)
            small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
            th, tw = small.shape[:2]
            oy, ox = y0 + pad + (thumb - th) // 2, x0 + pad + (thumb - tw) // 2
            canvas[oy:oy + th, ox:ox + tw] = small
        cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + cell - 2, y0 + cell - 2), col, 2)
        simtxt = "n/a" if sim is None else f"{sim:.3f}"
        cv2.putText(canvas, f"{verdict} {simtxt}", (x0 + 6, y0 + cell + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(name)[:26], (x0 + 6, y0 + cell + label_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def annotate(bgr, sim, verdict):
    """Return a copy with a score/verdict badge in the top-left corner."""
    out = bgr.copy()
    col = _COLOURS.get(verdict, (150, 150, 150))
    simtxt = "n/a" if sim is None else f"{sim:.3f}"
    txt = f"{verdict} {simtxt}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(out, (0, 0), (tw + 20, th + 20), (30, 30, 30), -1)
    cv2.rectangle(out, (0, 0), (tw + 20, th + 20), col, 2)
    cv2.putText(out, txt, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    return out
