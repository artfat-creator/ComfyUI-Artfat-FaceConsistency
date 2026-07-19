"""Artfat Face Consistency — ComfyUI nodes.

Two nodes over a shared cv2 YuNet+SFace core (CPU, zero VRAM):

  * Artfat Face Consistency (Sort)  — inline gate: score each generated frame
    against a reference identity, save pass/fail into two folders (score baked
    into the filename), log to CSV, pass the image through untouched.

  * Artfat Face Consistency (Batch) — script-style: point at a folder of ready
    images + reference(s), get a labelled contact-sheet (thumbnails + numbers),
    a CSV, and optional keep/reject copies.

Reference = an IMAGE batch (1..N) and/or a folder path; all of them are averaged
into one centroid (more refs = a steadier anchor). The Batch node falls back to
an auto-medoid when no reference is given; the inline Sort node requires one.
"""
import csv
import os
import shutil
from pathlib import Path

from . import face_core as fc


def _yolo_choices():
    m = list(fc.list_yolo_models().keys())
    return m if m else ["face_yolov8m.pt"]


def _det_kwargs(detector, yolo_model, detect_conf, crop_padding, min_face_frac):
    d = "yolo" if str(detector).lower().startswith("yolo") else "yunet"
    return dict(detector=d, yolo_model=yolo_model, yolo_conf=float(detect_conf),
                pad=float(crop_padding), min_face_frac=float(min_face_frac))


# reusable optional-input block for the detector controls (both nodes share it)
def _detector_inputs():
    return {
        "detector": (["YuNet (fast, CPU)", "YOLO (robust, full-body)"],
                     {"default": "YuNet (fast, CPU)",
                      "tooltip": "YuNet = fast cv2, great on portraits. YOLO = finds small faces in "
                                 "full-body/complex shots, then crops+re-aligns so the score is fair."}),
        "yolo_model": (_yolo_choices(), {"tooltip": "Face .pt model used when detector = YOLO."}),
        "detect_conf": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05,
                                  "tooltip": "YOLO confidence. LOWER = catches harder/smaller/angled faces."}),
        "crop_padding": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "Expand the detected face box by this fraction before scoring (more context)."}),
        "min_face_frac": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                                    "tooltip": "Ignore faces smaller than this fraction of the frame (0 = off). Skips background faces."}),
    }


def _collect_reference(reference_image, reference_folder, **det_kwargs):
    """Merge IMAGE-batch refs + folder refs into one centroid. -> (centroid, desc)."""
    bgr = []
    if reference_image is not None:
        bgr.extend(fc.comfy_to_bgr(reference_image))
    if reference_folder and str(reference_folder).strip():
        bgr.extend(img for _p, img in fc.load_folder_bgr(reference_folder))
    if not bgr:
        return None, "no reference"
    centroid, n_used, self_cons = fc.build_reference(bgr, **det_kwargs)
    if centroid is None:
        return None, "reference(s) had no detectable face"
    desc = f"{n_used} ref face(s)"
    if self_cons is not None:
        desc += f", self-consistency {self_cons:.3f}"
    return centroid, desc


def _next_index(folder, prefix):
    folder = Path(folder)
    if not folder.is_dir():
        return 1
    mx = 0
    for p in folder.iterdir():
        name = p.name
        if name.startswith(prefix + "_"):
            part = name[len(prefix) + 1:].split("_", 1)[0]
            if part.isdigit():
                mx = max(mx, int(part))
    return mx + 1


def _csv_append(csv_path, row):
    if not csv_path or not str(csv_path).strip():
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "similarity", "verdict", "passed"])
        if new:
            w.writeheader()
        w.writerow(row)


class ArtfatFaceConsistencySort:
    """Inline pass/fail gate — scores a frame vs reference and sorts it to disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Frame(s) to check — usually the VAE-decoded output."}),
                "min_similarity": ("FLOAT", {
                    "default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Cosine >= this -> pass, else fail. SFace same-id starts ~0.363; "
                               "for a strong persona LoRA 0.45-0.6 is a sensible gate.",
                }),
                "pass_dir": ("STRING", {"default": "", "tooltip": "Folder for images that PASS. Created if missing."}),
                "fail_dir": ("STRING", {"default": "", "tooltip": "Folder for images that FAIL (and no-face). Created if missing."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Reference identity, 1..N images (batch). Averaged into a centroid."}),
                "reference_folder": ("STRING", {"default": "", "tooltip": "Optional folder of reference images, merged with reference_image."}),
                "filename_prefix": ("STRING", {"default": "blanca"}),
                "csv_path": ("STRING", {"default": "", "tooltip": "Optional CSV log path (appended). Empty = no log."}),
                "save_files": ("BOOLEAN", {"default": True, "tooltip": "Off = only score/pass outputs, nothing written to disk."}),
                **_detector_inputs(),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "BOOLEAN", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "similarity", "passed", "verdict", "annotated")
    FUNCTION = "run"
    CATEGORY = "artfat"
    OUTPUT_NODE = True

    def run(self, image, min_similarity, pass_dir, fail_dir,
            reference_image=None, reference_folder="", filename_prefix="blanca",
            csv_path="", save_files=True, detector="YuNet (fast, CPU)",
            yolo_model="face_yolov8m.pt", detect_conf=0.5, crop_padding=0.3, min_face_frac=0.0):
        det = _det_kwargs(detector, yolo_model, detect_conf, crop_padding, min_face_frac)
        centroid, desc = _collect_reference(reference_image, reference_folder, **det)
        if centroid is None:
            raise ValueError(f"Artfat Face Consistency (Sort): {desc}. Connect a reference_image or set reference_folder.")

        frames = fc.comfy_to_bgr(image)
        first_sim, first_verdict, first_passed, first_annot = None, "noface", False, None

        for i, bgr in enumerate(frames):
            emb, _n = fc.embed_bgr(bgr, **det)
            sim = None if emb is None else fc.cosine(emb, centroid)
            passed = bool(sim is not None and sim >= min_similarity)
            verdict = "noface" if sim is None else ("pass" if passed else "fail")

            if save_files:
                target = pass_dir if passed else fail_dir
                if target and str(target).strip():
                    Path(target).mkdir(parents=True, exist_ok=True)
                    idx = _next_index(target, filename_prefix)
                    simtxt = "noface" if sim is None else f"{sim:.3f}"
                    fname = f"{filename_prefix}_{idx:05d}_sim{simtxt}_{verdict}.png"
                    fc.imwrite_unicode(str(Path(target) / fname), bgr)
                    _csv_append(csv_path, {"file": fname, "similarity": "" if sim is None else f"{sim:.4f}",
                                           "verdict": verdict, "passed": passed})

            if i == 0:
                first_sim = 0.0 if sim is None else sim
                first_verdict, first_passed = verdict, passed
                first_annot = fc.annotate(bgr, sim, verdict)

        annot_tensor = fc.bgr_to_comfy(first_annot) if first_annot is not None else image
        return (image, float(first_sim if first_sim is not None else 0.0), first_passed, first_verdict, annot_tensor)


class ArtfatFaceConsistencyBatch:
    """Score a whole folder vs reference(s) -> contact sheet + CSV (+ optional copies)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": "", "tooltip": "Folder of images to analyse."}),
                "keep_threshold": ("FLOAT", {"default": 0.363, "min": -1.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "Cosine >= this -> keep (SFace same-id = 0.363)."}),
                "reject_threshold": ("FLOAT", {"default": 0.28, "min": -1.0, "max": 1.0, "step": 0.01,
                                               "tooltip": "Cosine < this -> reject; between = borderline."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Reference identity 1..N (batch). Empty = auto-medoid of the input set."}),
                "reference_folder": ("STRING", {"default": "", "tooltip": "Optional folder of references, merged with reference_image."}),
                "sort_copies": ("BOOLEAN", {"default": False, "tooltip": "Also copy images into keep/borderline/reject/_noface subfolders."}),
                "contact_cols": ("INT", {"default": 6, "min": 1, "max": 12}),
                **_detector_inputs(),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("contact_sheet", "csv_path", "mean_score", "report")
    FUNCTION = "run"
    CATEGORY = "artfat"
    OUTPUT_NODE = True

    def run(self, input_folder, keep_threshold, reject_threshold,
            reference_image=None, reference_folder="", sort_copies=False, contact_cols=6,
            detector="YuNet (fast, CPU)", yolo_model="face_yolov8m.pt",
            detect_conf=0.5, crop_padding=0.3, min_face_frac=0.0):
        det = _det_kwargs(detector, yolo_model, detect_conf, crop_padding, min_face_frac)
        items = fc.load_folder_bgr(input_folder)
        if not items:
            raise ValueError(f"Artfat Face Consistency (Batch): no images in '{input_folder}'.")

        # embeddings for every input image
        embs = {}
        for p, bgr in items:
            e, n = fc.embed_bgr(bgr, **det)
            embs[p] = (e, n)

        # reference: explicit centroid, else auto-medoid over the set
        centroid, desc = _collect_reference(reference_image, reference_folder, **det)
        if centroid is None:
            mk, ms = fc.medoid_reference(embs)
            if mk is None:
                raise ValueError("Artfat Face Consistency (Batch): no faces found anywhere, cannot pick a reference.")
            centroid = embs[mk][0]
            desc = f"AUTO medoid = {mk.name} (self-sim {ms:.3f})"

        out_base = Path(input_folder) / "_face_consistency"
        out_base.mkdir(parents=True, exist_ok=True)
        if sort_copies:
            for k in ("keep", "borderline", "reject", "_noface"):
                (out_base / k).mkdir(exist_ok=True)

        rows, csv_rows, valid = [], [], []
        counts = {"keep": 0, "borderline": 0, "reject": 0, "noface": 0}
        for p, bgr in items:
            e, n = embs[p]
            sim = None if e is None else fc.cosine(e, centroid)
            verdict = fc.verdict_for(sim, keep_threshold, reject_threshold)
            counts[verdict] += 1
            if sim is not None and sim >= 0.1:   # drop detector-glitch near-zeros from the mean
                valid.append(sim)
            rows.append((bgr, sim, verdict, p.name))
            csv_rows.append({"file": p.name, "similarity": "" if sim is None else f"{sim:.4f}",
                             "verdict": verdict, "faces": n})
            if sort_copies:
                dst = out_base / ("_noface" if verdict == "noface" else verdict) / p.name
                shutil.copy2(str(p), str(dst))

        csv_rows.sort(key=lambda r: (r["similarity"] == "", r["similarity"]), reverse=True)
        csv_path = out_base / "scores.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "similarity", "verdict", "faces"])
            w.writeheader()
            w.writerows(csv_rows)

        rows.sort(key=lambda r: (-1.0 if r[1] is None else r[1]), reverse=True)
        sheet = fc.contact_sheet(rows, cols=int(contact_cols))

        mean_score = float(sum(valid) / len(valid)) if valid else 0.0
        report = (f"ref: {desc} | n={len(items)} | mean(valid,drop<0.1)={mean_score:.4f} | "
                  f"keep={counts['keep']} borderline={counts['borderline']} "
                  f"reject={counts['reject']} noface={counts['noface']}")
        return (fc.bgr_to_comfy(sheet), str(csv_path), mean_score, report)


NODE_CLASS_MAPPINGS = {
    "ArtfatFaceConsistencySort": ArtfatFaceConsistencySort,
    "ArtfatFaceConsistencyBatch": ArtfatFaceConsistencyBatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArtfatFaceConsistencySort": "Artfat Face Consistency (Sort)",
    "ArtfatFaceConsistencyBatch": "Artfat Face Consistency (Batch)",
}
