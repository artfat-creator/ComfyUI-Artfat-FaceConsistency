# ComfyUI — Artfat Face Consistency

Face-identity consistency scoring, built on OpenCV's **YuNet** (detector) +
**SFace** (recognizer). Pure `cv2` + `numpy` on **CPU** — it never touches VRAM,
so it runs happily alongside the sampler.

Two nodes, one shared scoring core (identical to the `face_consistency_sort.py`
script):

## Artfat Face Consistency (Sort) — inline gate
Score each generated frame against a reference identity and sort it to disk.

- **Inputs:** `image`, `min_similarity` (cosine pass threshold), `pass_dir`,
  `fail_dir`; optional `reference_image` (IMAGE batch 1..N), `reference_folder`,
  `filename_prefix`, `csv_path`, `save_files`.
- **Outputs:** `image` (passthrough), `similarity` (FLOAT), `passed` (BOOLEAN),
  `verdict` (STRING), `annotated` (IMAGE with a score badge).
- Passing frames go to `pass_dir`, failing/no-face to `fail_dir`, with the score
  baked into the filename (`blanca_00007_sim0.782_pass.png`) and appended to CSV.

## Artfat Face Consistency (Batch) — folder analysis
Point it at a folder of ready images + reference(s); get a labelled contact
sheet, a CSV, and (optionally) keep/reject copies.

- **Inputs:** `input_folder`, `keep_threshold`, `reject_threshold`; optional
  `reference_image`, `reference_folder`, `sort_copies`, `contact_cols`.
- **Outputs:** `contact_sheet` (IMAGE → Preview), `csv_path`, `mean_score`
  (FLOAT, near-zero detector glitches dropped), `report`.
- No reference given → auto-medoid of the input set.

## Reference handling
`reference_image` (an IMAGE batch of 1..N) and `reference_folder` are **merged
and averaged into one centroid**. More references = a steadier anchor — a
dataset centroid beats a single frame, which beats a self-medoid.

## Notes
- Near-zero cosine (`< 0.1`) usually means SFace failed to embed that crop
  (a detector glitch), **not** a real identity mismatch — the Batch mean drops
  these so one bad crop doesn't skew the score.
- Models (`yunet.onnx`, `sface.onnx`) ship in `models/`.
- Roadmap: a `Sample Until Consistent` node (owns the sampler, retries on a new
  seed until it passes) — deferred.

MIT.
