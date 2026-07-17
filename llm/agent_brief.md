# Using the v3 data release (brief for agents and humans)

All data lives in `/SAN/intelsys/discoviz/systematic/data_release_v3/` on the
CS cluster. These are the EXACT, already-filtered datasets used by the v3
caption-level campaign (July 2026). Use these files verbatim. Do NOT rebuild
them from the raw COCO manifests or re-apply any filtering — the whole point of
this release is that filtering is already done and frozen.

## Ground rules

1. Before first use, verify integrity:
   `cd /SAN/intelsys/discoviz/systematic/data_release_v3 && md5sum -c MANIFEST.md5`
   Every line must say OK. If any file fails, stop and report — do not "fix" it.
2. Never edit files in this directory. Copy if you need to transform.
3. A training run is "the same data" if and only if it reads one of the
   `train_*.jsonl` files below, whole, with no extra filtering or dedup.

## Training sets (pick exactly one file per run)

Format: JSON Lines; each line `{"cap_idx": int, "caption": str, "cocoid": int}`.
`cocoid` is the COCO image id; `cap_idx` numbers that image's captions (0-4+).

| file                               | rows     | meaning                                                                         |
| ---------------------------------- | -------- | ------------------------------------------------------------------------------- |
| train_karpathy_dedup.jsonl         | 545,481  | THE main training set (Karpathy train, benchmark-overlap images removed)        |
| train_karpathy_nodedup.jsonl       | 553,398  | ablation: nothing removed (contaminated on purpose)                             |
| train_karpathy_random_s0..s4.jsonl | ~545,5xx | ablation: 1,622 random images removed instead (control; match seed to run seed) |
| train_coco2017_dedup.jsonl         | 575,612  | official COCO-2017 split variant                                                |

Sanity check after loading: assert the row count matches this table exactly.

Images for a row: look up `cocoid` in
`/SAN/intelsys/discoviz/systematic/manifest_train.jsonl` (or `manifest_train2017.jsonl`)
→ field `image_path`. Precomputed CLIP ViT-B/32 features (what our runs used
instead of raw pixels): `/SAN/intelsys/discoviz/systematic/clip_cache/clip_pooled.pt`
= a torch dict `{cocoid: float16[512]}` (2017 variant: `clip_pooled_2017train.pt`).

## Eval sets (fixed item lists — score these, nothing more, nothing less)

- Retrieval: `eval_retrieval_val_karpathy.jsonl` (24,447 caption rows, same
  schema as training) over the 5,000 images in
  `eval_retrieval_val_karpathy_imgids.jsonl`. 2017 versions alongside
  (24,452 / 5,000). Protocol: text→image and image→text, R@1/5/10/20, an image
  counts as retrieved if any of its captions ranks it.
- Benchmarks: `eval_sugarcrepe.jsonl` (7,203), `eval_scpp.jsonl` (4,622),
  `eval_winoground.jsonl` (746), `eval_aro.jsonl` (8,868). Each line:
  `{"img": path, "true_caption": str, "false_caption": str, "category": str}`
  (ARO has `relation_name` instead of `category`). Score = does the model rank
  true_caption above false_caption for that image; count ties as 0.5.
  These lists are already restricted to items every v3 model was scored on —
  do not add or drop items.

## Optional extras (only if reproducing our exact training conditions)

- Parse trees (CCG): `/SAN/intelsys/discoviz/systematic/trees_train/*.jsonl`,
  lines `{cocoid, cap_idx, caption, tree}`; join to training rows on
  (cocoid, cap_idx). Stanford-parser variants: `trees_train_pcfg_x/`,
  `trees_train_stanza_x/` — same caption set, different trees.
- Hard negatives, READY TO USE: `negs_karpathy_dedup.jsonl` in this directory
  (539,088 captions, 1,952,644 candidate negatives) — see the π-sweep section
  below. (The raw leaf-index swap specs in `hard_negs_v2/` are superseded by
  this file for all normal use.)

## Training recipe — the encoder-independent contract

To run OUR experiment with a NEW text encoder, keep everything below fixed and
swap only the text tower:

- **Encoder contract**: text tower maps a caption string → one L2-NORMALIZED
  512-d vector. Score = cosine against the provided CLIP image features
  (re-normalize the fp16 features after loading). Vocabulary construction is
  the encoder's own business (ours came from the training captions).
- **Loss**: logits = cos × e^s with s a LEARNED scalar initialized to 2.5
  (s is a trainable parameter in the same optimizer). Base loss =
  0.5 · [CE(image→caption) + CE(caption→image)] over the B×B batch matrix.
  Hard-negative term (if π > 0): one EXTRA CE(image→text) over the matrix
  widened with the sampled negative columns, added at weight 1.0.
- **Optimization**: AdamW, weight decay 1e-5, gradient-norm clip 1.0,
  batch 512, 12 epochs, captions shuffled each epoch.
- **Learning rate — run the screen, don't guess**: 3-point grid × your encoder
  at seed 0 on `train_karpathy_dedup` with π=1; pick by best-epoch val rsum
  (sum of R@{1,5,10,20}, both retrieval directions, on the Karpathy val pool);
  a non-default LR must win by ≥2.0 rsum points (of ~800) to displace the
  default. Grids we used: from-scratch encoders {6e-4, 1.2e-3, 2.4e-3}
  (default 1.2e-3); pretrained/finetuned towers {3e-6, 1e-5, 3e-5}.
  For reference, our picks: all tree cells + transformer 6e-4;
  lstm/bilstm/gru/mean 1.2e-3; finetuned CLIP text 3e-5.
- **Selection & reporting**: compute val rsum every epoch; keep the best-epoch
  checkpoint for analysis, but REPORT benchmarks from the FINAL epoch (that is
  what all our tables do). Run ≥3 seeds (we used 5 for main results) and
  report mean±SEM over seeds.

## Metrics (report both, exactly like our tables)

- **Raw** benchmark accuracy: fraction of items with cos(true, image) >
  cos(false, image); ties count 0.5.
- **Gap** (grounded accuracy): raw MINUS the same accuracy computed with each
  item scored against a random OTHER item's image (one fixed permutation of
  the eval set's images, same permutation for true and false captions). The
  gap isolates what the model gets from actually looking at the image; the
  subtracted term is the text-prior baseline. Most of our conclusions
  (especially ARO) are stated on gaps — raw alone is misleading there.
- Retrieval: rank by cosine with strict >; R@k both directions on the fixed
  pools; image→text counts an image correct if ANY of its captions beats all
  other captions.

## Reproducing the π-sweep (hard-negative dose-response)

`negs_karpathy_dedup.jsonl` lines:
`{"cocoid", "cap_idx", "caption", "negs": [{"neg": str, "t": "obj"|"attr", "h": float}, ...]}`
`caption` is the tokenized lowercase form of the training caption (join of the
parse's leaves); each `neg` is that caption with two words exchanged — an
object swap ("a cat is holding a woman ...") or attribute swap. `h` ∈ [-1, 1]
is CLIP-text similarity of the two swapped words (higher = harder negative).
98.8% of `train_karpathy_dedup` captions have ≥1 candidate (mean 3.62); the
rest simply contribute no negative.

Protocol, exactly as our runs (π = the sweep variable):

1. Train under the recipe in "Training recipe" above (same batch, loss,
   temperature, optimizer, epochs; LR from your encoder's screen).
2. At every step, for each positive INDEPENDENTLY: with probability π, sample
   one negative from its candidate list; with probability 1−π, none.
3. Sampling rule (given the candidate list): drop candidates with h > 0.95
   (near-paraphrases; if that empties the list, keep all), then sample one with
   probability ∝ exp((h − mean_h) / 0.5) over the survivors.
4. Use the sampled negatives as EXTRA image→text in-batch negatives: append
   their text embeddings as columns to the B×B logit matrix and add a second
   cross-entropy(image→text) term over the widened matrix.
5. Our grid: π ∈ {0, 0.1, 0.25, 0.5, 1.0} (π=0 is training with no negatives).
   Read the effect on SugarCrepe swap-object / swap-attribute accuracy — the
   aggregate score dilutes it ~7×.

Encode `caption` and `neg` strings through the same text pipeline so the
positive/negative comparison is tokenization-consistent.

## Pitfalls

- All paths are absolute for this cluster; running elsewhere requires copying
  the referenced dirs and remapping the path prefix.
- `train_karpathy_dedup` is 1.4% smaller than nodedup BY DESIGN
  (decontamination). Do not "top it up".
- Winoground clean-subset analysis uses the id list in
  `/SAN/intelsys/discoviz/vlm_grounding/diwan_clean_ids.json` (key `notag`).

Provenance, schemas in full, and the campaign report:
`README.md` in this directory, and
`discoviz-repo/llm/report_socher_seq_redo_v3.md`.
