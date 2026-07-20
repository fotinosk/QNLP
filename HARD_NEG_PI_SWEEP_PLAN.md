# Plan: Hard-negative π-sweep on bobcat + tree-reader datasets

**Status: FINALIZED 2026-07-21 — ready for implementation. This file is the source of
truth; update it as work proceeds so any fresh agent context can resume from here.**

**DECISIONS (user-confirmed / defaulted):**
- Hard negatives: GENERATE OUR OWN (Phase A0) from our cached bobcat trees — user
  confirmed 2026-07-21. The colleague's negs jsonl is used only for validating the port
  (Phase A0 step 6), never as a training source.
- π grid: {0, 0.1, 0.25, 0.5, 1.0} for all 4 cells (20 runs); trim non-frozen cells to
  {0, 0.25, 1.0} only if queue time forces it.
- Loss: widen our existing i2t InfoNCE term with negative columns (labels unchanged,
  negatives never queries, t2i untouched). NOT copying the colleague's double-counted
  base+wide structure. π=0 must be bit-identical to current loss.
- NO EARLY STOPPING (decided 2026-07-21): every sweep run trains a FIXED epoch budget,
  identical across all cells; benchmarks reported from the FINAL epoch (best-epoch
  checkpoint kept for analysis only — colleague's protocol). Rationale: (a) job 7076697
  (epoch 34) is the only current-gen run above chance on ARO/SC while patience-stopped
  siblings (ep 9–20) all sat at chance — val loss is a bad stopping signal for sparse
  symbols; (b) π changes val dynamics, so patience would give cells different training
  lengths and confound the dose-response. Budget: ~35–40 epochs frozen; non-frozen set
  after post-speedup epoch-time measurement. Implement via config (patience=∞ or an
  ML_DISABLE_EARLY_STOPPING flag) — do NOT silently change existing scripts' defaults.
- The sweep is SELF-CONTAINED: π=0 cells regenerate all 4 linear baselines under the
  new protocol, superseding the early-stopped chance-level runs in RESULTS.md §A1.
  Nothing outside the 20 runs is needed (NLC out of scope).

## Task (from user, 2026-07-21)

A colleague generated hard negatives (v3 data release on the UCL CS cluster; guide at
`llm/agent_brief.md`). Integrate them into OUR training pipeline (NOT the colleague's —
our pipelines are separate; much of the brief does not apply) and run a grid:

- **Parsers:** bobcat-parsed dataset AND tree-reader-parsed dataset
- **π (hard-negative ratio):** varying — colleague's grid was π ∈ {0, 0.1, 0.25, 0.5, 1.0}
- **Image tower:** frozen AND non-frozen
- **Model:** linear only — IGNORE NLC (non-linear) training
- Grid = 2 parsers × |π| × 2 tower modes

Constraints:
- May ssh to cluster to LOOK ONLY — no changes there. (One ssh attempt was rejected by
  user mid-plan; re-confirm before ssh'ing again.)
- `llm/` dir has lots of info but much is OUTDATED — trust code over docs.
- Be token-efficient.

## Key facts from colleague's brief (llm/agent_brief.md)

- Data: `/SAN/intelsys/discoviz/systematic/data_release_v3/` on UCL CS cluster
  (ssh hosts in ~/.ssh/config: knuckles → beaker/vic via ProxyJump, user kinianlo).
- Hard negatives file: `negs_karpathy_dedup.jsonl` — lines
  `{"cocoid", "cap_idx", "caption", "negs": [{"neg": str, "t": "obj"|"attr", "h": float}]}`.
  `caption` is tokenized-lowercase (join of parse leaves). 539,088 captions,
  ~1.95M candidate negs, 98.8% coverage of train_karpathy_dedup, mean 3.62 cands.
- Sampling protocol (per positive, per step, independently): with prob π sample one neg;
  drop candidates with h > 0.95 (unless that empties the list); sample ∝ exp((h − mean_h)/0.5).
- Loss usage (adapt to OUR loss, keep the 3 invariants):
  negatives appear ONLY as extra columns in the image→text softmax; labels stay 0..B−1;
  negatives never act as queries and never enter caption→image term.
- Read the effect on SugarCrepe swap-object / swap-attribute (aggregate dilutes ~7×).
- Neg generation code: `/SAN/intelsys/discoviz/discoviz-repo/llm/systematic/harness.py`.
- Things in the brief that likely DON'T apply to us: their encoder contract (learned temp
  s=2.5 vs our fixed 0.07), their optimizer/LR screen, their exact train file — we use our
  own parsed datasets, InfoNCE symmetric with fixed temperature, TTN or CLIP image tower.

## Our pipeline facts (from COCO_EXPERIMENTS.md + memory; verify in code)

- Repo: /Users/fotinoskyriakides/Desktop/Dev/qnlp, branch `coco-training-nlc`.
- Text encoder: lambeq/DisCoCat einsum tensor nets (bobcat CCG) + a tree-reader variant
  ("tree_no_type"? — confirm dataset name). Loss: symmetric InfoNCE, temp 0.07.
- Frozen tower experiments used frozen CLIP ViT-B/32 in-memory cache (exp 11) and earlier
  frozen TTN (exp 2/5) — confirm which is current for "frozen".
- Recent perf work landed on this branch: same-diagram batched contraction +
  `_prepare_rescaled_weights` (see COCO_EXPERIMENTS.md 2026-07-19 entry).
- lambeq CCG compile leaks memory cumulatively (see memory: worker_batch_size ×
  max_tasks_per_child bounds it) — relevant because HARD-NEGATIVE STRINGS MUST BE PARSED
  through the same pipeline as positives → dataset-build cost is a real concern
  (~1.95M candidate strings if we pre-parse all; consider parsing only kept candidates
  h ≤ 0.95, or dedup identical neg strings).
- Cluster node arbuckle lacks AVX → use polars[rtcompat] (memory).
- Env: always `conda run -n qnlp`.

## Open design questions — ALL RESOLVED, see "Implementation plan" below (kept for history)

1. **Join key:** do our parsed dataset rows carry (cocoid, cap_idx) or raw caption to join
   against negs_karpathy_dedup.jsonl? Our dataset is COCO-Karpathy-based but possibly a
   different caption set than the colleague's filtered train_karpathy_dedup — expect
   partial overlap; rows without negs simply contribute none (matches their protocol).
2. **Parsing negatives:** negatives are word-swap variants of the caption — same leaves,
   different order. For bobcat: must re-parse (CCG changes with word order... actually the
   swap may keep POS structure — still re-parse to be safe). For tree reader: also re-parse.
   Decide: precompute parsed negs offline into a companion dataset file keyed by
   (cocoid, cap_idx, neg_idx).
3. **π sampling location:** at training time per step (their protocol) — dataset should ship
   ALL candidate negs (parsed) + h values; sampler picks per step. Alternative (cheaper,
   less faithful): pre-sample per epoch. Prefer faithful per-step.
4. **Loss integration:** extra columns in image→text logits only, labels unchanged; keep our
   fixed temperature. Do NOT add their double-counted base+wide structure unless our loss
   already differs — OUR loss stays symmetric InfoNCE, we only widen the image→text term.
   (Decide exactly once loss code is read.)
5. **Frozen tower:** which frozen variant do we run (CLIP cache vs frozen TTN)? Exp 11
   (frozen CLIP + NLC) failed at random — but this grid is LINEAR, different regime. Ask
   user if ambiguous.
6. **π grid:** default to colleague's {0, 0.1, 0.25, 0.5, 1.0}? 2×2×5 = 20 runs — check
   epoch time budget (linear frozen should be fast post-speedup). Maybe ask user.

## Progress log

- 2026-07-21: Read COCO_EXPERIMENTS.md, agent_brief.md. Explore agent launched over local
  repo (results pending — will fill "Pipeline ground truth" section below). SSH look-around
  NOT yet done (user rejected first attempt; ask before retrying).

## Pipeline ground truth (explored 2026-07-21)

User clarifications: agent_brief's train/eval datasets do NOT apply — we use OUR datasets.
Local data/datasets/ is OUTDATED; the cluster copies under
`/SAN/intelsys/discoviz/fotinos/QNLP/` are authoritative.

**Training code (the 4 grid cells already exist as scripts — only π/negatives are new):**
- bobcat linear non-frozen: `scripts/submit_coco_multi_caption_linear.sh` →
  `qnlp.scripts.coco_multi_caption.run` (dataset default `coco_single_caption`, TTN image tower)
- tree linear non-frozen: `scripts/submit_coco_multi_caption_tree_no_type_linear.sh`
- bobcat linear frozen: `scripts/submit_coco_single_caption_linear_frozen.sh` /
  multi-caption frozen → `qnlp.scripts.coco_multi_caption.run_frozen` (frozen CLIP ViT-B/32,
  in-memory cache, trains EinsumModel + linear text_head, ML_EMBEDDING_DIM=512)
- tree linear frozen: `scripts/submit_coco_multi_caption_tree_no_type_linear_frozen.sh`
  (env: PARSER_VERSION=tree_no_type, ML_DATASET_NAME=coco_single_caption_nlc_tree_no_type)
- Config: `qnlp/scripts/coco_multi_caption/config.py` — pydantic BaseSettings, env prefix
  `ML_`. Fields incl. use_non_linear_contractions, dataset_name, batch_size, temperature
  (0.07 fixed), max_epochs=50, patience=10. Add pi + neg-related fields here.
- run.py: DataLoaders via `get_dataloaders` (qnlp/domain/datasets/dataloader.py),
  `SimpleCaptionStep` computes loss from model outputs; loss =
  `SingleCaptionLoss(temperature, alignment_weight=0.0)` →
  `qnlp/core/training/losses/single_caption.py` wrapping `SymmetricInfoNCE`
  (losses/symmetric_infonce.py). Trainer in qnlp/core/training/trainer.py.
- run_frozen.py: own epoch loop (`_run_epoch`), `FrozenCOCODataset` (reads parquet directly,
  builds caption tuple (diagram, symbols[, path])), `CLIPImageCache`, TopologyBucketSampler
  in linear mode. Loss same SingleCaptionLoss.
- Text encoding input = (diagram, symbols) precompiled at DATASET BUILD time — the model
  never sees raw strings. So hard negatives MUST be parsed+compiled offline into the dataset.

**Dataset schema (parquet, both parsers):**
`sample_id` (e.g. "coco_1" — internal atlas id, NOT cocoid), `local_image_path`
(filename embeds cocoid: COCO_val2014_000000318219.jpg → 318219), `processed_text`
(original caption string), `text_hash`, `diagram`, `symbols` (serialized), `path`
(opt_einsum path; ignored by linear mode). Multi-caption = rows repeated per caption
with same sample_id.
- Dataset build: `qnlp.scripts.coco_single_caption.create_dataset(_tree_no_type)` →
  `create_train_val_test_datasets` (qnlp/core/data_engine/dataset_creator/dataset_generator.py)
  with `SingleCaptionStrategy` over atlas "derived" atoms. PARSER_VERSION env versions all
  derived dirs/caches/output names.
- Preprocessing pipelines: `qnlp/preprocessing_pipelines/coco/pipeline.py` (bobcat) and
  `pipeline_tree_no_type.py` (tree reader). lambeq tree compile leaks memory (bound with
  worker_batch_size × max_tasks_per_child — see memory note).

**Join to colleague's negs (`negs_karpathy_dedup.jsonl`):** extract cocoid from
local_image_path filename; match caption by normalized text (their `caption` is
tokenized-lowercase join of parse leaves; our `processed_text` is raw) — match within
cocoid on lowercase+strip-punct comparison rather than cap_idx (orderings may differ).
Rows without a match get no negatives (their protocol allows this).

## Cluster findings (2026-07-21, read-only via `ssh beaker`, bash -c wrapper — login shell is csh; /SAN not mounted on knuckles)

- Our authoritative datasets: `/SAN/intelsys/discoviz/fotinos/QNLP/data/datasets/`:
  `coco_single_caption_nlc_{train,val,test}.parquet` (bobcat, 2026-06-19, 41M train) and
  `coco_single_caption_nlc_tree_no_type_{train,val,test}.parquet` (tree, 2026-07-17, 46M train).
- Negs file `negs_karpathy_dedup.jsonl` (57M): `caption` is simply the LOWERCASED original
  COCO caption (punctuation kept, e.g. "cutting a cake."), NOT exotic tokenization.
  Verified cocoid 318219 caption "a young boy stares up at the computer monitor." matches
  our processed_text row exactly (modulo case). Join = cocoid (from image filename) +
  lowercase(processed_text) == caption. `negs` list: word-swap strings + h + t(obj/attr).
- **Overlap measured (2026-07-21):** negs file: 538,891 captions / 1,952,644 candidates.
  Our train sets: bobcat 388,985 rows (90,592 unique images), tree_no_type 452,783 rows
  (90,629 unique images). Exact (cocoid, lowercase-text) match: **66.0% / 66.2%** —
  the missing third is captions LemmatizeStep rewrote (inflected verbs etc.).
- **Join decision:** two-stage join — exact match first, then per-cocoid fuzzy match
  (each image has ≤5 negs-file captions; align remaining rows to the unmatched candidate
  with highest token-multiset overlap, threshold ~0.6; strip punctuation, lowercase).
  Expected coverage ≥95%. Neg strings are raw-form; they get re-lemmatized by our
  pipeline anyway, so form mismatch with the positive is resolved at parse time.
  Fallback if fuzzy coverage disappoints: regenerate swaps from OUR bobcat trees with
  colleague's `gen_hard_negs_v2.py` logic (enumerate obj/attr leaf swaps on CCG types;
  h = CLIP-text sim of swapped word pair) — feasible, but more work; only if needed.
- Colleague's sampling constants confirmed in harness.py: HARD_CAP=0.95, NEG_TEMP=0.5,
  softmax over (h − mean); `sample_swap` drops h>HARD_CAP with keep-all fallback.
- Colleague's repo has `gen_hard_negs_v2.py` (swap enumeration rules: OBJECT = two head
  nouns with predicate between, skip coordinate pairs; ATTRIBUTE = two adjectives on
  different head nouns, skip identical) and `export_negs_v3.py` (built the negs jsonl).

**Pipeline subtleties that affect the join (found 2026-07-21):**
- Both parsers share the bobcat CCG parse; tree_no_type differs only in diagram conversion
  (TreeReader) and reuses bobcat trees from a versioned diskcache
  (`pipeline_tree_no_type.py` docstring). Neg strings are new texts → full bobcat parse
  needed once; tree variant then reuses via its own diskcache.
- `LemmatizeStep` (qnlp/core/data_engine/processing/lemmatize_step.py) REWRITES captions
  into finite sentences: strips punctuation, capitalizes, appends ".", and INFLECTS verbs
  for captions lacking a finite verb ("a man riding a bike" → "A man rides a bike.").
  Parquet `processed_text` is the post-lemmatize text. Colleague's negs `caption` is the
  lowercased RAW caption → string equality will FAIL for every caption the lemmatizer
  altered (many COCO captions are participial). Overlap job measures the exact-match rate;
  expect it to understate the true joinable fraction.
- Join options, in order of preference depending on measured rate:
  a) exact lowercase match (works only for already-finite captions);
  b) cocoid + fuzzy match (word-multiset ignoring punctuation/inflection differences);
  c) REGENERATE negatives from OUR processed_text with the colleague's generation logic
     (readable at /SAN/intelsys/discoviz/discoviz-repo/llm/systematic/harness.py) —
     cleanest: swaps then operate on exactly the strings our models train on.
- Whatever the source, neg strings must run through the SAME steps as positives
  (RemoveTrailingDots → Lemmatize → CCGCompiler → UnifyEinsumRank) so
  tokenization/lemma treatment is consistent.

## Implementation plan

### Phase A0 — generating hard negatives OURSELVES (own-generation path)

**STATUS: IMPLEMENTED 2026-07-21 (locally, on branch coco-training-nlc; not yet run on
cluster).** Files:
- `qnlp/scripts/coco_multi_caption/generate_hard_negatives.py` — the generator.
  Reads both train parquets (unique by text_hash), fetches lambeq CCGTrees from the
  diskcaches (key = `str((TreebankWordTokenizer tokens, True, False))`, tries bobcat
  then bobcat_tree_no_type cache), enumerates obj/attr swaps (exact port of
  gen_hard_negs_v2 rules, MAXK=6, incl. FA-rule check for adjectives), scores h with
  HF CLIP (openai/clip-vit-base-patch32, template "a photo of a {}", identical to his
  score_negs_hardness.py), writes `data/datasets/coco_hard_neg_specs.parquet`
  (text_hash, processed_text, neg_text, t, w1, w2, h). Logs coverage/obj-attr/h stats
  vs his reference numbers (98.8% coverage, 3.62 cand/caption).
- `qnlp/scripts/coco_multi_caption/validate_hard_negatives.py` — port validation:
  on the ~66% exact-match subset, compares (t, {w1,w2}) candidate sets vs his negs
  jsonl (form-insensitive via lemma norm; recovers his pairs by 2-token diff).
  Reports identical-set % + mean Jaccard + example mismatches.
- `scripts/submit_generate_hard_negatives.sh` — SGE job (GPU for CLIP; 12h; runs
  generator then validator). Smoke mode: `qsub -v SMOKE=5000 ...` → writes
  `coco_hard_neg_specs_smoke.parquet`, skips validation.
- Verified locally: swap enumeration produces correct obj + attr swaps on a synthetic
  lambeq CCGTree ('A red dog chases a small cat' → dog↔cat obj, red↔small attr).
- RESUMABLE (added 2026-07-21): captions sorted by text_hash, processed in 50k-caption
  chunks, each written atomically (tmp+rename) to
  `data/datasets/coco_hard_neg_specs_parts/part_NNNNN.parquet`; on restart existing
  parts are skipped, so a killed job resumes at the first missing part. CLIP h-scoring
  runs once at the end over all parts; final output also written atomically. Parts are
  KEPT after success (cheap; delete `*_parts/` manually to force full regeneration —
  required if the input parquets ever change, since chunk boundaries are content-based).
- TO RUN (user, on cluster): sync repo to PROJECT_DIR, then
  `qsub -v SMOKE=5000 scripts/submit_generate_hard_negatives.sh` first; check the log
  (cache-miss count MUST be ~0, coverage/h stats sane), then full
  `qsub scripts/submit_generate_hard_negatives.sh`. Validation Jaccard ≥~0.8 expected;
  large HIS-ONLY buckets = port bug, investigate before Phase A.

#### Original design notes (kept)

Motivation: the colleague's negs are keyed to HIS caption set/forms; ours are lemmatized
rewrites (66% exact overlap). Generating from OUR data gives 100% coverage, trivial keying
by `text_hash`, and swaps consistent with the exact strings our models train on.

Inputs we already have:
- `processed_text` (post-lemmatize, the exact training strings) from both train parquets
  (union across bobcat + tree_no_type rows, keyed by `text_hash`).
- Bobcat CCG trees for every one of those strings: lambeq `CCGTree` objects in
  `CachedBobcatParser`'s diskcache at
  `/SAN/intelsys/discoviz/fotinos/QNLP/.cache/lambeq/bobcat/diskcache`, keyed
  `str((sentence, tokenised, suppress_exceptions))` (qnlp/discoviz/parser/cached_bobcat.py).
  Both parsers share these trees (tree_no_type only changes diagram conversion), so ONE
  generation pass serves both.

New script `qnlp/scripts/coco_multi_caption/generate_hard_negatives.py`:
1. Load union of train `processed_text`/`text_hash`; fetch each CCGTree from the diskcache
   (read-only open; no parsing — cache hits only, log any misses and skip).
2. Port the colleague's swap-enumeration rules (`gen_hard_negs_v2.py`, read from cluster,
   do not modify) onto lambeq CCGTree leaves in surface order:
   - OBJECT swap: two distinct head nouns (atomic type n/np leaf) with a predicate leaf
     strictly between them in surface order (type containing `s\` or the core-preposition
     type (np\np)/np); SKIP coordinate pairs (a conj between with no predicate).
   - ATTRIBUTE swap: two adjectives (type n/n) modifying DIFFERENT head nouns; skip
     identical adjective pairs.
   - Identity check on normalized words (lowercase, strip punctuation); cap MAXK=6 per type.
3. Materialize each swap: exchange the two surface tokens at leaf indices i,j, rejoin
   leaves with spaces → the negative string.
4. Hardness h: cosine similarity of the two swapped WORDS under the CLIP ViT-B/32 text
   encoder (same `clip` package as run_frozen.py; batch + dedup unique word pairs —
   cheap, vocabulary is small).
5. Output `data/datasets/coco_hard_neg_specs.parquet`:
   `text_hash`, `neg_text`, `t` (obj|attr), `h`. Same information as the colleague's
   jsonl, keyed directly to our rows. Report: captions with ≥1 candidate (his figure:
   98.8%, mean 3.62), obj/attr mix, h distribution — compare against his stats as a
   sanity check of the port.
6. Validation of the port: for a sample of captions that DO exact-match his negs file,
   compare our enumerated swap set against his `negs` list — expect near-identical
   strings modulo lemmatization; investigate systematic differences.

Cost: ~440k cache lookups + pure-python tree walks + batched CLIP word encodings — hours,
not days; no bobcat parsing in this phase. Runs on the cluster (or locally if the
diskcache is synced — cluster copy is authoritative).

DECIDED (2026-07-21): own-generation is THE path. The join alternative is retired
(kept above only as history/context for the 66% overlap finding).

### Phase A — offline: build parsed hard-negative companion datasets (per parser)

New script `qnlp/scripts/coco_multi_caption/build_hard_negatives.py` (PARSER_VERSION-aware),
plus 2 submit scripts (bobcat / tree_no_type). Steps:
1. Obtain (text_hash → candidate negs) mapping — EITHER from Phase A0 own-generation
   output (default; already keyed by text_hash) OR by joining the colleague's negs jsonl:
   (cocoid from `local_image_path` filename, `processed_text.lower()` == negs `caption`),
   exact then fuzzy (see Cluster findings). Unmatched rows → no negatives (allowed).
2. Apply the h>0.95 drop at build time (keep-all fallback if list empties).
3. Dedup identical neg strings globally before parsing; parse+compile each neg string
   through the SAME pipeline as positives (bobcat: `qnlp/preprocessing_pipelines/coco/pipeline.py`
   machinery; tree: `pipeline_tree_no_type.py`). Reuse the existing sentence→diagram
   processing code — find the exact reusable function during implementation.
   Mind the lambeq compile memory leak (bound via worker_batch_size × max_tasks_per_child).
   Expect parse failures on ungrammatical swaps (esp. bobcat) — drop failures, log rate.
4. Output `coco_hard_negs{_tree_no_type}_train.parquet`:
   columns `text_hash` (of the POSITIVE row), `neg_text`, `h`, `t`, `diagram`, `symbols`.
   Keyed by positive text_hash at train time.
- Volume concern: ~1.95M candidate strings before dedup (~3.6/caption). If parse cost is
  prohibitive, fallback knob: cap candidates per caption (sample by the softmax rule
  offline, keep top-K≈4) — decide after measuring parse throughput on a small slice.

### Phase B — training-time π sampling + loss widening

1. Config (`qnlp/scripts/coco_multi_caption/config.py`): add
   `hard_neg_pi: float = 0.0` (env ML_HARD_NEG_PI), `hard_negs_dataset: str | None = None`
   (env ML_HARD_NEGS_DATASET), `hard_neg_softmax_temp: float = 0.5`, `hard_neg_h_max: float = 0.95`
   (h filter already applied at build; keep temp param for sampling).
2. New `HardNegativeBank` (suggest `qnlp/domain/datasets/hard_negatives.py`):
   loads companion parquet, maps text_hash → list[(diagram, symbols, h)]; method
   `sample(text_hash, rng) -> caption_tuple | None` implementing: with prob π pick one
   with prob ∝ exp((h − mean_h)/0.5), else None. Per-row independent, per step (fresh
   each epoch — sampling lives in collate/getitem so every epoch resamples).
3. Batch plumbing: in both train loops, for each batch collect sampled neg caption tuples
   (M ≈ π·B). Encode through the SAME text tower + text head:
   - frozen (`run_frozen.py::_run_epoch`): `F.normalize(text_head(text_model(neg_caps)))`.
   - non-frozen (`run.py::SimpleCaptionStep`): encode via `model.text_model` + `model.text_head`
     directly (ContrastiveVLM forward untouched).
   Drop non-finite NEGATIVE embeddings separately (they only remove columns, not rows).
4. Loss (`qnlp/core/training/losses/symmetric_infonce.py` + `single_caption.py`):
   optional `negative_text_emb` arg. i2t term becomes CE over
   `logit_scale * img @ cat([txt, negs]).T` with labels still arange(B); t2i term unchanged
   (negatives never queries). DECISION: we widen our existing i2t term (no extra
   double-counted term like the colleague's base+wide — theirs is an implementation
   artifact; our 3 invariants match their contract: labels 0..B−1, columns-only, i2t-only).
   π=0 path must be bit-identical to current loss (negs arg None).
5. Metrics: log `n_hard_negs` per batch, and neg-vs-pos similarity gap for monitoring.

### Phase C — runs (grid)

Submit scripts: parametrize the 4 existing linear submit scripts with `PI` env →
`ML_HARD_NEG_PI=$PI`, `ML_HARD_NEGS_DATASET=...`, `RUN_NAME` suffixed `_pi{PI}`.
Either 4 new scripts reading `PI` from `qsub -v PI=0.25`, or SGE array over π.
- Grid: {bobcat, tree_no_type} × {frozen (run_frozen), non-frozen (run)} × π ∈ {0, 0.1, 0.25, 0.5, 1.0} = 20 runs. All linear (ML_USE_NON_LINEAR_CONTRACTIONS=false).
- π=0 reruns included for comparability (new code path/RNG).
- Budget note: frozen runs cheap (~6–15 min/epoch post-speedup); non-frozen TTN runs are
  the expensive half — if queue-limited, run non-frozen at π ∈ {0, 0.25, 1.0} first.
- Readout: SugarCrepe swap-obj/swap-attr (aggregate dilutes ~7×), ARO, Winoground,
  COCO retrieval — all already produced by run.py/run_frozen.py report.

### Verification

1. Unit-ish check: build script on a 1k-row slice locally (bobcat) — assert join rate,
   parse failure rate, output schema.
2. π=0 equivalence: short run old code vs new code, same seed → identical losses.
3. π=1 smoke run (frozen, few epochs): confirm M≈B negs per batch, loss finite, new
   symbols from negs registered in EinsumModel (IMPORTANT: `collect_symbol_sizes` must
   include the negs companion dataset — negs introduce NEW symbols (new word/CCG-type
   combos from swapped word order); model must allocate them).
4. Full grid submission; update COCO_EXPERIMENTS.md with a new experiment entry
   (per standing instruction, log experiments + results there).

### Open items

All user-facing decisions resolved (see DECISIONS at top). Remaining implementation-time
measurements (not blockers):
- Parse cost of the generated neg strings (expect ~1.5–2M pre-dedup) — measure bobcat
  throughput on a 1k slice first; if prohibitive, cap candidates per caption at build
  time (top-K by the sampling weight) via a config knob.
- Verify diskcache hit rate for train captions in Phase A0 step 1 (expect ~100%).

### Execution order

1. Phase A0 generation script + validation vs colleague's negs (cluster, read-only inputs).
2. Phase A parse jobs (bobcat first, tree reuses trees) → 2 companion parquets.
3. Phase B code (config, HardNegativeBank, loss widening, both train loops) + π=0
   bit-identity check + π=1 frozen smoke run.
4. Phase C: submit 20-run grid; log new experiment entry + results in COCO_EXPERIMENTS.md.
