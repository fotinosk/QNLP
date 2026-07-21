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

### ⭐⭐ NEW PHASE 0 (2026-07-21) — unify bobcat/tree train-val-test split, READ BEFORE THE REVISION BELOW

**Not yet implemented. Proposed by user, verified feasible, recommended — do this
BEFORE hard-negative generation, it simplifies it.**

**Problem this solves:** bobcat and tree_no_type currently have DIFFERENT
train/val/test splits (see the correction below: only ~59% overlap between their
train sets). This isn't just an inconvenience for hard-neg generation — it's a
methodological confound for the whole π-sweep: bobcat vs tree cells are currently
trained on different image pools, not just different parsers, so any bobcat-vs-tree
difference in results is partly attributable to that, not the parser alone.

**Root cause (verified, read-only on beaker):**
`qnlp/core/data_engine/dataset_creator/dataset_generator.py::split_by_groups`
(called from `create_train_val_test_datasets`) already splits at the correct level
— image (`group_column="sample_id"` default) — via `_split_ids`: deterministic
`np.random.default_rng(seed)` shuffle + cutoff, default `ratios=(0.8,0.1,0.1)`,
default `seed=42`. Both bobcat (built 2026-06-19) and tree_no_type (built
2026-07-17) used this SAME function with the SAME seed — but the atlas is ingested
incrementally (`load_coco_to_atlas.py::ingest_data_from_remote`), so tree's build
had more images available than bobcat's did a month earlier. A shuffle+cutoff split
is sensitive to the size of the pool being shuffled, so identical seed+ratios over
different-sized universes produces different membership — this fully explains the
divergence (not a bug, just a consequence of incremental ingestion between builds).

**Verified (read-only on beaker) — bobcat's full pool is a STRICT SUBSET of tree's:**
```
bobcat full pool (train+val+test): 463,075
tree full pool (train+val+test):   542,040
intersection:                       463,075   (= 100% of bobcat's pool)
bobcat-only:                        0
tree-only:                          78,965 (14.6% of tree's pool)
```
Every caption bobcat has ever compiled already exists somewhere in tree's data too
(just possibly in a different split file) — so a matched split can be built with
**zero re-parsing or re-compiling**, purely by reshuffling existing rows.

**Proposed Phase 0 steps:**
1. Common pool = bobcat's full 463,075-caption pool (the limiting/smaller side;
   tree's extra 78,965 captions are simply unused for this purpose, no compute lost
   since they were never bobcat-compiled anyway).
2. Run `split_by_groups`-equivalent ONE time over this common pool (reuse the
   existing function/defaults for consistency: `ratios=(0.8,0.1,0.1)`, `seed=42`,
   `group_column="sample_id"`) → one shared train/val/test image assignment.
   ⚠️ NEEDS USER CONFIRMATION: defaulting to reusing the existing ratios/seed
   convention unless told otherwise.
3. For EACH parser, pull existing compiled rows (already has `diagram`/`symbols`
   from its own existing `_train/_val/_test.parquet`) for exactly the common-pool
   captions, and reassign them into NEW files per the new split assignment — a pure
   filter+regroup of existing rows, no recomputation.
4. Write as NEW files, do NOT overwrite the originals (standing rule — see "MUST
   create NEW output files" below): suggest
   `coco_single_caption_nlc_matched_{train,val,test}.parquet` (bobcat) and
   `coco_single_caption_nlc_tree_no_type_matched_{train,val,test}.parquet` (tree).
   Naming not finalized — confirm before implementing.
5. Consequence for hard-neg generation: SIMPLIFIES it back to the original design —
   since both parsers now train on the IDENTICAL caption set by construction, the
   per-parser conditional-compile logic (added in the correction below to handle
   divergent splits) becomes unnecessary; every caption in the new unified train set
   gets both diagrams compiled unconditionally again. Keep the conditional-compile
   CODE PATH anyway (cheap safety net) but it should become a no-op once Phase 0 is
   in place — every candidate will be in both hash sets by construction.
6. Consequence for Phase C (the 20-run grid): submit scripts need
   `ML_DATASET_NAME` pointed at the new matched dataset names instead of the
   originals — a small change, not yet made (submit scripts still reference the
   original names as of this writing).

**Ratios/seed CONFIRMED by user 2026-07-21: keep existing convention (0.8/0.1/0.1,
seed=42).** File naming still open — will confirm at implementation time.

**Verification plan — run ALL of these immediately after generating the matched
files, BEFORE starting hard-negative generation on top of them (a bug here would
silently invalidate everything built after it):**

**A. Structural integrity (per parser, per split file)**
1. No row loss / no duplication: `len(matched_train) + len(matched_val) + len(matched_test) == 463,075`
   (bobcat) and likewise for tree restricted to the common pool — every common-pool
   caption appears in EXACTLY ONE split file, none dropped, none duplicated.
2. `text_hash` values in the matched files are a SUBSET of (for tree) or EQUAL to
   (for bobcat) the union of `text_hash` across that parser's ORIGINAL
   train+val+test — confirms we only ever reused existing compiled rows, never
   invented or recomputed anything.
3. Tree-only captions (the 78,965 not in bobcat's pool) are ABSENT from all three
   tree matched files — confirms the common-pool restriction was applied correctly,
   not just to train but to val/test too.

**B. Cross-parser consistency (the actual point of Phase 0 — check this hardest)**
4. Build `sample_id -> split` maps from bobcat-matched and tree-matched separately;
   assert they are IDENTICAL for every sample_id in the common pool. This is the
   core correctness property: same image, same split, in both parsers.
5. Pick ~20 random `sample_id`s with multiple captions in the common pool; for each,
   confirm ALL of that image's captions (across both parsers) landed in the SAME
   split file. Directly validates image-level (not caption-level) grouping was
   preserved through the reslice — a caption-level bug here would leak an image's
   other captions across train/test, a real eval-validity failure that row-count
   checks alone would NOT catch.
6. No `sample_id` appears in more than one of {train, val, test} for either parser
   (the fundamental no-leakage property — verify directly, don't just trust
   `split_by_groups`'s docstring claim).

**C. Data fidelity (didn't corrupt anything while reslicing)**
7. For a random sample of ~50 text_hashes present in both the OLD and NEW files for
   a parser, assert `diagram` and `symbols` bytes are byte-identical between old and
   new — confirms the reslice only filtered/regrouped rows and never touched the
   compiled payload.
8. Schema match: matched files have identical column names/dtypes to the originals
   (`sample_id, local_image_path, processed_text, text_hash, diagram, symbols[, path]`)
   — required for the training code (`get_dataloaders`, `FrozenCOCODataset`, etc.)
   to consume them as a drop-in replacement via `ML_DATASET_NAME`.

**D. Reproducibility**
9. Re-run the Phase 0 split step twice with the same seed; assert the resulting
   `sample_id -> split` assignment is bit-identical both times. Guards against
   accidentally introducing nondeterminism (e.g. relying on dict/set iteration
   order, which isn't guaranteed stable) when reimplementing `split_by_groups`'s
   logic rather than calling it directly — PREFER calling the existing function
   directly over reimplementing it, precisely to avoid this risk.

**E. End-to-end smoke test (the check that catches what data-only checks can't)**
10. Point `ML_DATASET_NAME` at the new matched dataset and run a FEW BATCHES (not a
    full training run) through both `run.py` and `run_frozen.py` for both parsers —
    confirms `collect_symbol_sizes`, `TopologyBucketSampler`, and the dedup val/test
    loaders all work unmodified against the new files, before committing to a full
    20-run grid that depends on this.

**F. Process guard rail (prevents a future silent regression, not a one-time check)**
11. Once matched files are verified, explicitly grep the sweep's submit scripts to
    confirm ALL FOUR linear cells' `ML_DATASET_NAME` (or default) point at the
    matched files, not the originals — easy to forget for one cell and silently
    train it on the old, unmatched split. Do this check again right before
    launching Phase C, not just once at Phase 0 completion.

---

### ⭐ ARCHITECTURE REVISION 2026-07-21 — READ THIS SECOND, SUPERSEDES Phase A0 + Phase A BELOW

**Decision: merge Phase A0 (enumerate swaps) and Phase A (compile negatives to
diagrams) into ONE combined pipeline that reuses a single CCG parse per caption for
everything — no re-parsing of swapped strings, no separate compile pass.** Not yet
implemented (next session's task). The Phase A0 / Phase A sections further below are
KEPT for historical/technical reference (they contain code paths, constants, and
rule-port details still needed) but their *staging* (separate enumerate→specs-file→
recompile) is obsolete. Read this section, then pull needed details from below.

**Why this is correct, not just simpler:** a swap only exchanges two leaves that
already share a CCG type (two `n` nouns, or two `n/n` adjectives) — so the swap
NEVER changes the derivation tree structure, only two leaves' text. This means the
negative's CCGTree can be built by deep-copying the POSITIVE's already-parsed tree
and mutating two leaves — no fresh CCG search needed for the negative sentence at
all (which also sidesteps worrying whether a swapped sentence is independently
grammatical/parseable — it inherits the positive's valid derivation by construction).

**Critical correctness fix this caught (would have been a silent bug in the old
staged plan):** trained tensor symbol names are `{lemma}_{index}__{CCG_type}` —
lemma-based, not surface-token-based. Confirmed by reading
`qnlp/discoviz/models/bobcat_text_processor.py::BobcatTextProcessor.lemmatize_tree`:
it deep-copies the tree and overwrites EVERY leaf's `._text` with its NLTK lemma
(`node._text = next(lemma_iter)`) before compilation — this is what the real
positive-compilation pipeline does. The OLD (currently-running-on-cluster,
job 7081533) design swaps raw SURFACE tokens for the materialized `neg_text`
string, not lemmas. If that had fed into a separate re-compile step, the negative's
compiled symbols could mismatch the model's existing lemma-based vocabulary. The
combined design fixes this structurally: lemmatize the tree ONCE (right after
parsing, before enumeration), enumerate swaps on the now-lemma leaves, and swap
LEMMA text directly — so compiled negative symbols are guaranteed consistent.
(Side effect: this also makes moot the earlier open question of "should
LemmatizeStep re-run on the swapped text" — negatives never go back through raw-text
processing at all; everything happens at the tree level from the positive's parse.)

**Combined pipeline per caption (one worker-pool job, CPU-bound, same resumable
part-file pattern as before):**
1. Tokenize (`Tokenizer.tokenize`, from `bobcat_text_processor.py`).
2. Parse: `CachedBobcatParser(cache_path=BOBCAT_CACHE, load_parser=True).sentences2trees(tokens, tokenised=True, suppress_exceptions=True)`
   — as now; hits the diskcache when available, parses (and caches) on miss. This is
   ADDITIVE to the existing `bobcat/diskcache` — safe to keep writing into it (content-
   addressed by exact token sequence, no collision/corruption risk with existing entries).
3. Lemmatize the tree: NLTK lemmas via `Tokenizer.lemmatize(tokens)`, then apply the
   SAME leaf-overwrite as `BobcatTextProcessor.lemmatize_tree` (deepcopy + traverse +
   `node._text = lemma`) — reuse that exact method by instantiating a
   `BobcatTextProcessor` per worker (simplest: call its `.lemmatize_tree()` directly
   rather than reimplementing).
4. Enumerate obj/attr swaps on the LEMMATIZED tree's leaves (same rules as before:
   object = two `n` nouns with a predicate between, not coordinate; attribute = two
   `n/n` adjectives under forward application on different head nouns; MAXK=6/type;
   ported from `gen_hard_negs_v2.py`, already implemented and validated in the current
   `generate_hard_negatives.py` — reuse `object_swaps`/`attribute_swaps`/`_leaves`
   almost as-is, just fed the lemmatized tree instead of a raw one).
5. Per swap (i,j,t): deep-copy the LEMMATIZED tree, swap `leaf._text` at i,j (this
   now correctly swaps lemma-identity). Compile that tree TWICE, no re-parsing:
   - **bobcat**: `diagram = tree.to_diagram()`; `diagram = Rewriter(rules)(diagram).remove_snakes()`
     where `rules = ["auxiliary","connector","determiner","postadverb","preadverb","prepositional_phrase","coordination","object_rel_pronoun","subject_rel_pronoun"]`
     (exact list from `CCGCompilerStep.__init__`); `circuit = ansatz(diagram)`;
     `einsum_inputs = tn_to_einsum(circuit)` (`tn_to_einsum` lives in
     `bobcat_text_processor.py`).
   - **tree_no_type**: `diagram = TreeReader.tree2diagram(tree, mode=TreeReaderMode.NO_TYPE)`
     (NO rewriter — tree diagrams have no cups/snakes, per `pipeline_tree_no_type.py`);
     `circuit = ansatz(diagram)`; `einsum_inputs = tn_to_einsum(circuit)`.
   - `ansatz = CustomMPSAnsatz({AtomicType.SENTENCE: Dim(512), AtomicType.NOUN: Dim(512), AtomicType.PREPOSITIONAL_PHRASE: Dim(512)}, bond_dim=10)`
     from `qnlp.discoviz.parser.asnsatz.CustomMPSAnsatz` — embedding_dim=512, bond_dim=10
     matches `constants.embedding_dim`/`constants.bond_dim` AND every linear submit
     script's `ML_EMBEDDING_DIM`/`ML_BOND_DIM` in this sweep. ONE shared ansatz instance
     serves both diagram types (only diagram conversion differs).
   - Symbols serialization — copy exactly from `compiler_step.py::_worker_process_batch`
     so it's byte-compatible with what `collect_symbol_sizes`/`EinsumModel` expect:
     `symbols = [[asdict(x[0]), x[1]] for x in einsum_inputs[1]]` (needs
     `from dataclasses import asdict`); `diagram_out = einsum_inputs[0]`.
   - **No contraction-path computation** — this sweep is linear-only, so skip whatever
     `compute_contraction_paths=True` does; just diagram+symbols columns, matching the
     bobcat `coco_single_caption_train.parquet` (no `_nlc` suffix, no `path` column)
     schema. (Tree's own `coco_single_caption_nlc_tree_no_type_train.parquet` DOES carry
     a path column, but linear training ignores it — so it's fine/simplest for OUR
     negative companion files to omit `path` for BOTH parsers.)
   - `neg_text` for logging/validation/hardness = `" ".join(leaf.text for leaf in _leaves(swapped_lemma_tree))`
     — this is now the lemma-form sentence (consistent with what's actually compiled/trained on).

**⚠️ MUST create NEW output files — do NOT override or touch existing data:**
- Do NOT write to `data/sentence_mapping/` (LMDB) — that's the positives' authoritative
  compiled-diagram store; negatives never go there.
- Do NOT modify/overwrite `coco_single_caption*_train.parquet` /
  `coco_single_caption_nlc*_train.parquet` (bobcat or tree_no_type) — those existing
  training datasets stay exactly as they are; negatives are purely ADDITIVE companion
  files, joined at training time via `text_hash` (Phase B, not yet implemented).
- OK to keep reading/writing `bobcat/diskcache` (the raw-tree parse cache) — additive,
  content-addressed, no risk to existing entries.
- New output files (names TBD at implementation time, suggest):
  `data/datasets/coco_hard_negs_train.parquet` (bobcat: text_hash, processed_text,
  neg_text, t, w1, w2, diagram, symbols) and
  `data/datasets/coco_hard_negs_tree_no_type_train.parquet` (same, tree_no_type
  diagram/symbols). Confirmed 2026-07-21: these columns are sufficient to compute
  and join h — w1/w2 (already lemma-normalized) are exactly the CLIP scoring input,
  text_hash/processed_text trace back to the source caption. `h` (hardness) joined
  on afterward — CLIP-scoring stays a SEPARATE, short GPU pass over the accumulated
  unique (w1,w2) pairs across all parts (unchanged from the old design), run once,
  joined onto both per-parser outputs by **(w1, w2) alone** — `t` (obj/attr) is NOT
  part of the join key, since hardness is purely a function of the two words'
  CLIP similarity and doesn't depend on which swap rule produced the pair (a
  correction from an earlier, imprecise "(t, w1, w2)" note — including t is
  harmless/redundant, not required).
- Resumability: keep the same 50k-caption-chunk part-file pattern + worker pool with
  `maxtasksperchild` recycling (lambeq CCG memory leak mitigation), but each part now
  carries BOTH compiled outputs per candidate. Use a NEW parts-dir name distinct from
  the old `coco_hard_neg_specs*_parts/` (e.g. `coco_hard_negs_compiled_parts/`) so
  there's no risk of the new job resuming into old-design leftovers (see the earlier
  "resume gotcha" — still no automatic guard, per user's earlier explicit preference;
  just use a different name).

**Operational note — the OLD design's job is now obsolete:**
Job 7081533 (started 2026-07-21 02:59, old string-swap-only `generate_hard_negatives.py`)
does not produce compiled diagrams and swaps raw surface tokens, not lemmas — its
output cannot feed the new combined design. It was ~5h in / 3 of 11 parts done at
last check. User has NOT yet decided whether to `qdel` it or let it run to
completion (its output has no further use either way under this revision) — I have
NOT killed it myself, that's the user's call on their own compute budget. The old
`coco_hard_neg_specs*.parquet` / `coco_hard_neg_specs*_parts/` artifacts (and
`qnlp/scripts/coco_multi_caption/generate_hard_negatives.py` +
`validate_hard_negatives.py` + the two `submit_generate_hard_negatives_*.sh` scripts)
should be treated as SUPERSEDED once the combined script exists — either rewritten
in place or replaced by new files; not yet decided which.

**⚠️ CORRECTION 2026-07-21 — the two parsers' train splits are NOT the same set of
captions (verified, read-only on beaker):**

| | unique captions |
|---|---|
| bobcat train | 372,073 |
| tree_no_type train | 435,358 |
| **intersection** | 299,373 |
| bobcat-only | 72,700 (19.5% of bobcat train) |
| tree-only | 135,985 (31.2% of tree train) |
| union | 508,058 |

Only ~59% of the union is shared. Compiling BOTH diagrams unconditionally for every
caption in the union (as the pipeline description above implies) would waste compute
on ~136k bobcat-diagram compiles and ~73k tree-diagram compiles that are never
looked up by the parser that doesn't actually have that caption in its train set —
and worse, would leave each output file ambiguously scoped (containing rows for
captions not actually in that parser's own train set).

**FIX — compile step must be conditional per parser:** the shared part stays shared
(parse + lemmatize + enumerate over the union of 508,058 captions — the CCG
derivation doesn't care which split a caption belongs to, so this part is correctly
deduplicated). Before the main loop, build two hash sets — `bobcat_train_hashes`
and `tree_train_hashes` — from the two source `*_train.parquet` files. Then, per
swap candidate: compile the bobcat diagram ONLY IF `text_hash ∈ bobcat_train_hashes`;
compile the tree diagram ONLY IF `text_hash ∈ tree_train_hashes`. A caption in only
one parser's train set gets only that parser's diagram compiled (no wasted work);
a caption in both (the 299,373 intersection) gets both, still from the single shared
parse. Each output file ends up precisely scoped: `coco_hard_negs_train.parquet` has
entries for AT MOST 372,073 captions (bobcat's own train set exactly), and
`coco_hard_negs_tree_no_type_train.parquet` for AT MOST 435,358 (tree's own train
set exactly) — matching each parser's actual training data 1:1, with no ambiguity
about which rows are valid for which parser.

**Why read the existing train parquets rather than start from raw COCO data
(considered and rejected 2026-07-21):** it's not a shortcut, it's the only version
guaranteed consistent with what's actually trained on. (1) We don't know the exact
train/val/test split logic/seed used at dataset-creation time — reconstructing it
ourselves from raw data risks silently misclassifying a val/test caption as train,
i.e. an eval-leak bug that wouldn't be obvious. (2) The join key is
`text_hash = sha256(processed_text)`, and `processed_text` is `LemmatizeStep`'s
output (spaCy-based rewrite — verb conjugation, capitalization, restructuring); any
tiny discrepancy from recomputing it ourselves (library version, tokenizer
edge case) produces a different hash and the negatives silently fail to join onto
the real training rows — no error, just zero matches. Reading `processed_text`
straight from the parquet is byte-identical by construction. (3) No compute is
saved anyway — the CCG parse (the expensive step) needs `processed_text` regardless
of how we got there, so redoing lemmatization from raw data just adds cost for no
benefit while adding both risks above. Conclusion: always source captions/hashes
from the existing `*_train.parquet` files, never re-derive from raw COCO data.

**Output scope — confirmed 2026-07-21, no dataset pipeline / splits involved:**
The combined script is a standalone flat-file generator, NOT a run through the
atlas/`derived_v1`/LMDB dataset-pipeline machinery — no manifest, no atlas metadata.
Input is explicitly restricted to `coco_single_caption*_train.parquet` (both parsers)
— val/test parquets are NEVER read. So the output (`coco_hard_negs_train.parquet` /
`..._tree_no_type_train.parquet`) is train-only BY CONSTRUCTION, not something that
needs a train/val/test split step afterward — it's a `text_hash → candidate
negatives` lookup table that only ever contains entries for captions already in the
train split. Nothing to partition.
At training time (Phase B), the `HardNegativeBank` lookup must be wired ONLY into
the TRAIN forward/loss step — val/test evaluation code paths (`_dedup_loader`,
`_collect_retrieval_metrics`, benchmark evaluators in `evaluate.py`) are separate
and must never call it. NOTE the join key is `text_hash` (hash of the caption
STRING), not `(cocoid, cap_idx)` — if some generic caption string happened to
appear verbatim in val/test for a different image it would share a `text_hash`
with a train row, which is harmless ONLY because the lookup is exclusively
consulted from the train step. Be deliberate about this when wiring Phase B: do
not create a shared "look up negatives for this row" utility that both train and
eval code could accidentally call.

**Next step (this is what "we will implement it" refers to):** write the combined
enumerate+compile script (working name `generate_hard_negatives.py` v3, or a new
file), reusing `object_swaps`/`attribute_swaps`/`_leaves`/`MAXK` from the current
implementation almost unchanged, adding the lemmatize-tree step and the dual-compile
step described above, and new submit script(s). Validation
(`validate_hard_negatives.py`'s Jaccard-vs-colleague's-negs check) should still run,
pointed at the new output's `(w1, w2)` columns (unchanged semantics, new file path).

**STATUS: v3 combined script IMPLEMENTED 2026-07-21 (rewritten in place; old
enumerate/score-only design is gone from git history but recoverable, not kept
side-by-side).** `qnlp/scripts/coco_multi_caption/generate_hard_negatives.py` now
does everything described above in one `enumerate` stage:
- `_worker_init` loads `CachedBobcatParser` (plain `bobcat/diskcache`, unversioned
  — parses+caches on miss, same as before), a `Tokenizer`, ONE shared
  `CustomMPSAnsatz(embedding_dim=512, bond_dim=10)` (hardcoded as module constants
  `EMBEDDING_DIM`/`BOND_DIM`, matching every linear submit script's
  `ML_EMBEDDING_DIM`/`ML_BOND_DIM` — verified by reading the 4 scripts directly,
  not just trusting the doc above), and `Rewriter(REWRITE_RULES)` where
  `REWRITE_RULES` is copy-pasted verbatim from `CCGCompilerStep.__init__`.
  Also loads `bobcat_train_hashes`/`tree_train_hashes` as plain Python `set`s by
  reading only the `text_hash` column of each parser's `*_train.parquet` — one
  read per worker at startup, not per batch.
- `_worker_process_batch`: tokenize -> `parser.sentences2trees(...)` (batched) ->
  per caption: skip if `text_hash` in neither hash set; else NLTK-lemmatize
  (`Tokenizer.lemmatize`) and build the lemmatized tree via a direct port of
  `BobcatTextProcessor.lemmatize_tree` (`_lemmatize_tree` — deepcopy + leaf-text
  overwrite, avoids depending on a full `BobcatTextProcessor` instance); enumerate
  obj/attr swaps on the lemmatized leaves (unchanged rule logic); per swap,
  `_swap_tree` deep-copies the lemmatized tree ONCE and swaps two leaves' `_text`,
  shared for both compiles; `_compile_bobcat` (`to_diagram` -> `Rewriter(...).
  remove_snakes()` -> ansatz -> `tn_to_einsum`) if the caption is in bobcat's
  train set, `_compile_tree_no_type` (`TreeReader.tree2diagram(NO_TYPE)` -> ansatz
  -> `tn_to_einsum`) if in tree's — both gated independently per the "CORRECTION"
  conditional-compile design above.
- **New fix found while implementing, not in the plan doc above:** the real
  pipeline (`compiler_step.py::_worker_process_batch` + `dataset_generator.py`'s
  `UnifyEinsumRankStep`) doesn't just emit whatever einsum string
  `tn_to_einsum` produces — it truncates the diagram's output signature to a
  SINGLE index (tracing out the rest) whenever the CCG resolves to >1 open wire,
  and drops rows that still aren't rank-1 after that. The plan doc's compile
  steps above didn't mention this. Ported directly (`_unify_rank`/`_is_1d`,
  regex-equivalent to `UnifyEinsumRankStep`'s string replace, verified against it
  line-by-line) and applied inside `_compile_diagram` before a row is kept —
  skipping this would have produced diagrams with the wrong output rank for
  `EinsumModel` to consume, a silent shape-mismatch bug at training time, not a
  crash at generation time.
- Output: two independent resumable part-file streams per 50k-caption chunk,
  `part_bobcat_{k}.parquet` / `part_tree_{k}.parquet` under
  `data/datasets/coco_hard_negs_compiled_parts/` (new dir, per the plan's "use a
  different parts-dir name" rule — cannot collide with the old design's
  `coco_hard_neg_specs*_parts/`). A chunk resumes only when BOTH files for that
  chunk exist. Columns: `text_hash, processed_text, neg_text, t, w1, w2, diagram,
  symbols` (`symbols` stored as an `orjson`-serialized JSON string, matching how
  the real pipeline stores it in the training parquets).
- `score` stage: gathers the UNION of `(w1, w2)` pairs across both parts sets
  (not just one), CLIP-scores once, joins `h` onto each parser's parts separately,
  writes `data/datasets/coco_hard_negs_train.parquet` and
  `data/datasets/coco_hard_negs_tree_no_type_train.parquet`.
- `scripts/submit_generate_hard_negatives_enumerate.sh` and
  `..._score.sh` updated in place: new `--parts-dir`/output paths, updated echo
  text (no longer describes a "Phase A0 enumerate-only" job). `SMOKE` mode now
  writes to `coco_hard_negs_compiled_smoke_parts/`.
  `validate_hard_negatives.py`'s default `SPECS` path updated to
  `coco_hard_negs_train.parquet` (still reads the same `(text_hash, t, w1, w2)`
  columns, so its Jaccard-vs-colleague's-negs logic needed no other changes).
- **Verified locally (no cluster, `qnlp` conda env), 2026-07-21:**
  1. Parsed `"A red dog chases a small cat"` with a real `CachedBobcatParser` and
     confirmed, post-lemmatization, `object_swaps`/`attribute_swaps` still recover
     `dog↔cat` (obj) and `red↔small` (attr) — the lemmatize-before-enumerate
     reordering didn't break the rule port.
  2. Ran `_compile_bobcat` and `_compile_tree_no_type` on the swapped tree end to
     end: both produced valid rank-1 einsum diagrams (`...->i`, `...->W` — single
     output index each), confirming the ansatz/rewriter/rank-truncation wiring is
     correct, not just importable.
  3. Unit-checked `_unify_rank`/`_is_1d` against a synthetic multi-output-wire
     einsum string (`ab,cd->bcd` -> `ab,cd->b`) — matches `UnifyEinsumRankStep`'s
     behavior exactly.
  4. Ran the actual `enumerate` + `score` CLI stages end to end locally (2
     synthetic captions, 2 workers, real multiprocessing pool + part-file
     writing + CLIP scoring) — **PASSED**, after finding and fixing two real bugs
     along the way (see "Bugs found and fixed" below). Final output inspected
     directly: `coco_hard_negs_train.parquet`-equivalent has 3 rows for the 2
     captions (`h1` -> obj `dog↔cat` h=0.931, attr `red↔small` h=0.860; `h2` -> obj
     `bike↔man` h=0.848 — `h2` "a man ride a bike" only appears in the bobcat
     output, correctly excluded from tree output since it wasn't in the tiny
     `tree_train` fixture, confirming the per-parser conditional-compile gating
     works), tree output has 2 rows (only `h1`, correctly excluded `h2`); both have
     the full `text_hash, processed_text, neg_text, t, w1, w2, diagram, symbols, h`
     schema; `neg_text` values are correctly swapped ("a red cat chase a small
     dog", "a small dog chase a red cat", "a bike ride a man"); diagram strings
     end in a single output index (rank-1, e.g. `...->i`, `...->W`) confirming the
     `_unify_rank` port is wired correctly end to end, not just unit-correct.

**Bugs found and fixed while running the above (both real, neither anticipated by
the plan doc above):**

1. **Cluster crash — diskcache write race under concurrent workers (hit twice on
   the cluster: smoke job 7082632, batch 8/10, ~09:57-10:00). All `max_workers`
   processes point `CachedBobcatParser` at the SAME shared `bobcat/diskcache`
   directory and write every cache miss back to it; since every caption in this
   job is unique (deduped by `text_hash` up front), essentially every parse is a
   miss, so this is unusually write-heavy vs. the cache's normal mostly-read
   pattern. Under that write pressure, diskcache's internal `_cull`/`reset`
   accounting hit a transient race (`ValueError: not enough values to unpack
   (expected 1, got 0)` inside `diskcache/core.py`, from `self._cache[key] =
   result` in `cached_bobcat.py::sentences2trees`) and killed the job — the parse
   result itself had already been computed successfully; only the optional
   write-back failed.
   **Tried:** wrapped that write in try/except (log + keep the parsed result
   regardless) in `qnlp/discoviz/parser/cached_bobcat.py`.
   **REVERTED at user's request (2026-07-21):** job 7082661 (the FULL run) was
   submitted 7 minutes after 7082632 (the smoke run) and has NOT hit this error —
   still running cleanly at the time this was checked. User's read: this was
   likely triggered by running the smoke and full jobs CONCURRENTLY (2 jobs × 5
   workers = 10 processes hammering the same shared diskcache at once, not just 5
   from a single job), not a flaw that shows up in normal single-job operation.
   Reverted `cached_bobcat.py` back to its original (unwrapped) state — do not
   reapply without new evidence this recurs from a single job running alone. If
   it recurs under normal (non-overlapping) operation, the fix above is ready to
   reapply from this doc's history.
2. **Local-only hang, looked like a stall but was actually a crash-loop.** The
   module hardcodes `BOBCAT_CACHE = "/SAN/intelsys/discoviz/fotinos/QNLP/.cache/
   lambeq/bobcat/diskcache"`, which only exists on the cluster — locally `/SAN`
   doesn't exist at all. Each worker's `_worker_init` failed trying to create the
   cache dir there, and `multiprocessing.Pool` kept spawning replacement workers
   to maintain its process count (a known Pool gotcha when an initializer raises)
   — ~250 rapid respawns over ~28 minutes with zero progress, easy to mistake for
   a slow cold-start rather than a crash-loop (confirmed via the giveaway: one
   fresh timestamped log file created every few seconds, since `setup_logger` runs
   at module import and each respawn re-imports the module). **Fix (kept):** added
   a `--cache-path` CLI override (defaults to the same cluster path, so cluster
   behavior is unchanged) so this is actually testable locally.
3. **Job 7082661 (the full run) stalled at the end of part 1/11, memory-blown —
   found 2026-07-21 ~13:00-14:00 while monitoring it live.** Progressed cleanly
   through all 250 batches of part 1 (`CHUNK` was still 50,000 at the time —
   250 batches × 200/batch = 50,000 captions, steady ~0.17-0.18s/caption avg,
   no errors), then went silent for 30-40+ minutes after batch 250 with the parts
   dir still empty. `qstat -j 7082661` showed `maxvmem=90.277G` against a
   `tmem=16G` × 5 slots (`pe smp 5`) request — i.e. peaked at/near/over the total
   memory reservation. Root cause (reasoned from the code, not directly observed
   via a cluster debugger): `enumerate_stage` accumulates ALL of a chunk's output
   rows (`bobcat_rows`, `tree_rows` — Python lists of tuples, ~150-200k rows for a
   50k-caption chunk at the ~3.5-3.6 cand/caption reference rate) in the MAIN
   process across all 250 batches, and only builds+writes at the very end via
   `pl.DataFrame(rows, schema=_ROW_SCHEMA, orient="row")` — a row-oriented
   construction that has to hold the source list AND the new columnar frame
   simultaneously. This is a purely structural memory peak (list monotonically
   grows to its max size exactly at the last batch, then roughly doubles during
   the DataFrame build) — NOT the lambeq-tree-leak explanation first suspected;
   that would show up as gradual/periodic slowdown across batches (bounded by
   `maxtasksperchild` worker recycling), not a hard stop precisely at the chunk
   boundary, which is what was actually observed. This step was also completely
   unlogged before the fix below, so there was no way to tell "slow" from "stuck"
   from the log alone.
   **Fix:** `CHUNK` reduced 50,000 -> 5,000 (bounds the worst-case accumulated
   list and DataFrame-build size to ~1/10th, i.e. peak memory should drop
   roughly proportionally). Added logging around the previously-silent end-of-
   chunk step: row counts accumulated, DataFrame-build duration, and per-file
   write duration/row-count, so a future stall is immediately diagnosable by
   which specific sub-step it's in rather than by "guess and check `ps`/`qstat`."
   Verified locally (2-caption smoke test) that the new log lines fire correctly
   end to end. **Job 7082661 should be `qdel`'d — it is expected to be
   stuck/thrashing indefinitely at the memory ceiling, not merely slow — before
   resubmitting with this fix.**

**NOT yet done (must happen before the real cluster run):**
- `qdel 7082661` (memory-blown, stuck since ~12:29) and resubmit fresh with the
  `CHUNK=5,000` + logging fix. `cached_bobcat.py` stays UNCHANGED from its
  original state (fix #1 above reverted) — do not resubmit believing that's
  fixed, it isn't; watch for the diskcache race recurring specifically if
  multiple jobs against this script are ever run concurrently again.
- Cluster smoke test needed to (a) confirm coverage/candidate-rate stats still
  match the ~99%/3.5-3.6-per-caption reference now that lemmatize+dual-compile
  runs inside the same worker call, and (b) measure real per-caption throughput of
  the COMBINED parse+compile pipeline, which is new cost the earlier
  (~1h/50k-caption-part) throughput numbers from the enumerate-only design did NOT
  include — do not assume the old timing extrapolation still holds before
  checking. Also re-check the new memory ceiling at the smaller `CHUNK=5,000`
  before trusting the full 11-part (now more chunks at the smaller size) run not
  to hit the same wall.
- Decide fate of the OLD job 7081533 (enumerate-only design, string-swap-only,
  still possibly running on the cluster as of this writing) — asked the user,
  not yet confirmed either way (kill vs let finish and ignore output). User said
  (2026-07-21) they'll let it keep running since it doesn't hurt anything; its
  output remains not directly usable by the new design (see "operational note"
  above), at most useful later as an independent cross-check.
- Phase 0 (unified bobcat/tree split) is still NOT implemented — this combined
  script's per-parser conditional-compile logic is currently load-bearing (real
  ~59% train-set divergence), not the no-op safety net it becomes once Phase 0
  lands.

---

### Phase A0 — generating hard negatives OURSELVES (own-generation path) [STAGING SUPERSEDED — see revision above; technical details below still apply]

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

**SMOKE TEST RESULT (job 7081219, 2026-07-20): FOUND A DESIGN BUG, NOW FIXED.**
386/500 (77%) cache misses — the "no parsing happens, cache hits only" assumption
was wrong. Root cause (confirmed by direct key comparison against the real
diskcache): in `compiler_step.py::CCGCompilerStep.process()`, the **LMDB diagram
store (keyed by text_hash) is checked FIRST**, and the worker pool (hence
`CachedBobcatParser`, hence the tree diskcache) is only invoked for captions NOT
already in LMDB at compile time. So the tree diskcache is an incomplete,
history-dependent subset of the training set — not a full mirror — and never was.
The enumeration logic itself was fine: of the 114/500 captions that DID have a
cached tree, candidate yield (3.45/caption) and h distribution matched his
reference (3.62/caption) closely.

**REDESIGNED 2026-07-20 — two-stage, now a real parsing job:**
`generate_hard_negatives.py` is now `enumerate`/`score` subcommands (both required
via CLI positional arg):
- `enumerate` (CPU only): worker pool mirroring `CCGCompilerStep`'s pattern
  (`mp.get_context("spawn").Pool`, `maxtasksperchild` recycling — same lambeq
  memory-leak mitigation as [[project-lambeq-tree-memory-leak]]). Each worker
  holds ONE real `CachedBobcatParser(load_parser=True)` targeting
  `bobcat/diskcache` and calls `sentences2trees()` per batch — this checks the
  cache internally AND parses+caches on a miss, so coverage becomes complete
  rather than whatever fraction happened to miss LMDB historically. Trees never
  cross the process boundary as raw objects — swap enumeration happens INSIDE
  the worker (`_enumerate_from_tree`), only lightweight tuples come back. Writes
  the same resumable 50k-caption part files as before (no `h` column yet).
- `score` (GPU, short): loads all parts, CLIP-embeds the unique swapped words
  once, joins `h`, writes final `coco_hard_neg_specs.parquet` atomically. Kept
  separate so the (many-hour) CPU parsing stage never holds a GPU allocation idle.
- Only `bobcat/diskcache` is used now (dropped the separate tree_no_type cache
  path) — both parsers share the same underlying CCG trees, this generation step
  doesn't care which store a tree came from, and using one cache halves the
  redundant parsing work versus checking two.
- SCOPE vs the real preprocessing pipeline (clarified 2026-07-21): A0 redoes only
  the CCG PARSE (tagging + CKY search) — NOT the ansatz/rewriter/einsum-compile
  steps that turn a tree into a training diagram (those are unavoidably
  parser-specific and belong to Phase A, applied to the negative strings this
  step produces). The parse itself is parser-INDEPENDENT (bobcat and tree_no_type
  share identical CCG trees; tree_no_type only changes diagram conversion,
  downstream of parsing) and captions are deduped by text_hash across both train
  parquets, so each caption is parsed ONCE regardless of which dataset(s) it's
  in — not twice. Same for CLIP hardness scoring (one pass over unique words).
  Phase A, by contrast, DOES need to run twice (once per parser) since diagram
  compilation genuinely differs — that's real, unavoidable duplication, but over
  the cheaper compile step, not the expensive parse.
- Verified locally (no cluster needed): synthetic-tree swap enumeration still
  correct after the refactor into `_enumerate_from_tree`; `CCGTree` confirmed
  plain-attribute (safe for `to_json`/pickling, though we no longer need to move
  tree objects across processes at all — enumeration moved inside the worker
  instead); module imports cleanly; both submit scripts pass `bash -n`.
  NOT verified: real per-sentence bobcat parse throughput (a probe job to time
  this stalled/timed out against the cluster mid-session — cluster was reported
  unresponsive; harmless to skip, the 72h budget has generous headroom, but
  CHECK THE FIRST SMOKE RUN'S elapsed-time-per-caption to see if `max-workers`/
  `worker-batch-size` need tuning before the full run).

Scripts: `scripts/submit_generate_hard_negatives_enumerate.sh` (CPU, 16G tmem,
5 slots, 72h — mirrors `submit_coco_create_dataset_tree_no_type.sh`'s resource
shape) and `scripts/submit_generate_hard_negatives_score.sh` (GPU, 2h). The old
single-script `submit_generate_hard_negatives.sh` is REMOVED (superseded).

TO RUN (user):
```
qsub -v SMOKE=2000 scripts/submit_generate_hard_negatives_enumerate.sh   # smoke
# check log: coverage should approach ~99%, not 22.8%; note elapsed time/caption
qsub scripts/submit_generate_hard_negatives_enumerate.sh                 # full (resumable — rerun same cmd if killed)
qsub scripts/submit_generate_hard_negatives_score.sh                     # after all parts exist; also runs validation
```

**SMOKE TEST v2 RESULT (2026-07-21, after delete+rerun with the redesigned enumerate
stage): CORRECTNESS CONFIRMED, THROUGHPUT UNKNOWN.** 500 captions: 99.2% coverage
(496/500, vs his 98.8% reference), 3.19 obj/cap + 0.35 attr/cap = 3.54 cand/cap total
(vs his 3.62) — the parser-based fix works and the rule port is validated.
BUT: took 2007s (33 min) wall time, and a standalone probe of a single cold parse call
earlier measured 226s for ONE sentence (clearly a one-time model-load cost, not
steady-state). With only 3 batches (worker_batch_size=200) across 4 workers, this smoke
test is too small to separate pool-startup cost from real per-caption throughput — and
taken at face value, 500 caps/2007s extrapolated to the ~391k captions still needing a
real parse (77% of 508k unique, from the original miss-rate finding) would be ~2-3 WEEKS,
far over the 72h budget. Added per-batch timing logs (first batch includes startup,
later batches show steady-state) to `enumerate_stage` so a single run can distinguish
them. NEXT STEP: rerun a LARGER smoke test (suggest SMOKE=5000, several batches per
worker) and check the per-batch log — if batches after the first settle to a much lower
s/caption, we're fine; if not, need more workers or a rethink (e.g. batch multiple
sentences into one `sentences2trees` call more aggressively, or accept a much longer
walltime / split the 72h job into several resumed submissions).

**THROUGHPUT RESOLVED (2026-07-21, full run job 7081533, launched directly instead of
the bigger-smoke-test step above — worked out fine).** Real per-part timings from the
job log: part 1 9226s (≈2h34m, pays one-time 4-worker pool startup/model-load cost),
part 2 4152s, part 3 3727s (≈1h/part) — clearly converged to steady state by part 2.
Extrapolated total enumerate-stage wall time for all 11 parts (508,058 unique
captions): ≈13.5h, well inside the 72h budget. Quality holds at scale: ~178k
candidates/50k-caption part (3.56/cap) and ~98.6–98.7% coverage per part, consistent
with the smoke test (3.54/cap, 99.2%) and the colleague's reference (3.62/cap, 98.8%).
No further action needed — job runs to completion unattended; if SGE kills it before
then, resubmit the same script and it resumes from the first missing part.

⚠️ **Resume gotcha (hit once, 2026-07-21, decided NOT to auto-guard):** resuming only
checks whether a `part_NNNNN.parquet` file exists — NOT whether it was produced by the
current code. The first smoke run (job 7081219, pre-redesign dict-lookup version, 77%
miss) left `coco_hard_neg_specs_smoke_parts/part_00000.parquet`; rerunning the redesigned
script silently reused that stale/wrong part instead of regenerating it ("Part 1/1 exists
— skipping"). Considered adding a version-marker check; user explicitly declined
(prefers manual delete-and-rerun over that machinery). RULE: whenever
`generate_hard_negatives.py`'s enumerate logic changes, manually delete the relevant
`*_parts/` directory before rerunning — smoke and full runs use separate parts dirs
(`coco_hard_neg_specs_smoke_parts/` vs `coco_hard_neg_specs_parts/`) so they can't
cross-contaminate each other, but a stale dir from an OLDER version of the same stage
will not be detected automatically.

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

### Phase A — offline: build parsed hard-negative companion datasets (per parser) [SUPERSEDED — folded into the combined pipeline in the revision section above; this whole phase as a SEPARATE re-parse step is gone. Kept only for the "no contraction path needed" / output-schema notes, already carried into the revision above.]

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
