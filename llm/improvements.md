# Training Improvements

## Systematic Errors

### 1. Projectors never enter eval mode
The `Trainer` calls `model.eval()` but the projectors (`image_projector`, `text_projector`) are
standalone modules not registered inside `ContrastiveVLM`. During validation and test, their
`BatchNorm` layers use batch statistics instead of running statistics, producing inconsistent
metrics.

**Fix:** register projectors as children of the model wrapper, or call `.eval()` / `.train()`
explicitly in `COCOSingleCaptionStep` based on the `train` flag.

### 2. `cosine_similarity` metric is computed on projector outputs, not backbone
`VICRegLoss` computes `cos_sim` from `z_img`/`z_txt` — the projector outputs. The projector is a
throw-away head; at evaluation time the backbone embeddings are used directly. A projector that
collapses dimensions aggressively will look great on this metric while the backbone learns nothing.
`monitor_metric="cosine_similarity"` is therefore not selecting the best backbone checkpoint.

**Fix:** pass raw backbone embeddings through the step as a separate key, compute
`cosine_similarity` before projecting, and return it alongside the VICReg metrics.

### 3. Projectors not checkpointed
Only `model` (ContrastiveVLM) is saved in the trainer checkpoint. If the best epoch is epoch 11
and training runs to epoch 30, test evaluation uses epoch-30 projector weights with an epoch-11
backbone — they are not calibrated to each other.

**Fix:** include projector state dicts in the checkpoint, or register projectors inside the model
wrapper so they are saved automatically.

---

## ML Design Improvements

### 4. Switch to in-batch SymmetricInfoNCE (highest priority)
With `batch_size=512` and a single-caption-per-image COCO dataset, each batch naturally contains
512 distinct (image, caption) pairs — 511 in-batch negatives per sample with no extra annotation
required. This is CLIP-style training. InfoNCE failed in the ARO contrastive setup because the
parquet gave only 1 explicit negative per image (effectively batch_size=2 for the contrastive
task). COCO single-caption does not have that problem.

**Fix:** replace VICReg with `SymmetricInfoNCE` for COCO single-caption training. Remove the
projector heads (they are not needed for InfoNCE). With B=512 the loss landscape is rich enough
for stable learning.

### 5. Add LR warmup + cosine decay
Neither backbone has a scheduler. EinsumModel is particularly sensitive to early gradient noise
because its symbol tensors are initialised randomly. A 5-epoch linear warmup followed by cosine
decay would stabilise the first few epochs where most instability occurs.

### 6. Decouple projector and backbone schedulers
The projector converges much faster than the backbone. Tying their schedulers means the projector
is either underfitted or the backbone is pushed too hard early. Use separate `param_group` entries
with different schedulers or at minimum different warmup lengths.

---

## Tensor Network Specific

### 7. Add a linear adapter after EinsumModel
EinsumModel output depends on the CCG diagram structure and the learned symbol tensors. The
gradient path is long and sparse — each word contributes through one tensor contraction. Adding a
single `Linear(embedding_dim, embedding_dim)` after EinsumModel (before the projector or loss)
gives the optimiser a denser gradient path and lets the network compensate for structural mismatch
between the CCG parse and the image feature space.

### 8. The 78% ARO accuracy comes from the triplet loss, not InfoNCE
From `train_aro_clean.py`: `loss = infonce_loss + 40000 * hard_neg_loss`. The InfoNCE with
batch_size=128 provides structure but the triplet term at weight 40000 is doing the heavy lifting.

**Recommended training trajectory:**
1. Train on COCO single-caption with in-batch InfoNCE to bootstrap general image-text alignment.
2. Fine-tune on an ARO-style contrastive dataset with InfoNCE + weighted TripletMarginLoss to
   push hard-negative discrimination.
