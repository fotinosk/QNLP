# discoviz text models

How a parsed caption becomes a unit sentence-vector for contrastive image–text
training. Two designs, same output space (Rᵈ), different composition.

## `einsum_model.py` — EinsumModel
One free learnable tensor per symbol; the sentence vector is the `einsum`
contraction of the diagram's tensors. **Params grow with vocabulary** (~0.3–1B
on COCO). High capacity, poor generalisation — memorises (train→0.99, val→chance).

## `tree_neural_composer.py` — TreeNeuralComposer
Walks a CCG parse tree bottom-up:
- **leaf** → learned word embedding (`nn.Embedding`, + an UNK row)
- **binary node** → small MLP `(left, right) → parent`, keyed by the CCG rule
  (`SHAPE=ccg`) or one shared MLP (`SHAPE=ccg_single`, default)
- **unary** → identity · **n-ary** → left-fold · **root** → L2-normalise

The tree is the *wiring*; the embeddings + MLPs are the *weights* — **shared
across all captions and fixed w.r.t. sentence length** (~10M params). That
sharing is why it generalises and scales with data where EinsumModel doesn't.

### Tree format
Serialized CCG tree (`extract_trees_coco_cmp.py`): nested dicts — `{leaf, lemma,
type}` at leaves, `{rule, type, children}` internally. The composer keys on
`rule` + tree shape and ignores `type`.

`"2 giraffes grazing in the open savannah."` composes as:
```
g( [2·giraffes] , [grazing · [in · [the · [open·savannah.]]]] )
```
i.e. leaves → embeddings, every bracket → the rule MLP `g`, root normalised.

## Training
`tree_neural_coco_run.py`: TreeNeuralComposer + frozen CLIP-ViT image tower,
plugged into the shared `ContrastiveVLM` / `ContrastiveLoss` / `Trainer`.
Env knobs: `SHAPE`, `SUBSET`, `ART_*`.

Scaling result and rationale: `llm/model_evolution.md`.
