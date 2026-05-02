import polars as pl
import pyinflect  # noqa: F401 — registers token._.inflect with spaCy
import spacy

from qnlp.core.data_engine.processing.pipeline import PipelineStep
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="lemmatize_step")

# Load model once at module level.
# Disable pipeline components we don't need for a massive speedup.
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "lemmatizer", "custom"])


class LemmatizeStep(PipelineStep):
    """
    Transforms caption text into grammatically finite sentences.
    This guarantees that the downstream CCG Parser evaluates the text
    as an 'S' type, forcing the compiled tensor diagram to be Rank-1.
    """

    def __init__(self, text_column: str = "processed_text", batch_size: int = 1000):
        self.text_column = text_column
        self.batch_size = batch_size

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Starting LemmatizeStep for chunk of size {len(df)}")
        if self.text_column not in df.columns:
            return df

        # 1. Fast vectorised text cleaning natively in Polars
        df = df.with_columns(pl.col(self.text_column).str.replace_all(r"[^\w\s]", "").alias("__cleaned_text"))

        texts = df["__cleaned_text"].to_list()
        processed_texts = [None] * len(texts)

        # 2. Filter out nulls or empty strings to avoid processing overhead
        valid_data = [(i, t) for i, t in enumerate(texts) if t]

        for i, t in enumerate(texts):
            if not t:
                processed_texts[i] = t

        if not valid_data:
            return df.drop("__cleaned_text")

        valid_indices, valid_texts = zip(*valid_data)

        # 3. Batch process with SpaCy's optimized pipe (C-level batching)
        for i, doc in zip(valid_indices, nlp.pipe(valid_texts, batch_size=self.batch_size)):
            has_formal_verb = any(t.tag_ in ("VBZ", "VBP", "VBD") for t in doc)
            if has_formal_verb:
                sentence = doc.text.strip().capitalize()
                processed_texts[i] = sentence + ("." if not sentence.endswith(".") else "")
                continue

            new_tokens = []
            replace = False
            for token in doc:
                is_aux = any(child.dep_ == "aux" for child in token.children)
                if token.tag_ == "VBG" and not is_aux and not replace:
                    # Find the subject to determine singular vs plural
                    is_plural = False
                    for child in token.head.children:
                        if child.dep_ in ("nsubj", "nsubjpass") and child.morph.get("Number") == ["Plur"]:
                            is_plural = True
                            break

                    # Inflect based on number (VBZ for singular, VBP for plural)
                    target_tag = "VBP" if is_plural else "VBZ"
                    finite_verb = token._.inflect(target_tag)
                    new_tokens.append(finite_verb if finite_verb else token.text)
                    replace = True
                else:
                    new_tokens.append(token.text)

            sentence = "".join(
                [" " + t if not t.startswith(("'s", "n't", ",", ".")) else t for t in new_tokens]
            ).strip()
            processed_texts[i] = sentence.capitalize() if sentence.endswith(".") else (sentence.capitalize() + ".")

        # 4. Update the dataframe
        return df.with_columns(pl.Series(self.text_column, processed_texts)).drop("__cleaned_text")
