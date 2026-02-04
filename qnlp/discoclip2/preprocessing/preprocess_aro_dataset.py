import gc
import json
import pandas as pd
from dataclasses import asdict
import concurrent.futures
import psutil
from qnlp.discoclip2.trainers.frozen.train_aro import BOND_DIM, EMBEDDING_DIM, TRAIN_DATA_PATH


# Global variables for worker processes
_processor = None
_parser = None
_ansatz = None

def worker_init(bond_dim, embedding_dim):
    """
    Initialise heavy objects ONCE per worker process.
    """
    global _processor, _parser, _ansatz
    from lambeq import AtomicType, Rewriter
    from lambeq.backend.tensor import Dim
    from qnlp.discoclip2.parser.asnsatz import CustomMPSAnsatz
    from qnlp.discoclip2.models.bobcat_text_processor import BobcatTextProcessor
    from qnlp.discoclip2.parser.cached_bobcat import CachedBobcatParser

    _ansatz = CustomMPSAnsatz(
        {AtomicType.SENTENCE: Dim(embedding_dim), AtomicType.NOUN: Dim(embedding_dim), AtomicType.PREPOSITIONAL_PHRASE: Dim(embedding_dim)},
        bond_dim=bond_dim
    )
    
    rules = [
        "auxiliary",
        "connector",
        "determiner",
        "postadverb",
        "preadverb",
        "prepositional_phrase",
        "coordination",
        "object_rel_pronoun",
        "subject_rel_pronoun",
    ]
    
    _parser = CachedBobcatParser(device="mps")
    _processor = BobcatTextProcessor(ccg_parser=_parser, ansatz=_ansatz, rewriter=Rewriter(rules),)

def process_batch_standalone(batch_data):
    """
    Process a batch of data using the pre-initialised processor.
    """
    global _processor
    if _processor is None:
        raise RuntimeError("Processor not initialised in worker process.")
    
    result = _processor(batch_data)
    import gc
    gc.collect()
    return result

def preprocess_aro(data_path: str, max_line_to_process:int = 4000):
    dataset = pd.read_json(data_path)
    # filter out bad sentences
    dataset = dataset[(~dataset['true_caption'].str.contains("pasture")) & (~dataset['false_caption'].str.contains("pasture"))]
    captions = set(dataset['true_caption'].unique().tolist() + dataset['false_caption'].unique().tolist())

    output_path = f"{data_path.split('.')[0]}_processed_{EMBEDDING_DIM}.jsonl"
    print(output_path)
    processed_dataset = pd.read_json(output_path, lines=True)
    if not processed_dataset.empty:
        processed_captions = set(processed_dataset['caption'].unique())
        captions = list(captions - processed_captions)
        print(f"{len(captions)} captions left to process, already processed {len(processed_captions)}")
    else:
        captions = list(captions)

    captions = captions[:min(len(captions), max_line_to_process)]
    print(f"Processing {len(captions)} captions")
    
    # Smaller chunks (e.g., 50) allow more frequent RAM clearing
    chunks = [captions[i:i + 50] for i in range(0, len(captions), 50)]
    
    
    print(f"Processing {len(captions)} lines in {len(chunks)} chunks...")

    # max_workers=1 or 2 is safest for Bobcat on a standard laptop
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, initializer=worker_init, initargs=(BOND_DIM, EMBEDDING_DIM)) as executor:
        # We use executor.map because it yields results in order and 
        # doesn't store all futures in a massive dictionary upfront.
        results_iterator = executor.map(process_batch_standalone, chunks)

        with open(output_path, "a") as f:
            for i, batch_result in enumerate(results_iterator):
                einsum_inputs = batch_result['einsum_inputs']
                sentences = batch_result['sentences']
                
                for c, out in zip(sentences, einsum_inputs):
                    line = {
                        "caption": c,
                        "diagram": out[0],
                        "symbols": [[asdict(x[0]), x[1]] for x in out[1]]
                    }
                    f.write(json.dumps(line) + "\n")
                
                f.flush() # Force write to disk
                print(f"Batch {i+1}/{len(chunks)} written to disk. (RAM: {psutil.Process().memory_info().rss/(1024**2):.0f}MB)")
                
                # Explicit trigger for the main process
                gc.collect()

if __name__ == "__main__":
    
    preprocess_aro(TRAIN_DATA_PATH)
