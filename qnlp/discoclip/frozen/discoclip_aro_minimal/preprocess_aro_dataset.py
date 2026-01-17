import gc
import json
import pandas as pd
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from lambeq import AtomicType, Rewriter
from lambeq.backend.tensor import Dim

from qnlp.discoclip.ansatz import CustomMPSAnsatz
from qnlp.discoclip.text_processor import BobcatTextProcessor
from qnlp.discoclip.cached_parser import CachedBobcatParser

worker_transform = None

def chunk_list(lst, chunk_size):
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]
        
def create_text_transform(dim, bond_dim):
    ansatz = CustomMPSAnsatz(
    {
        AtomicType.SENTENCE: Dim(dim),
        AtomicType.NOUN: Dim(dim),
        AtomicType.PREPOSITIONAL_PHRASE: Dim(dim),
    },
    bond_dim=bond_dim,
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
    rewriter = Rewriter(rules)
    bobcat_parser = CachedBobcatParser()
    text_transform = BobcatTextProcessor(
        ccg_parser=bobcat_parser,
        ansatz=ansatz,
        rewriter=rewriter,
    )
    return text_transform

def init_worker(dim, bond_dim):
    global worker_transform
    worker_transform = create_text_transform(dim, bond_dim)

def process_batch(batch):
    # text_transform = create_text_transform(dim, bond_dim)
    global worker_transform
    return worker_transform(batch) # type: ignore
    # try: 
    #     res = worker_transform(batch)
    #     return res
    # finally: 
    #     del worker_transform
    #     gc.collect()

def preprocess_aro(data_path: str, dim: int, bond_dim: int):
    dataset = pd.read_json(data_path)
    captions = dataset['true_caption'].unique().tolist() + dataset['false_caption'].unique().tolist()
    
    batches = list(chunk_list(captions, 100))
    dir_data_path, file_name = data_path.rsplit("/", 1)
    file_name = file_name.split(".")[0]
    processed_file_name = f"{dir_data_path}/processed_{file_name}.jsonl"
    with open(processed_file_name, "w") as f: 
        print(f"Streaming results to {processed_file_name}")
            
        with ProcessPoolExecutor(max_workers=2, initializer=init_worker, initargs=(dim, bond_dim)) as executor:
            future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
            # Collect results as they complete
            for future in as_completed(future_to_batch, timeout=60):
                batch_result = future.result()
                einsum_inputs = batch_result['einsum_inputs']
                captions_2 = batch_result['sentences']
                
                for c, out in zip(captions_2, einsum_inputs):
                    line = {
                        "caption": c,
                        "diagram": out[0],
                        "symbols": [[asdict(x[0]), x[1]] for x in out[1]]
                    }
                    f.write(json.dumps(line) + "\n")
                    f.flush()
                print("Finished processing batch")
    print("Finished!")
    

if __name__ == "__main__":
    from qnlp.discoclip.frozen.discoclip_aro_minimal.train_aro import VAL_DATA_PATH, BOND_DIM, EMBEDDING_DIM
    preprocess_aro(VAL_DATA_PATH, EMBEDDING_DIM, BOND_DIM)