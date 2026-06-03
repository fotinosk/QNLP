import torch

from qnlp.constants import constants
from qnlp.core.non_linear_contraction.einsum_interface import contract_einsum_non_linearly

DATASETS_PATH = constants.datasets_path

VAL_PARQUET = DATASETS_PATH / "coco_short_caption_val.parquet"

if __name__ == "__main__":
    # df = pl.read_parquet(VAL_PARQUET)

    # print(df.head(1).to_dicts())

    svo_str = "b,bcd,d->c"
    s = torch.randn(3)
    v = torch.randn(3, 3, 3)
    o = torch.randn(3)
    # print(torch.einsum(svo_str, s, v, o))

    res = contract_einsum_non_linearly(svo_str, [s, v, o], torch.nn.Sigmoid())
    print(res)
