from datetime import datetime
import torch

from torch.utils.data import DataLoader

from qnlp.discoclip2.dataset.aro_dataset import ProcessedARODataset, aro_tn_collate_fn

from qnlp.discoclip2.models.einsum_model import EinsumModel
from qnlp.discoclip2.models.image_model import TTNImageModel
from qnlp.discoclip2.models.loss import create_loss_functions
from qnlp.utils.logging import get_log_file_path, setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.discoclip2.trainers.unfrozen.train_aro_clean import ModelSettings, evaluate_models



EXPERIMENT_NAME = "test_vlm_on_aro_attr"
ts_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = setup_logger(log_name=EXPERIMENT_NAME, ts_string=ts_string)
log_file_path = get_log_file_path(logger)
hyperparams = ModelSettings()


DEVICE = get_device()
set_seed()

DATA_PATH = "data/aro/processed/visual_genome_attribution/test.json"
IMAGES_PATH = "data/aro/raw/images/"
MODEL_PATH = "runs/checkpoints/train_vlm_on_aro/2026-02-23_20-02-49/best_model.pt"


if __name__ == "__main__":
    best_checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    text_model = EinsumModel()
    text_model.load_state_dict(best_checkpoint["text_model_state_dict"])
    text_model = text_model.to(DEVICE)

    image_model = TTNImageModel(hyperparams.embedding_dim)
    image_model.load_state_dict(best_checkpoint["image_model_state_dict"])
    image_model = image_model.to(DEVICE)

    test_ds = ProcessedARODataset(
        data_path=DATA_PATH, image_dir_path=IMAGES_PATH, return_images=True, is_train=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        collate_fn=aro_tn_collate_fn,
    )

    contrastive_loss, hard_neg_loss = create_loss_functions(
        temperature=hyperparams.temperature,
        hard_neg_distance_function=hyperparams.hard_neg_distance_function,
        margin=hyperparams.hard_neg_margin,
        swap=hyperparams.hard_neg_swap,
    )

    with setup_mlflow_run(EXPERIMENT_NAME, hyperparams.model_dump(), 8080) as run:
        acc = test_acc = evaluate_models(
            text_model=text_model,
            image_model=image_model,
            dataloader=test_loader,
            contrastive_criterion=contrastive_loss,
            hard_neg_criterion=hard_neg_loss,
            hard_neg_loss_weight=hyperparams.hard_neg_loss_weight,
            epoch=hyperparams.epochs + 1,
            usage="test",
        )
        logger.info(f"Final acc on test set: {acc}")
