import mlflow.pyfunc
import torch

# makes sure infer_code_paths works
from qnlp.discoclip2.models.cp_node import CPQuadRankLayer  # noqa
from qnlp.discoclip2.models.einsum_model import EinsumModel
from qnlp.discoclip2.models.image_model import TTNImageModel


class VLM_Wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def load_context(self, context):
        """
        Runs automatically when the model is loaded from MLflow.
        Recovers the architecture and weights from saved artifacts.
        """

        checkpoint = torch.load(context.artifacts["best_model"], map_location="cpu")

        # 2. Reconstruct architectures
        self.text_model = EinsumModel()
        self.image_model = TTNImageModel(self.embedding_dim)

        # 3. Load the specific state dicts from the checkpoint object
        self.text_model.load_state_dict(checkpoint["text_model_state_dict"])
        self.image_model.load_state_dict(checkpoint["image_model_state_dict"])

        self.text_model.eval()
        self.image_model.eval()

    def predict(self, context, model_input):
        """
        Defines the inference logic.
        Expects a dict or list with 'images' and 'text'.
        """
        images = model_input.get("images")
        texts = model_input.get("text")

        with torch.no_grad():
            image_embeddings = self.image_model(images)
            text_embeddings = self.text_model(texts)

        return {"image_embeddings": image_embeddings, "text_embeddings": text_embeddings}
