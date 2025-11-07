import numpy as np
import torch
import torch.optim as optim
from torch.nn import CosineEmbeddingLoss
import matplotlib.pyplot as plt

from lambeq.backend.tensor import Dim
from lambeq import BobcatParser, SpiderAnsatz, PytorchModel, AtomicType


def prepare_all_data():
    """Prepare all data and split into train/eval sets."""
    all_sentences1 = [
        'The cat sleeps on the mat.',
        'A dog runs in the park.',
        'This is a happy kitty.',
        'That hound is fast.',
        'The feline rests on the floor.',
        'A puppy plays in the garden.',
        'This is a content cat.',
        'That canine is speedy.',
        'Birds fly in the sky.',
        'The sun shines brightly.'
    ]
    all_sentences2 = [
        'A kitty naps on the rug.',      # Similar to sentence 1
        'A car drives on the road.',     # Dissimilar to sentence 2
        'The cat is joyful.',            # Similar to sentence 3
        'The dog is quick.',             # Similar to sentence 4
        'A cat sleeps on the ground.',   # Similar to sentence 5
        'A truck moves on the highway.', # Dissimilar to sentence 6  
        'The kitty is pleased.',         # Similar to sentence 7
        'The hound is rapid.',           # Similar to sentence 8
        'Planes soar in the air.',       # Similar to sentence 9
        'The moon glows dimly.'          # Dissimilar to sentence 10
    ]
    all_labels_float = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    all_labels = (all_labels_float * 2) - 1  # Convert to [-1, 1] for CosineEmbeddingLoss
    
    train_sentences1 = all_sentences1[:5]
    train_sentences2 = all_sentences2[:5]
    train_labels = all_labels[:5]
    
    eval_sentences1 = all_sentences1[5:]
    eval_sentences2 = all_sentences2[5:]
    eval_labels = all_labels[5:]
    
    return (train_sentences1, train_sentences2, train_labels, 
            eval_sentences1, eval_sentences2, eval_labels)


def process_sentences(sentences):
    """Process a list of sentences into diagrams."""
    reader = BobcatParser(verbose='suppress')
    diagrams = reader.sentences2diagrams(sentences)
    return diagrams


def create_circuits(sentences1, sentences2, embedding_dim=4):
    """Create quantum circuits from sentences using lambeq pipeline."""
    diagrams1 = process_sentences(sentences1)
    diagrams2 = process_sentences(sentences2)

    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    ansatz = SpiderAnsatz(
        ob_map={N: Dim(2), S: Dim(embedding_dim)},  # Map types to feature dimensions
        max_order=2,
    )

    circuits1 = [ansatz(d) for d in diagrams1]
    circuits2 = [ansatz(d) for d in diagrams2]
    
    return circuits1, circuits2


def create_model_from_all_circuits(train_sentences1, train_sentences2, eval_sentences1, eval_sentences2, embedding_dim=4):
    """Create the lambeq pipeline and model using ALL circuits (train + eval)."""
    # Create circuits for both training and evaluation
    train_circuits1, train_circuits2 = create_circuits(train_sentences1, train_sentences2, embedding_dim)
    eval_circuits1, eval_circuits2 = create_circuits(eval_sentences1, eval_sentences2, embedding_dim)
    
    # Combine all circuits for model creation
    all_circuits = train_circuits1 + train_circuits2 + eval_circuits1 + eval_circuits2
    
    # Create and initialize model
    model = PytorchModel.from_diagrams(all_circuits)
    model.initialise_weights()
    
    return model, train_circuits1, train_circuits2, eval_circuits1, eval_circuits2


def train_model(model, circuits1, circuits2, labels, epochs=100):
    """Train the model."""
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = CosineEmbeddingLoss()
    
    losses = []
    print("\n--- Starting Unsupervised *Classical* Training ---")

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        embeddings1 = model(circuits1)
        embeddings2 = model(circuits2)
        
        loss = loss_fn(embeddings1, embeddings2, labels)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3}, Loss: {loss.item():.4f}')

    print("--- Training Finished ---")
    return losses


def calculate_cosine_similarity(a, b):
    """Calculate cosine similarity between two tensors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return (a_flat @ b_flat) / (torch.norm(a_flat) * torch.norm(b_flat))


def evaluate_model(model, circuits1, circuits2, labels):
    """Evaluate the trained model."""
    model.eval()
    with torch.no_grad():
        e1_final = model(circuits1)
        e2_final = model(circuits2)

    similarities = []
    for i in range(len(e1_final)):
        sim = calculate_cosine_similarity(e1_final[i], e2_final[i])
        similarities.append(sim.item())
    
    print("\n--- Evaluation (Classical Model) ---")
    print(f"Embeddings shape: {e1_final.shape}")
    print(f"\nExample Embedding (Sentence 1):\n {e1_final[0].numpy()}")
    
    for i, (sim, label) in enumerate(zip(similarities, labels.numpy())):
        target = "1.0 (similar)" if label == 1 else "-1.0 (dissimilar)"
        print(f"Similarity (Pair {i}): {sim:.4f} (Target: {target})")

    # Calculate accuracy
    correct = 0
    for sim, label in zip(similarities, labels.numpy()):
        if (sim > 0 and label == 1) or (sim <= 0 and label == -1):
            correct += 1
    accuracy = correct / len(similarities)
    print(f"\nEvaluation Accuracy: {accuracy:.2%} ({correct}/{len(similarities)})")
    
    return similarities


def plot_results(train_losses, eval_similarities, eval_labels):
    """Plot training loss and evaluation similarities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # Plot evaluation similarities
    x_pos = np.arange(len(eval_similarities))
    colors = ['green' if label == 1 else 'red' for label in eval_labels.numpy()]
    ax2.bar(x_pos, eval_similarities, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_title('Evaluation Cosine Similarities')
    ax2.set_xlabel('Sentence Pair Index')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Eval {i}' for i in x_pos])
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Prepare all data and split into train/eval
    (train_sentences1, train_sentences2, train_labels,
     eval_sentences1, eval_sentences2, eval_labels) = prepare_all_data()
    
    print(f"Training set: {len(train_sentences1)} pairs")
    print(f"Evaluation set: {len(eval_sentences1)} pairs")
    print(f"Training labels: {train_labels.numpy()}")
    
    # Create model using ALL circuits (both train and eval)
    model, train_circuits1, train_circuits2, eval_circuits1, eval_circuits2 = create_model_from_all_circuits(
        train_sentences1, train_sentences2, eval_sentences1, eval_sentences2
    )
    
    # Train the model on training circuits
    losses = train_model(model, train_circuits1, train_circuits2, train_labels, epochs=100)
    
    # Evaluate the model on evaluation circuits
    similarities = evaluate_model(model, eval_circuits1, eval_circuits2, eval_labels)
    
    # Plot results
    plot_results(losses, similarities, eval_labels)