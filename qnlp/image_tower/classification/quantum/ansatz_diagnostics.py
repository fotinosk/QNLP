import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

# Define the number of qubits for a single QTTN quad-node
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)

# Define configurations with varying entangling gate counts (CNOTs)
# Each configuration defines which CNOT gates to apply
cnot_configs = {
    "0 CNOTs (Separable)": [],
    "1 CNOT": [(1, 0)],
    "2 CNOTs": [(1, 0), (2, 0)],
    "3 CNOTs (All children to parent)": [(1, 0), (2, 0), (3, 0)],
    "4 CNOTs (Ring entanglement)": [(1, 0), (2, 0), (3, 0), (3, 1)],
    "6 CNOTs (Highly entangled)": [(1, 0), (2, 0), (3, 0), (0, 1), (1, 2), (2, 3)],
}


@qml.qnode(dev)
def node_circuit(inputs, weights, cnot_pairs):
    """
    Simulates a single quad-tree node.
    - inputs: 4 rotation angles for AngleEmbedding
    - weights: 1-qubit rotation angles (RZ-RY-RZ) for all 4 wires
    - cnot_pairs: list of (control, target) tuples for CNOTs
    """
    # 1. State preparation (dense angle encoding approximation)
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")

    # 2. Local rotations (Equivalent to CP factors)
    for i in range(num_qubits):
        qml.RZ(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
        qml.RZ(weights[i, 2], wires=i)

    # 3. Entanglement (Equivalent to CP core interaction)
    for ctrl, tgt in cnot_pairs:
        qml.CNOT(wires=[ctrl, tgt])

    # We return the density matrix of the surviving "parent" qubit (wire 0)
    return qml.density_matrix(wires=[0])


def compute_fidelity(rho1, rho2):
    """
    Computes the quantum fidelity between two single-qubit density matrices.
    F(rho1, rho2) = (Tr(sqrt(sqrt(rho1) * rho2 * sqrt(rho1))))^2
    For 2x2 density matrices, there is a simplified analytical formula:
    F = Tr(rho1 * rho2) + 2 * sqrt(det(rho1) * det(rho2))
    """
    tr_prod = np.real(np.trace(np.dot(rho1, rho2)))
    det1 = np.linalg.det(rho1)
    det2 = np.linalg.det(rho2)
    # Ensure no negative value under sqrt due to numerical precision
    det_term = 2 * np.sqrt(max(0.0, np.real(det1 * det2)))
    return tr_prod + det_term


def calculate_kl_divergence(fidelities, num_bins=20):
    """
    Calculates KL divergence from the uniform distribution (Haar-random 1-qubit fidelity).
    For a single qubit, the Haar-random state fidelity distribution is flat (uniform on [0, 1]).
    """
    # Compute the probability density function (PDF)
    hist, bin_edges = np.histogram(fidelities, bins=num_bins, range=(0.0, 1.0), density=True)
    # Convert to probabilities
    probs = hist / np.sum(hist)
    # Avoid log(0) by adding a small epsilon
    eps = 1e-12
    probs = probs + eps
    probs = probs / np.sum(probs)

    # Uniform distribution probabilities
    uniform_prob = 1.0 / num_bins

    # KL(P || U) = sum( P(x) * log(P(x) / U(x)) )
    kl = np.sum(probs * np.log(probs / uniform_prob))
    return kl


def run_experiment(num_samples=100, num_fidelity_pairs=500):
    entropy_results = {}
    expressibility_results = {}

    np.random.seed(42)

    for name, cnots in cnot_configs.items():
        print(f"Running experiments for: {name}...")
        entropies = []
        density_matrices = []

        # 1. Collect density matrices for entropy and fidelity
        for _ in range(num_samples):
            # Generate random inputs (RGB patch data mapping)
            inputs = np.random.uniform(0, 2 * np.pi, size=num_qubits)
            # Generate random parameter weights
            weights = np.random.uniform(0, 2 * np.pi, size=(num_qubits, 3))

            # Execute circuit
            rho = node_circuit(inputs, weights, cnots)
            density_matrices.append(rho)

            # Compute Von Neumann entropy of rho
            # S(rho) = -sum(eigval * log(eigval))
            eigenvalues = np.linalg.eigvalsh(rho)
            # Filter out zero eigenvalues to avoid log(0)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            entropy = -np.sum(eigenvalues * np.log(eigenvalues))
            entropies.append(entropy)

        mean_entropy = np.mean(entropies)
        entropy_results[name] = mean_entropy

        # 2. Compute pairwise fidelities to evaluate expressibility
        fidelities = []
        for _ in range(num_fidelity_pairs):
            idx1, idx2 = np.random.choice(num_samples, size=2, replace=False)
            f = compute_fidelity(density_matrices[idx1], density_matrices[idx2])
            fidelities.append(f)

        kl_div = calculate_kl_divergence(fidelities)
        expressibility_results[name] = kl_div

        # Plot fidelity distributions
        plt.hist(fidelities, bins=20, range=(0, 1), alpha=0.5, density=True, label=f"{len(cnots)} CNOTs")

    # Finalize and save the fidelity distribution plot
    plt.axhline(y=1.0, color="r", linestyle="--", label="Haar (Ideal Flat)")
    plt.xlabel("Fidelity")
    plt.ylabel("Probability Density")
    plt.title("Fidelity Distribution vs. Haar Uniformity")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.savefig("qnlp/image_tower/classification/quantum/results/fidelity_distributions.png")
    plt.close()

    # Return results
    return entropy_results, expressibility_results


if __name__ == "__main__":
    import os

    os.makedirs("qnlp/image_tower/classification/quantum/results", exist_ok=True)

    print("====================================================")
    # Lower number of samples for speed
    entropies, expressibility = run_experiment(num_samples=150, num_fidelity_pairs=1000)
    print("====================================================\n")

    print("RESULTS:")
    print("-" * 75)
    print(f"{'Ansatz Configuration':<40} | {'Mean Entropy':<12} | {'KL Div (Expressibility)':<22}")
    print("-" * 75)
    for name in cnot_configs.keys():
        print(f"{name:<40} | {entropies[name]:.4f}       | {expressibility[name]:.4f}")
    print("-" * 75)
    print("\nNote: Lower KL divergence means the ansatz is MORE expressible (closer to Haar-random coverage).")
    print("Higher entropy means more information gets entangled from children into the parent qubit.")
    print(
        "Fidelity distribution plot saved to: qnlp/image_tower/classification/quantum/results/fidelity_distributions.png"
    )
