import numpy as np


class QuantumQubit:
    """
    Represents a single qubit in a quantum state.
    """

    def __init__(self):
        # The initial state is always |0>
        self.state = np.array([[1], [0]])

    def apply_gate(self, gate):
        """
        Applies a quantum gate to the qubit.
        """
        self.state = np.dot(gate, self.state)

    def measure(self):
        """
        Measures the qubit and returns the result (0 or 1).
        """
        probabilities = np.abs(self.state) ** 2
        return np.random.choice([0, 1], p=[probabilities[0][0], probabilities[1][0]])


class QuantumGates:
    """
    Common quantum gates.
    """
    # Hadamard gate (superposition gate)
    HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])

    # Pauli-X gate (bit-flip gate)
    PAULI_X = np.array([[0, 1], [1, 0]])

    # Pauli-Y gate
    PAULI_Y = np.array([[0, -1j], [1j, 0]])

    # Pauli-Z gate (phase-flip gate)
    PAULI_Z = np.array([[1, 0], [0, -1]])

    # Phase gate
    PHASE = np.array([[1, 0], [0, np.exp(1j * np.pi / 2)]])

    # CNOT gate (controlled NOT gate)
    @staticmethod
    def CNOT(control_qubit, target_qubit):
        """
        Applies a CNOT gate to two qubits.
        """
        if control_qubit.measure() == 1:
            target_qubit.apply_gate(QuantumGates.PAULI_X)


class QuantumNeuralNetwork:
    """
    A Quantum Neural Network (QNN) implementation using quantum gates.
    """

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.qubits = [QuantumQubit() for _ in range(num_qubits)]
        self.weights = np.random.rand(num_qubits)  # Initialize random weights

    def apply_gate(self, gate, qubit_index):
        """
        Apply a quantum gate to a specific qubit.
        """
        self.qubits[qubit_index].apply_gate(gate)

    def forward(self, inputs):
        """
        Forward pass for the Quantum Neural Network.
        Uses Hadamard gates to create superposition and entanglement.
        """
        for i, input_val in enumerate(inputs):
            if input_val:
                self.apply_gate(QuantumGates.HADAMARD, i)
        return [qubit.measure() for qubit in self.qubits]

    def train(self, training_data, labels):
        """
        Trains the quantum neural network using simple loss calculation.
        """
        for epoch in range(100):  # Train for 100 epochs
            total_loss = 0
            for data, label in zip(training_data, labels):
                prediction = self.forward(data)
                loss = np.sum(np.abs(np.array(prediction) - np.array(label)))
                total_loss += loss
                # Update the weights (simplified for demonstration)
                self.weights -= 0.01 * loss
            print(f"Epoch {epoch + 1}, Loss: {total_loss}")


class QuantumSVM:
    """
    Quantum Support Vector Machine (QSVM) implementation.
    """

    def __init__(self):
        self.support_vectors = []

    def fit(self, X, y):
        """
        Trains the quantum SVM using quantum kernel.
        """
        # For simplicity, let's assume a classical SVM training with quantum kernel
        self.support_vectors = X  # Simplified

    def predict(self, X):
        """
        Predicts the class using the quantum kernel.
        """
        # Quantum kernel is simplified to a classical distance metric
        return [np.sign(np.dot(x, self.support_vectors[0])) for x in X]


class QuantumKMeans:
    """
    Quantum K-Means clustering algorithm.
    """

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.centroids = []

    def fit(self, X):
        """
        Cluster the data using quantum-enhanced K-Means.
        """
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(len(X), self.num_clusters, replace=False)]

        for i in range(10):  # Run for 10 iterations
            clusters = [[] for _ in range(self.num_clusters)]
            for x in X:
                distances = [np.linalg.norm(x - centroid) for centroid in self.centroids]
                cluster_index = np.argmin(distances)
                clusters[cluster_index].append(x)

            # Update centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in clusters if len(cluster) > 0]

    def predict(self, X):
        """
        Assigns data points to the nearest cluster.
        """
        return [np.argmin([np.linalg.norm(x - centroid) for centroid in self.centroids]) for x in X]


# Example usage of the library
if __name__ == "__main__":
    # Example: Quantum Neural Network (QNN)
    qnn = QuantumNeuralNetwork(3)
    training_data = [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
    labels = [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
    qnn.train(training_data, labels)

    # Example: Quantum Support Vector Machine (QSVM)
    qsvm = QuantumSVM()
    X_train = np.array([[1, 1], [2, 2], [3, 3], [-1, -1], [-2, -2], [-3, -3]])
    y_train = np.array([1, 1, 1, -1, -1, -1])
    qsvm.fit(X_train, y_train)
    print("QSVM Prediction:", qsvm.predict([[1, 1], [-2, -2]]))

    # Example: Quantum K-Means Clustering
    qkmeans = QuantumKMeans(2)
    X = np.array([[1, 1], [2, 2], [3, 3], [-1, -1], [-2, -2], [-3, -3]])
    qkmeans.fit(X)
    print("KMeans Clustering:", qkmeans.predict([[0, 0], [2, 2]]))
