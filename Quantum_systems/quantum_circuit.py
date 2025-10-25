from qiskit import QuantumCircuit

import matplotlib.pyplot as plt

# Step 1: Create a 3-qubit quantum circuit
qc = QuantumCircuit(3)

# Step 2: Apply Hadamard on the first qubit
qc.h(0)

# Step 3: Apply CNOTs to entangle all qubits
qc.cx(0, 1)
qc.cx(0, 2)

# Step 4: Draw the circuit
qc.draw('mpl')
plt.show()