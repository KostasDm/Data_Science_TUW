
try:
  from qiskit import QuantumCircuit
  import matplotlib.pyplot as plt
except Exception:
   print('Some important dependecy is missing, please intall all libraries as described in requirements.txt')

   
# Activate environment e.g my_quantum_env
# Simply run:
# python Konstantinos_Damanakis_Exercise_7.py

def exercise_7():
    
    # hidden string, arbitrary selection
    s = "00000001"

    n = len(s)

    #Define n+1 qubits and n classical qubits for measurement
    qc = QuantumCircuit(n+1, n) 

    #X gate to last qubit for setting it to state 1 if it is defined in state 0
    qc.x(n)

    #Hadamard gate to all qubits
    qc.h(range(n+1))

    # Oracle represented by CNOTs
    for i, bit in enumerate(s):
         qc.cx(i, n)

    # Undo hadamard
    qc.h(range(n))

    # Measurement
    qc.measure(range(n), range(n))
    

    qc.draw('mpl')
    plt.savefig('exercise7.png')
    plt.clf()



if __name__ == '__main__':

    exercise_7()