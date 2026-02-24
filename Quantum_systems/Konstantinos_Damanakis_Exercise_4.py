try:

   from qiskit import QuantumCircuit  
   from qiskit.circuit.library import QFT
   from qiskit.quantum_info import Statevector
   import matplotlib.pyplot as plt
   import numpy as np

except Exception:
   print('Some important dependecy is missing, please intall all libraries as described in requirements.txt')

# Activate environment e.g my_quantum_env
# Simply run:
# python Konstantinos_Damanakis_Exercise_4.py


def exercise_4():

  #Number of qubits 

  N = 3 

  # Create QFT object for N=3
  qft3 = QFT(num_qubits=N, do_swaps=True).decompose() 
  qft3.draw('mpl')

  plt.show()
  # sum from 0 to 2**N-1, thus range(2**N)
  for x in range(2**N):
  
    vector_in = Statevector.from_label(format(x, f"0{N}b"))

    #application of QFT - evolve system
    vector_out = vector_in.evolve(qft3)
   
    print(f"|{x}> --> QFT(|{x}>) =\n{vector_out}\n")


if __name__ == '__main__':

    exercise_4()
