try:

  from qiskit import QuantumCircuit
  import matplotlib.pyplot as plt

except Exception:
   print('Some important dependecy is missing, please intall all libraries as described in requirements.txt')

   
# Activate environment e.g my_quantum_env
# Simply run:
# python Konstantinos_Damanakis_Exercise_2.py

def exercise_2():

  # Create two qubits and two classical bits for measurement

  qc = QuantumCircuit(2,2)
  
  #Add Hadamard on qubit0, CNOT 0->1 
  qc.h(0)
  qc.cx(0, 1)

  # Redo Hadamard
  qc.h(0)

  #qubit i -> classical bit i
  qc.measure(0, 0)  
  qc.measure(1, 1)  

  qc.draw('mpl')

  plt.savefig('exercise2.png')



if __name__ == '__main__':

    exercise_2()

