try:

  from qiskit import QuantumCircuit
  import matplotlib.pyplot as plt
  from qiskit.circuit.library import RZGate
  from qiskit.circuit import Parameter

except Exception:
   print('Some important dependecy is missing, please intall all libraries as described in requirements.txt')

# Activate environment e.g my_quantum_env
# Simply run:
# python Konstantinos_Damanakis_Exercise_5.py

def exercise_5():
   
   # Only circuit sketch, no measurement
   qc = QuantumCircuit(3)
   
   # X1X2Z3

   # apply hadamard gates and then CNOT gates ón the q0, q1 with target qubit q2
   qc.h(0)  
   qc.h(1)  

   qc.cx(0, 2)  
   qc.cx(1, 2) 

   #  Apply Rz(theta) on q2
   theta = Parameter('theta')
   qc.rz(theta, 2)

   # Undo CNOT and then hadamard
   qc.cx(1, 2)
   qc.cx(0, 2)

   qc.h(1)
   qc.h(0)

   qc.draw('mpl')
   plt.savefig('exercise5_1.png')
   plt.clf()


   # Z1Y2X3
   
   qc1 = QuantumCircuit(3)
   
   #Hadamard on q1, q2 
   qc1.h(1)  
   qc1.h(2)   

   #Hermitian S on q1 based on Z = HS^{dag}YSH
   qc1.sdg(1)  
   
   #CNOT on qubits q0, q1 with target qubit q2
   qc1.cx(0, 2)
   qc1.cx(1, 2)

   #  Apply Rz(theta)
   qc1.rz(theta, 2)

   # Undo CNOTs and rest of gates
   qc1.cx(1, 2)
   qc1.cx(0, 2)

   qc1.s(1) 
   qc1.h(2)
   qc1.h(1)

   qc1.draw('mpl')
   plt.savefig('exercise5_2.png')
   plt.clf()


   # Y1X2X3
   qc2 = QuantumCircuit(3)
    
   # Apply Hadamard on all qubits 
   
   # Change Y to Z
   qc2.h(0)
   qc2.sdg(0)

   qc2.h(1)
   qc2.h(2)

   qc2.cx(0, 2)
   qc2.cx(1, 2)

   qc2.rz(theta, 2)

   qc2.cx(1, 2)
   qc2.cx(0, 2)

   qc2.h(1)
   qc2.h(2)
   qc2.s(0)
   qc2.h(0)

   qc2.draw('mpl')
   plt.savefig('exercise5_3.png')
   plt.clf()


  # Z1X2Y3
  
   qc3 = QuantumCircuit(3)
  
   qc3.h(1)

   # Change Y to Z
   qc3.h(2)
   qc3.sdg(2)

   qc3.cx(0, 2)
   qc3.cx(1, 2)

   qc3.rz(theta, 2)

   qc3.cx(1, 2)
   qc3.cx(0, 2)

  
   qc3.s(2)
   qc3.h(2)
   qc3.h(1)
  
   qc3.draw('mpl')
   plt.savefig('exercise5_4.png')



if __name__ == '__main__':

    exercise_5()
