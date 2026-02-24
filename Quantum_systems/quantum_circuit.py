from qiskit import QuantumCircuit, ClassicalRegister

import matplotlib.pyplot as plt
import numpy as np
from qiskit.visualization import plot_histogram,  plot_state_city

from qiskit_aer.primitives import Estimator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeManilaV2 
from qiskit.circuit.library import RZGate
from qiskit.circuit import Parameter

from qiskit.quantum_info import Statevector, Operator, Pauli



def exercise_2():

  qc = QuantumCircuit(2,2)
  
  #Add Hadamard on qubit0, CNOT 0->1 and redo Hadamard
  qc.h(0)
  qc.cx(0, 1)

  qc.h(0)

  #qubit i -> classical bit i
  qc.measure(0, 0)  
  qc.measure(1, 1)  

  qc.draw('mpl')

  plt.show()
  plt.clf()



def exercise_3():

  n = 4

  #n qubit, n classical bits
  qc = QuantumCircuit(n, n)  

  qc.h(0)     

  #CNOT with control q0 and target all the rest       
  for i in range(1, n):
     qc.cx(0, i)     

  all_qubits = range(n)
  qc.measure(all_qubits, all_qubits)  

  qc.draw('mpl')
  plt.show()

  simulator = AerSimulator()

  job1 = simulator.run(qc)  # qc is your QuantumCircuit
  result = job1.result()

  counts = result.get_counts(qc)

  # Plot
  plot_histogram(counts)
  plt.show()
  
  plt.clf()

  fake_noise =  FakeManilaV2()
  noise_model = NoiseModel.from_backend(fake_noise)

  job = simulator.run(qc, noise_model=noise_model, shots=1024)
  result_noisy = job.result()
  counts_noisy = result_noisy.get_counts()

  # Plot histogram with built in function of qiskit
  plot_histogram([counts, counts_noisy], color=['blue', 'red'], legend=['No noise', 'Noise'])

  plt.title('Comparison of simulation with and without fake noise.')
  plt.show()
  plt.clf()


def exercise_4():

  from qiskit.circuit.library import QFT


  N = 3  # number of qubits

  qft3 = QFT(num_qubits=N, do_swaps=True).decompose() 

  for x in range(2**N):
  

    sv_in = Statevector.from_label(format(x, f"0{N}b"))

    #application of QFT
    sv_out = sv_in.evolve(qft3)
   
    print(f"|{x}>  →  QFT(|{x}>) =\n{sv_out}\n")




def exercise_5():
   
   from qiskit.circuit.library import RZGate
   from qiskit.circuit import Parameter

   qc = QuantumCircuit(3)
   
   # X1X2Z3

   # apply hadamard gates and then CNOT gates ón the qubits
   qc.h(0)  
   qc.h(1)  

  
   qc.cx(0, 2)  
   qc.cx(1, 2) 

    #  Apply Rz(theta) on qubit 3 ---
   theta = Parameter('theta')
   qc.rz(theta, 2)

    # Undo CNOT and then hadamard
   qc.cx(1, 2)
   qc.cx(0, 2)

   qc.h(1)
   qc.h(0)

   #qc.draw('mpl')
   #plt.show()



   # Z1Y2X3
   
   qc1 = QuantumCircuit(3)
   
   #Hadamard on qubits 2,3
   qc1.h(1)  
   qc1.h(2)   

   #Hermitian S on qubit 2
   qc1.sdg(1)  
   
   #CNOT on qubit 3 --> 2 and 3 --> 1
   qc1.cx(0, 2)
   qc1.cx(1, 2)

   #  Apply Rz
   qc1.rz(theta, 2)

   # Undo CNOTs and rest of gates
   qc1.cx(1, 2)
   qc1.cx(0, 2)

   qc1.s(1) 
   qc1.h(2)
   qc1.h(1)

   #qc1.draw('mpl')
   #plt.show()
   #plt.clf()


   # Y1X2X3
   qc2 = QuantumCircuit(3)
    
   qc2.h(0)

   qc2.h(1)

   qc2.h(2)
   qc2.sdg(0)

   qc2.cx(0, 1)
   qc2.cx(1, 2)

   qc2.rz(theta, 2)

   qc2.cx(1, 2)
   qc2.cx(0, 1)

   qc2.s(0)
   qc2.h(0)

   qc2.h(1)
   qc2.h(2)
   
   #qc2.draw('mpl')
   #plt.show()
   #plt.clf()

  # Z1X2Y3
  
   qc3 = QuantumCircuit(3)
  
   qc3.h(1)
   qc3.h(2)
   qc3.sdg(2)

   qc3.cx(0, 1)
   qc3.cx(1, 2)

   qc3.rz(theta, 2)

   qc3.cx(1, 2)
   qc3.cx(0, 1)

   qc3.h(1)

   qc3.h(2)
   qc3.s(2)
  
   qc3.draw('mpl')
   plt.show()
    


#exercise 7
def exercise_7():
    

    s = "00000001"

    n = len(s)

    #Define n+1 qubits and n classical qubits for measurement
    qc = QuantumCircuit(n+1, n) 

    #X gate to last qubit
    qc.x(n)

    #Hadamard gate to all qubits
    qc.h(range(n+1))

    # Oracle
    for i, bit in enumerate(s):
            qc.cx(i, n)

    # Undo hadamard
    qc.h(range(n))

    # Measurement
    qc.measure(range(n), range(n))
    

    qc.draw('mpl')
    plt.show()
    plt.clf()



if __name__ == '__main__':

    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    exercise_7()
