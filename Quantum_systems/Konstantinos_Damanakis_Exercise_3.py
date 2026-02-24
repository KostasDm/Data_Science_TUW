
try:
   from qiskit import QuantumCircuit
   import matplotlib.pyplot as plt
   from qiskit.visualization import plot_histogram

   from qiskit_aer.primitives import Estimator
   from qiskit_aer import AerSimulator
   from qiskit_aer.noise import NoiseModel
   from qiskit_ibm_runtime.fake_provider import FakeManilaV2 

except Exception:
   print('Some important dependecy is missing, please intall all libraries as described in requirements.txt')

# Activate environment e.g my_quantum_env
# Simply run:
# python Konstantinos_Damanakis_Exercise_3.py

def exercise_3():

  n = 4

  #n qubit, n classical bits
  qc = QuantumCircuit(n, n)  

  # Create superposition state
  qc.h(0)     

  #CNOT with control q0 and target all the rest       
  for i in range(1, n):
     qc.cx(0, i)     

  all_qubits = range(n)
  qc.measure(all_qubits, all_qubits)  

  # plot circuit
  qc.draw('mpl')
  plt.savefig('exercise3_circuit.png')
  plt.clf()

  #Create simulator object
  simulator = AerSimulator()

  #Run system without noise
  job1 = simulator.run(qc)  
  result = job1.result()

  counts = result.get_counts(qc)

  # Plot counts distribution without noise
  plot_histogram(counts)
  plt.savefig('exercise3_no_noise.png')
  
  plt.clf()


  # Create object of fake noise - FakeManila
  fake_noise =  FakeManilaV2()
  noise_model = NoiseModel.from_backend(fake_noise)

  # Simulate system with noise model ans same number of shots
  job = simulator.run(qc, noise_model=noise_model, shots=1024)

  result_noisy = job.result()
  counts_noisy = result_noisy.get_counts()

  # Plot histogram of counts with and without noise by using built-in function of qiskit
  plot_histogram([counts, counts_noisy], color=['blue', 'red'], legend=['No noise', 'Noise'])

  plt.title('Comparison of simulation with and without fake noise.')
  plt.savefig('exercise3_with_noise.png')
  plt.clf()



if __name__ == '__main__':

   exercise_3()