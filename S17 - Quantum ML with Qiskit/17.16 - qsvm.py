from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel

# IRIS Data Set
from sklearn import datasets
iris = datasets.load_iris()
print(iris.data)
print(iris.target)
print(iris.target_names)
dimension = len(iris.data[0])

seed = 98765 # any number
algorithm_globals.random_seed = seed
feature_map = ZZFeatureMap(feature_dimension=dimension, 
                           reps=2, entanglement="linear")
backend = QuantumInstance(
    BasicAer.get_backend("qasm_simulator"), shots=1024, 
    seed_simulator=seed, seed_transpiler=seed
)
kernel = QuantumKernel(feature_map=feature_map, 
                       quantum_instance=backend)
qsvc = QSVC(quantum_kernel=kernel)
qsvc.fit(iris.data,iris.target)

print(qsvc.predict([[5.0, 3.3, 1.5, 0.3], 
                    [6.0, 2.9, 5.2, 1.7]]))
