# QubitRBM
A Quantum circuit simulator based on Restricted Boltzmann Machines, focusing on the Quantum Approximate Optimization Algorithm (QAOA). This code is associated with the following paper: 

**Classical variational simulation of the Quantum Approximate Optimization Algorithm** ([arXiv:2009.01760](https://arxiv.org/abs/2009.01760)).

Examples can be seen in the [examples](./examples/) folder and all functions/classes have been documented within the source code. We describe some of the basic functionality here:

## Restricted Boltzmann Machines (RBMs)

We provide a `RBM` class with custom methods implementing ansatz/gradients evaluations,

```python
import numpy as np
import networkx as nx
from qubitrbm.rbm import RBM

logpsi = RBM(n_visible=12)

B = np.random.rand(100, 12) > 0.5 # 100 random bitstrings
log_vals = logpsi(B)
grad_log_vals = logpsi.grad_log(B)
```

different gate applications (by modifying variational parameters in-place),

```python
logpsi.X(n=0) # Apply the Pauli X gate to qubit 0.
logpsi.Y(n=0) # Apply the Pauli Y gate to qubit 0.
logpsi.Z(n=0) # Apply the Pauli Z gate to qubit 0.

logpsi.RZZ(0, 1, phi=0.1)
#Apply the two-qubit ZZ rotstion on qubits 0 and 1 with angle 0.1.

G = nx.random_regular_graph(3, 16, seed=123)
logpsi.UC(G, gamma=0.1)
# Applying the QAOA U_C gate (a series of ZZ rotations) with angle gamma=0.1 on a given networkx graph G.
```

sampling the ansatz using the single-spin flip Metropolis-Hastings algorithm,

```python
samples = logpsi.get_samples(
    n_steps=1000, n_chains=5, warmup=100, step=16)
```

perform stochastic optimizations to apply more complicated QAOA gates:

```python
from qubitrbm.optim import Optimizer

optim = Optimizer(
    logpsi, n_steps=1000, n_chains=4, warmup=1000, step=16)

for n in range(len(G)):
    params, history = optim.sr_rx(
        n=n, beta=0.1, resample_phi=3, verbose=True)
    optim.machine.params = params
```

and more. For a more examples, take a look at the examples folder or the documentation within the source code. For mathematical background, please refer to the original [paper](https://arxiv.org/abs/2009.01760).

## The QAOA class

We provide a simple `QAOA` class that wraps around some of the basic operations on smaller QAOA instances.

```python
from qubitrbm.qaoa import QAOA

G = nx.random_regular_graph(3, 16, seed=123)
qaoa = QAOA(G, p=1)
```

[Cirq](https://github.com/quantumlib/Cirq) simulators are used under the hood to provide some basic functionality such as sampling the circuit or calculating the output state vector:

```python
gamma, beta = np.random.rand(2)

psi = qaoa.simulate(gamma, beta).final_state_vector
samples = qaoa.sample(gamma, beta, n_samples=100)
```

Perhaps more importantly, the QAOA class makes calculating optimal angles easy (for moderate circuit sizes and/or depths):

```python
angles, costs = qaoa.optimize(init=[np.pi/8, np.pi/8], tol=1e-5)
```

For QAOA depths of p=1, the exact formula (derived in the paper) is used to evaluate costs and their gradients efficiently. As long as one keeps p=1, very high qubit counts are achievable (on the order of 1000).

Switching to p>1, direct simulation is used for gradient estimation which is substantially slower.