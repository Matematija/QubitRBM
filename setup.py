from setuptools import setup, find_namespace_packages

setup(
    name = 'qubitrbm',
    version = '0.1.0',   
    description = "RBM quantum circuit simulator",
    long_description="""
        An approximate Quantum circuit simulator based on Restricted Boltzmann Machines,
        focusing on the Quantum Approximate Optimization Algorithm (QAOA).
        """,
    url = 'https://github.com/Matematija/QubitRBM',
    author = 'Matija Medvidovic',
    author_email = 'matija.medvidovic@columbia.edu',
    license = 'Apache License Version 2.0',
    packages = find_namespace_packages(),
    install_requires = [
        'numpy>=1.17.3',
        'scipy>=1.3.1',
        'numba>=0.49.1',
        'cirq>=0.8.0',
        'networkx>=2.4',
        'sympy'
    ],
    classifiers = [
        'License :: OSI Approved :: Apache Software License',  
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)