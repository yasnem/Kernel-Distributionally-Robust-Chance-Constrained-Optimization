from setuptools import setup

setup(
    name='kernel-dr-mpc',
    version='0.0.1',
    packages=['drccp', 'drcc_mpc', 'drccp_utils'],
    package_dir={'': 'src'},
    url='',
    license='',
    author='ynemmour',
    author_email='yasnemou@gmail.com',
    description='Kernel DR Chance Constraint Programming',
    install_requires=[
        'casadi',
        'cvxpy',
        'matplotlib',
        'numpy',
        'scipy',
        'gpytorch',
        'scikit-learn',
        'torch'
    ]
)
