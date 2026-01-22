from setuptools import setup, find_packages

setup(
    name='MAD-defense',
    version='1.0.0',
    description='Manifold-Adaptive Defense against Backdoor Attacks',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'scipy>=1.11.0',
        'pandas>=2.0.0',
        'tqdm>=4.65.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
