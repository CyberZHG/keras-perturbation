from setuptools import setup

setup(
    name='keras-perturbation',
    version='0.2',
    packages=['keras_perturbation'],
    url='https://github.com/CyberZHG/keras-perturbation',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='A demonstration of perturbation of data',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'Keras',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
