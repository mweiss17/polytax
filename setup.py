import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="polytax",
    version="0.0.1",
    author="Martin Weiss",
    author_email="martin.clyde.weiss@gmail.com",
    description="A model-parallel neural network package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        'torch==1.9.1',
        'numpy',
        'tensorboardX',
        'tensorflow',
        'tfds-nightly',
        'mesh_tensorflow',
        't5',
        'seqio',
        'clu',
        'tbp-nightly',
        'google-cloud',
        'wandb',
        'tensorboardX',
        'dill',
        'torch_lr_scheduler',
        'speedrun @ git+https://git@github.com/inferno-pytorch/speedrun@dev#egg=speedrun',
        'transformers @ git+https://github.com/Arka161/transformers@master#egg=transformers',
        ],
    extra_requires={
        "xla": 'torch-xla @ https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.9-cp38-cp38-linux_x86_64.whl'
    }
)

