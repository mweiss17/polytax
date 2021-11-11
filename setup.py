import subprocess
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
        "torch==1.9.1",
        "numpy",
        "tensorboardX",
        "tensorflow==2.6.0",
        "keras==2.6.0",
        "tensorflow-estimator==2.6.0",
        "tfds-nightly",
        "mesh_tensorflow",
        "t5",
        "seqio",
        "clu",
        "tbp-nightly",
        "google-cloud",
        "wandb",
        "tensorboardX",
        "dill",
        "google-cloud-storage",
        "torch_lr_scheduler",
        "speedrun @ git+https://git@github.com/inferno-pytorch/speedrun@dev#egg=speedrun",
        "transformers @ git+https://github.com/Arka161/transformers@master#egg=transformers",
    ],
    extras_require={
        "xla": [
            "torch-xla @ https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.9-cp38-cp38-linux_x86_64.whl"
        ],
    },
)

# If XLA is installed, then we setup some TPU-specific environment stuff
try:
    import torch_xla

    print("XLA support enabled")
    subprocess.call(
        'export WANDB_API_KEY=$(curl "http://metadata.google.internal/computeMetadata/v1/project/attributes/wandb_api_key" -H "Metadata-Flavor: Google")'.split()
    )
    subprocess.call('echo export WANDB_API_KEY="${WANDB_API_KEY}" >> ~/.bashrc'.split())
    subprocess.call(
        'git config --global user.email "martin.clyde.weiss@gmail.com"'.split()
    )
    subprocess.call('git config --global user.name "Martin Weiss"'.split())
    subprocess.call("export PATH=$PATH:/home/$USER/.local/bin".split())
    subprocess.call("unset LD_PRELOAD".split())
    subprocess.call('export XRT_TPU_CONFIG="localservice;0;localhost:51011"'.split())
    subprocess.call(
        'echo export XRT_TPU_CONFIG="localservice\;0\;localhost:51011" >> ~/.bashrc'.split()
    )
    subprocess.call(
        'python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint="localhost:2379"  train.py experiments/$expname --inherit $templatename'.split()
    )
except ImportError:
    print("XLA support disabled")
