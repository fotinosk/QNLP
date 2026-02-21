from setuptools import find_packages, setup

setup(
    name="qnlp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pillow",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "ruff",
        ],
    },
    python_requires=">=3.11",
)
