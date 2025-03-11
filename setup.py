from setuptools import setup, find_packages

setup(
    name="fused",
    version="0.1.0",
    description="FUSED: Foundation-based Unified Sequential Embedding Design",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="FUSED Team",
    author_email="team@fused-project.org",
    url="https://github.com/fused-project/fused",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "torchdiffeq>=0.2.3",  # For Neural ODEs
        "einops>=0.4.1",       # For tensor operations
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
            "mypy>=0.9.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
        ],
        "tracking": [
            "mlflow>=2.0.0",
            "wandb>=0.12.0",
        ],
        "hyperopt": [
            "optuna>=3.0.0",
            "ray[tune]>=2.0.0",
        ],
        "interpretability": [
            "shap>=0.40.0",
        ],
        "serving": [
            "flask>=2.0.0",
            "grpcio>=1.40.0",
            "onnx>=1.10.0",
            "onnxruntime>=1.10.0",
        ],
        "documentation": [
            "nbformat>=5.0.0",
        ],
        "all": [
            "mlflow>=2.0.0",
            "wandb>=0.12.0",
            "optuna>=3.0.0",
            "ray[tune]>=2.0.0",
            "shap>=0.40.0",
            "flask>=2.0.0",
            "grpcio>=1.40.0",
            "onnx>=1.10.0",
            "onnxruntime>=1.10.0",
            "nbformat>=5.0.0",
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
        ],
    },
)
