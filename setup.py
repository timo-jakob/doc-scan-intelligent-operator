"""Setup configuration for doc-scan-intelligent-operator."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="doc-scan-intelligent-operator",
    version="0.1.0",
    description="Analyses documents and organises them intelligently using MLX and Hugging Face models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "mlx>=0.0.0",
        "mlx-lm>=0.0.0",
        "transformers>=4.30.0",
        "PyYAML>=6.0",
        "pypdf>=3.0.0",
        "python-docx>=0.8.11",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "ocr": [
            "pytesseract>=0.3.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "docscan=docscan.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
