from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "XAI-powered Network Intrusion Detection System"

# Read requirements file
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
            return requirements
    except FileNotFoundError:
        return [
            "pandas>=1.5.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "xgboost>=1.7.0",
            "shap>=0.42.0",
            "lime>=0.2.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pyyaml>=6.0",
            "click>=8.1.0"
        ]

setup(
    # Basic package information
    name="xai-intrusion-detection",
    version="0.1.0",
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="Explainable AI-powered Network Intrusion Detection System with real-time monitoring and adversarial robustness testing",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/xai-intrusion-detection",  # Replace with your repo
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Core dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies for different use cases
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
            "isort>=5.12.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "ipywidgets>=8.1.0",
            "notebook>=7.0.0",
        ],
        "monitoring": [
            "mlflow>=2.5.0",
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
        "deployment": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "streamlit>=1.25.0",
            "docker>=6.1.0",
        ],
        "security": [
            "scapy>=2.5.0",
            "cryptography>=41.0.0",
            "adversarial-robustness-toolbox>=1.15.0",
        ],
        "all": [
            # Include all optional dependencies
            "pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.0.0", "flake8>=6.0.0",
            "jupyter>=1.0.0", "ipykernel>=6.25.0", "ipywidgets>=8.1.0",
            "mlflow>=2.5.0", "wandb>=0.15.0", "fastapi>=0.100.0", "uvicorn>=0.23.0",
            "streamlit>=1.25.0", "scapy>=2.5.0", "cryptography>=41.0.0",
        ]
    },
    
    # Command line entry points
    entry_points={
        "console_scripts": [
            "xai-ids=main:main",
            "xai-ids-train=model_trainer:main",
            "xai-ids-detect=real_time_detector:main",
            "xai-ids-explain=explainer:main",
            "xai-ids-analyze=security_analyzer:main",
        ],
    },
    
    # Project classification
    classifiers=[
        # Development Status
        "Development Status :: 3 - Alpha",
        
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Information Technology",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: System :: Networking :: Monitoring",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # Natural Language
        "Natural Language :: English",
    ],
    
    # Keywords for easier discovery
    keywords=[
        "intrusion-detection", "cybersecurity", "machine-learning", 
        "explainable-ai", "xai", "network-security", "anomaly-detection",
        "deep-learning", "adversarial-robustness", "real-time-monitoring"
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/xai-intrusion-detection/issues",
        "Source": "https://github.com/yourusername/xai-intrusion-detection",
        "Documentation": "https://github.com/yourusername/xai-intrusion-detection/docs",
        "Funding": "https://github.com/sponsors/yourusername",
    },
    
    # Zip safety
    zip_safe=False,
    
    # Platforms
    platforms=["any"],
)
