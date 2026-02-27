from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tinyml-deployer",
    version="0.1.0",
    author="Alex Odhiambo Haya",
    author_email="alexhaya4@gmail.com",
    description="Deploy TensorFlow Lite models to ESP32 and STM32 microcontrollers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexhaya4/tinyml-deployer",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow",
        "numpy",
        "click",
        "rich",
    ],
    entry_points={
        "console_scripts": [
            "tinyml-deployer=tinyml_deployer.cli:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Embedded Systems",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
