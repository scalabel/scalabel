"""Package setup."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scalabel",  # Replace with your own username
    version="0.3.0",
    author="Fisher Yu",
    author_email="i@yf.io",
    description="Scalabel Python Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://scalabel.ai/",
    project_urls={
        "Documentation": "https://doc.scalabel.ai",
        "Source": "https://github.com/scalabel/scalabel",
        "Tracker": "https://github.com/scalabel/scalabel/issues",
    },
    package_data={"scalabel": ["py.typed", "label/configs.toml"]},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "boto3",
        "Flask-Cors",
        "Flask",
        "joblib",
        "matplotlib",
        "motmetrics",
        "nanoid",
        "numpy",
        "pandas",
        "Pillow",
        "plyfile",
        "psutil",
        "pycocotools",
        "pydantic",
        "pyparsing",
        "python-dateutil",
        "PyYAML",
        "requests",
        "scikit-learn",
        "scikit-image",
        "tabulate",
        "tqdm",
        "toml",
    ],
    include_package_data=True,
)
