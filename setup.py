import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    REQUIRED_PCKGS = fh.read().replace("\r\n", "\n").split("\n")

setuptools.setup(
    name="xswem",
    version="1.0.0",
    author="Kieran Litschel",
    author_email="kieran.litschel@outlook.com",
    description="A simple and explainable deep learning model for NLP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KieranLitschel/XSWEM",
    download_url='https://github.com/KieranLitschel/XSWEM/tags',
    license='MIT',
    packages=["xswem"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
    install_requires=REQUIRED_PCKGS,
    keywords='nlp fast machine learning deep simple tensorflow model word embeddings keras glove explainable swem '
             'global local explanations'
)
