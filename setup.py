import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autophrasex",
    version="0.1.0",
    description="Automated Phrase Mining from Massive Text Corpora in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luozhouyang/AutoPhraseX",
    author="ZhouYang Luo",
    author_email="zhouyang.luo@gmail.com",
    packages=setuptools.find_packages(),
    # include_package_data=True,
    package_data={

    },
    install_requires=[
        "jieba",
        "lac",
        "naive-stopwords",
        "scikit-learn",
    ],
    dependency_links=[

    ],
    extras_require={

    },
    license="Apache Software License",
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    )
)
