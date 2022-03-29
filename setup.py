import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bert-search-relevance",
    version="0.0.1",
    author="John Dukatz",
    author_email="johndukatz@gmail.com",
    description="Final project for Stanford XCS224U",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jdukatz/bert-search-relevance",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)
