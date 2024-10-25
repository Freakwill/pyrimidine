from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="bobocep",
    version="1.7",
    author="Freakwill",
    author_email="Williamxxoo@gmail.com",
    description="An extensible framework of genetic algorithm by python. "
                "designed in OOP and 'Algebra-insprited' what I call it.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/Freakwill/bobocep",
    keywords=[
        "Genetic algorithm",
        "Evolutionary algorithms",
        "Stochastic Optimization",
        "Iterative algorithms",
        "Algebra-insprited Programming",
        "Meta Programming"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Artificial Life",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: Mature",
    ],
    install_requires=install_requires,
    packages=find_packages(include=["pyrimidine"]),
    test_suite="tests",
    zip_safe=False
)
