import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuralnetw", # Replace with your own username
    version="0.0.1",
    author="Victor Ivamoto",
    author_email="vivoguard-abc@yahoo.com.br",
    description="Some NN algorithms in NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vivamoto",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',          # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3',     # Specify which pyhton versions that you want to support',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
