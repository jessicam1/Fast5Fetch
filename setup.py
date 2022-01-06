import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
        name="Fast5Fetch",
        version="1.1",
        author="Jessica Martin",
        author_email="jessimartin1996@gmail.com",
        description="Generators and related functions for using raw signal data from ONT with neural nets and analysis",
        url="https://github.com/jessicam1/cnn-train",
        license="MIT",
        packages=["Fast5Fetch"],        
        install_requires=["tensorflow", "scipy", "ont_fast5_api", "pathlib", "numpy"],
)


