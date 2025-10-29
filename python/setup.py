from setuptools import setup, find_packages
import os
import subprocess

# Build the Go shared library
def build_go_library():
    """Build the Go shared library using CGO"""
    go_src = "../pkg/embedcache"
    lib_dir = "embedcache/lib"

    os.makedirs(lib_dir, exist_ok=True)

    # Determine platform
    import platform
    system = platform.system()

    if system == "Darwin":
        lib_name = "libembedcache.dylib"
    elif system == "Linux":
        lib_name = "libembedcache.so"
    elif system == "Windows":
        lib_name = "embedcache.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    # Build command
    cmd = [
        "go", "build",
        "-buildmode=c-shared",
        "-o", os.path.join(lib_dir, lib_name),
        go_src
    ]

    print(f"Building Go library: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))

    if result.returncode != 0:
        raise RuntimeError("Failed to build Go library")

# Build before installing
build_go_library()

with open("../README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="embedcache",
    version="0.1.0",
    author="Nathan Samson",
    description="Zero-config embedding cache for serverless & edge",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bostesa/VectorLite",
    packages=find_packages(),
    package_data={
        "embedcache": ["lib/*"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
)
