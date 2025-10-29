# Build Instructions

## Requirements

- Go 1.21+
- Python 3.8+
- CGO enabled (usually default)

## Quick Build

```bash
# Clone
git clone https://github.com/Bostesa/VectorLite.git
cd vectorlite

# Build shared library (platform-specific)
# macOS:
go build -buildmode=c-shared -o python/embedcache/lib/libembedcache.dylib ./pkg/embedcache

# Linux:
go build -buildmode=c-shared -o python/embedcache/lib/libembedcache.so ./pkg/embedcache

# Windows:
go build -buildmode=c-shared -o python/embedcache/lib/embedcache.dll ./pkg/embedcache

# Install Python package
cd python
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

## Test

```bash
# Go tests
go test ./pkg/storage -v

# Python tests
cd python && source venv/bin/activate
pytest tests/ -v
```

## Run Examples

```bash
cd python && source venv/bin/activate
python ../examples/basic_usage.py
```

## Platform Notes

**macOS**
- Uses `.dylib` extension
- May need code signing: `codesign -s - python/embedcache/lib/libembedcache.dylib`

**Linux**
- Uses `.so` extension
- May need `LD_LIBRARY_PATH` set

**Windows**
- Uses `.dll` extension
- Requires MinGW or similar for CGO

## Build Script

Save as `build.sh`:

```bash
#!/bin/bash
set -e

# Detect platform
case "$(uname -s)" in
    Darwin*) LIB="libembedcache.dylib" ;;
    Linux*)  LIB="libembedcache.so" ;;
    *)       LIB="embedcache.dll" ;;
esac

# Build
echo "Building Go library..."
mkdir -p python/embedcache/lib
go build -buildmode=c-shared -o "python/embedcache/lib/$LIB" ./pkg/embedcache

# Install
echo "Installing Python package..."
cd python
python -m venv venv
source venv/bin/activate
pip install -e .

echo "Done! Activate: source python/venv/bin/activate"
```

Run: `chmod +x build.sh && ./build.sh`

## Troubleshooting

**CGO errors**: Make sure you have a C compiler (gcc/clang)
**Import errors**: Check the shared library exists in `python/embedcache/lib/`
**Permission errors**: Try `chmod +x` on the library file
