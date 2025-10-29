import ctypes
import platform
from pathlib import Path


def get_library_path():
    system = platform.system()

    if system == "Darwin":
        lib_name = "libembedcache.dylib"
    elif system == "Linux":
        lib_name = "libembedcache.so"
    elif system == "Windows":
        lib_name = "embedcache.dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    lib_dir = Path(__file__).parent / "lib"
    lib_path = lib_dir / lib_name

    if not lib_path.exists():
        raise FileNotFoundError(
            f"Could not find shared library at {lib_path}. "
            "Please run 'python setup.py build' first."
        )

    return str(lib_path)


def load_library():
    lib_path = get_library_path()
    lib = ctypes.CDLL(lib_path)

    lib.OpenDB.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.OpenDB.restype = ctypes.c_int

    lib.CloseDB.argtypes = [ctypes.c_int]
    lib.CloseDB.restype = ctypes.c_int

    lib.Insert.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    lib.Insert.restype = ctypes.c_int

    lib.Get.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_int)
    ]
    lib.Get.restype = ctypes.c_int

    lib.FindSimilar.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.FindSimilar.restype = ctypes.c_int

    lib.GetStats.argtypes = [ctypes.c_int]
    lib.GetStats.restype = ctypes.c_char_p

    lib.FreeVector.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.FreeVector.restype = None

    lib.FreeString.argtypes = [ctypes.c_char_p]
    lib.FreeString.restype = None

    return lib
