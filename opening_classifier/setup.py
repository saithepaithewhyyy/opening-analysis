import sys
import subprocess
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

if sys.platform == "win32":
    compile_args = ["/O2", "/std:c++20"]
    link_args = []
    extra_includes = []
elif sys.platform == "darwin":
    try:
        libomp_prefix = subprocess.check_output(
            ["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        libomp_prefix = "/opt/homebrew/opt/libomp"  # fallback for Apple Silicon
    compile_args = ["-O2", "-std=c++20", "-Xpreprocessor", "-fopenmp"]
    link_args = [f"-L{libomp_prefix}/lib", "-lomp"]
    extra_includes = [f"{libomp_prefix}/include"]
else:
    compile_args = ["-O2", "-std=c++20", "-fopenmp"]
    link_args = ["-fopenmp"]
    extra_includes = []

ext = Pybind11Extension(
    "chess_classifier",
    sources=[
        "src/bindings.cpp",
        "src/classifier.cpp",
        "src/movegen.cpp",
        "src/board.cpp",
    ],
    include_dirs=["src", pybind11.get_include()] + extra_includes,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    language="c++",
)

setup(
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)