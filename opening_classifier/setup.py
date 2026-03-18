from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext = Pybind11Extension(
    "chess_classifier",
    sources=[
        "src/bindings.cpp",
        "src/classifier.cpp",
        "src/movegen.cpp",
        "src/board.cpp",
    ],
    include_dirs=["src", pybind11.get_include()],
    extra_compile_args=["/O2", "/std:c++20"],
    language="c++",
)

setup(
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)