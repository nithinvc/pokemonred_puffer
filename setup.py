"""Builds the FastBoy emulator core (C port of PyBoy's DMG core) as a native extension.

`uv pip install -e .` (or `python setup.py build_ext --inplace`) compiles it.
Set FASTBOY_CFLAGS to override optimization flags (e.g. PGO).
"""

import os

from Cython.Build import cythonize
from setuptools import Extension, setup

SRC = "pokemonred_puffer/fastboy"
cflags = os.environ.get("FASTBOY_CFLAGS", "-O3").split() + ["-std=c11", "-Wno-unused-function"]

ext = Extension(
    "pokemonred_puffer.fastboy._core",
    sources=[f"{SRC}/_core.pyx", f"{SRC}/gbcore.c"],
    include_dirs=[SRC],
    depends=[f"{SRC}/gbcore.h", f"{SRC}/opcodes_gen.h"],
    extra_compile_args=cflags,
    extra_link_args=os.environ.get("FASTBOY_LDFLAGS", "").split(),
)

setup(ext_modules=cythonize([ext], language_level=3))
