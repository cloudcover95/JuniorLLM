import os, shutil, sys
from setuptools import setup, Extension
from Cython.Build import cythonize

# Path definitions
core_dir = "junior_memsys_suite/core"
source = os.path.join(core_dir, "omni_math.py")
target = os.path.join(core_dir, "omni_math.pyx")

# Ensure source exists as .pyx for Cython
if os.path.exists(source):
    if os.path.exists(target): os.remove(target)
    shutil.move(source, target)
elif not os.path.exists(target):
    print(f"CRITICAL: No source found at {source} or {target}")
    sys.exit(1)

# Explicitly define the extension to match the package structure
ext = Extension(
    name="junior_memsys_suite.core.omni_math",
    sources=[target]
)

setup(
    ext_modules=cythonize([ext], compiler_directives={'language_level': "3"}),
    script_args=['build_ext', '--inplace']
)
