#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import setuptools
import numpy as np

from distutils.core import setup, Extension
from Cython.Distutils import build_ext

pkg_name = 'blue_zero'
author = "Sean P. Cornelius"
author_email = "spcornelius@gmail.com"

install_requires = ['argparse', 'contextlib2',
                    'numpy', 'pygame', 'torch', 'tqdm', 'gym']

args = sys.argv[1:]

# We want to always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext") + 1, "--inplace")

clusters_ext = Extension("blue_zero.clusters.clusters",
                         ["blue_zero/clusters/clusters.pyx"],
                         include_dirs=[np.get_include()],
                         extra_compile_args=['-O3']
                         )

if __name__ == '__main__':
    setup(
        name=pkg_name.lower(),
        description="Blue Zero",
        author=author,
        author_email=author_email,
        packages=setuptools.find_packages(),
        python_requires='>=3.8',
        install_requires=install_requires,
        cmdclass=dict(build_ext=build_ext),
        ext_modules=[clusters_ext]
    )
