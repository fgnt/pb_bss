import setuptools
try:
    import numpy as np
    from Cython.Build import cythonize

    import scipy  # nessesary for cython files
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("""
This package has some Cython files that will be compilled,\n
when you install this package. The Cython files use numpy and scipy.\n
Please install them before you install this packges:\n
    'conda install numpy Cython scipy'
or
    'pip install numpy Cython scipy'
""") from e


setuptools.setup(
    name="pb_bss",

    author="Lukas Drude",
    author_email="mail@lukas-drude.de",

    description="EM algorithms for integrated spatial and spectral models.",
    long_description=open('README.md', encoding='utf-8').read(),

    packages=setuptools.find_packages(),

    install_requires=[
        'dataclasses',
        'matplotlib',
        'scikit-learn',
        'cached_property',
        'einops',
        'sympy',  # Bingham mixture model symbolic solution dependency
        # Metric dependencies
        'mir_eval',
        'pystoi',
        'pesq'
    ],

    extras_require={
        'all': [
            'soundFile',
            'nara_wpe',
            'lazy_dataset',
            'pytest',
            'nose',
            'parameterized',
            'pytest-rerunfailures',
            'paderbox @ git+https://github.com/fgnt/paderbox',
        ]
    },

    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],

    ext_modules=cythonize([
        'pb_bss/extraction/cythonized/get_gev_vector.pyx',
        'pb_bss/extraction/cythonized/c_eig.pyx',
    ]),
    include_dirs=[np.get_include()],
)
