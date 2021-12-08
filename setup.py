from setuptools import find_packages, setup

setup(
    name='anomalydetector',
    version='0.1.1',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    url='https://github.com/jspnet/AnomalyDetector',
    install_requires=[
        'tensorflow',
        'numpy',
        'librosa',
        'tqdm',
        'pandas',
        'matplotlib',
        'soundfile',
        'scipy'
    ],
    entry_points={
        "console_scripts": [
            "adpreproc = anomalydetector.preprocessor:main",
            "admixer = anomalydetector.mixer:main",
            "adtrainer = anomalydetector.trainer:main",
            "adevaluator = anomalydetector.evaluator:main",
        ]
    },
    license='GPLv3',
    author='JSP co.',
    author_email='masahiko.hashimoto@jspnet.co.jp',
    description=''
)
