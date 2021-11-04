from distutils.core import setup

setup(
    name='anomalydetector',
    version='https://github.com/jspnet/AnomalyDetector',
    packages=['anomalydetector'],
    url='',
    install_requires=[
        'tensorflow',
        'numpy',
        'librosa',
        'tqdm',
        'pandas',
        'matplotlib',
        'soundfile'
    ],
    license='GPLv3',
    author='JSP co.',
    author_email='masahiko.hashimoto@jspnet.co.jp',
    description=''
)
