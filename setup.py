import os
from setuptools import setup
from setuptools import find_packages

ld = {}
if os.path.exists("README.md"):
    ld['filename'] = "README.md"
    ld['content_type'] = "text/markdown"
elif os.path.exists("readme_src.org"):
    ld['filename'] = "readme_src.org"
    ld['content_type'] = "text/plain"

with open(file=ld['filename'], mode="r") as readme_f:
    ld['data'] = readme_f.read()

setup(
    # Metadata
    name="deep_gesture",
    author="Philipp Denzel",
    author_email="phdenzel@gmail.com",
    version="0.0.dev1",
    description=("An LSTM gesture recognition neural net which can easily be trained to categorize any number of gestures!"),
    long_description=ld['data'],
    long_description_content_type=ld['content_type'],
    license='GNU General Public License v3.0',
    url="https://github.com/phdenzel/deep_gesture",
    keywords="action recognition, neural network, machine learning, real-time",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Multimedia :: Graphics :: Capture :: Digital Camera',
        'Topic :: Multimedia :: Video :: Capture',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
    ],

    # Package
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'opencv-python',
                      'mediapipe',
                      'sklearn',
                      'tensorflow',
                      'tensorflow-gpu'],
    package_dir={"": "deep_gesture"},
    packages=find_packages(where='deep_gesture'),
    py_modules=['deep_gesture'],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'deep_gesture = deep_gesture.__main__:main',
        ],
    },

)
