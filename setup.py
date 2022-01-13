from setuptools import setup, find_packages

setup(
  name = 'rela-transformer',
  packages = find_packages(exclude=[]),
  version = '0.0.5',
  license='MIT',
  description = 'ReLA Transformer',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/rela-transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention-mechanism',
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
