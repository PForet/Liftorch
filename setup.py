from setuptools import setup, find_packages

setup(name='Liftorch',
      version='0.1',
      description='Lifted neural networks for Pytorch',
      author='Pierre Foret',
      author_email='pierre_foret@berkeley.edu',
      url='https://github.com/PForet/Liftorch',
      packages = find_packages(exclude=['*.tests*']),
      license='MIT',
      install_requires=['numpy',
			'pytorch'],
      classifiers=[
          'Development Status :: 1 - Planning',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ])