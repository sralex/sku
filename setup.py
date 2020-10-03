from setuptools import setup

setup(name='sku',
      version='0.2.4',
      description='This project only contains one function to extract the features from a pipeline in sklearn',
      url='',
      author='A Maldonado',
      license='MIT',
      packages=['sku'],
      python_requires='>=3.6.9',
      install_requires=[
          'scikit-learn>=0.23.1',
          'numpy>=1.19.1',
          'pandas>=1.1.0',
      ],
      zip_safe=False)
