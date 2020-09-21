from setuptools import setup

setup(name='sku',
      version='0.2',
      description='This project only contains one function to extract the features from a pipeline in sklearn',
      url='',
      author='A Maldonado',
      license='MIT',
      packages=['sku'],
      python_requires='>=3.7',
      install_requires=[
          'scikit-learn==0.23.1',
      ],
      zip_safe=False)
