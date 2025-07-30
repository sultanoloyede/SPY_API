from setuptools import setup, find_packages

setup(
    name='DataFusionTrader',
    version='0.1.0',
    description='A ports-and-adapters based trading system combining various data sources to perform market orders',
    author='Josue Dazogbo, Sultan Oloyede',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
