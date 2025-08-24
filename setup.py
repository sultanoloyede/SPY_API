from setuptools import setup, find_packages

setup(
    name='OTE',
    version='0.1.0',
    description='A modular and extensible trading strategy creation platform built with scalable design patterns, supporting live trading (Interactive Brokers) and backtesting (custom/Yahoo adapters) for stocks, forex, and more.',
    author='Josue Dazogbo, Sultan Oloyede, Jedidiah Ange-Emmanuel Kouakou, Racine Kane',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
