from setuptools import setup, find_packages

setup(
    name="item-recommender",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here
    ],
    entry_points={
        'console_scripts': [
            'item-recommender = item_recommender.recommender:main',
        ],
    },
)