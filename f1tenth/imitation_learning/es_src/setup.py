from setuptools import setup

setup(
    name='es_src',
    version='1.0',
    packages=['es'],  # same as name
    scripts=[
        'scripts/experiments.py'
    ]
)
