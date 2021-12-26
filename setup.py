from setuptools import setup
from pip.req import parse_requirements

install_reqs = parse_requirements("requirements.txt")

# reqs is a list of requirement
reqs = [str(ir.req) for ir in install_reqs]


setup(name='VolterraConvolutions',
      version='0.1',
      description='Data, models and train scripts for Volterra Convolutions',
      url='https://github.com/AGiannoutsos/Volterra-Convolutions',
      author='Andreas Giannoutsos',
      license='MIT',
      packages=['VolterraConvolutions'],
      install_requires=reqs,
      zip_safe=False)
