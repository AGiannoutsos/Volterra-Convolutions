from setuptools import setup


reqs = [
      "numpy",
      "matplotlib",
      "torch",
      "torchvision",
      "wandb",
      "pytorch-lightning",
      "torchmetrics",
      "benedict",
]


setup(name='VolterraConvolutions',
      version='0.1',
      description='Data, models and train scripts for Volterra Convolutions',
      url='https://github.com/AGiannoutsos/Volterra-Convolutions',
      author='Andreas Giannoutsos',
      license='MIT',
      packages=['VolterraConvolutions'],
      install_requires=reqs,
      zip_safe=False)
