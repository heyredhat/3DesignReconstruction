# **V**ar**i**ance-**b**ound**e**d **s**tate-**s**paces: VIBES

If you're not familiar with `python`, I'd start by downloading and installing `anaconda` from https://www.anaconda.com/. Then open up the terminal and create a new `conda` environment: this will give you a playground to install packages in and make sure everything is nice and easy. To do this, use the command

`conda create --name spring2025`

or you can call it whatever you like. Then activate the environment with

`conda activate spring2025`

I like using `pip` to install software so install `pip` first:

`conda install pip`

Then install our libraries:

`pip install numpy scipy jax matplotlib jupyter ipympl`

We'll use `numpy` for linear algebra, `scipy` for optimization and more, `jax` for just-in-time compiling and taking derivatives of functions in order to optimize efficiently, `matplotlib` to visualize things, `jupyter` so we can use notebooks, and finally `ipympl` so we can have interactive matplotlib widgets in notebooks.

Finally, to open up a jupyter notebook, navigate to the directory which contains the notebook and run

`jupyter notebook`

We collect useful methods in `vibes.py`. Add `from vibes import *` to your python file, and make sure `vibes.py` is in the working directory.
