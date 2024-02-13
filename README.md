# ElphonPy: A simple python interface expediting electron-phonon calculations using EPW.

## Tutorial & full documentation coming soon!

Currently ElphonPy does not have a distributable, but this repository can be directly cloned to your home directory and import into your python files or Jupyter notebook as follows:

```python
import sys

sys.path.append('PATH-TO-ELPHONPY')
```

Prerequisite packages:
pymatgen, numpy, matplotlib.pyplot

Programs that are used in some ElphonPy functions:

read_relax_output --> pwo2xsf.sh (found in ~/qe-installation/PW/tools, and should be added to your .bashrc PATH.)
