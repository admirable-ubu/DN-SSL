Unveiling the Power of Disturbing Neighbors: A Comparative Study of Ensemble Methods for Semi-Supervised Learning
==============================
This is the official repository of the paper "Unveiling the Power of Disturbing Neighbors: A Comparative Study of Ensemble Methods for Semi-Supervised Learning".


<!--Badged-->
![License - MIT](https://img.shields.io/github/license/admirable-ubu/DN-SSL?style=flat-square)
![Python Version: 3.10.8](https://img.shields.io/badge/python-3.10.8-blue?style=flat-square)
![DOI](https://img.shields.io/badge/DOI-PENDING-red?style=flat-square)

Citation
--------
If you use the code, please cite the following paper:

```
PENDING PUBLICATION
```

Structure of the repository
---------------------------

This repository contains:
- The source code of the Disturbing Neighbors (DN) algorithm -> From the repository [jlgarridol/admirable-methods](https://github.com/jlgarridol/admirable-methods) (License: BSD-3-Clause)
- The source code of the experiments
- The datasets used in the experiments
- The results of the experiments
- The Jupyter Notebooks used to generate the tests and the plots of the paper

The repository is structured as follows:
- `datasets/`: Contains the datasets used in the experiments compressed in a tar.xz file. They are from UCI Machine Learning Repository and can be downloaded from [here](https://archive.ics.uci.edu/). The format of the datasets are `.csv`.
- `results/`: Contains the results of the experiments and the Jupyter Notebooks used to generate the tests and the plots of the paper. The resutls are two files: `dn_results.pkl` and `dn_f1_results.pkl`. The first file contains the accuracy of the experiments and the second file contains the F1-score of the experiments. Both are in [`dill`](https://pypi.org/project/dill/) format.
- `experiments/`: Contains the source code of the experiments. The experiments are launched over the `framework.py` file and configured in `config.json`. The file `experiments_done.db` is a SQLite database that contains the experiments already done. The `requeriments.txt` file contains the required packages to run the experiments. The Python version used is **3.10.8**.

Fundings
--------

This work was supported through the Junta de Castilla y León (JCyL) (regional government) under project BU055P20 (JCyL/FEDER, UE), the Spanish Ministry of Science and Innovation under project PID2020-119894GB-I00 co-financed through European Union FEDER funds, and project TED2021-129485B-C43 funded by MCIN/AEI/10.13039/501100011033 and the European Union NextGenerationEU/PRTR. J.L. Garrido-Labrador is supported through Consejería de Educación of the Junta de Castilla y León and the European Social Fund through a pre-doctoral grant EDU/875/2021 (Spain).


<!--Add the funding picture-->
![Funding](funding/funding_project.svg)

<img hspace="1.3%" align="center" width="22%" src="funding/FEDER.svg">
<img hspace="1.3%" align="center" width="22%" src="funding/JCYL.svg">
<img hspace="1.3%" align="center" width="22%" src="funding/JCYL_impulsa.svg"><img hspace="1.3%" align="center" width="22%" src="funding/fondo-social-europeo.svg">


