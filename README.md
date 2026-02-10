# Internship-SST
Data and code for 2026 internship to apply our methodology to SST.


## Installation


Prerequisites:
- Check *cdo* existence.
- Install all python packages in *Utilities/requirements.txt* or *Utilities/environment.yml*.
- Install [NSSEA](https://github.com/Occitane-Barbaux/NSSEA) and [SDFC](https://github.com/Occitane-Barbaux/SDFC-python) manually from github.
- Check package installation in environement by running *Test_Packages_installation.ipynb* once.
- Install [CMDStan](https://cmdstanpy.readthedocs.io/en/v1.3.0-post2/installation.html#cmdstan-installation).
- Modify path to CMDSTan location in *Utilities/Full_run_Model.py*, line 10.



## Content

### Folders:
- Data_Brutes: GCM and observations data for SST applications
- Example: Data for application to MAUGUIO using variable tasmax
- Utilities: Functions code files

### Jupyter files:
- *Test_Packages_installation.ipynb* : Try loading all packages used at least once.
- *Example_Application_tasmax.ipynb* : Run all caclulation for application to MAUGUIO using variable tasmax


## Other useful commands:

#### To add an env to jupyter:
In EnvName:
> conda install ipykernel 
> python -m ipykernel install --user --name EnvName

## Bibliography

* Barbaux, O., Naveau, P., Bertrand, N., & Ribes, A. (2025). Integrating non-stationarity and uncertainty in design life levels based on climatological time series. Weather and Climate Extremes, 100807. [doi](https://www.sciencedirect.com/science/article/pii/S2212094725000659)
* Guinaldo, T., Voldoire, A., Waldman, R., Saux Picart, S., & Roquet, H. (2023). Response of the sea surface temperature to heatwaves during the France 2022 meteorological summer. Ocean Science, 19(3), 629-647.Guinaldo, T., Cassou, C., Sallée, J. B., & Liné, A. (2025). Internal variability effect doped by climate change drove the 2023 marine heat extreme in the North Atlantic. Communications Earth & Environment, 6(1), 291, [doi](https://www.nature.com/articles/s43247-025-02197-1)
* Barbaux, O., Naveau,P, Ribes, A, Bertrand, N, 2025. Extreme Temperatures for the 21st century. [Thesis defended, manuscript available on demand](https://theses.fr/s350251)
* Robin, Y., Ribes, A., 2020. Nonstationary extreme value analysis for event attribution combining climate models and observations. Advances in Statistical Climatology, Meteorology and Oceanography 6, 205–221, [doi](https://doi.org/10.5194/ascmo-6-205-2020)


