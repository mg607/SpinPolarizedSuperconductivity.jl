# SpinPolarizedSuperconductivity.jl: Chiral superconductivity in spin-polarized electron gas from screened Coulomb repulsion

This repository contains the code to compute the chiral superconducting state emerging from screened Coulomb repulsion in rhomohedral graphene, as detailed in our article [M Geier, M Davydova, L Fu, Nature Communications 17, 232 (2026)]. 

The mechanism for superconductivity arises from screening of the Coulomb interaction by the two-dimensional electron gas: Close to the band bottom where the density-of-states is large, the screening leads to Friedel oscillations in the electron density around each particle due to the wave-nature of electrons. These density oscillations lead to a strong, effective attraction when the Thomas-Fermi screening length is short compared to the Fermi wavelength. 

This mechanism is captured by the numerical calculations in Random Phase Approximation contained here.

The code allows to solve for different angular momentum channels, the including of Berry phase effects. 

By default, the system solves for a model dispersion for multilayer rhombohedral graphene and Keldysh interaction between the electrons. Keldysh interaction is Coulomb interaction that includes dielectric screening due to the difference in dielectric constant in the graphene sheet to the surrouding hBN dielectric. 

All system parameters, dispersion, interaction, and Berry phases, can be modified in the code.

The self-consistency are solved numerically by an iterative procedure as detailed in the Supplementary Material of our article [M Geier, M Davydova, L Fu, Nature Communications 17, 232 (2026)].

The iterative solution of the self-consistency equations may encounter instabilities that can be resolved with tuning of the numerical discretization of the momentum grid, large-momentum cutoff, critical temperature values, and zero-temperature gap magnitudes. 

Author: Max Geier, Massachusetts Institute of Technology (2025)

## Giving Credit

The calculations and numerical results are discussed in our Nature Communications article:

M Geier, M Davydova, L Fu, "Chiral and topological superconductivity in isospin polarized multilayer graphene" Nature Communications 17, 232 (2026)
```
@article{geier2025chiral,
	author = {Geier, Max and Davydova, Margarita and Fu, Liang},
	date = {2025/12/01},
	doi = {10.1038/s41467-025-66902-6},
	id = {Geier2025},
	isbn = {2041-1723},
	journal = {Nature Communications},
	number = {1},
	pages = {232},
	title = {Chiral and topological superconductivity in isospin polarized multilayer graphene},
	url = {https://doi.org/10.1038/s41467-025-66902-6},
	volume = {17},
	year = {2025},
	bdsk-url-1 = {https://doi.org/10.1038/s41467-025-66902-6}}
```

Please cite our repository as:

```
@software{spinpolarizedsuperconductivity_github,
  author = {Max Geier},
  title = {{SpinPolarizedSuperconductivity.jl}},
  url = {https://github.com/mg607/SpinPolarizedSuperconductivity.jl},
  year = {2025},
}
```

