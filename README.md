# AbstractSphericalHarmonics.jl

Provide a uniform interface to other packages that implement
spin-weighted spherical harmonic transforms.

* [![Documenter](https://img.shields.io/badge/docs-dev-blue.svg)](https://eschnett.github.io/AbstractSphericalHarmonics.jl/dev)
* [![GitHub
  CI](https://github.com/eschnett/AbstractSphericalHarmonics.jl/workflows/CI/badge.svg)](https://github.com/eschnett/AbstractSphericalHarmonics.jl/actions)
* [![Codecov](https://codecov.io/gh/eschnett/AbstractSphericalHarmonics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/eschnett/AbstractSphericalHarmonics.jl)

## Example

Plot the `s=+1, l=1, m=+1` spin-weighted spherical harmonic function:
```Julia
using AbstractSphericalHarmonics

lmax = 40

phis = ash_phis(lmax)
thetas = ahs_thetas(lmax)

flm = zeros(Complex{Float64}, ash_nmodes(lmax));
flm[ash_mode_index(+1, 1, +1)] = 1;

f = ash_evaluate(flm, +1, lmax);

################################################################################

using GLMakie

fig = Figure(; resolution=(1000, 300));

Axis(fig[1, 1]; title="real(f)")
Axis(fig[1, 3]; title="imag(f)")
hm = heatmap!(fig[1, 1], phis, thetas, real.(ash_grid_as_phi_theta(f)); colormap=:magma)
Colorbar(fig[1, 2], hm)
hm = heatmap!(fig[1, 3], phis, thetas, imag.(ash_grid_as_phi_thetaf)); colormap=:magma)
Colorbar(fig[1, 4], hm)
rowsize!(fig.layout, 1, Aspect(1, 1 / 2))

display(fig)
```

![s=_1, l=1, m=+1 mode](https://github.com/eschnett/AbstractSphericalHarmonics.jl/blob/main/figures/sYlm.png)

## Supported packages

- [FastSphericalHarmonics.jl](https://github.com/eschnett/FastSphericalHarmonics.jl)
- [ssht.jl](https://github.com/eschnett/ssht.jl)

## Literature

- R. Gomez, L. Lehner, P. Papadopoulos, J. Winicour, *The eth
  formalism in numerical relativity*, DOI 10.1088/0264-9381/14/4/013,
  Class. Quantum Grav. **14** 977,
  [arXiv:gr-qc/9702002](https://arxiv.org/abs/gr-qc/9702002).

- Florian Beyer, Boris Daszuta, Jörg Frauendiener, Ben Whale,
  "Numerical evolutions of fields on the 2-sphere using a spectral
  method based on spin-weighted spherical harmonics", [arXiv:1308.4729
  [physics.comp-ph]](https://arxiv.org/abs/1308.4729)

- Florian Beyer, Boris Daszuta, Jörg Frauendiener, "A spectral method
  for half-integer spin fields based on spin-weighted spherical
  harmonics", [arXiv:1502.07427
  [gr-qc]](https://arxiv.org/abs/1502.07427)
