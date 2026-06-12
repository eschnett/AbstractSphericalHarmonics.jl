# AbstractSphericalHarmonics.jl

A backend-agnostic interface for spin-weighted spherical harmonic
transforms on the sphere, plus a tensor calculus (`Tensor`/`SpinTensor`)
built on top of it.

* [![Documenter](https://img.shields.io/badge/docs-dev-blue.svg)](https://eschnett.github.io/AbstractSphericalHarmonics.jl/dev)
* [![GitHub
  CI](https://github.com/eschnett/AbstractSphericalHarmonics.jl/workflows/CI/badge.svg)](https://github.com/eschnett/AbstractSphericalHarmonics.jl/actions)
* [![Codecov](https://codecov.io/gh/eschnett/AbstractSphericalHarmonics.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/eschnett/AbstractSphericalHarmonics.jl)

## Design

This package follows the AbstractFFTs model: it owns the generic
interface functions, and transform providers implement them via package
extensions.  The choice of backend is a value — a *grid object* — rather
than a package import:

| Grid type            | Sampling                              | Backend package          |
|----------------------|---------------------------------------|--------------------------|
| `DriscollHealyGrid`  | Driscoll–Healy, `(2L−1) × 2L` (ϕ, θ)  | `SSHT`                   |
| `EquiangularGrid`    | θ midpoints, `L × (2L−1)` (θ, ϕ)      | `FastSphericalHarmonics` |

with `L = lmax + 1`.  Load a backend with `import` (not `using`; the
extension activates on loading, and `import` keeps your namespace clean):

```julia
using AbstractSphericalHarmonics
import SSHT                     # activates DriscollHealyGrid
import FastSphericalHarmonics   # activates EquiangularGrid

grid = DriscollHealyGrid(40)
```

Both backends can be active in the same session, and all interface
functions dispatch on the grid: `ash_transform`, `ash_evaluate`,
`ash_eth`, `ash_ethbar`, `ash_grid_size`, `ash_point_coord`,
`ash_point_delta`, `ash_nmodes`, `ash_mode_index`, `ash_mode_numbers`,
`ash_thetas`, `ash_phis`, `ash_lmax`.

Canonical conventions (independent of the backend): coefficients of a
spin-`s` field are a `Vector{Complex}` of length `(lmax+1)²` with linear
mode index `l²+l+m+1`; the spin-weighted harmonics and the `ð`/`ð̄`
operators follow the
[Wikipedia](https://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics)
normalization, `ð ₛYₗₘ = +√((l−s)(l+s+1)) ₛ₊₁Yₗₘ`.  Backend-specific
layouts and phase conventions are reconciled inside the extensions, and
cross-backend consistency is enforced by the test suite.

## Example

Evaluate the `s=+1, l=1, m=+1` spin-weighted spherical harmonic:

```julia
using AbstractSphericalHarmonics
import SSHT

grid = DriscollHealyGrid(40)

flm = zeros(Complex{Float64}, ash_nmodes(grid));
flm[ash_mode_index(grid, +1, 1, +1)] = 1;

f = ash_evaluate(grid, flm, +1);
```

## Tensor calculus

`Tensor{D}` stores rank-`D` tensor fields in components of the
orthonormal coordinate dyad of the unit sphere (pole-regular), and
`SpinTensor{D}` stores their spin-weighted spectral coefficients;
`tensor_gradient` is the covariant derivative with respect to the unit
round sphere.  These types carry their grid and work with any backend.

Note (changed in version 1.1.0): `conj(::SpinTensor)` returns the
spectral representation of the conjugated field, i.e.
`Tensor(conj(st)) ≈ conj(Tensor(st))` — not an elementwise `conj` of the
coefficient arrays.

## Adding a backend

Implement, in a package extension, methods for a new `SphereGrid`
subtype: `ash_grid_size`, `ash_point_coord`, `ash_point_delta`,
`ash_thetas`, `ash_phis`, `ash_transform!`, `ash_evaluate!` — converting
to the canonical coefficient layout and harmonic conventions if
necessary.  Mode indexing and the eth operators are provided centrally
and need no backend code.

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
