# Backend for `EquiangularGrid`, provided by the FastSphericalHarmonics
# package (FastTransforms).
#
# FSH uses θ midpoints (N = lmax+1 points), grid layout (ntheta, nphi), and
# stores spin-s coefficients in a 2-D array indexed by spinsph_mode(s,l,m);
# this extension converts to/from the canonical linear coefficient layout
# of AbstractSphericalHarmonics.  FSH's (FastTransforms') spin-weighted
# harmonic basis differs from the canonical (Wikipedia) one by the per-mode
# sign (−1)^max(m,−s) (independent of l; determined empirically against the
# analytic sYlm and verified by the cross-backend tests); the conversion
# applies this sign in both directions.
#
# Integration weights are Fejér's first quadrature rule (exact for
# integrands polynomial in cos θ up to degree N−1).

module AbstractSphericalHarmonicsFastSphericalHarmonicsExt

using AbstractSphericalHarmonics
const ASH = AbstractSphericalHarmonics
import FastSphericalHarmonics
const FSH = FastSphericalHarmonics

_N(grid::EquiangularGrid) = grid.lmax + 1

ASH.ash_grid_size(grid::EquiangularGrid) = (_N(grid), 2 * _N(grid) - 1)
ASH.ash_ntheta(grid::EquiangularGrid) = _N(grid)
ASH.ash_nphi(grid::EquiangularGrid) = 2 * _N(grid) - 1

function ASH.ash_point_coord(grid::EquiangularGrid, ij::Union{CartesianIndex{2},NTuple{2,Int}})
    N = _N(grid)
    t, p = Tuple(ij)
    return (π / N * (t - 1 / 2), 2π / (2N - 1) * (p - 1))
end

function ASH.ash_point_delta(grid::EquiangularGrid, ij::Union{CartesianIndex{2},NTuple{2,Int}})
    N = _N(grid)
    t, p = Tuple(ij)
    θ = π / N * (t - 1 / 2)
    # Fejér's first rule for ∫₀^π g(θ) sin(θ) dθ at θ midpoints:
    # weight w_t = (2/N) (1 − 2 Σ_{k=1}^{⌊N/2⌋} cos(2kθ)/(4k²−1)) for the
    # measure d(cos θ); convert to the (dθ, dϕ) convention with the
    # integrand carrying an explicit sin(θ) factor.
    w = (2 / N) * (1 - 2 * sum(cos(2k * θ) / (4k^2 - 1) for k in 1:(N ÷ 2); init=0.0))
    dtheta = w / sin(θ)
    dphi = 2π / (2N - 1)
    return (dtheta, dphi)
end

ASH.ash_thetas(grid::EquiangularGrid) = FSH.sph_points(_N(grid))[1]
ASH.ash_phis(grid::EquiangularGrid) = FSH.sph_points(_N(grid))[2]

# Plan caches, keyed (s, N) as in FSH
const SPINSPH_CACHE = FSH.SpinSphPlanCache{Complex{Float64}}()

"Sign relating FSH's harmonic basis to the canonical (Wikipedia) one"
_basis_sign(s::Int, m::Int) = isodd(max(m, -s)) ? -1 : 1

function ASH.ash_transform!(
    grid::EquiangularGrid, flm::AbstractVector{<:Complex}, f::AbstractMatrix{<:Complex}, s::Integer
)
    C = FSH.spinsph_transform(Matrix{Complex{Float64}}(f), Int(s); cache=SPINSPH_CACHE)
    fill!(flm, 0)
    for l in abs(Int(s)):(grid.lmax), m in (-l):l
        flm[l^2 + l + m + 1] = _basis_sign(Int(s), m) * C[FSH.spinsph_mode(Int(s), l, m)]
    end
    return flm
end

function ASH.ash_evaluate!(
    grid::EquiangularGrid, f::AbstractMatrix{<:Complex}, flm::AbstractVector{<:Complex}, s::Integer
)
    N = _N(grid)
    C = zeros(Complex{Float64}, N, 2N - 1)
    for l in abs(Int(s)):(grid.lmax), m in (-l):l
        C[FSH.spinsph_mode(Int(s), l, m)] = _basis_sign(Int(s), m) * flm[l^2 + l + m + 1]
    end
    f .= FSH.spinsph_evaluate(C, Int(s); cache=SPINSPH_CACHE)
    return f
end

end
