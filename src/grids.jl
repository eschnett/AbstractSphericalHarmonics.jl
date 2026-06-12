# The backend-agnostic interface of AbstractSphericalHarmonics.
#
# A *grid object* describes a collocation grid on the sphere together with
# the transform backend that operates on it.  All interface functions take
# the grid as their first argument; backends (package extensions) add
# methods for their grid types.  This follows the AbstractFFTs model: this
# package owns the generic functions, providers implement them.
#
# Canonical conventions (backends must adapt to these):
#
# - Spherical harmonic coefficients of a spin-s field are stored as a
#   `Vector{Complex}` of length `(lmax+1)^2` with the linear mode index
#   `l^2 + l + m + 1`.  Entries with `l < |s|` are unused (treated as zero).
# - Grid values are stored as a `Matrix{Complex}` of size
#   `ash_grid_size(grid)`; the layout is backend-specific, and consumers
#   iterate over `CartesianIndices` and query `ash_point_coord(grid, ij)`.
# - The eth operators follow the Wikipedia normalization,
#       ð ₛYₗₘ = +√((l−s)(l+s+1)) ₛ₊₁Yₗₘ ,
#       ð̄ ₛYₗₘ = −√((l+s)(l−s+1)) ₛ₋₁Yₗₘ ,
#   and are implemented here, in mode space, independent of the backend.

export SphereGrid
"""
    abstract type SphereGrid end

A collocation grid on the sphere, tied to a spherical-harmonic transform
backend.  Concrete subtypes: [`DriscollHealyGrid`](@ref) (backend: SSHT),
[`EquiangularGrid`](@ref) (backend: FastSphericalHarmonics).
"""
abstract type SphereGrid end

export DriscollHealyGrid
"""
    DriscollHealyGrid(lmax)

Driscoll–Healy sampled grid with `ntheta = 2(lmax+1)`, `nphi = 2(lmax+1)−1`,
grid layout `(nphi, ntheta)`.  Backend: the SSHT package; load it
(`import SSHT`) to activate the methods.
"""
struct DriscollHealyGrid <: SphereGrid
    lmax::Int
    function DriscollHealyGrid(lmax::Integer)
        lmax ≥ 0 || throw(DomainError(lmax, "Need lmax ≥ 0"))
        return new(Int(lmax))
    end
end

export EquiangularGrid
"""
    EquiangularGrid(lmax)

Equiangular grid with θ midpoints, `ntheta = lmax+1`, `nphi = 2(lmax+1)−1`,
grid layout `(ntheta, nphi)`.  Backend: the FastSphericalHarmonics package;
load it (`import FastSphericalHarmonics`) to activate the methods.
"""
struct EquiangularGrid <: SphereGrid
    lmax::Int
    function EquiangularGrid(lmax::Integer)
        lmax ≥ 0 || throw(DomainError(lmax, "Need lmax ≥ 0"))
        return new(Int(lmax))
    end
end

Base.:(==)(g1::SphereGrid, g2::SphereGrid) = typeof(g1) == typeof(g2) && g1.lmax == g2.lmax

export ash_lmax
"Band limit of the grid"
ash_lmax(grid::SphereGrid) = grid.lmax

################################################################################
# Backend interface: geometry and transforms.  Stubs only; package
# extensions add methods for their grid types.

_backend_error(grid) = error(
    "No transform backend loaded for $(typeof(grid)). " *
    "Load the backend package, e.g. `import SSHT` for DriscollHealyGrid " *
    "or `import FastSphericalHarmonics` for EquiangularGrid.",
)

export ash_grid_size, ash_ntheta, ash_nphi, ash_thetas, ash_phis, ash_point_coord, ash_point_delta

"Size of the grid-value arrays"
ash_grid_size(grid::SphereGrid) = _backend_error(grid)
"Coordinates (θ, ϕ) of the grid point at index `ij`"
ash_point_coord(grid::SphereGrid, ij::Union{CartesianIndex{2},NTuple{2,Int}}) = _backend_error(grid)
"Integration weights (dθ, dϕ) at `ij`: ∮ f ≈ Σ f sin(θ) dθ dϕ, exact for band-limited f"
ash_point_delta(grid::SphereGrid, ij::Union{CartesianIndex{2},NTuple{2,Int}}) = _backend_error(grid)
"All θ values of the grid"
ash_thetas(grid::SphereGrid) = _backend_error(grid)
"All ϕ values of the grid"
ash_phis(grid::SphereGrid) = _backend_error(grid)
ash_ntheta(grid::SphereGrid) = _backend_error(grid)
ash_nphi(grid::SphereGrid) = _backend_error(grid)

export ash_transform!, ash_transform, ash_evaluate!, ash_evaluate

"Transform grid values of a spin-`s` field to coefficients (canonical layout)"
ash_transform!(grid::SphereGrid, flm::AbstractVector{<:Complex}, f::AbstractMatrix{<:Complex}, s::Integer) = _backend_error(grid)
"Evaluate coefficients of a spin-`s` field on the grid"
ash_evaluate!(grid::SphereGrid, f::AbstractMatrix{<:Complex}, flm::AbstractVector{<:Complex}, s::Integer) = _backend_error(grid)

function ash_transform(grid::SphereGrid, f::AbstractMatrix{<:Complex}, s::Integer)
    return ash_transform!(grid, Vector{ComplexF64}(undef, ash_nmodes(grid)...), f, s)
end
function ash_evaluate(grid::SphereGrid, flm::AbstractVector{<:Complex}, s::Integer)
    return ash_evaluate!(grid, Matrix{ComplexF64}(undef, ash_grid_size(grid)...), flm, s)
end

################################################################################
# Canonical mode indexing: grid-independent, implemented here

export ash_nmodes, ash_mode_index, ash_mode_numbers

"Size of the coefficient array (a 1-tuple)"
ash_nmodes(grid::SphereGrid) = ((grid.lmax + 1)^2,)

"Index of the (l, m) mode of a spin-`s` field in the coefficient vector"
function ash_mode_index(grid::SphereGrid, s::Integer, l::Integer, m::Integer)
    abs(s) ≤ l ≤ grid.lmax || throw(DomainError(l, "Need abs(s) ≤ l ≤ lmax"))
    -l ≤ m ≤ l || throw(DomainError(m, "Need -l ≤ m ≤ l"))
    return CartesianIndex(l^2 + l + m + 1)
end

"Mode numbers (l, m) for a coefficient index"
function ash_mode_numbers(grid::SphereGrid, s::Integer, ind::Union{CartesianIndex{1},NTuple{1,<:Integer},Integer})
    i = ind isa Integer ? Int(ind) : Int(ind[1])
    1 ≤ i ≤ ash_nmodes(grid)[1] || throw(DomainError(i, "Mode index out of range"))
    l = isqrt(i - 1)
    m = (i - 1) - l^2 - l
    return (l, m)::NTuple{2,Int}
end

################################################################################
# Eth operators: pure mode-space operations in the canonical layout,
# independent of the backend

export ash_eth!, ash_eth, ash_ethbar!, ash_ethbar

"ð: raise the spin weight; ð ₛYₗₘ = +√((l−s)(l+s+1)) ₛ₊₁Yₗₘ"
function ash_eth!(grid::SphereGrid, ðflm::AbstractVector{<:Complex}, flm::AbstractVector{<:Complex}, s::Integer)
    @assert length(ðflm) == length(flm) == ash_nmodes(grid)[1]
    s′ = s + 1
    for l in 0:(grid.lmax), m in (-l):l
        i = l^2 + l + m + 1
        ðflm[i] = l ≥ max(abs(s), abs(s′)) ? sqrt(Float64((l - s) * (l + s + 1))) * flm[i] : zero(eltype(ðflm))
    end
    return ðflm
end
ash_eth(grid::SphereGrid, flm::AbstractVector{<:Complex}, s::Integer) = ash_eth!(grid, similar(flm), flm, s)

"ð̄: lower the spin weight; ð̄ ₛYₗₘ = −√((l+s)(l−s+1)) ₛ₋₁Yₗₘ"
function ash_ethbar!(grid::SphereGrid, ðflm::AbstractVector{<:Complex}, flm::AbstractVector{<:Complex}, s::Integer)
    @assert length(ðflm) == length(flm) == ash_nmodes(grid)[1]
    s′ = s - 1
    for l in 0:(grid.lmax), m in (-l):l
        i = l^2 + l + m + 1
        ðflm[i] = l ≥ max(abs(s), abs(s′)) ? -sqrt(Float64((l + s) * (l - s + 1))) * flm[i] : zero(eltype(ðflm))
    end
    return ðflm
end
ash_ethbar(grid::SphereGrid, flm::AbstractVector{<:Complex}, s::Integer) = ash_ethbar!(grid, similar(flm), flm, s)
