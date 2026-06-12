# Backend for `DriscollHealyGrid`, provided by the SSHT package.
#
# Uses SSHT's low-level API (core_dh_*, sampling_*) only, so it works
# independently of whether SSHT still carries its legacy `ash_*` layer.
# SSHT's native conventions coincide with the canonical ones of
# AbstractSphericalHarmonics (coefficient index l²+l+m+1, Wikipedia sYlm
# normalization), so no conversions are needed.

module AbstractSphericalHarmonicsSSHTExt

using AbstractSphericalHarmonics
const ASH = AbstractSphericalHarmonics
import SSHT

_L(grid::DriscollHealyGrid) = grid.lmax + 1

function ASH.ash_grid_size(grid::DriscollHealyGrid)
    L = _L(grid)
    return (SSHT.sampling_dh_nphi(L), SSHT.sampling_dh_ntheta(L))::NTuple{2,Int}
end
ASH.ash_nphi(grid::DriscollHealyGrid) = ASH.ash_grid_size(grid)[1]
ASH.ash_ntheta(grid::DriscollHealyGrid) = ASH.ash_grid_size(grid)[2]

function ASH.ash_point_coord(grid::DriscollHealyGrid, ij::Union{CartesianIndex{2},NTuple{2,Int}})
    L = _L(grid)
    p, t = Tuple(ij)
    return (SSHT.sampling_dh_t2theta(t, L), SSHT.sampling_dh_p2phi(p, L))
end

function ASH.ash_point_delta(grid::DriscollHealyGrid, ij::Union{CartesianIndex{2},NTuple{2,Int}})
    L = _L(grid)
    p, t = Tuple(ij)
    theta = SSHT.sampling_dh_t2theta(t, L)
    dtheta = SSHT.sampling_weight_dh(theta, L) / sin(theta)
    dphi = 2π / SSHT.sampling_dh_nphi(L)
    return (dtheta, dphi)
end

ASH.ash_thetas(grid::DriscollHealyGrid) = [SSHT.sampling_dh_t2theta(t, _L(grid)) for t in 1:ASH.ash_ntheta(grid)]
ASH.ash_phis(grid::DriscollHealyGrid) = [SSHT.sampling_dh_p2phi(p, _L(grid)) for p in 1:ASH.ash_nphi(grid)]

function ASH.ash_transform!(
    grid::DriscollHealyGrid, flm::AbstractVector{<:Complex}, f::AbstractMatrix{<:Complex}, s::Integer
)
    SSHT.core_dh_forward_sov!(flm, f, _L(grid), Int(s))
    # canonical layout: entries with l < |s| are zero
    for l in 0:min(abs(Int(s)) - 1, grid.lmax), m in (-l):l
        flm[l^2 + l + m + 1] = 0
    end
    return flm
end

function ASH.ash_evaluate!(
    grid::DriscollHealyGrid, f::AbstractMatrix{<:Complex}, flm::AbstractVector{<:Complex}, s::Integer
)
    return SSHT.core_dh_inverse_sov!(f, flm, _L(grid), Int(s))
end

end
