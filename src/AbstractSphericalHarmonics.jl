module AbstractSphericalHarmonics

using ComputedFieldTypes
using LinearAlgebra
using StaticArrays

################################################################################

# FastSphericalHarmonics requires julia >= 1.7
# SSHT requires julia >= 1.6

# using FastSphericalHarmonics
using SSHT

export ash_grid_size, ash_nmodes
export ash_ntheta, ash_nphi, ash_thetas, ash_phis, ash_point_coord, ash_point_delta, ash_grid_as_phi_theta
export ash_mode_index, ash_mode_numbers
export ash_transform!, ash_transform, ash_evaluate!, ash_evaluate
export ash_eth!, ash_eth, ash_ethbar!, ash_ethbar

################################################################################

bitsign(i::Integer) = isodd(i) ? -one(i) : one(i)

################################################################################

include("sylm.jl")
include("tensor.jl")

end
