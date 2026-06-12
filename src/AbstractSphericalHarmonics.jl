module AbstractSphericalHarmonics

using ComputedFieldTypes
using LinearAlgebra
using StaticArrays

################################################################################

bitsign(i::Integer) = isodd(i) ? -one(i) : one(i)

################################################################################

include("grids.jl")
include("sylm.jl")
include("tensor.jl")

end
