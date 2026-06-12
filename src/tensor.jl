stensor(D::Integer) = SArray{Tuple{ntuple(d -> 2, D)...}}
stensor(D::Integer, T::Type) = stensor(D){T}

export Tensor
@computed struct Tensor{D,T} # <: AbstractArray{D,T}
    values::AbstractArray{stensor(D, T),2}
    grid::SphereGrid
end

@doc """
    struct Tensor{D,T}

Tensor of rank `D` with elements of type `T`, stored component-wise.

The components are stored "normalized", i.e. projected onto the dyad
vectors `eθ^a` and `eϕ^a`. This removes singularities at the poles by
multiplying the `ϕ` components with respective powers of `sin θ`. This
also means that covariant (index down) and contravariant (index up)
indices are represented in the same way.

eθ^a = [1, 0]           eθ_a = [1, 0]
eϕ^a = [0, 1/(sin θ)]   eϕ_a = [0, sin θ]

eθ and eϕ are orthogonal, and both have length 1.

g_ab = diag[1, (sin θ)^2]
g_ab = eθ_a eθ_b + eϕ_a eϕ_b = (m_a m̄_b + m̄_a m_b) / 2

Example:

Scalar `s`: stored as is
Vector `v^a`: store `[eθ_a v^a, eϕ_a v^a]`
Tensor `t_ab`: store `t[1,1] = eθ^a eθ^b t_ab`
                     `t[1,2] = eθ^a eϕ^b t_ab`
                     `t[2,1] = eϕ^a eθ^b t_ab`
                     `t[2,2] = eϕ^a eϕ^b t_ab`
""" Tensor

export SpinTensor
struct SpinTensor{D,T} # <: AbstractVector{T}
    coeffs::AbstractArray{<:AbstractArray{T},D}
    grid::SphereGrid
end

@doc """
    struct SpinTensor{D,T}

Tensor of rank `D` with elements of type `T`, stored separated by spin
weights.

Grouping tensor components by their spin weight allows representing
tensors via spin-weighted spherical harmonics, and allows calculating
covariant derivatives (covariant with respect to the unit sphere).

Note that we use a convention where `m^a m̄_a = 2`.
""" SpinTensor

# Convenience constructors
Tensor{D}(values::AbstractArray{<:SArray{X,T} where {X},2}, grid::SphereGrid) where {D,T<:Number} = Tensor{D,T}(values, grid)
Tensor{0}(values::AbstractArray{<:Number,2}, grid::SphereGrid) = Tensor{0}(stensor(0).(values), grid)
Tensor(values::AbstractArray{<:SArray,2}, grid::SphereGrid) = Tensor{ndims(zero(eltype(values)))}(values, grid)
Tensor(values::AbstractArray{<:Number,2}, grid::SphereGrid) = Tensor{0}(values, grid)

SpinTensor{D}(coeffs::AbstractArray{<:AbstractArray{T},D}, grid::SphereGrid) where {D,T<:Number} = SpinTensor{D,T}(coeffs, grid)
SpinTensor{0}(coeffs::AbstractArray{<:Number}, grid::SphereGrid) = SpinTensor{0}(stensor(0)((coeffs,)), grid)
SpinTensor(coeffs::AbstractArray{<:AbstractArray,D}, grid::SphereGrid) where {D} = SpinTensor{D}(coeffs, grid)
SpinTensor(coeffs::AbstractArray{<:Number}, grid::SphereGrid) = SpinTensor{0}(coeffs, grid)

# Basic operations
Base.eltype(::Tensor{D,T}) where {D,T} = T
Base.:(==)(t1::Tensor{D}, t2::Tensor{D}) where {D} = t1.grid == t2.grid && t1.values == t2.values
Base.isapprox(t1::Tensor{D}, t2::Tensor{D}; kws...) where {D} = t1.grid == t2.grid && isapprox(t1.values, t2.values; kws...)
LinearAlgebra.norm(t::Tensor; kws...) = norm(t.values; kws...)
Base.copy(t::Tensor{D}) where {D} = Tensor{D}(copy(t.values), t.grid)
Base.zero(t::Tensor{D}) where {D} = Tensor{D}(zero(t.values), t.grid)
Base.:-(t::Tensor{D}) where {D} = Tensor{D}(-t.values, t.grid)
# (via Tuple since broadcast/map on 0-dim SArrays does not preserve the type)
Base.conj(t::Tensor{D}) where {D} = Tensor{D}(map(v -> typeof(v)(map(conj, Tuple(v))), t.values), t.grid)
function Base.:+(t1::Tensor{D}, t2::Tensor{D}) where {D}
    t1.grid == t2.grid || throw(DimensionMismatch())
    return Tensor{D}(t1.values + t2.values, t1.grid)
end
function Base.:-(t1::Tensor{D}, t2::Tensor{D}) where {D}
    t1.grid == t2.grid || throw(DimensionMismatch())
    return Tensor{D}(t1.values - t2.values, t1.grid)
end
Base.:*(a::Number, t::Tensor{D}) where {D} = Tensor{D}(a * t.values, t.grid)
Base.:*(t::Tensor{D}, a::Number) where {D} = Tensor{D}(t.values * a, t.grid)
Base.:/(t::Tensor{D}, a::Number) where {D} = Tensor{D}(t.values / a, t.grid)

Base.eltype(::SpinTensor{D,T}) where {D,T} = T
Base.:(==)(t1::SpinTensor{D}, t2::SpinTensor{D}) where {D} = t1.grid == t2.grid && t1.coeffs == t2.coeffs
Base.isapprox(t1::SpinTensor{D}, t2::SpinTensor{D}; kws...) where {D} =
    t1.grid == t2.grid && isapprox(t1.coeffs, t2.coeffs; kws...)
LinearAlgebra.norm(t::SpinTensor; kws...) = norm(t.coeffs; kws...)
Base.copy(t::SpinTensor{D}) where {D} = SpinTensor{D}(copy.(t.coeffs), t.grid)
Base.zero(t::SpinTensor{D}) where {D} = SpinTensor{D}(zero.(t.coeffs), t.grid)
Base.:-(t::SpinTensor{D}) where {D} = SpinTensor{D}(-t.coeffs, t.grid)
"""
    conj(t::SpinTensor)

Spectral representation of the complex conjugate field, i.e.
`Tensor(conj(t)) ≈ conj(Tensor(t))`.

Conjugating the field swaps the `m` and `m̄` directions in every tensor
slot and maps the coefficients via the parity relation
`conj(ₛYₗₘ) = (-1)^(s+m) ₋ₛYₗ₋ₘ`. This is *not* an elementwise `conj` of
the coefficient arrays.
"""
function Base.conj(spintensor::SpinTensor{D}) where {D}
    T = eltype(spintensor)
    @assert T <: Complex
    coeffs = spintensor.coeffs
    grid = spintensor.grid
    twos = ntuple(d -> 2, D)
    weights = SVector{2}(+1, -1)
    coeffs′ = SArray{Tuple{twos...}}([
        begin
            # Component `ij` of the conjugate comes from the slot-flipped
            # (m ↔ m̄) component of the input, with
            #     conj(f)_{l,m} = (-1)^(s+m) conj(f_{l,-m}) ;
            # the sign is the same whether `s` is the input or output spin
            # since (-1)^s = (-1)^(-s)
            ij′ = CartesianIndex(ntuple(d -> 3 - ij[d], D))
            cs = coeffs[ij′]
            @assert size(cs) == ash_nmodes(grid)
            s = sum(SVector{D,Int}(weights[i] for i in Tuple(ij)))
            [
                begin
                    l, m = ash_mode_numbers(grid, s, ind)
                    abs(s) ≤ l ? bitsign(s + m) * conj(cs[ash_mode_index(grid, s, l, -m)]) : zero(T)
                end for ind in CartesianIndices(cs)
            ]
        end for ij in CartesianIndices(twos)
    ])
    return SpinTensor{D}(coeffs′, grid)::SpinTensor{D,T}
end
function Base.:+(t1::SpinTensor{D}, t2::SpinTensor{D}) where {D}
    t1.grid == t2.grid || throw(DimensionMismatch())
    return SpinTensor{D}(t1.coeffs + t2.coeffs, t1.grid)
end
function Base.:-(t1::SpinTensor{D}, t2::SpinTensor{D}) where {D}
    t1.grid == t2.grid || throw(DimensionMismatch())
    return SpinTensor{D}(t1.coeffs - t2.coeffs, t1.grid)
end
Base.:*(a::Number, t::SpinTensor{D}) where {D} = SpinTensor{D}(a * t.coeffs, t.grid)
Base.:*(t::SpinTensor{D}, a::Number) where {D} = SpinTensor{D}(t.coeffs * a, t.grid)
Base.:/(t::SpinTensor{D}, a::Number) where {D} = SpinTensor{D}(t.coeffs / a, t.grid)

export filter_modes
"""
    filter_modes(spintensor::SpinTensor; lfilter=ash_lmax(grid)*2÷3)

Zero all modes with `l > lfilter` in every spin component
(anti-aliasing/de-aliasing filter). The default cutoff `2*lmax÷3`
implements the standard "two-thirds rule" suitable for quadratic
nonlinearities.
"""
function filter_modes(spintensor::SpinTensor{D}; lfilter::Integer=ash_lmax(spintensor.grid) * 2 ÷ 3) where {D}
    T = eltype(spintensor)
    @assert T <: Complex
    coeffs = spintensor.coeffs
    grid = spintensor.grid
    twos = ntuple(d -> 2, D)
    weights = SVector{2}(+1, -1)
    coeffs′ = SArray{Tuple{twos...}}([
        begin
            cs = coeffs[ij]
            @assert size(cs) == ash_nmodes(grid)
            s = sum(SVector{D,Int}(weights[i] for i in Tuple(ij)))
            [
                begin
                    l, m = ash_mode_numbers(grid, s, ind)
                    abs(s) ≤ l ≤ lfilter && abs(m) ≤ l ? c : zero(c)
                end for (ind, c) in zip(CartesianIndices(cs), cs)
            ]
        end for ij in CartesianIndices(twos)
    ])
    return SpinTensor{D}(coeffs′, grid)::SpinTensor{D,T}
end

"Convert `Tensor` to `SpinTensor`"
function SpinTensor{D}(tensor::Tensor{D,T}) where {D,T<:Complex}
    grid = tensor.grid
    # This represents `m` in terms of `eθ` and `eϕ`
    #     m^a = eθ^a + im eϕ^a
    #     m^a m_a = 0
    #     m^a m̄_a = 2
    m = SVector{2}(1, im)
    m̄ = conj(m)
    mm̄ = SVector{2}(m, m̄)
    mm̄::SVector{2,SVector{2,Complex{Int}}}
    weights = SVector{2}(+1, -1)
    twos = ntuple(d -> 2, D)
    coeffss = [
        begin
            values = [
                sum(prod(SVector{D,Complex{Int}}(mm̄[ij[d]][ab[d]] for d in 1:D)) * v[ab] for ab in CartesianIndices(twos)) for
                v in tensor.values
            ]
            s = sum(SVector{D,Int}(weights[i] for i in Tuple(ij)))
            coeffs = ash_transform(grid, values, s)
            coeffs
        end for ij in CartesianIndices(twos)
    ]
    spintensor = SpinTensor{D}(SArray{Tuple{twos...}}(coeffss), grid)
    return spintensor::SpinTensor{D,T}
end
SpinTensor{D}(tensor::Tensor{D,<:Real}) where {D} = SpinTensor{D}(Tensor{D}(map(x -> Complex.(x), tensor.values), tensor.grid))
SpinTensor(tensor::Tensor{D}) where {D} = SpinTensor{D}(tensor)

"Convert `SpinTensor` to `Tensor`"
function Tensor{D}(spintensor::SpinTensor{D}) where {D}
    T = eltype(spintensor)
    @assert T <: Complex
    spintensor::SpinTensor{D,T}
    grid = spintensor.grid
    # See above
    m = SVector{2}(1, im)
    m̄ = conj(m)
    m̄m = SVector{2}(m̄, m)
    m̄m::SVector{2,SVector{2,Complex{Int}}}
    weights = SVector{2}(+1, -1)
    twos = ntuple(d -> 2, D)
    valuess = [
        begin
            s = sum(SVector{D,Int}(weights[i] for i in Tuple(ij)))
            values = ash_evaluate(grid, spintensor.coeffs[ij], s)
            values
        end for ij in CartesianIndices(twos)
    ]
    values = [
        SArray{Tuple{twos...}}(
            sum(valuess[ij][n] * prod(SVector{D,Complex{Int}}(m̄m[ij[d]][ab[d]] for d in 1:D)) for ij in CartesianIndices(twos)) /
            2^D for ab in CartesianIndices(twos)
        ) for n in CartesianIndices(valuess[begin])
    ]
    tensor = Tensor{D}(values, grid)
    return tensor::Tensor{D,T}
end
Tensor(spintensor::SpinTensor{D}) where {D} = Tensor{D}(spintensor)

export tensor_gradient!
"""
    tensor_gradient!(dspintensor::SpinTensor{D+1}, spintensor::SpinTensor{D})

Calculate the covariant derivative (with respect to the unit sphere) of a
rank-`D` tensor, storing the result in `dspintensor`. See
[`tensor_gradient`](@ref) for the conventions.
"""
function tensor_gradient!(dspintensor::SpinTensor{D1,R}, spintensor::SpinTensor{D,T}) where {D1,R<:Complex,D,T<:Complex}
    @assert D1 == D + 1
    grid = spintensor.grid
    @assert dspintensor.grid == grid

    weights = SVector{2}(+1, -1)
    twos = ntuple(d -> 2, D)
    coeffs = spintensor.coeffs
    dcoeffs = dspintensor.coeffs
    for ab in CartesianIndices(twos)
        s = sum(SVector{D,Int}(weights[ab[d]] for d in 1:D))
        ash_eth!(grid, dcoeffs[ab, 1], coeffs[ab], s)
        dcoeffs[ab, 1] .*= -1
        ash_ethbar!(grid, dcoeffs[ab, 2], coeffs[ab], s)
        dcoeffs[ab, 2] .*= -1
    end
    return dspintensor::SpinTensor{D + 1,R}
end

export tensor_gradient
"""
    tensor_gradient(spintensor::SpinTensor{D}) -> SpinTensor{D+1}

Calculate the covariant derivative (with respect to the unit sphere) of a
rank-`D` tensor.

Conventions:
- The derivative index is appended *last*: `∇_c T_{ab...}` is stored as
  `dT[a, b, ..., c]`.
- For every index, slot 1 is the contraction with `m^a = eθ^a + i eϕ^a`
  (spin weight +1) and slot 2 the contraction with `m̄^a` (spin weight −1),
  normalized such that `m^a m̄_a = 2`. (References that use
  `m = (eθ + i eϕ)/√2` with `m·m̄ = 1` differ by factors of `√2`.)
- The stored derivative components are `(∇_m, ∇_m̄) = (−ð, −ð̄)` under the
  eth convention `ð = −(sin θ)^s (∂_θ + i csc(θ) ∂_ϕ) (sin θ)^(−s)`
  (see `ash_eth`); e.g. for a scalar `f`, `∇_m f = m^a ∂_a f = −ð f`.
"""
function tensor_gradient(spintensor::SpinTensor{D,T}) where {D,T<:Complex}
    grid = spintensor.grid
    weights = SVector{2}(+1, -1)
    twos = ntuple(d -> 2, D)
    coeffs = spintensor.coeffs
    dcoeffs = [
        begin
            s = sum(SVector{D,Int}(weights[ab[d]] for d in 1:D))
            if ab1 == 1
                -ash_eth(grid, coeffs[ab], s)
            else
                -ash_ethbar(grid, coeffs[ab], s)
            end
        end for ab in CartesianIndices(twos), ab1 in 1:2
    ]
    dspintensor = SpinTensor{D + 1}(stensor(D + 1)(dcoeffs), grid)
    return dspintensor::SpinTensor{D + 1,T}
end
