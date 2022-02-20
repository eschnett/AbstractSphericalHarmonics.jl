stensor(D::Integer) = SArray{Tuple{ntuple(d -> 2, D)...}}
stensor(D::Integer, T::Type) = stensor(D){T}

export Tensor
@computed struct Tensor{D,T} # <: AbstractArray{D,T}
    values::AbstractArray{stensor(D, T),2}
    lmax::Int
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
eϕ^a = [0, 1/(sin θ)]   eϕ_a = [1, sin θ]

eθ and eϕ are orthogonal, and both have length 1.

g_ab = diag[1, (sin θ)^2]
g_ab = eθ_a eϕ_b + eθ_b eϕ_a = m_a m̄_b + m̄_a m_b

Example:

Scalar `s`: stored as is
Vector `v^a`: store `[eθ_a v^a, eϕ_a v_a]`
Tensor `t_ab`: store `t[1,1] = eθ^a eθ^b t_ab`
                     `t[1,2] = eθ^a eϕ^b t_ab`
                     `t[2,1] = eϕ^a eθ^b t_ab`
                     `t[2,2] = eϕ^a eϕ^b t_ab`
""" Tensor

export SpinTensor
struct SpinTensor{D,T} # <: AbstractVector{T}
    coeffs::AbstractArray{<:AbstractArray{T},D}
    lmax::Int
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
Tensor{D}(values::AbstractArray{<:SArray{X,T} where {X},2}, lmax::Int) where {D,T<:Number} = Tensor{D,T}(values, lmax)
Tensor{0}(values::AbstractArray{<:Number,2}, lmax::Int) = Tensor{0}(stensor(0).(values), lmax)
Tensor(values::AbstractArray{<:SArray,2}, lmax::Int) = Tensor{ndims(zero(eltype(values)))}(values, lmax)
Tensor(values::AbstractArray{<:Number,2}, lmax::Int) = Tensor{0}(values, lmax)

SpinTensor{D}(coeffs::AbstractArray{<:AbstractArray{T},D}, lmax::Int) where {D,T<:Number} = SpinTensor{D,T}(coeffs, lmax)
SpinTensor{0}(coeffs::AbstractArray{<:Number}, lmax::Int) = SpinTensor{0}(stensor(0)((coeffs,)), lmax)
SpinTensor(coeffs::AbstractArray{<:AbstractArray,D}, lmax::Int) where {D} = SpinTensor{D}(coeffs, lmax)
SpinTensor(coeffs::AbstractArray{<:Number}, lmax::Int) = SpinTensor{0}(coeffs, lmax)

# Basic operations
Base.eltype(::Tensor{D,T}) where {D,T} = T
Base.:(==)(t1::Tensor{D}, t2::Tensor{D}) where {D} = t1.lmax == t2.lmax && t1.values == t2.values
Base.isapprox(t1::Tensor{D}, t2::Tensor{D}; kws...) where {D} = t1.lmax == t2.lmax && isapprox(t1.values, t2.values; kws...)
LinearAlgebra.norm(t::Tensor; kws...) where {D} = norm(t.values; kws...)
Base.copy(t::Tensor{D}) where {D} = Tensor{D}(copy(t.values), t.lmax)
Base.zero(t::Tensor{D}) where {D} = Tensor{D}(zero(t.values), t.lmax)
Base.:-(t::Tensor{D}) where {D} = Tensor{D}(-t.values, t.lmax)
Base.conj(t::Tensor{D}) where {D} = Tensor{D}(conj(t.values), t.lmax)
function Base.:+(t1::Tensor{D}, t2::Tensor{D}) where {D}
    t1.lmax == t2.lmax || throw(DimensionMismatch())
    return Tensor{D}(t1.values + t2.values, t1.lmax)
end
function Base.:-(t1::Tensor{D}, t2::Tensor{D}) where {D}
    t1.lmax == t2.lmax || throw(DimensionMismatch())
    return Tensor{D}(t1.values - t2.values, t1.lmax)
end
Base.:*(a::Number, t::Tensor{D}) where {D} = Tensor{D}(a * t.values, t.lmax)
Base.:*(t::Tensor{D}, a::Number) where {D} = Tensor{D}(t.values * a, t.lmax)
Base.:/(t::Tensor{D}, a::Number) where {D} = Tensor{D}(t.values / a, t.lmax)

Base.eltype(::SpinTensor{D,T}) where {D,T} = T
Base.:(==)(t1::SpinTensor{D}, t2::SpinTensor{D}) where {D} = t1.lmax == t2.lmax && t1.coeffs == t2.coeffs
Base.isapprox(t1::SpinTensor{D}, t2::SpinTensor{D}; kws...) where {D} = t1.lmax == t2.lmax && isapprox(t1.coeffs, t2.coeffs; kws...)
LinearAlgebra.norm(t::SpinTensor; kws...) where {D} = norm(t.coeffs; kws...)
Base.copy(t::SpinTensor{D}) where {D} = SpinTensor{D}(copy.(t.coeffs), t.lmax)
Base.zero(t::SpinTensor{D}) where {D} = SpinTensor{D}(zero.(t.coeffs), t.lmax)
Base.:-(t::SpinTensor{D}) where {D} = SpinTensor{D}(-t.coeffs, t.lmax)
Base.conj(t::SpinTensor{D}) where {D} = SpinTensor{D}(conj(t.coeffs), t.lmax)
function Base.:+(t1::SpinTensor{D}, t2::SpinTensor{D}) where {D}
    t1.lmax == t2.lmax || throw(DimensionMismatch())
    return SpinTensor{D}(t1.coeffs + t2.coeffs, t1.lmax)
end
function Base.:-(t1::SpinTensor{D}, t2::SpinTensor{D}) where {D}
    t1.lmax == t2.lmax || throw(DimensionMismatch())
    return SpinTensor{D}(t1.coeffs - t2.coeffs, t1.lmax)
end
Base.:*(a::Number, t::SpinTensor{D}) where {D} = SpinTensor{D}(a * t.coeffs, t.lmax)
Base.:*(t::SpinTensor{D}, a::Number) where {D} = SpinTensor{D}(t.coeffs * a, t.lmax)
Base.:/(t::SpinTensor{D}, a::Number) where {D} = SpinTensor{D}(t.coeffs / a, t.lmax)

export filter_modes
function filter_modes(spintensor::SpinTensor{D}) where {D}
    T = eltype(spintensor)
    @assert T <: Complex
    coeffs = spintensor.coeffs
    lmax = spintensor.lmax
    twos = ntuple(d -> 2, D)
    weights = SVector{2}(+1, -1)
    lfilter = lmax * 2 ÷ 3
    coeffs′ = SArray{Tuple{twos...}}([
        begin
            cs = coeffs[ij]
            @assert size(cs) == ash_nmodes(lmax)
            s = sum(SVector{D,Int}(weights[i] for i in Tuple(ij)))
            [
                begin
                    l, m = ash_mode_numbers(s, ind, lmax)
                    abs(s) ≤ l ≤ lfilter && abs(m) ≤ l ? c : zero(c)
                end for (ind, c) in zip(CartesianIndices(cs), cs)
            ]
        end for ij in CartesianIndices(twos)
    ])
    return SpinTensor{D}(coeffs′, lmax)::SpinTensor{D,T}
end

"Convert `Tensor` to `SpinTensor`"
function SpinTensor{D}(tensor::Tensor{D,T}) where {D,T<:Complex}
    lmax = tensor.lmax
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
            coeffs = ash_transform(values, s, lmax)
            coeffs
        end for ij in CartesianIndices(twos)
    ]
    spintensor = SpinTensor{D}(SArray{Tuple{twos...}}(coeffss), lmax)
    return spintensor::SpinTensor{D,T}
end
SpinTensor{D}(tensor::Tensor{D,<:Real}) where {D} = SpinTensor{D}(Tensor{D}(map(x -> Complex.(x), tensor.values), tensor.lmax))
SpinTensor(tensor::Tensor{D}) where {D} = SpinTensor{D}(tensor)

"Convert `SpinTensor` to `Tensor`"
function Tensor{D}(spintensor::SpinTensor{D}) where {D}
    T = eltype(spintensor)
    @assert T <: Complex
    spintensor::SpinTensor{D,T}
    lmax = spintensor.lmax
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
            values = ash_evaluate(spintensor.coeffs[ij], s, lmax)
            values
        end for ij in CartesianIndices(twos)
    ]
    values = [
        SArray{Tuple{twos...}}(
            sum(valuess[ij][n] * prod(SVector{D,Complex{Int}}(m̄m[ij[d]][ab[d]] for d in 1:D)) for ij in CartesianIndices(twos)) /
            2^D for ab in CartesianIndices(twos)
        ) for n in CartesianIndices(valuess[begin])
    ]
    tensor = Tensor{D}(values, lmax)
    return tensor::Tensor{D,T}
end
Tensor(spintensor::SpinTensor{D}) where {D} = Tensor{D}(spintensor)

export tensor_gradient!
"Calculate gradient"
function tensor_gradient!(dspintensor::SpinTensor{D1,R}, spintensor::SpinTensor{D,T}) where {D1,R<:Complex,D,T<:Complex}
    @assert D1 == D + 1
    lmax = spintensor.lmax
    @assert dspintensor.lmax == lmax

    weights = SVector{2}(+1, -1)
    twos = ntuple(d -> 2, D)
    coeffs = spintensor.coeffs
    dcoeffs = dspintensor.coeffs
    for ab in CartesianIndices(twos)
        s = sum(SVector{D,Int}(weights[ab[d]] for d in 1:D))
        ash_eth!(dcoeffs[ab, 1], coeffs[ab], s, lmax)
        dcoeffs[ab, 1] .*= -1
        ash_ethbar!(dcoeffs[ab, 2], coeffs[ab], s, lmax)
        dcoeffs[ab, 2] .*= -1
    end
    return dspintensor::SpinTensor{D + 1,R}
end

export tensor_gradient
"Calculate gradient"
function tensor_gradient(spintensor::SpinTensor{D,T}) where {D,T<:Complex}
    lmax = spintensor.lmax
    weights = SVector{2}(+1, -1)
    twos = ntuple(d -> 2, D)
    coeffs = spintensor.coeffs
    dcoeffs = [
        begin
            s = sum(SVector{D,Int}(weights[ab[d]] for d in 1:D))
            if ab1 == 1
                -ash_eth(coeffs[ab], s, lmax)
            else
                -ash_ethbar(coeffs[ab], s, lmax)
            end
        end for ab in CartesianIndices(twos), ab1 in 1:2
    ]
    dspintensor = SpinTensor{D + 1}(stensor(D + 1)(dcoeffs), lmax)
    return dspintensor::SpinTensor{D + 1,T}
end
