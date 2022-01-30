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
@computed struct SpinTensor{D,T} # <: AbstractVector{T}
    # TODO: We want `<:AbstractArray{T}` instead
    coeffs::stensor(D, AbstractArray{T})
    lmax::Int
end

@doc """
    struct SpinTensor{D,T}

Tensor of rank `D` with elements of type `T`, stored separated by spin
weights.

Grouping tensor components by their spin weight allows representing
tensors via spin-weighted spherical harmonics, and allows calculating
covariant derivatives (covariant with respect to the unit sphere).
""" SpinTensor

# Convenience constructors
function Tensor{D}(values::AbstractArray{<:SArray{X,T} where {X},2}, lmax::Int) where {D,T}
    return Tensor{D,T}(values, lmax)
end
Tensor{0}(values::AbstractArray{<:Number,2}, lmax::Int) = Tensor{0}(stensor(0).(values), lmax)

function SpinTensor{D}(coeffs::SArray{X,<:AbstractArray{T}} where {X}, lmax::Int) where {D,T}
    return SpinTensor{D,T}(coeffs, lmax)
end
function SpinTensor{0}(coeffs::AbstractArray{<:Number,2}, lmax::Int)
    return SpinTensor{0}(stensor(0)(coeffs), lmax)
end

# Basic operations
Base.eltype(::Tensor{D,T}) where {D,T} = T
Base.:(==)(t1::Tensor{D}, t2::Tensor{D}) where {D} = t1.lmax == t2.lmax && t1.values == t2.values
function Base.isapprox(t1::Tensor{D}, t2::Tensor{D}; kws...) where {D}
    return t1.lmax == t2.lmax && isapprox(t1.values, t2.values; kws...)
end
LinearAlgebra.norm(t::Tensor; kws...) where {D} = norm(t.values; kws...)
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
function Base.:(==)(t1::SpinTensor{D}, t2::SpinTensor{D}) where {D}
    return t1.lmax == t2.lmax && t1.coeffs == t2.coeffs
end
function Base.isapprox(t1::SpinTensor{D}, t2::SpinTensor{D}; kws...) where {D}
    return t1.lmax == t2.lmax && isapprox(t1.coeffs, t2.coeffs; kws...)
end
LinearAlgebra.norm(t::SpinTensor; kws...) where {D} = norm(t.coeffs; kws...)
Base.zero(t::SpinTensor{D}) where {D} = SpinTensor{D}(zero(t.coeffs), t.lmax)
Base.:-(t::SpinTensor{D}) where {D} = SpinTensor{D}(-t.coefffs, t.lmax)
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

"Convert `Tensor` to `SpinTensor`"
function SpinTensor{D}(tensor::Tensor{D}) where {D}
    T = eltype(tensor)
    @assert T <: Number
    CT = typeof(Complex(zero(T)))
    tensor::Tensor{D,T}
    lmax = tensor.lmax
    # This represents `m` in terms of `eθ` and `eϕ`
    #     m^a = eθ^a + im eϕ^a
    #     m^a m_a = 0
    #     m^a m̄_a = 2
    m = SVector{2}(1, im)
    m̄ = conj(m)
    if D == 0
        # Avoid real-valued spin spherical harmonics
        values = [Complex(v[]) for v in tensor.values]
        coeffs = ash_transform(values, 0, lmax)
        return SpinTensor{D}(Scalar(coeffs), lmax)::SpinTensor{D,CT}
    end
    if D == 1
        values_m = [sum(m[a] * v[a] for a in 1:2) for v in tensor.values]::Array{CT,2}
        values_m̄ = [sum(m̄[a] * v[a] for a in 1:2) for v in tensor.values]::Array{CT,2}
        coeffs_m = ash_transform(values_m, +1, lmax)
        coeffs_m̄ = ash_transform(values_m̄, -1, lmax)
        return SpinTensor{D}(SVector{2}(coeffs_m, coeffs_m̄), lmax)::SpinTensor{D,CT}
    end
    if D == 2
        values_mm = [sum(m[a] * m[b] * v[a, b] for a in 1:2, b in 1:2) for v in tensor.values]
        values_mm̄ = [sum(m[a] * m̄[b] * v[a, b] for a in 1:2, b in 1:2) for v in tensor.values]
        values_m̄m = [sum(m̄[a] * m[b] * v[a, b] for a in 1:2, b in 1:2) for v in tensor.values]
        values_m̄m̄ = [sum(m̄[a] * m̄[b] * v[a, b] for a in 1:2, b in 1:2) for v in tensor.values]
        coeffs_mm = ash_transform(values_mm, +2, lmax)
        coeffs_mm̄ = ash_transform(values_mm̄, 0, lmax)
        coeffs_m̄m = ash_transform(values_m̄m, 0, lmax)
        coeffs_m̄m̄ = ash_transform(values_m̄m̄, -2, lmax)
        return SpinTensor{D}(SMatrix{2,2}(coeffs_mm, coeffs_m̄m, coeffs_mm̄, coeffs_m̄m̄), lmax)::SpinTensor{D,CT}
    end
    @assert false
end
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
    if D == 0
        coeffs = spintensor.coeffs[]
        values = ash_evaluate(coeffs, 0, lmax)
        return Tensor{D}(Scalar.(values), lmax)::Tensor{D,T}
    end
    if D == 1
        values_m = ash_evaluate(spintensor.coeffs[1], +1, lmax)
        values_m̄ = ash_evaluate(spintensor.coeffs[2], -1, lmax)
        values = [SVector{2}((vm * m̄[a] + vm̄ * m[a]) / 2 for a in 1:2) for (vm, vm̄) in zip(values_m, values_m̄)]
        return Tensor{D}(values, lmax)::Tensor{D,T}
    end
    if D == 2
        values_mm = ash_evaluate(spintensor.coeffs[1, 1], +2, lmax)
        values_mm̄ = ash_evaluate(spintensor.coeffs[1, 2], 0, lmax)
        values_m̄m = ash_evaluate(spintensor.coeffs[2, 1], 0, lmax)
        values_m̄m̄ = ash_evaluate(spintensor.coeffs[2, 2], -2, lmax)
        values = [SMatrix{2,2}((vmm * m̄[a] * m̄[b] + vmm̄ * m̄[a] * m[b] + vm̄m * m[a] * m̄[b] + vm̄m̄ * m[a] * m[b]) / 4
                               for a in 1:2, b in 1:2)
                  for (vmm, vmm̄, vm̄m, vm̄m̄) in zip(values_mm, values_mm̄, values_m̄m, values_m̄m̄)]
        return Tensor{D}(values, lmax)::Tensor{D,T}
    end
    @assert false
end
Tensor(spintensor::SpinTensor{D}) where {D} = Tensor{D}(spintensor)

export tensor_gradient
"Calculate gradient"
function tensor_gradient(spintensor::SpinTensor{D}) where {D}
    T = eltype(spintensor)
    @assert T <: Complex
    spintensor::SpinTensor{D,T}
    lmax = spintensor.lmax
    if D == 0
        coeffs = spintensor.coeffs[]
        dcoeffs_m = -ash_eth(coeffs, 0, lmax)
        dcoeffs_m̄ = -ash_ethbar(coeffs, 0, lmax)
        return SpinTensor{D + 1}(stensor(D + 1)(dcoeffs_m, dcoeffs_m̄), lmax)::SpinTensor{D + 1,T}
    end
    if D == 1
        coeffs_m = spintensor.coeffs[1]
        coeffs_m̄ = spintensor.coeffs[2]
        dcoeffs_mm = -ash_eth(coeffs_m, 1, lmax)
        dcoeffs_mm̄ = -ash_ethbar(coeffs_m, 1, lmax)
        dcoeffs_m̄m = -ash_eth(coeffs_m̄, -1, lmax)
        dcoeffs_m̄m̄ = -ash_ethbar(coeffs_m̄, -1, lmax)
        return SpinTensor{D + 1}(stensor(D + 1)(dcoeffs_mm, dcoeffs_m̄m, dcoeffs_mm̄, dcoeffs_m̄m̄), lmax)::SpinTensor{D + 1,T}
    end
    @assert false
end
