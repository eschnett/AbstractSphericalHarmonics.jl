# Direct, naive translation of the Goldberg et al. sum. Exact in exact
# arithmetic (use `big` arguments), but suffers catastrophic cancellation in
# floating point for l ≳ 50, and does not work at poles (cot is infinite).
# Kept as an independent reference implementation for the tests.
function sYlm0(::Val{s}, ::Val{l}, ::Val{m}, θ::Real, ϕ::Real) where {s,l,m}
    return bitsign(m) *
           sqrt(factorial(big(l + m)) * factorial(big(l - m)) * (2l + 1) /
                (factorial(big(l + s)) * factorial(big(l - s)) * 4 * oftype(big(one(θ)), π))) *
           sin(θ / 2)^2l *
           sum(
               binomial(big(l - s), r) * binomial(big(l + s), r + s - m) * bitsign(l - r - s) * cis(m * ϕ) *
               cot(θ / 2)^(2r + s - m) for r in max(0, m - s):min(l - s, l + m)
           )
end

# Wigner small-d function d^l_{μν}(θ) in the standard (Wikipedia/Varshalovich)
# convention, evaluated by upward three-term recursion in the degree l. The
# recursion follows the dominant solution, so it is stable for all θ ∈ [0, π]
# (this is the spin-weighted generalization of the standard recursion for
# fully normalized associated Legendre functions).
#
# The seed at l₀ = max(|μ|, |ν|) is the single product
#     d^{l₀}_{μν} = (−1)^{max(0,μ−ν)} √((2l₀)!/(a!b!)) sin^a(θ/2) cos^b(θ/2)
# with a = |μ−ν|, b = |μ+ν| (no alternating sum, hence no cancellation). The
# binomial and the half-angle powers are accumulated in one interleaved
# product so intermediate values never stray far from the final magnitude.
#
# In Float64, the seed underflows to zero very close to the poles when
# l₀ ≳ 1900; for smaller degrees the result is accurate to a few l⋅eps.
function wigner_d(μ::Int, ν::Int, l::Int, sinθ2::T, cosθ2::T) where {T<:Real}
    l0 = max(abs(μ), abs(ν))
    @assert l ≥ l0
    a = abs(μ - ν)
    b = abs(μ + ν)

    # Seed: interleave the √((2l₀)!/(a!b!)) = √(binomial(a+b,a)) factors with
    # the sin/cos powers (Bresenham-style) to avoid overflow for large l₀
    d = one(T)
    i = j = 0
    while i < a || j < b
        if j * a ≤ i * b && j < b
            j += 1
            d *= cosθ2
        else
            i += 1
            d *= sqrt(T(b + i) / i) * sinθ2
        end
    end
    d *= bitsign(max(0, μ - ν))
    l == l0 && return d

    x = (cosθ2 - sinθ2) * (cosθ2 + sinθ2) # cos θ
    if l0 == 0
        # First step of the μ = ν = 0 recursion, where the generic step below
        # would evaluate μν/(l(l+1)) = 0/0
        dm1 = d
        d *= x
        l == 1 && return d
        lcur = 1
    else
        dm1 = zero(T)
        lcur = l0
    end

    # A_{l+1} d^{l+1} = (2l+1) (x − μν/(l(l+1))) d^l − A_l d^{l−1},
    # starting from d^{l₀−1} = 0 (A_{l₀} vanishes automatically)
    A(k) = sqrt(T((k - μ) * (k + μ)) * T((k - ν) * (k + ν))) / k
    while lcur < l
        k = lcur
        dp1 = ((2k + 1) * (x - T(μ) * ν / (k * (k + 1))) * d - A(k) * dm1) / A(k + 1)
        dm1 = d
        d = dp1
        lcur += 1
    end
    return d
end

export sYlm
"""
    sYlm(s::Integer, l::Integer, m::Integer, θ::Real, ϕ::Real)
    sYlm(::Val{s}, ::Val{l}, ::Val{m}, θ::Real, ϕ::Real)

Evaluate the spin-weighted spherical harmonic ``ₛYₗₘ(θ, ϕ)``.

Conventions: Goldberg et al. (1967), which includes the Condon–Shortley
phase ``(-1)^m``; equivalently
``ₛYₗₘ = (-1)^s √((2l+1)/4π) d^l_{m,-s}(θ) e^{imϕ}`` with the Wigner
d-function in the standard (Wikipedia/Varshalovich) convention. For `s = 0`
this reduces to the standard spherical harmonics ``Yₗₘ``. The parity
relation is ``conj(ₛYₗₘ) = (-1)^{s+m} ₋ₛYₗ₋ₘ``.

The evaluation uses a stable Wigner-d recursion and is accurate to a few
``l⋅eps`` for all degrees (in Float64 up to ``l ≈ 1900`` even at the poles).
Generic over the argument type: pass `BigFloat` angles for higher precision.
Throws `DomainError` unless ``|s| ≤ l`` and ``|m| ≤ l``.
"""
function sYlm(s::Int, l::Int, m::Int, θ::Real, ϕ::Real)
    abs(s) ≤ l || throw(DomainError(s, "Need abs(s) ≤ l"))
    -l ≤ m ≤ l || throw(DomainError(m, "Need -l ≤ m ≤ l"))
    T = float(typeof(zero(θ) * zero(ϕ)))
    sinθ2, cosθ2 = sincos(T(θ) / 2)
    d = wigner_d(m, -s, l, sinθ2, cosθ2)
    return bitsign(s) * sqrt((2l + 1) / (4 * T(π))) * d * cis(m * T(ϕ))
end
sYlm(s::Integer, l::Integer, m::Integer, θ::Real, ϕ::Real) = sYlm(Int(s), Int(l), Int(m), θ, ϕ)
sYlm(::Val{s}, ::Val{l}, ::Val{m}, θ::Real, ϕ::Real) where {s,l,m} = sYlm(Int(s), Int(l), Int(m), θ, ϕ)
