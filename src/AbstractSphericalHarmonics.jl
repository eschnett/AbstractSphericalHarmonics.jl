module AbstractSphericalHarmonics

# using FastSphericalHarmonics
using ssht

export ash_grid_size, ash_nmodes
export ash_ntheta, ash_nphi, ash_thetas, ash_phis, ash_point_coord, ash_point_delta, ash_grid_as_phi_theta
export ash_mode_index
export ash_transform!, ash_transform, ash_evaluate!, ash_evaluate
export ash_eth!, ash_eth, ash_ethbar!, ash_ethbar

################################################################################

bitsign(i::Integer) = isodd(i) ? -one(i) : one(i)

# Direct, naive translation, works only for small (s, l, m), does not work at poles
function sYlm0(::Val{s}, ::Val{l}, ::Val{m}, θ::Real, ϕ::Real) where {s,l,m}
    T = typeof(zero(θ) * zero(ϕ))
    return bitsign(m) *
           sqrt(factorial(l + m) * factorial(l - m) * (2l + 1) / (factorial(l + s) * factorial(l - s) * 4 * T(π))) *
           sin(θ / 2)^2l *
           sum(binomial(l - s, r) * binomial(l + s, r + s - m) * bitsign(l - r - s) * cis(m * ϕ) * cot(θ / 2)^(2r + s - m)
               for r in 0:(l - s))
end

@generated function sYlm(::Val{s}, ::Val{l}, ::Val{m}, θ::Real, ϕ::Real) where {s,l,m}
    stmts = []
    push!(stmts, :((sinθ2, cosθ2) = sincos(θ / 2)))
    T = typeof(one(θ) / one(θ))
    quot = (factorial(big(l + m)) * factorial(big(l - m))) // (factorial(big(l + s)) * factorial(big(l - s)))
    scale = bitsign(m) * sqrt(T(quot * (2l + 1)) / (4 * T(π)))
    push!(stmts, :(res = zero($T)))
    for r in 0:(l - s)
        α = bitsign(l - r - s) * binomial(l - s, r) * binomial(l + s, r + s - m)
        if α ≠ 0
            cospower = 2r + s - m
            sinpower = 2l - cospower
            push!(stmts, :(res += $α * sinθ2^$sinpower * cosθ2^$cospower))
        end
    end
    push!(stmts, :(return $scale * res * cis(m * ϕ)))
    return quote
        $((stmts)...)
    end
end
sYlm(s::Int, l::Int, m::Int, θ::Real, ϕ::Real) = sYlm(Val(s), Val(l), Val(m), θ, ϕ)
sYlm(s::Integer, l::Integer, m::Integer, θ::Real, ϕ::Real) = sYlm(Int(s), Int(l), Int(m), θ, ϕ)

export sYlm

end
