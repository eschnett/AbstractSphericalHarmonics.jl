# Direct, naive translation, works only for small (s, l, m), does not work at poles
function sYlm0(::Val{s}, ::Val{l}, ::Val{m}, θ::Real, ϕ::Real) where {s,l,m}
    T = typeof(zero(θ) * zero(ϕ))
    return bitsign(m) *
           sqrt(factorial(l + m) * factorial(l - m) * (2l + 1) / (factorial(l + s) * factorial(l - s) * 4 * T(π))) *
           sin(θ / 2)^2l *
           sum(
               binomial(l - s, r) * binomial(l + s, r + s - m) * bitsign(l - r - s) * cis(m * ϕ) * cot(θ / 2)^(2r + s - m) for
               r in 0:(l - s)
           )
end

export sYlm
@generated function sYlm(::Val{s}, ::Val{l}, ::Val{m}, θ::Real, ϕ::Real) where {s,l,m}
    stmts = []
    push!(stmts, :((sinθ2, cosθ2) = sincos(θ / 2)))
    T = typeof(one(θ) / one(θ))
    quot = (factorial(big(l + m)) * factorial(big(l - m)))//(factorial(big(l + s)) * factorial(big(l - s)))
    scale = bitsign(m) * sqrt(T(quot * (2l + 1)) / (4 * T(π)))
    push!(stmts, :(res = zero($T)))
    # TODO: Use Horner scheme. Each summand differs by cos(θ/2)^2 / sin(θ/2)^2 from the next.
    for r in 0:(l - s)
        α = bitsign(l - r - s) * binomial(big(l - s), r) * binomial(big(l + s), r + s - m)
        if α ≠ 0
            cospower = 2r + s - m
            sinpower = 2l - cospower
            push!(stmts, :(res += $(T(α)) * sinθ2^$sinpower * cosθ2^$cospower))
        end
    end
    push!(stmts, :(return $scale * res * cis(m * ϕ)))
    return quote
        $((stmts)...)
    end
end

sYlm(s::Int, l::Int, m::Int, θ::Real, ϕ::Real) = sYlm(Val(s), Val(l), Val(m), θ, ϕ)
sYlm(s::Integer, l::Integer, m::Integer, θ::Real, ϕ::Real) = sYlm(Int(s), Int(l), Int(m), θ, ϕ)
