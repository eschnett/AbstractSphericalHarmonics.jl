using AbstractSphericalHarmonics
using Random
using Test

bitsign(b::Bool) = b ? -1 : 1
bitsign(i::Integer) = bitsign(isodd(i))

chop(x) = abs2(x) < 100eps(x) ? zero(x) : x
chop(x::Complex) = Complex(chop(real(x)), chop(imag(x)))

function integrate(f::AbstractMatrix, g::AbstractMatrix, lmax::Integer)
    sz = ash_grid_size(lmax)
    @assert size(f) == size(g) == sz

    s = zero(eltype(f)) * zero(eltype(g)) * zero(Float64)
    for pt in CartesianIndices(sz)
        theta, phi = ash_point_coord(pt, lmax)
        dtheta, dphi = ash_point_delta(pt, lmax)
        s += conj(f[pt]) * g[pt] * sin(theta) * dtheta * dphi
    end

    return s
end

################################################################################

# Half angle formulae:
#     sin(θ/2)^2 = (1-cos(θ))/2
#     cos(θ/2)^2 = (1+cos(θ))/2

# These sYlm are taken from Wikipedia and black-holes.org:
# <https://en.wikipedia.org/wiki/Spherical_harmonics> and
# <https://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics>
# <https://www.black-holes.org/SpinWeightedSphericalHarmonics.nb>
function sYlm(s::Integer, l::Integer, m::Integer, θ::Real, ϕ::Real)
    @assert abs(s) ≤ l
    @assert -l ≤ m ≤ l

    # Parity:
    s < 0 && return bitsign(s + m) * conj(sYlm(-s, l, -m, θ, ϕ))

    (s, l, m) == (0, 0, 0) && return sqrt(1 / 4π)
    (s, l, m) == (0, 1, -1) && return sqrt(3 / 8π) * sin(θ) * cis(-ϕ)
    (s, l, m) == (0, 1, 0) && return sqrt(3 / 4π) * cos(θ)
    (s, l, m) == (0, 1, +1) && return -sqrt(3 / 8π) * sin(θ) * cis(ϕ)
    (s, l, m) == (0, 2, -2) && return sqrt(15 / 2π) * cos(θ / 2)^2 * sin(θ / 2)^2 * cis(-2ϕ)
    (s, l, m) == (0, 2, -1) && return -sqrt(15 / 2π) * cos(θ / 2) * sin(θ / 2) * (-cos(θ / 2)^2 + sin(θ / 2)^2) * cis(-ϕ)
    (s, l, m) == (0, 2, 0) && return sqrt(5 / 4π) * (cos(θ / 2)^4 - 4 * cos(θ / 2)^2 * sin(θ / 2)^2 + sin(θ / 2)^4)
    (s, l, m) == (0, 2, +1) && return -sqrt(15 / 2π) * cos(θ / 2) * sin(θ / 2) * (cos(θ / 2)^2 - sin(θ / 2)^2) * cis(ϕ)
    (s, l, m) == (0, 2, +2) && return sqrt(15 / 2π) * cos(θ / 2)^2 * sin(θ / 2)^2 * cis(2ϕ)

    (s, l, m) == (+1, 1, -1) && return -sqrt(3 / 16π) * (1 + cos(θ)) * cis(-ϕ)
    (s, l, m) == (+1, 1, 0) && return sqrt(3 / 8π) * sin(θ)
    (s, l, m) == (+1, 1, +1) && return -sqrt(3 / 16π) * (1 - cos(θ)) * cis(ϕ)
    (s, l, m) == (+1, 2, -2) && return -sqrt(5 / π) * cos(θ / 2)^3 * sin(θ / 2) * cis(-2ϕ)
    (s, l, m) == (+1, 2, -1) && return -sqrt(5 / 4π) * cos(θ / 2)^2 * (cos(θ / 2)^2 - 3 * sin(θ / 2)^2) * cis(-ϕ)
    (s, l, m) == (+1, 2, 0) && return sqrt(15 / 2π) * cos(θ / 2) * sin(θ / 2) * (cos(θ / 2)^2 - sin(θ / 2)^2)
    (s, l, m) == (+1, 2, +1) && return sqrt(5 / 4π) * sin(θ / 2)^2 * (-3 * cos(θ / 2)^2 + sin(θ / 2)^2) * cis(ϕ)
    (s, l, m) == (+1, 2, +2) && return sqrt(5 / π) * cos(θ / 2) * sin(θ / 2)^3 * cis(2ϕ)

    (s, l, m) == (+2, 2, -2) && return sqrt(5 / 4π) * cos(θ / 2)^4 * cis(-2ϕ)
    (s, l, m) == (+2, 2, -1) && return -sqrt(5 / π) * cos(θ / 2)^3 * sin(θ / 2) * cis(-ϕ)
    (s, l, m) == (+2, 2, 0) && return sqrt(15 / 2π) * cos(θ / 2)^2 * sin(θ / 2)^2
    (s, l, m) == (+2, 2, +1) && return -sqrt(5 / π) * cos(θ / 2) * sin(θ / 2)^3 * cis(ϕ)
    (s, l, m) == (+2, 2, +2) && return sqrt(5 / 4π) * sin(θ / 2)^4 * cis(2ϕ)

    @assert false
end

# ð F = - (sin θ)^s (∂_θ + i / sin(θ) ∂_ϕ) (sin θ)^-s F
# (Calculated manually from sYlm above)
function ðsYlm(s::Integer, l::Integer, m::Integer, θ::Real, ϕ::Real)
    @assert abs(s) ≤ l
    @assert -l ≤ m ≤ l

    (s, l, m) == (0, 0, 0) && return 0
    (s, l, m) == (0, 1, -1) && return -sqrt(3 / 8π) * (1 + cos(θ)) * cis(-ϕ)
    (s, l, m) == (0, 1, 0) && return sqrt(3 / 4π) * sin(θ)
    (s, l, m) == (0, 1, +1) && return -sqrt(3 / 8π) * (1 - cos(θ)) * cis(ϕ)
    (s, l, m) == (0, 2, -2) && return -sqrt(15 / 8π) * (1 + cos(θ)) * sin(θ) * cis(-2ϕ)
    (s, l, m) == (0, 2, -1) && return -sqrt(15 / 2π) * cos(θ / 2)^2 * (-1 + 2 * cos(θ)) * cis(-ϕ)
    (s, l, m) == (0, 2, 0) && return sqrt(45 / 16π) * sin(2θ)
    (s, l, m) == (0, 2, +1) && return -sqrt(15 / 2π) * (1 + 2 * cos(θ)) * sin(θ / 2)^2 * cis(ϕ)
    (s, l, m) == (0, 2, +2) && return -sqrt(15 / 8π) * (-1 + cos(θ)) * sin(θ) * cis(2ϕ)

    (s, l, m) == (+1, 1, -1) && return 0
    (s, l, m) == (+1, 1, 0) && return 0
    (s, l, m) == (+1, 1, +1) && return 0
    (s, l, m) == (+1, 2, -2) && return sqrt(5 / π) * cos(θ / 2)^4 * cis(-2ϕ)
    (s, l, m) == (+1, 2, -1) && return -sqrt(5 / π) * cos(θ / 2)^2 * sin(θ) * cis(-ϕ)
    (s, l, m) == (+1, 2, 0) && return sqrt(15 / 8π) * sin(θ)^2
    (s, l, m) == (+1, 2, +1) && return -sqrt(5 / π) * sin(θ / 2)^2 * sin(θ) * cis(ϕ)
    (s, l, m) == (+1, 2, +2) && return sqrt(5 / π) * sin(θ / 2)^4 * cis(2ϕ)

    (s, l, m) == (+2, 2, -2) && return 0
    (s, l, m) == (+2, 2, -1) && return 0
    (s, l, m) == (+2, 2, 0) && return 0
    (s, l, m) == (+2, 2, +1) && return 0
    (s, l, m) == (+2, 2, +2) && return 0

    @assert false
end

# ð̄ F = - (sin θ)^-s (∂_θ - i / sin(θ) ∂_ϕ) (sin θ)^s F
# (Calculated manually from sYlm above)
function ð̄sYlm(s::Integer, l::Integer, m::Integer, θ::Real, ϕ::Real)
    @assert abs(s) ≤ l
    @assert -l ≤ m ≤ l

    (s, l, m) == (0, 0, 0) && return 0
    (s, l, m) == (0, 1, -1) && return sqrt(3 / 8π) * (1 - cos(θ)) * cis(-ϕ)
    (s, l, m) == (0, 1, 0) && return sqrt(3 / 4π) * sin(θ)
    (s, l, m) == (0, 1, +1) && return sqrt(3 / 8π) * (1 + cos(θ)) * cis(ϕ)
    (s, l, m) == (0, 1, +1) && return -sqrt(3 / 8π) * (1 - cos(θ)) * cis(ϕ)
    (s, l, m) == (0, 2, -2) && return -sqrt(15 / 8π) * (-1 + cos(θ)) * sin(θ) * cis(-2ϕ)
    (s, l, m) == (0, 2, -1) && return sqrt(15 / 2π) * (1 + 2 * cos(θ)) * sin(θ / 2)^2 * cis(-ϕ)
    (s, l, m) == (0, 2, 0) && return sqrt(45 / 16π) * sin(2θ)
    (s, l, m) == (0, 2, +1) && return sqrt(15 / 2π) * cos(θ / 2)^2 * (-1 + 2 * cos(θ)) * cis(ϕ)
    (s, l, m) == (0, 2, +2) && return -sqrt(15 / 8π) * (1 + cos(θ)) * sin(θ) * cis(2ϕ)

    (s, l, m) == (+1, 1, -1) && return -sqrt(3 / 4π) * sin(θ) * cis(-ϕ)
    (s, l, m) == (+1, 1, 0) && return -sqrt(3 / 2π) * cos(θ)
    (s, l, m) == (+1, 1, +1) && return sqrt(3 / 4π) * sin(θ) * cis(ϕ)
    (s, l, m) == (+1, 2, -2) && return -sqrt(45 / 16π) * sin(θ)^2 * cis(-2ϕ)
    (s, l, m) == (+1, 2, -1) && return -sqrt(45 / 16π) * sin(2θ) * cis(-ϕ)
    (s, l, m) == (+1, 2, 0) && return -sqrt(15 / 32π) * (1 + 3 * cos(2θ))
    (s, l, m) == (+1, 2, +1) && return sqrt(45 / 16π) * sin(2θ) * cis(ϕ)
    (s, l, m) == (+1, 2, +2) && return -sqrt(45 / 16π) * sin(θ)^2 * cis(2ϕ)

    (s, l, m) == (+2, 2, -2) && return sqrt(5 / 4π) * (1 + cos(θ)) * sin(θ) * cis(-2ϕ)
    (s, l, m) == (+2, 2, -1) && return sqrt(5 / 4π) * (cos(θ) + cos(2θ)) * cis(-ϕ)
    (s, l, m) == (+2, 2, 0) && return -sqrt(15 / 2π) * cos(θ) * sin(θ)
    (s, l, m) == (+2, 2, +1) && return sqrt(5 / 4π) * (cos(θ) - cos(2θ)) * cis(ϕ)
    (s, l, m) == (+2, 2, +2) && return -sqrt(5 / π) * sin(θ / 2)^2 * sin(θ) * cis(2ϕ)

    @assert false
end

################################################################################

# TODO: Test phase and parity of sYlm
#
# {\displaystyle Y_{\ell }^{m}(\theta ,\phi )\to Y_{\ell }^{m}(\pi -\theta ,\pi +\phi )=(-1)^{\ell }Y_{\ell }^{m}(\theta ,\phi )}
#
# <math display="block">Y_\ell^m(\theta,\phi) \to Y_\ell^m(\pi-\theta,\pi+\phi) = (-1)^\ell Y_\ell^m(\theta,\phi)</math>
#
# {}_s\bar Y_{l m} &= \left(-1\right)^{s+m}{}_{-s}Y_{l(-m)}\\
#
# {}_sY_{l m}(\pi-\theta,\phi+\pi) &= \left(-1\right)^l {}_{-s}Y_{l m}(\theta,\phi).

Random.seed!(100)
modes = [(name="(s=$s,l=$l,m=$m)", spin=s, el=l, m=m, fun=(θ, ϕ) -> sYlm(s, l, m, θ, ϕ), modes=(l′, m′) -> l′ == l && m′ == m)
         for s in -2:+2 for l in abs(s):2 for m in (-l):l]
@testset "Simple transforms: $(mode.name)" for mode in modes
    for lmax in (mode.el):(mode.el) #TODO 20
        sz = ash_grid_size(lmax)
        f = Array{Complex{Float64}}(undef, sz)
        for ij in CartesianIndices(sz)
            θ, ϕ = ash_point_coord(ij, lmax)
            f[ij] = mode.fun(θ, ϕ)
        end
        # function setvalue(ij::CartesianIndex{2})
        #     θ, ϕ = ash_point_coord(ij, lmax)
        #     return Complex{Float64}(mode.fun(θ, ϕ))
        # end
        # f = map(setvalue, CartesianIndices(sz))

        flm = ash_transform(f, mode.spin, lmax)
        @test all(isapprox(flm[ash_mode_index(mode.spin, l, m, lmax)], mode.modes(l, m); atol=100eps()) for l in
                                                                                                            abs(mode.spin):lmax
                  for m in (-l):l)

        f′ = ash_evaluate(flm, mode.spin, lmax)
        @test isapprox(f′, f; atol=100eps())
    end
end

Random.seed!(100)
@testset "Linearity" begin
    for iter in 1:100
        lmax = rand(0:100)
        sz = ash_grid_size(lmax)
        nmodes = ash_nmodes(lmax)

        spin = rand(-4:4)

        f = randn(Complex{Float64}, sz)
        g = randn(Complex{Float64}, sz)
        α = randn(Complex{Float64})
        h = f + α * g

        flm = ash_transform(f, spin, lmax)
        glm = ash_transform(g, spin, lmax)
        hlm = ash_transform(h, spin, lmax)

        if !(flm + α * glm ≈ hlm)
            @show iter lmax sz nmodes spin
            @show α
            @show any(isnan, f) any(isnan, g) any(isnan, h)
            @show any(isnan, flm) any(isnan, glm) any(isnan, hlm)
            @show f[1] g[1] h[1]
            @show flm[1] glm[1] hlm[1]
        end
        @test flm + α * glm ≈ hlm

        flm = randn(Complex{Float64}, nmodes)
        glm = randn(Complex{Float64}, nmodes)
        hlm = flm + α * glm

        f = ash_evaluate(flm, spin, lmax)
        g = ash_evaluate(glm, spin, lmax)
        h = ash_evaluate(hlm, spin, lmax)

        @test f + α * g ≈ h
    end
end

# Phase: conj(Ylm) = (-1)^m Yl(-m)

Random.seed!(100)
@testset "Orthonormality transforms" begin
    for iter in 1:100
        lmax = rand(0:100)
        nmodes = ash_nmodes(lmax)

        smax = min(4, lmax)
        spin = rand((-smax):smax)

        flm = zeros(Complex{Float64}, nmodes)
        glm = zeros(Complex{Float64}, nmodes)

        lf = rand(abs(spin):lmax)
        mf = rand((-lf):lf)
        lg = rand(abs(spin):lmax)
        mg = rand((-lg):lg)

        flm[ash_mode_index(spin, lf, mf, lmax)] = 1
        glm[ash_mode_index(spin, lg, mg, lmax)] = 1

        f = ash_evaluate(flm, spin, lmax)
        g = ash_evaluate(glm, spin, lmax)

        @test isapprox(integrate(f, f, lmax), 1; atol=1 / lmax^2)
        @test isapprox(integrate(f, g, lmax), (lf == lg) * (mf == mg); atol=1 / lmax^2)

        h = conj(f) .* f
        hlm = ash_transform(h, 0, lmax)
        @test isapprox(hlm[ash_mode_index(0, 0, 0, lmax)], sqrt(1 / 4π); atol=sqrt(eps()))

        h = conj(f) .* g
        hlm = ash_transform(h, 0, lmax)
        @test isapprox(hlm[ash_mode_index(0, 0, 0, lmax)], (lf == lg) * (mf == mg) * sqrt(1 / 4π); atol=sqrt(eps()))
    end
end

Random.seed!(100)
modes = [(name="(s=$s,l=$l,m=$m)", spin=s, el=l, fun=(θ, ϕ) -> sYlm(s, l, m, θ, ϕ), ðfun=(θ, ϕ) -> ðsYlm(s, l, m, θ, ϕ),
          ð̄fun=(θ, ϕ) -> ð̄sYlm(s, l, m, θ, ϕ)) for s in 0:+2 for l in abs(s):2 for m in (-l):l]
@testset "Simple derivatives (eth, eth-bar): $(mode.name)" for mode in modes
    for lmax in (mode.el):20
        sz = ash_grid_size(lmax)
        f = Array{Complex{Float64}}(undef, sz)
        ðf₀ = Array{Complex{Float64}}(undef, sz)
        ð̄f₀ = Array{Complex{Float64}}(undef, sz)
        for ij in CartesianIndices(sz)
            θ, ϕ = ash_point_coord(ij, lmax)
            f[ij] = mode.fun(θ, ϕ)
            ðf₀[ij] = mode.ðfun(θ, ϕ)
            ð̄f₀[ij] = mode.ð̄fun(θ, ϕ)
        end

        flm = ash_transform(f, mode.spin, lmax)

        ðflm = ash_eth(flm, mode.spin, lmax)
        ðf = ash_evaluate(ðflm, mode.spin + 1, lmax)
        @test isapprox(ðf, ðf₀; atol=10000eps())

        ð̄flm = ash_ethbar(flm, mode.spin, lmax)
        ð̄f = ash_evaluate(ð̄flm, mode.spin - 1, lmax)
        @test isapprox(ð̄f, ð̄f₀; atol=10000eps())
    end
end

Random.seed!(100)
@testset "Eigenvectors of Laplacian" begin
    for iter in 1:100
        lmax = rand(0:100)
        nmodes = ash_nmodes(lmax)

        flm = zeros(Complex{Float64}, nmodes)

        l = rand(0:lmax)
        m = rand((-l):l)
        flm[ash_mode_index(0, l, m, lmax)] = 1

        ðflm = ash_eth(flm, 0, lmax)
        ð̄ðflm = ash_ethbar(ðflm, +1, lmax)

        ð̄flm = ash_ethbar(flm, 0, lmax)
        ðð̄flm = ash_eth(ð̄flm, -1, lmax)

        f = ash_evaluate(flm, 0, lmax)
        ð̄ðf = ash_evaluate(ð̄ðflm, 0, lmax)
        ðð̄f = ash_evaluate(ðð̄flm, 0, lmax)

        @test isapprox(ð̄ðf, -l * (l + 1) * f; atol=(lmax + 1)^2 * 100eps())
        @test isapprox(ðð̄f, -l * (l + 1) * f; atol=(lmax + 1)^2 * 100eps())
    end
end
