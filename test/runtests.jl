using AbstractSphericalHarmonics
using LinearAlgebra: norm
using Random
using StaticArrays
using Test
import FastSphericalHarmonics
import SSHT

const ASH = AbstractSphericalHarmonics
const bitsign = ASH.bitsign

chop(x) = abs2(x) < 100eps(x) ? zero(x) : x
chop(x::Complex) = Complex(chop(real(x)), chop(imag(x)))
chop(x::SArray) = chop.(x)

function integrate(f::AbstractMatrix, g::AbstractMatrix, grid::SphereGrid)
    sz = ash_grid_size(grid)
    @assert size(f) == size(g) == sz

    s = zero(eltype(f)) * zero(eltype(g)) * zero(Float64)
    for pt in CartesianIndices(sz)
        theta, phi = ash_point_coord(grid, pt)
        dtheta, dphi = ash_point_delta(grid, pt)
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

function rand_tensor(::Val{D}, ::Type{T}, grid::SphereGrid) where {D,T}
    sz = ash_grid_size(grid)
    Dims = Tuple{[2 for d in 1:D]...}
    f = randn(SArray{Dims,T}, sz)
    return f
end

function const_tensor(::Val{D}, ::Type{T}, grid::SphereGrid) where {D,T}
    sz = ash_grid_size(grid)
    Dims = Tuple{[2 for d in 1:D]...}
    α = randn(SArray{Dims,T})
    f = fill(α, sz)
    return f
end

################################################################################
# Pointwise sYlm evaluation (backend-independent)

Random.seed!(100)
@testset "sYlm accuracy: small l vs closed forms" begin
    # Closed forms from the table above, including the poles
    for s in -2:2, l in abs(s):2, m in (-l):l, θ in (0.0, 0.3, π / 2, π - 0.3, 1.0π), ϕ in (0.0, 0.7, 2.1)
        @test isapprox(ASH.sYlm(s, l, m, θ, ϕ), sYlm(s, l, m, θ, ϕ); atol=10eps())
    end
    # Independent naive implementation (exact in BigFloat, breaks at poles)
    for s in -4:4, l in abs(s):10, m in (-l):l
        θ, ϕ = π * (0.05 + 0.9 * rand()), 2π * rand()
        @test isapprox(ASH.sYlm(s, l, m, θ, ϕ), Complex{Float64}(ASH.sYlm0(Val(s), Val(l), Val(m), big(θ), big(ϕ)));
                       atol=100eps())
    end
end

Random.seed!(100)
@testset "sYlm accuracy: large l" begin
    # The reference itself suffers cancellation of ~2l bits; give it headroom
    bigref(s, l, m, θ, ϕ) = setprecision(BigFloat, 2l + 512) do
        Complex{Float64}(ASH.sYlm0(Val(s), Val(l), Val(m), big(θ), big(ϕ)))
    end
    for l in (40, 60, 80, 200, 400)
        s = rand(-4:4)
        for m in unique((0, 1, -1, l, -l, l ÷ 2, -l ÷ 2)), θ in (0.01, 1.234, π - 0.01)
            ϕ = 2π * rand()
            @test isapprox(ASH.sYlm(s, l, m, θ, ϕ), bigref(s, l, m, θ, ϕ); atol=l * 1000eps(), rtol=l * 1000eps())
        end
    end
    # Parity at large l
    for l in (100, 300), iter in 1:5
        s = rand(-4:4)
        m = rand((-l):l)
        θ, ϕ = π * rand(), 2π * rand()
        @test conj(ASH.sYlm(s, l, m, θ, ϕ)) ≈ bitsign(s + m) * ASH.sYlm(-s, l, -m, θ, ϕ)
    end
    # Generic over the argument type
    let θ = big(1.234), ϕ = big(0.567)
        v = ASH.sYlm(2, 300, 150, θ, ϕ)
        @test v isa Complex{BigFloat}
        vref = setprecision(() -> ASH.sYlm0(Val(2), Val(300), Val(150), θ, ϕ), BigFloat, 4096)
        @test abs(v - vref) ≤ 1000 * eps(BigFloat)
    end
end

@testset "sYlm at the poles" begin
    for s in -3:3, l in (abs(s), 10, 100)
        for m in unique((-l, -s, 0, s, l))
            abs(m) ≤ l || continue
            f0 = ASH.sYlm(s, l, m, 0.0, 0.3)
            @test isfinite(f0)
            # Only m = -s is nonzero at θ = 0 (exact: sin(0/2) is exactly zero)
            m ≠ -s && @test f0 == 0
        end
    end
end

@testset "sYlm error paths" begin
    @test_throws DomainError ASH.sYlm(3, 2, 0, 1.0, 1.0)
    @test_throws DomainError ASH.sYlm(0, 2, 5, 1.0, 1.0)
    @test_throws DomainError ASH.sYlm(0, 2, -3, 1.0, 1.0)
end

################################################################################

backends = [
    (name="DriscollHealy (SSHT)", mkgrid=DriscollHealyGrid, maxl=100),
    (name="Equiangular (FastSphericalHarmonics)", mkgrid=EquiangularGrid, maxl=32),
]

@testset "Backend: $(backend.name)" for backend in backends
mkgrid = backend.mkgrid
maxl = backend.maxl

@testset "Mode indices" begin
    for lmax in 0:20
        grid = mkgrid(lmax)
        nmodes = ash_nmodes(grid)
        for s in -4:4, l in abs(s):lmax, m in (-l):l
            ind = ash_mode_index(grid, s, l, m)
            @test length(nmodes) ≡ length(ind)
            @test all(1 ≤ ind[d] ≤ nmodes[d] for d in 1:length(nmodes))
            l′, m′ = ash_mode_numbers(grid, s, ind)
            @test l′ == l && m′ == m
        end
    end

    let grid = mkgrid(4)
        @test_throws DomainError ash_mode_index(grid, 0, 5, 0)   # l > lmax
        @test_throws DomainError ash_mode_index(grid, 2, 1, 0)   # l < |s|
        @test_throws DomainError ash_mode_index(grid, 0, 2, 3)   # |m| > l
        @test_throws DimensionMismatch Tensor{1}(fill(SMatrix{2,2,Complex{Float64}}(1, 0, 0, 1), ash_grid_size(grid)), grid)
    end
end

Random.seed!(100)
modes = [
    (name="(s=$s,l=$l,m=$m)", spin=s, el=l, m=m, fun=(θ, ϕ) -> sYlm(s, l, m, θ, ϕ), modes=(l′, m′) -> l′ == l && m′ == m) for
    s in -2:+2 for l in abs(s):2 for m in (-l):l
]
@testset "Simple transforms: $(mode.name)" for mode in modes
    for lmax in (mode.el):20
        grid = mkgrid(lmax)
        sz = ash_grid_size(grid)
        f = Array{Complex{Float64}}(undef, sz)
        for ij in CartesianIndices(sz)
            θ, ϕ = ash_point_coord(grid, ij)
            f[ij] = mode.fun(θ, ϕ)
        end
        # function setvalue(ij::CartesianIndex{2})
        #     θ, ϕ = ash_point_coord(grid, ij)
        #     return Complex{Float64}(mode.fun(θ, ϕ))
        # end
        # f = map(setvalue, CartesianIndices(sz))

        flm = ash_transform(grid, f, mode.spin)
        @test all(
            isapprox(flm[ash_mode_index(grid, mode.spin, l, m)], mode.modes(l, m); atol=100eps()) for l in abs(mode.spin):lmax for
            m in (-l):l
        )

        f′ = ash_evaluate(grid, flm, mode.spin)
        @test isapprox(f′, f; atol=1000eps())
    end
end

Random.seed!(100)
@testset "Parity" begin
    for iter in 1:20
        lmax = rand(0:maxl)
        grid = mkgrid(lmax)
        sz = ash_grid_size(grid)
        nmodes = ash_nmodes(grid)

        s = rand(-4:4)
        abs(s) ≤ lmax || continue
        l = rand(abs(s):lmax)
        m = rand((-l):l)

        flm = zeros(Complex{Float64}, nmodes)
        flm[ash_mode_index(grid, s, l, m)] = 1

        flm′ = zeros(Complex{Float64}, nmodes)
        flm′[ash_mode_index(grid, -s, l, -m)] = 1

        f = ash_evaluate(grid, flm, s)
        f′ = ash_evaluate(grid, flm′, -s)

        # Phase: conj(sYlm) = (-1)^(s+m) (-s)Yl(-m)
        @test conj(f) ≈ bitsign(s + m) * f′
    end
end

Random.seed!(100)
@testset "Linearity" begin
    for iter in 1:20
        lmax = rand(0:maxl)
        grid = mkgrid(lmax)
        sz = ash_grid_size(grid)
        nmodes = ash_nmodes(grid)

        spin = rand(-4:4)

        f = randn(Complex{Float64}, sz)
        g = randn(Complex{Float64}, sz)
        α = randn(Complex{Float64})
        h = f + α * g

        flm = ash_transform(grid, f, spin)
        glm = ash_transform(grid, g, spin)
        hlm = ash_transform(grid, h, spin)

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

        f = ash_evaluate(grid, flm, spin)
        g = ash_evaluate(grid, glm, spin)
        h = ash_evaluate(grid, hlm, spin)

        @test f + α * g ≈ h
    end
end

Random.seed!(100)
@testset "Orthonormality" begin
    for iter in 1:20
        lmax = rand(0:maxl)
        grid = mkgrid(lmax)
        nmodes = ash_nmodes(grid)

        smax = min(4, lmax ÷ 2)
        spin = rand((-smax):smax)

        flm = zeros(Complex{Float64}, nmodes)
        glm = zeros(Complex{Float64}, nmodes)

        # The produce will have lh = lf + lg
        lf = rand(abs(spin):(lmax ÷ 2))
        mf = rand((-lf):lf)
        lg = rand(abs(spin):(lmax - lf))
        mg = rand((-lg):lg)

        flm[ash_mode_index(grid, spin, lf, mf)] = 1
        glm[ash_mode_index(grid, spin, lg, mg)] = 1

        f = ash_evaluate(grid, flm, spin)
        g = ash_evaluate(grid, glm, spin)

        @test isapprox(integrate(f, f, grid), 1; atol=20 / (lmax + 1)^2)
        @test isapprox(integrate(f, g, grid), (lf == lg) * (mf == mg); atol=1 / (lmax + 1)^2)

        h = conj(f) .* f
        hlm = ash_transform(grid, h, 0)
        @test isapprox(hlm[ash_mode_index(grid, 0, 0, 0)], sqrt(1 / 4π); atol=sqrt(eps()))

        h = conj(f) .* g
        hlm = ash_transform(grid, h, 0)
        @test isapprox(hlm[ash_mode_index(grid, 0, 0, 0)], (lf == lg) * (mf == mg) * sqrt(1 / 4π); atol=sqrt(eps()))
    end
end

Random.seed!(100)
modes = [
    (
        name="(s=$s,l=$l,m=$m)",
        spin=s,
        el=l,
        fun=(θ, ϕ) -> sYlm(s, l, m, θ, ϕ),
        ðfun=(θ, ϕ) -> ðsYlm(s, l, m, θ, ϕ),
        ð̄fun=(θ, ϕ) -> ð̄sYlm(s, l, m, θ, ϕ),
    ) for s in 0:+2 for l in abs(s):2 for m in (-l):l
]
@testset "Simple derivatives (eth, eth-bar): $(mode.name)" for mode in modes
    for lmax in (mode.el):20
        grid = mkgrid(lmax)
        sz = ash_grid_size(grid)
        f = Array{Complex{Float64}}(undef, sz)
        ðf₀ = Array{Complex{Float64}}(undef, sz)
        ð̄f₀ = Array{Complex{Float64}}(undef, sz)
        for ij in CartesianIndices(sz)
            θ, ϕ = ash_point_coord(grid, ij)
            f[ij] = mode.fun(θ, ϕ)
            ðf₀[ij] = mode.ðfun(θ, ϕ)
            ð̄f₀[ij] = mode.ð̄fun(θ, ϕ)
        end

        flm = ash_transform(grid, f, mode.spin)

        ðflm = ash_eth(grid, flm, mode.spin)
        ðf = ash_evaluate(grid, ðflm, mode.spin + 1)
        @test isapprox(ðf, ðf₀; atol=10000eps())

        ð̄flm = ash_ethbar(grid, flm, mode.spin)
        ð̄f = ash_evaluate(grid, ð̄flm, mode.spin - 1)
        @test isapprox(ð̄f, ð̄f₀; atol=10000eps())
    end
end

Random.seed!(100)
@testset "Eigenvectors of Laplacian" begin
    for iter in 1:20
        lmax = rand(0:maxl)
        grid = mkgrid(lmax)
        nmodes = ash_nmodes(grid)

        flm = zeros(Complex{Float64}, nmodes)

        l = rand(0:lmax)
        m = rand((-l):l)
        flm[ash_mode_index(grid, 0, l, m)] = 1

        ðflm = ash_eth(grid, flm, 0)
        ð̄ðflm = ash_ethbar(grid, ðflm, +1)

        ð̄flm = ash_ethbar(grid, flm, 0)
        ðð̄flm = ash_eth(grid, ð̄flm, -1)

        f = ash_evaluate(grid, flm, 0)
        ð̄ðf = ash_evaluate(grid, ð̄ðflm, 0)
        ðð̄f = ash_evaluate(grid, ðð̄flm, 0)

        @test isapprox(ð̄ðf, -l * (l + 1) * f; atol=(lmax + 1)^2 * 100eps())
        @test isapprox(ðð̄f, -l * (l + 1) * f; atol=(lmax + 1)^2 * 100eps())
    end
end

Random.seed!(100)
@testset "Spin-s Laplacian and eth commutator" begin
    # ð̄ð ₛYₗₘ = -(l-s)(l+s+1) ₛYₗₘ,  ðð̄ ₛYₗₘ = -(l+s)(l-s+1) ₛYₗₘ,
    # hence (ð̄ð - ðð̄) f = 2s f for any spin-s field f.
    # These are exact mode-space identities.
    for iter in 1:20
        lmax = rand(0:maxl)
        grid = mkgrid(lmax)
        nmodes = ash_nmodes(grid)

        s = rand(-4:4)
        abs(s) ≤ lmax || continue

        flm = randn(Complex{Float64}, nmodes)
        # zero the unused l < |s| entries
        for l in 0:(abs(s) - 1), m in (-l):l
            flm[CartesianIndex(l^2 + l + m + 1)] = 0
        end

        ð̄ðflm = ash_ethbar(grid, ash_eth(grid, flm, s), s + 1)
        ðð̄flm = ash_eth(grid, ash_ethbar(grid, flm, s), s - 1)

        for l in abs(s):lmax, m in (-l):l
            ind = ash_mode_index(grid, s, l, m)
            @test ð̄ðflm[ind] ≈ -(l - s) * (l + s + 1) * flm[ind]
            @test ðð̄flm[ind] ≈ -(l + s) * (l - s + 1) * flm[ind]
        end
        @test ð̄ðflm - ðð̄flm ≈ 2 * s * flm
    end
end

@testset "Quadrature weights" begin
    # ∮ 1 = Σ sin(θ) dθ dϕ = 4π
    for lmax in (0, 1, 2, 5, 20, maxl)
        grid = mkgrid(lmax)
        q = 0.0
        for ij in CartesianIndices(ash_grid_size(grid))
            θ, _ = ash_point_coord(grid, ij)
            dθ, dϕ = ash_point_delta(grid, ij)
            q += sin(θ) * dθ * dϕ
        end
        @test isapprox(q, 4π; rtol=1000eps())
    end
end

################################################################################

Random.seed!(100)
@testset "Arbitrary modes" begin
    for iter in 1:20
        lmax = rand(0:maxl)
        grid = mkgrid(lmax)
        nmodes = ash_nmodes(grid)

        flm = zeros(Complex{Float64}, nmodes)

        s = rand(-4:4)
        abs(s) ≤ lmax || continue
        l = rand(abs(s):lmax)
        m = rand((-l):l)
        flm[ash_mode_index(grid, s, l, m)] = 1

        f = ash_evaluate(grid, flm, s)

        f′ = [
            begin
                θ, ϕ = ash_point_coord(grid, ij)
                ASH.sYlm(s, l, m, θ, ϕ)
            end for ij in CartesianIndices(f)
        ]

        @test f ≈ f′
    end
end

################################################################################

Random.seed!(100)
@testset "Tensors on the sphere (rank $D)" for D in 0:4
    for iter in 1:20
        lmax = rand(0:maxl)
        grid = mkgrid(lmax)

        f = rand_tensor(Val(D), Complex{Float64}, grid)

        t = Tensor{D}(f, grid)
        st = SpinTensor{D}(t)
        t′ = Tensor{D}(st)
        st′ = SpinTensor{D}(t′)
        t″ = Tensor{D}(st′)

        @test st′ ≈ st
        @test t″ ≈ t′
    end
end

Random.seed!(100)
@testset "Linearity of tensors on the sphere (rank $D)" for D in 0:4
    for iter in 1:20
        lmax = rand(0:maxl)
        grid = mkgrid(lmax)

        f = rand_tensor(Val(D), Complex{Float64}, grid)
        g = rand_tensor(Val(D), Complex{Float64}, grid)
        α = randn(Complex{Float64})

        h = f + α * g

        t = Tensor{D}(f, grid)
        u = Tensor{D}(g, grid)
        v = Tensor{D}(h, grid)

        st = SpinTensor{D}(t)
        su = SpinTensor{D}(u)
        sv = SpinTensor{D}(v)

        @test sv ≈ st + α * su
    end
end

Random.seed!(100)
@testset "Conjugation of tensors on the sphere (rank $D)" for D in 0:4
    for iter in 1:10
        lmax = rand(0:min(20, maxl))
        grid = mkgrid(lmax)

        t = Tensor{D}(rand_tensor(Val(D), Complex{Float64}, grid), grid)
        u = Tensor{D}(rand_tensor(Val(D), Complex{Float64}, grid), grid)
        α = randn(Complex{Float64})

        st = SpinTensor{D}(t)
        su = SpinTensor{D}(u)

        # conj represents the conjugated field
        @test isapprox(Tensor{D}(conj(st)), conj(Tensor{D}(st)); atol=(lmax + 1)^2 * 100eps())
        # involution
        @test conj(conj(st)) ≈ st
        # antilinearity
        @test conj(st + α * su) ≈ conj(st) + conj(α) * conj(su)
    end

    if D == 0
        # Single-mode coefficient check: conj(f)_{l,m} = (-1)^m conj(f_{l,-m})
        lmax = 8
        grid = mkgrid(lmax)
        nmodes = ash_nmodes(grid)
        for (l, m) in ((0, 0), (3, 2), (5, -4), (8, 8))
            c = 0.7 + 1.3im
            coeffs = zeros(Complex{Float64}, nmodes)
            coeffs[ash_mode_index(grid, 0, l, m)] = c
            expected = zeros(Complex{Float64}, nmodes)
            expected[ash_mode_index(grid, 0, l, -m)] = bitsign(m) * conj(c)
            @test conj(SpinTensor(coeffs, grid)).coeffs[] == expected
        end
    end
end

Random.seed!(100)
@testset "Real-valued tensors on the sphere (rank $D)" for D in 0:2
    for iter in 1:5
        lmax = rand(0:min(20, maxl))
        grid = mkgrid(lmax)

        f = rand_tensor(Val(D), Float64, grid)
        t = Tensor{D}(f, grid)
        tc = Tensor{D}(map(x -> Complex.(x), f), grid)

        st = SpinTensor{D}(t)
        stc = SpinTensor{D}(tc)
        @test st ≈ stc

        # round trip: the projected real field reproduces itself
        t′ = Tensor{D}(st)
        st′ = SpinTensor{D}(t′)
        @test st′ ≈ st
    end
end

@testset "Simple derivatives of tensors on the sphere" begin
    for lmax in 1:20
        grid = mkgrid(lmax)
        sz = ash_grid_size(grid)

        # 1

        s = Tensor{0}(fill(Scalar{Complex{Float64}}(1), sz), grid)
        ds₀ = Tensor{1}(fill(SVector{2,Complex{Float64}}(0, 0), sz), grid)

        s̃ = SpinTensor(s)
        ds̃ = tensor_gradient(s̃)
        ds = Tensor(ds̃)

        @test isapprox(ds, ds₀; atol=(lmax + 1)^2 * 1000eps())

        # x

        x = Tensor{0}([
            begin
                θ, ϕ = ash_point_coord(grid, ij)
                Scalar{Complex{Float64}}(sin(θ) * cos(ϕ))
            end for ij in CartesianIndices(sz)
        ], grid)
        dx₀ = Tensor{1}([
            begin
                θ, ϕ = ash_point_coord(grid, ij)
                SVector{2,Complex{Float64}}(cos(θ) * cos(ϕ), -sin(ϕ))
            end for ij in CartesianIndices(sz)
        ], grid)

        x̃ = SpinTensor(x)
        dx̃ = tensor_gradient(x̃)
        dx = Tensor(dx̃)

        @test isapprox(dx, dx₀; atol=(lmax + 1)^2 * 100eps())

        # y

        y = Tensor{0}([
            begin
                θ, ϕ = ash_point_coord(grid, ij)
                Scalar{Complex{Float64}}(sin(θ) * sin(ϕ))
            end for ij in CartesianIndices(sz)
        ], grid)
        dy₀ = Tensor{1}([
            begin
                θ, ϕ = ash_point_coord(grid, ij)
                SVector{2,Complex{Float64}}(cos(θ) * sin(ϕ), cos(ϕ))
            end for ij in CartesianIndices(sz)
        ], grid)

        ỹ = SpinTensor(y)
        dỹ = tensor_gradient(ỹ)
        dy = Tensor(dỹ)

        @test isapprox(dy, dy₀; atol=(lmax + 1)^2 * 100eps())

        # z

        z = Tensor{0}([
            begin
                θ, ϕ = ash_point_coord(grid, ij)
                Scalar{Complex{Float64}}(cos(θ))
            end for ij in CartesianIndices(sz)
        ], grid)
        dz₀ = Tensor{1}([
            begin
                θ, ϕ = ash_point_coord(grid, ij)
                SVector{2,Complex{Float64}}(-sin(θ), 0)
            end for ij in CartesianIndices(sz)
        ], grid)

        z̃ = SpinTensor(z)
        dz̃ = tensor_gradient(z̃)
        dz = Tensor(dz̃)

        @test isapprox(dz, dz₀; atol=(lmax + 1)^2 * 100eps())

        if lmax ≥ 2

            # grad x
            #
            # q_θθ = 1
            # q_ϕϕ = sin(ϕ)^2
            #
            # Γ^θ_ϕϕ = - cos(θ) sin(θ)
            # Γ^ϕ_θϕ = cos(θ) / sin(θ)
            # Γ^ϕ_ϕθ = cos(θ) / sin(θ)
            #
            # x = sin(θ) cos(ϕ)
            # y = sin(θ) sin(ϕ)
            # z = cos(θ)
            #
            # ∇x_θ = cos(θ) cos(ϕ)
            # ∇x_ϕ = - sin(θ) sin(ϕ)
            #
            # ∇∇x_θθ = - sin(θ) cos(ϕ)
            # ∇∇x_ϕϕ = - sin(θ)^3 cos(ϕ)

            gradx = Tensor{1}([
                begin
                    θ, ϕ = ash_point_coord(grid, ij)
                    SVector{2,Complex{Float64}}(cos(θ) * cos(ϕ), -sin(ϕ))
                end for ij in CartesianIndices(sz)
            ], grid)
            dgradx₀ = Tensor{2}(
                [
                    begin
                        θ, ϕ = ash_point_coord(grid, ij)
                        SMatrix{2,2,Complex{Float64}}(-sin(θ) * cos(ϕ), 0, 0, -sin(θ) * cos(ϕ))
                    end for ij in CartesianIndices(sz)
                ], grid
            )

            gradx̃ = SpinTensor(gradx)
            dgradx̃ = tensor_gradient(gradx̃)
            dgradx = Tensor(dgradx̃)

            @test isapprox(dgradx, dgradx₀; atol=(lmax + 1)^2 * 100eps())

            ddgradx₀ = Tensor{3}(
                [
                    begin
                        θ, ϕ = ash_point_coord(grid, ij)
                        SArray{Tuple{2,2,2},Complex{Float64}}(-cos(θ) * cos(ϕ), 0, 0, -cos(θ) * cos(ϕ), sin(ϕ), 0, 0, sin(ϕ))
                    end for ij in CartesianIndices(sz)
                ],
                grid,
            )

            ddgradx̃ = tensor_gradient(dgradx̃)
            ddgradx = Tensor(ddgradx̃)

            @test isapprox(ddgradx, ddgradx₀; atol=(lmax + 1)^2 * 1000eps())

            # curl x
            #
            # curl x_θ = sin(ϕ)
            # curl x_ϕ = cos(θ) sin(θ) cos(ϕ)
            #
            # ∇ curl x_θθ = - sin(θ) cos(ϕ)
            # ∇ curl x_ϕϕ = - sin(θ)^3 cos(ϕ)

            curlx = Tensor{1}([
                begin
                    θ, ϕ = ash_point_coord(grid, ij)
                    SVector{2,Complex{Float64}}(sin(ϕ), cos(θ) * cos(ϕ))
                end for ij in CartesianIndices(sz)
            ], grid)
            dcurlx₀ = Tensor{2}(
                [
                    begin
                        θ, ϕ = ash_point_coord(grid, ij)
                        SMatrix{2,2,Complex{Float64}}(0, -sin(θ) * cos(ϕ), sin(θ) * cos(ϕ), 0)
                    end for ij in CartesianIndices(sz)
                ], grid
            )

            curlx̃ = SpinTensor(curlx)
            dcurlx̃ = tensor_gradient(curlx̃)
            dcurlx = Tensor(dcurlx̃)

            @test isapprox(dcurlx, dcurlx₀; atol=(lmax + 1)^2 * 100eps())
        end
    end
end

Random.seed!(100)
@testset "Derivatives of tensors on the sphere (rank $D)" for D in 0:3
    for iter in 1:20
        lmax = rand(0:maxl)
        grid = mkgrid(lmax)

        f = rand_tensor(Val(D), Complex{Float64}, grid)
        g = rand_tensor(Val(D), Complex{Float64}, grid)
        α = randn(Complex{Float64})

        # Linear combination
        s = f + α * g

        tf = Tensor{D}(f, grid)
        tg = Tensor{D}(g, grid)
        ts = Tensor{D}(s, grid)

        stf = SpinTensor{D}(tf)
        stg = SpinTensor{D}(tg)
        sts = SpinTensor{D}(ts)

        dstf = tensor_gradient(stf)
        dstg = tensor_gradient(stg)
        dsts = tensor_gradient(sts)

        dtf = Tensor{D + 1}(dstf)
        dtg = Tensor{D + 1}(dstg)
        dts = Tensor{D + 1}(dsts)

        @test dts ≈ dtf + α * dtg

        dstf′ = zero(dstf)
        dstg′ = zero(dstg)
        dsts′ = zero(dsts)
        tensor_gradient!(dstf′, stf)
        tensor_gradient!(dstg′, stg)
        tensor_gradient!(dsts′, sts)
        @test dstf′ ≈ dstf
        @test dstg′ ≈ dstg
        @test dsts′ ≈ dsts

        if D == 0
            # Constant function (derivative is zero)
            c = const_tensor(Val(D), Complex{Float64}, grid)

            tc = Tensor{D}(c, grid)
            stc = SpinTensor{D}(tc)
            dstc = tensor_gradient(stc)
            dtc = Tensor{D + 1}(dstc)

            dtc′ = Tensor{D + 1}(zero(dtc.values), grid)
            @test isapprox(dtc, dtc′; atol=sqrt(eps()))

            # Product rule (Leibniz). The inputs must be band-limited to
            # lmax÷2 so that their product is band-limited to lmax and hence
            # exactly representable on the grid.
            lcut = lmax ÷ 2
            fblm = zeros(Complex{Float64}, ash_nmodes(grid))
            gblm = zeros(Complex{Float64}, ash_nmodes(grid))
            for l in 0:lcut, m in (-l):l
                fblm[ash_mode_index(grid, 0, l, m)] = randn(Complex{Float64})
                gblm[ash_mode_index(grid, 0, l, m)] = randn(Complex{Float64})
            end
            fb = ash_evaluate(grid, fblm, 0)
            gb = ash_evaluate(grid, gblm, 0)

            tfb = Tensor{0}(fb, grid)
            tgb = Tensor{0}(gb, grid)
            tpb = Tensor{0}(fb .* gb, grid)

            dtfb = Tensor{1}(tensor_gradient(SpinTensor{0}(tfb)))
            dtgb = Tensor{1}(tensor_gradient(SpinTensor{0}(tgb)))
            dtpb = Tensor{1}(tensor_gradient(SpinTensor{0}(tpb)))

            dtpb′ = Tensor{1}(map((df, x) -> df * x[], dtfb.values, tgb.values) +
                              map((x, dg) -> x[] * dg, tfb.values, dtgb.values), grid)
            # f, g ~ lcut, d(fg) ~ lmax lcut², plus an lmax² factor for the
            # conditioning of the transforms
            @test isapprox(dtpb, dtpb′; atol=(lmax + 1)^4 * (lcut + 1) * 100eps())
        end
    end
end

Random.seed!(100)
@testset "Filtering tensors on the sphere D=$D" for D in 0:4
    for lmax in 0:20
        grid = mkgrid(lmax)
        sz = ash_grid_size(grid)
        t = Tensor{D}(rand_tensor(Val(D), Complex{Float64}, grid), grid)
        st = SpinTensor(t)
        st′ = filter_modes(st)
        # This is only a weak test
        lfilter = lmax * 2 ÷ 3
        for (ij, cs) in zip(CartesianIndices(st′.coeffs), st′.coeffs)
            s = count(Tuple(ij) .== 1) - count(Tuple(ij) .== 2)
            for l in abs(s):lmax, m in (-l):l
                if l ≤ lfilter
                    @test cs[ash_mode_index(grid, s, l, m)] ≠ 0
                else
                    @test abs(cs[ash_mode_index(grid, s, l, m)]) ≤ 1000eps()
                end
            end
        end
        # Default cutoff is the two-thirds rule
        @test filter_modes(st) == filter_modes(st; lfilter=lmax * 2 ÷ 3)
        # Filtering at lmax is the identity
        @test filter_modes(st; lfilter=lmax) == st
        # Filtering is a projection
        @test filter_modes(filter_modes(st)) == filter_modes(st)
        # Filtering is linear
        su = SpinTensor(Tensor{D}(rand_tensor(Val(D), Complex{Float64}, grid), grid))
        α = randn(Complex{Float64})
        @test filter_modes(st + α * su) ≈ filter_modes(st) + α * filter_modes(su)
        # Filtering commutes with the gradient (the eth operators are
        # diagonal in l)
        if D ≤ 2
            @test filter_modes(tensor_gradient(st)) ≈ tensor_gradient(filter_modes(st))
        end
    end
end

Random.seed!(100)
@testset "Calculate Ricci scalar" begin
    for iter in 1:20
        lmax = rand(2:maxl)
        grid = mkgrid(lmax)
        sz = ash_grid_size(grid)

        # q_ab = (1, 0, 0, 1)
        q = Tensor{2}(fill(SMatrix{2,2,Complex{Float64}}(1, 0, 0, 1), sz), grid)
        q̃ = SpinTensor(q)
        # qu^ab
        qu = q
        # q_ab,c
        dq̃ = tensor_gradient(q̃)
        dq = Tensor(dq̃)
        # Γ^a_bc = q^ad (q_dc,b + q_bd,c - q_bc,d) / 2
        Γ = Tensor{3}(
            [
                SArray{Tuple{2,2,2}}(
                    sum(qu[a, d] * (dq[d, c, b] + dq[b, d, c] - dq[b, c, d]) / 2 for d in 1:2) for a in 1:2, b in 1:2, c in 1:2
                ) for (qu, dq) in zip(qu.values, dq.values)
            ],
            grid,
        )
        # dΓ^a_bc,d
        Γ̃ = SpinTensor(Γ)
        dΓ̃ = tensor_gradient(Γ̃)
        dΓ = Tensor(dΓ̃)
        # R_ab = dΓ^c_ab,c - dΓ^c_ac,b + Γ^c_ab Γ^d_cd - Γ^c_ad Γ^d_bc
        R = Tensor{2}(
            [
                SMatrix{2,2}(
                    sum(dΓ[c, a, b, c] - dΓ[c, a, c, b] for c in 1:2) +
                    sum(Γ[c, a, b] * Γ[d, c, d] - Γ[c, a, d] * Γ[d, b, c] for c in 1:2, d in 1:2) for a in 1:2, b in 1:2
                ) for (Γ, dΓ) in zip(Γ.values, dΓ.values)
            ],
            grid,
        )
        Rsc = Tensor{0}([Scalar(sum(qu[a, b] * R[a, b] for a in 1:2, b in 1:2)) for (qu, R) in zip(qu.values, R.values)], grid)

        @test isapprox(map(x -> x[], Rsc.values), zeros(sz); atol=(lmax + 1)^4 * 10eps())

        # println("q:")
        # display(map(x -> chop.(x), q.values))
        # println()
        # println("Γ:")
        # display(map(x -> chop.(x), Γ.values))
        # println()
        # println("R:")
        # display(map(x -> chop.(x), R.values))
        # println()
        # println("Rsc:")
        # display(map(x -> chop.(x)[], Rsc.values))
        # println()
    end
end

end # backend loop

################################################################################

Random.seed!(100)
@testset "Cross-backend consistency" begin
    for iter in 1:10
        lmax = rand(0:32)
        grids = (DriscollHealyGrid(lmax), EquiangularGrid(lmax))

        s = rand(-4:4)
        # need 2 max(|s|, l) ≤ lmax for exact product quadrature below
        abs(s) ≤ lmax ÷ 2 || continue

        # random coefficients, band-limited to lmax÷2 so that quadrature of
        # products is exact on both grids (the equiangular Fejér rule is
        # exact only up to degree lmax)
        flm = zeros(Complex{Float64}, ash_nmodes(grids[1]))
        for l in abs(s):max(abs(s), lmax ÷ 2), m in (-l):l
            flm[ash_mode_index(grids[1], s, l, m)] = randn(Complex{Float64})
        end

        # evaluate on each grid, transform back: coefficients must agree
        flm1 = ash_transform(grids[1], ash_evaluate(grids[1], flm, s), s)
        flm2 = ash_transform(grids[2], ash_evaluate(grids[2], flm, s), s)
        @test maximum(abs.(flm1 - flm2)) ≤ (lmax + 1)^2 * 1000eps()

        # quadrature of |f|² agrees between the backends and with Parseval
        n2 = sum(abs2, flm)
        for grid in grids
            f = ash_evaluate(grid, flm, s)
            q = 0.0
            for ij in CartesianIndices(f)
                θ, _ = ash_point_coord(grid, ij)
                dθ, dϕ = ash_point_delta(grid, ij)
                q += abs2(f[ij]) * sin(θ) * dθ * dϕ
            end
            @test isapprox(q, n2; atol=(lmax + 1)^2 * 1000eps() * max(1, n2))
        end
    end

    # the same analytic field on both grids has the same coefficients
    for (s, l, m) in ((0, 1, 0), (1, 2, -1), (-2, 3, 2), (2, 4, 0))
        lmax = 8
        coeffss = map((DriscollHealyGrid(lmax), EquiangularGrid(lmax))) do grid
            sz = ash_grid_size(grid)
            f = [Complex{Float64}(AbstractSphericalHarmonics.sYlm(Val(s), Val(l), Val(m), ash_point_coord(grid, ij)...)) for ij in CartesianIndices(sz)]
            ash_transform(grid, f, s)
        end
        @test isapprox(coeffss[1], coeffss[2]; atol=10000eps())
        @test isapprox(coeffss[1][ash_mode_index(DriscollHealyGrid(lmax), s, l, m)], 1; atol=10000eps())
    end
end
