using AbstractSphericalHarmonics
using LinearAlgebra: norm
using Random
using StaticArrays
using Test

const bitsign = AbstractSphericalHarmonics.bitsign

chop(x) = abs2(x) < 100eps(x) ? zero(x) : x
chop(x::Complex) = Complex(chop(real(x)), chop(imag(x)))
chop(x::SArray) = chop.(x)

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

function rand_tensor(::Val{D}, ::Type{T}, lmax::Int) where {D,T}
    sz = ash_grid_size(lmax)
    Dims = Tuple{[2 for d in 1:D]...}
    f = randn(SArray{Dims,T}, sz)
    return f
end

function const_tensor(::Val{D}, ::Type{T}, lmax::Int) where {D,T}
    sz = ash_grid_size(lmax)
    Dims = Tuple{[2 for d in 1:D]...}
    α = randn(SArray{Dims,T})
    f = fill(α, sz)
    return f
end

################################################################################

@testset "Mode indices" begin
    for lmax in 0:20
        nmodes = ash_nmodes(lmax)
        for s in -4:4, l in abs(s):lmax, m in (-l):l
            ind = ash_mode_index(s, l, m, lmax)
            @test length(nmodes) ≡ length(ind)
            @test all(1 ≤ ind[d] ≤ nmodes[d] for d in 1:length(nmodes))
            l′, m′ = ash_mode_numbers(s, ind, lmax)
            @test l′ == l && m′ == m
        end
    end
end

Random.seed!(100)
modes = [
    (name="(s=$s,l=$l,m=$m)", spin=s, el=l, m=m, fun=(θ, ϕ) -> sYlm(s, l, m, θ, ϕ), modes=(l′, m′) -> l′ == l && m′ == m) for
    s in -2:+2 for l in abs(s):2 for m in (-l):l
]
@testset "Simple transforms: $(mode.name)" for mode in modes
    for lmax in (mode.el):20
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
        @test all(
            isapprox(flm[ash_mode_index(mode.spin, l, m, lmax)], mode.modes(l, m); atol=100eps()) for l in abs(mode.spin):lmax for
            m in (-l):l
        )

        f′ = ash_evaluate(flm, mode.spin, lmax)
        @test isapprox(f′, f; atol=1000eps())
    end
end

Random.seed!(100)
@testset "Parity" begin
    for iter in 1:20
        lmax = rand(0:100)
        sz = ash_grid_size(lmax)
        nmodes = ash_nmodes(lmax)

        s = rand(-4:4)
        abs(s) ≤ lmax || continue
        l = rand(abs(s):lmax)
        m = rand((-l):l)

        flm = zeros(Complex{Float64}, nmodes)
        flm[ash_mode_index(s, l, m, lmax)] = 1

        flm′ = zeros(Complex{Float64}, nmodes)
        flm′[ash_mode_index(-s, l, -m, lmax)] = 1

        f = ash_evaluate(flm, s, lmax)
        f′ = ash_evaluate(flm′, -s, lmax)

        # Phase: conj(sYlm) = (-1)^(s+m) (-s)Yl(-m)
        @test conj(f) ≈ bitsign(s + m) * f′
    end
end

Random.seed!(100)
@testset "Linearity" begin
    for iter in 1:20
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

Random.seed!(100)
@testset "Orthonormality" begin
    for iter in 1:20
        lmax = rand(0:100)
        nmodes = ash_nmodes(lmax)

        smax = min(4, lmax ÷ 2)
        spin = rand((-smax):smax)

        flm = zeros(Complex{Float64}, nmodes)
        glm = zeros(Complex{Float64}, nmodes)

        # The produce will have lh = lf + lg
        lf = rand(abs(spin):(lmax ÷ 2))
        mf = rand((-lf):lf)
        lg = rand(abs(spin):(lmax - lf))
        mg = rand((-lg):lg)

        flm[ash_mode_index(spin, lf, mf, lmax)] = 1
        glm[ash_mode_index(spin, lg, mg, lmax)] = 1

        f = ash_evaluate(flm, spin, lmax)
        g = ash_evaluate(glm, spin, lmax)

        @test isapprox(integrate(f, f, lmax), 1; atol=20 / lmax^2)
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
    for iter in 1:20
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

################################################################################

Random.seed!(100)
@testset "Arbitrary modes" begin
    for iter in 1:20
        lmax = rand(0:100)
        nmodes = ash_nmodes(lmax)

        flm = zeros(Complex{Float64}, nmodes)

        s = rand(-4:4)
        abs(s) ≤ lmax || continue
        l = rand(abs(s):lmax)
        m = rand((-l):l)
        flm[ash_mode_index(s, l, m, lmax)] = 1

        f = ash_evaluate(flm, s, lmax)

        f′ = [
            begin
                θ, ϕ = ash_point_coord(ij, lmax)
                # We need the increased precision of `BigFloat` for `l >≈ 50`
                # AbstractSphericalHarmonics.sYlm(Val(s), Val(l), Val(m), θ, ϕ)
                Complex{Float64}(AbstractSphericalHarmonics.sYlm(Val(s), Val(l), Val(m), big(θ), big(ϕ)))
            end for ij in CartesianIndices(f)
        ]

        @test f ≈ f′
    end
end

################################################################################

Random.seed!(100)
@testset "Tensors on the sphere (rank $D)" for D in 0:4
    for iter in 1:20
        lmax = rand(0:100)

        f = rand_tensor(Val(D), Complex{Float64}, lmax)

        t = Tensor{D}(f, lmax)
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
        lmax = rand(0:100)

        f = rand_tensor(Val(D), Complex{Float64}, lmax)
        g = rand_tensor(Val(D), Complex{Float64}, lmax)
        α = randn(Complex{Float64})

        h = f + α * g

        t = Tensor{D}(f, lmax)
        u = Tensor{D}(g, lmax)
        v = Tensor{D}(h, lmax)

        st = SpinTensor{D}(t)
        su = SpinTensor{D}(u)
        sv = SpinTensor{D}(v)

        @test sv ≈ st + α * su
    end
end

@testset "Simple derivatives of tensors on the sphere" begin
    for lmax in 1:20
        sz = ash_grid_size(lmax)

        # 1

        s = Tensor{0}(fill(Scalar{Complex{Float64}}(1), sz), lmax)
        ds₀ = Tensor{1}(fill(SVector{2,Complex{Float64}}(0, 0), sz), lmax)

        s̃ = SpinTensor(s)
        ds̃ = tensor_gradient(s̃)
        ds = Tensor(ds̃)

        @test isapprox(ds, ds₀; atol=(lmax + 1)^2 * 1000eps())

        # x

        x = Tensor{0}([
            begin
                θ, ϕ = ash_point_coord(ij, lmax)
                Scalar{Complex{Float64}}(sin(θ) * cos(ϕ))
            end for ij in CartesianIndices(sz)
        ], lmax)
        dx₀ = Tensor{1}([
            begin
                θ, ϕ = ash_point_coord(ij, lmax)
                SVector{2,Complex{Float64}}(cos(θ) * cos(ϕ), -sin(ϕ))
            end for ij in CartesianIndices(sz)
        ], lmax)

        x̃ = SpinTensor(x)
        dx̃ = tensor_gradient(x̃)
        dx = Tensor(dx̃)

        @test isapprox(dx, dx₀; atol=(lmax + 1)^2 * 100eps())

        # y

        y = Tensor{0}([
            begin
                θ, ϕ = ash_point_coord(ij, lmax)
                Scalar{Complex{Float64}}(sin(θ) * sin(ϕ))
            end for ij in CartesianIndices(sz)
        ], lmax)
        dy₀ = Tensor{1}([
            begin
                θ, ϕ = ash_point_coord(ij, lmax)
                SVector{2,Complex{Float64}}(cos(θ) * sin(ϕ), cos(ϕ))
            end for ij in CartesianIndices(sz)
        ], lmax)

        ỹ = SpinTensor(y)
        dỹ = tensor_gradient(ỹ)
        dy = Tensor(dỹ)

        @test isapprox(dy, dy₀; atol=(lmax + 1)^2 * 100eps())

        # z

        z = Tensor{0}([
            begin
                θ, ϕ = ash_point_coord(ij, lmax)
                Scalar{Complex{Float64}}(cos(θ))
            end for ij in CartesianIndices(sz)
        ], lmax)
        dz₀ = Tensor{1}([
            begin
                θ, ϕ = ash_point_coord(ij, lmax)
                SVector{2,Complex{Float64}}(-sin(θ), 0)
            end for ij in CartesianIndices(sz)
        ], lmax)

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
                    θ, ϕ = ash_point_coord(ij, lmax)
                    SVector{2,Complex{Float64}}(cos(θ) * cos(ϕ), -sin(ϕ))
                end for ij in CartesianIndices(sz)
            ], lmax)
            dgradx₀ = Tensor{2}(
                [
                    begin
                        θ, ϕ = ash_point_coord(ij, lmax)
                        SMatrix{2,2,Complex{Float64}}(-sin(θ) * cos(ϕ), 0, 0, -sin(θ) * cos(ϕ))
                    end for ij in CartesianIndices(sz)
                ], lmax
            )

            gradx̃ = SpinTensor(gradx)
            dgradx̃ = tensor_gradient(gradx̃)
            dgradx = Tensor(dgradx̃)

            @test isapprox(dgradx, dgradx₀; atol=(lmax + 1)^2 * 100eps())

            ddgradx₀ = Tensor{3}(
                [
                    begin
                        θ, ϕ = ash_point_coord(ij, lmax)
                        SArray{Tuple{2,2,2},Complex{Float64}}(-cos(θ) * cos(ϕ), 0, 0, -cos(θ) * cos(ϕ), sin(ϕ), 0, 0, sin(ϕ))
                    end for ij in CartesianIndices(sz)
                ],
                lmax,
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
                    θ, ϕ = ash_point_coord(ij, lmax)
                    SVector{2,Complex{Float64}}(sin(ϕ), cos(θ) * cos(ϕ))
                end for ij in CartesianIndices(sz)
            ], lmax)
            dcurlx₀ = Tensor{2}(
                [
                    begin
                        θ, ϕ = ash_point_coord(ij, lmax)
                        SMatrix{2,2,Complex{Float64}}(0, -sin(θ) * cos(ϕ), sin(θ) * cos(ϕ), 0)
                    end for ij in CartesianIndices(sz)
                ], lmax
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
        lmax = rand(0:100)

        f = rand_tensor(Val(D), Complex{Float64}, lmax)
        g = rand_tensor(Val(D), Complex{Float64}, lmax)
        α = randn(Complex{Float64})

        # Linear combination
        s = f + α * g

        tf = Tensor{D}(f, lmax)
        tg = Tensor{D}(g, lmax)
        ts = Tensor{D}(s, lmax)

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

        if D == 0
            # Constant function (derivative is zero)
            c = const_tensor(Val(D), Complex{Float64}, lmax)
            # Product of two functions
            p = map(.*, f, g)

            tc = Tensor{D}(c, lmax)
            tp = Tensor{D}(p, lmax)

            stc = SpinTensor{D}(tc)
            stp = SpinTensor{D}(tp)

            dstc = tensor_gradient(stc)
            dstp = tensor_gradient(stp)

            dtc = Tensor{D + 1}(dstc)
            dtp = Tensor{D + 1}(dstp)

            dtc′ = Tensor{D + 1}(zero(dtc.values), lmax)
            @test isapprox(dtc, dtc′; atol=sqrt(eps()))
            dtp′ = Tensor{D + 1}(
                map((dx, x) -> dx * x[], dtf.values, tg.values) + map((x, dx) -> x[] * dx, tf.values, dtg.values), lmax
            )
            # TODO: This needs smooth input, not noise
            # if !isapprox(dtp, dtp′; atol=1 / lmax^2)
            #     @show D iter lmax
            #     @show maximum(map(x -> maximum(abs.(x)), dtp.values - dtp′.values))
            # end
            @test_skip isapprox(dtp, dtp′; atol=1 / lmax^2)
        end
    end
end

Random.seed!(100)
@testset "Filtering tensors on the sphere D=$D" for D in 0:4
    for lmax in 0:20
        sz = ash_grid_size(lmax)
        t = Tensor{D}(rand_tensor(Val(D), Complex{Float64}, lmax), lmax)
        st = SpinTensor(t)
        st′ = filter_modes(st)
        # This is only a weak test
        lfilter = lmax * 2 ÷ 3
        for (ij, cs) in zip(CartesianIndices(st′.coeffs), st′.coeffs)
            s = count(Tuple(ij) .== 1) - count(Tuple(ij) .== 2)
            for l in abs(s):lmax, m in (-l):l
                if l ≤ lfilter
                    @test cs[ash_mode_index(s, l, m, lmax)] ≠ 0
                else
                    @test abs(cs[ash_mode_index(s, l, m, lmax)]) ≤ 1000eps()
                end
            end
        end
        # Ideas:
        # - test effect of filtering on particular modes
        # - test that filtering and derivatives commute
        # - test that filtering is linear
    end
end

Random.seed!(100)
@testset "Calculate Ricci scalar" begin
    for iter in 1:20
        lmax = rand(2:100)
        sz = ash_grid_size(lmax)

        # q_ab = (1, 0, 0, 1)
        q = Tensor{2}(fill(SMatrix{2,2,Complex{Float64}}(1, 0, 0, 1), sz), lmax)
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
            lmax,
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
            lmax,
        )
        Rsc = Tensor{0}([Scalar(sum(qu[a, b] * R[a, b] for a in 1:2, b in 1:2)) for (qu, R) in zip(qu.values, R.values)], lmax)

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
