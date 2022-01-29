using AbstractSphericalHarmonics
using Random
using StaticArrays
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

#TODO Random.seed!(100)
#TODO modes = [(name="(s=$s,l=$l,m=$m)", spin=s, el=l, m=m, fun=(θ, ϕ) -> sYlm(s, l, m, θ, ϕ), modes=(l′, m′) -> l′ == l && m′ == m)
#TODO          for s in -2:+2 for l in abs(s):2 for m in (-l):l]
#TODO @testset "Simple transforms: $(mode.name)" for mode in modes
#TODO     for lmax in (mode.el):(mode.el) #TODO 20
#TODO         sz = ash_grid_size(lmax)
#TODO         f = Array{Complex{Float64}}(undef, sz)
#TODO         for ij in CartesianIndices(sz)
#TODO             θ, ϕ = ash_point_coord(ij, lmax)
#TODO             f[ij] = mode.fun(θ, ϕ)
#TODO         end
#TODO         # function setvalue(ij::CartesianIndex{2})
#TODO         #     θ, ϕ = ash_point_coord(ij, lmax)
#TODO         #     return Complex{Float64}(mode.fun(θ, ϕ))
#TODO         # end
#TODO         # f = map(setvalue, CartesianIndices(sz))
#TODO 
#TODO         flm = ash_transform(f, mode.spin, lmax)
#TODO         @test all(isapprox(flm[ash_mode_index(mode.spin, l, m, lmax)], mode.modes(l, m); atol=100eps()) for l in
#TODO                                                                                                             abs(mode.spin):lmax
#TODO                   for m in (-l):l)
#TODO 
#TODO         f′ = ash_evaluate(flm, mode.spin, lmax)
#TODO         @test isapprox(f′, f; atol=100eps())
#TODO     end
#TODO end
#TODO 
#TODO Random.seed!(100)
#TODO @testset "Parity" begin
#TODO     for iter in 1:100
#TODO         lmax = rand(0:100)
#TODO         sz = ash_grid_size(lmax)
#TODO         nmodes = ash_nmodes(lmax)
#TODO 
#TODO         s = rand(-4:4)
#TODO         abs(s) ≤ lmax || continue
#TODO         l = rand(abs(s):lmax)
#TODO         m = rand((-l):l)
#TODO 
#TODO         flm = zeros(Complex{Float64}, nmodes)
#TODO         flm[ash_mode_index(s, l, m, lmax)] = 1
#TODO 
#TODO         flm′ = zeros(Complex{Float64}, nmodes)
#TODO         flm′[ash_mode_index(-s, l, -m, lmax)] = 1
#TODO 
#TODO         f = ash_evaluate(flm, s, lmax)
#TODO         f′ = ash_evaluate(flm′, -s, lmax)
#TODO 
#TODO         # Phase: conj(sYlm) = (-1)^(s+m) (-s)Yl(-m)
#TODO         @test conj(f) ≈ bitsign(s + m) * f′
#TODO     end
#TODO end
#TODO 
#TODO Random.seed!(100)
#TODO @testset "Linearity" begin
#TODO     for iter in 1:100
#TODO         lmax = rand(0:100)
#TODO         sz = ash_grid_size(lmax)
#TODO         nmodes = ash_nmodes(lmax)
#TODO 
#TODO         spin = rand(-4:4)
#TODO 
#TODO         f = randn(Complex{Float64}, sz)
#TODO         g = randn(Complex{Float64}, sz)
#TODO         α = randn(Complex{Float64})
#TODO         h = f + α * g
#TODO 
#TODO         flm = ash_transform(f, spin, lmax)
#TODO         glm = ash_transform(g, spin, lmax)
#TODO         hlm = ash_transform(h, spin, lmax)
#TODO 
#TODO         if !(flm + α * glm ≈ hlm)
#TODO             @show iter lmax sz nmodes spin
#TODO             @show α
#TODO             @show any(isnan, f) any(isnan, g) any(isnan, h)
#TODO             @show any(isnan, flm) any(isnan, glm) any(isnan, hlm)
#TODO             @show f[1] g[1] h[1]
#TODO             @show flm[1] glm[1] hlm[1]
#TODO         end
#TODO         @test flm + α * glm ≈ hlm
#TODO 
#TODO         flm = randn(Complex{Float64}, nmodes)
#TODO         glm = randn(Complex{Float64}, nmodes)
#TODO         hlm = flm + α * glm
#TODO 
#TODO         f = ash_evaluate(flm, spin, lmax)
#TODO         g = ash_evaluate(glm, spin, lmax)
#TODO         h = ash_evaluate(hlm, spin, lmax)
#TODO 
#TODO         @test f + α * g ≈ h
#TODO     end
#TODO end
#TODO 
#TODO Random.seed!(100)
#TODO @testset "Orthonormality transforms" begin
#TODO     for iter in 1:100
#TODO         lmax = rand(0:100)
#TODO         nmodes = ash_nmodes(lmax)
#TODO 
#TODO         smax = min(4, lmax)
#TODO         spin = rand((-smax):smax)
#TODO 
#TODO         flm = zeros(Complex{Float64}, nmodes)
#TODO         glm = zeros(Complex{Float64}, nmodes)
#TODO 
#TODO         lf = rand(abs(spin):lmax)
#TODO         mf = rand((-lf):lf)
#TODO         lg = rand(abs(spin):lmax)
#TODO         mg = rand((-lg):lg)
#TODO 
#TODO         flm[ash_mode_index(spin, lf, mf, lmax)] = 1
#TODO         glm[ash_mode_index(spin, lg, mg, lmax)] = 1
#TODO 
#TODO         f = ash_evaluate(flm, spin, lmax)
#TODO         g = ash_evaluate(glm, spin, lmax)
#TODO 
#TODO         @test isapprox(integrate(f, f, lmax), 1; atol=1 / lmax^2)
#TODO         @test isapprox(integrate(f, g, lmax), (lf == lg) * (mf == mg); atol=1 / lmax^2)
#TODO 
#TODO         h = conj(f) .* f
#TODO         hlm = ash_transform(h, 0, lmax)
#TODO         @test isapprox(hlm[ash_mode_index(0, 0, 0, lmax)], sqrt(1 / 4π); atol=sqrt(eps()))
#TODO 
#TODO         h = conj(f) .* g
#TODO         hlm = ash_transform(h, 0, lmax)
#TODO         @test isapprox(hlm[ash_mode_index(0, 0, 0, lmax)], (lf == lg) * (mf == mg) * sqrt(1 / 4π); atol=sqrt(eps()))
#TODO     end
#TODO end
#TODO 
#TODO Random.seed!(100)
#TODO modes = [(name="(s=$s,l=$l,m=$m)", spin=s, el=l, fun=(θ, ϕ) -> sYlm(s, l, m, θ, ϕ), ðfun=(θ, ϕ) -> ðsYlm(s, l, m, θ, ϕ),
#TODO           ð̄fun=(θ, ϕ) -> ð̄sYlm(s, l, m, θ, ϕ)) for s in 0:+2 for l in abs(s):2 for m in (-l):l]
#TODO @testset "Simple derivatives (eth, eth-bar): $(mode.name)" for mode in modes
#TODO     for lmax in (mode.el):20
#TODO         sz = ash_grid_size(lmax)
#TODO         f = Array{Complex{Float64}}(undef, sz)
#TODO         ðf₀ = Array{Complex{Float64}}(undef, sz)
#TODO         ð̄f₀ = Array{Complex{Float64}}(undef, sz)
#TODO         for ij in CartesianIndices(sz)
#TODO             θ, ϕ = ash_point_coord(ij, lmax)
#TODO             f[ij] = mode.fun(θ, ϕ)
#TODO             ðf₀[ij] = mode.ðfun(θ, ϕ)
#TODO             ð̄f₀[ij] = mode.ð̄fun(θ, ϕ)
#TODO         end
#TODO 
#TODO         flm = ash_transform(f, mode.spin, lmax)
#TODO 
#TODO         ðflm = ash_eth(flm, mode.spin, lmax)
#TODO         ðf = ash_evaluate(ðflm, mode.spin + 1, lmax)
#TODO         @test isapprox(ðf, ðf₀; atol=10000eps())
#TODO 
#TODO         ð̄flm = ash_ethbar(flm, mode.spin, lmax)
#TODO         ð̄f = ash_evaluate(ð̄flm, mode.spin - 1, lmax)
#TODO         @test isapprox(ð̄f, ð̄f₀; atol=10000eps())
#TODO     end
#TODO end
#TODO 
#TODO Random.seed!(100)
#TODO @testset "Eigenvectors of Laplacian" begin
#TODO     for iter in 1:100
#TODO         lmax = rand(0:100)
#TODO         nmodes = ash_nmodes(lmax)
#TODO 
#TODO         flm = zeros(Complex{Float64}, nmodes)
#TODO 
#TODO         l = rand(0:lmax)
#TODO         m = rand((-l):l)
#TODO         flm[ash_mode_index(0, l, m, lmax)] = 1
#TODO 
#TODO         ðflm = ash_eth(flm, 0, lmax)
#TODO         ð̄ðflm = ash_ethbar(ðflm, +1, lmax)
#TODO 
#TODO         ð̄flm = ash_ethbar(flm, 0, lmax)
#TODO         ðð̄flm = ash_eth(ð̄flm, -1, lmax)
#TODO 
#TODO         f = ash_evaluate(flm, 0, lmax)
#TODO         ð̄ðf = ash_evaluate(ð̄ðflm, 0, lmax)
#TODO         ðð̄f = ash_evaluate(ðð̄flm, 0, lmax)
#TODO 
#TODO         @test isapprox(ð̄ðf, -l * (l + 1) * f; atol=(lmax + 1)^2 * 100eps())
#TODO         @test isapprox(ðð̄f, -l * (l + 1) * f; atol=(lmax + 1)^2 * 100eps())
#TODO     end
#TODO end
#TODO 
#TODO ################################################################################
#TODO 
#TODO Random.seed!(100)
#TODO @testset "Arbitrary modes" begin
#TODO     for iter in 1:100
#TODO         lmax = rand(0:100)
#TODO         nmodes = ash_nmodes(lmax)
#TODO 
#TODO         flm = zeros(Complex{Float64}, nmodes)
#TODO 
#TODO         s = rand(-4:4)
#TODO         abs(s) ≤ lmax || continue
#TODO         l = rand(abs(s):lmax)
#TODO         m = rand((-l):l)
#TODO         flm[ash_mode_index(s, l, m, lmax)] = 1
#TODO 
#TODO         f = ash_evaluate(flm, s, lmax)
#TODO 
#TODO         f′ = [begin
#TODO                   θ, ϕ = ash_point_coord(ij, lmax)
#TODO                   # We need the increased precision of `BigFloat` for `l >≈ 50`
#TODO                   # AbstractSphericalHarmonics.sYlm(Val(s), Val(l), Val(m), θ, ϕ)
#TODO                   Complex{Float64}(AbstractSphericalHarmonics.sYlm(Val(s), Val(l), Val(m), big(θ), big(ϕ)))
#TODO               end
#TODO               for ij in CartesianIndices(f)]
#TODO 
#TODO         @test f ≈ f′
#TODO     end
#TODO end

################################################################################

function rand_tensor(::Val{D}, ::Type{T}, lmax::Int) where {D,T}
    sz = ash_grid_size(lmax)
    Dims = Tuple{[2 for d in 1:D]...}
    f = randn(SArray{Dims,Complex{Float64}}, sz)
    return f
end

function const_tensor(::Val{D}, ::Type{T}, lmax::Int) where {D,T}
    sz = ash_grid_size(lmax)
    Dims = Tuple{[2 for d in 1:D]...}
    α = randn(SArray{Dims,Complex{Float64}})
    f = fill(α, sz)
    return f
end

#TODO Random.seed!(100)
#TODO @testset "Tensors on the sphere (rank $D)" for D in 0:2
#TODO     for iter in 1:100
#TODO         lmax = rand(0:100)
#TODO 
#TODO         f = rand_tensor(Val(D), Complex{Float64}, lmax)
#TODO 
#TODO         t = Tensor{D}(f, lmax)
#TODO         st = SpinTensor{D}(t)
#TODO         t′ = Tensor{D}(st)
#TODO         st′ = SpinTensor{D}(t′)
#TODO         t″ = Tensor{D}(st′)
#TODO 
#TODO         @test st′ ≈ st
#TODO         @test t″ ≈ t′
#TODO     end
#TODO end
#TODO 
#TODO Random.seed!(100)
#TODO @testset "Linearity of tensors on the sphere (rank $D)" for D in 0:2
#TODO     for iter in 1:100
#TODO         lmax = rand(0:100)
#TODO 
#TODO         f = rand_tensor(Val(D), Complex{Float64}, lmax)
#TODO         g = rand_tensor(Val(D), Complex{Float64}, lmax)
#TODO         α = randn(Complex{Float64})
#TODO 
#TODO         h = f + α * g
#TODO 
#TODO         t = Tensor{D}(f, lmax)
#TODO         u = Tensor{D}(g, lmax)
#TODO         v = Tensor{D}(h, lmax)
#TODO 
#TODO         st = SpinTensor{D}(t)
#TODO         su = SpinTensor{D}(u)
#TODO         sv = SpinTensor{D}(v)
#TODO 
#TODO         @test sv ≈ st + α * su
#TODO     end
#TODO end

Random.seed!(100)
@testset "Derivatives of tensors on the sphere (rank $D)" for D in 0:1
    for iter in 1:100
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
            dtp′ = Tensor{D + 1}(map((dx, x) -> dx * x[], dtf.values, tg.values) + map((x, dx) -> x[] * dx, tf.values, dtg.values),
                                 lmax)
            # TODO: This needs smooth input, not noise
            # if !isapprox(dtp, dtp′; atol=1 / lmax^2)
            #     @show D iter lmax
            #     @show maximum(map(x -> maximum(abs.(x)), dtp.values - dtp′.values))
            # end
            @test_skip isapprox(dtp, dtp′; atol=1 / lmax^2)
        end
    end
end
