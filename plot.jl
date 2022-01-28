using ssht

lmax = 40

L = lmax + 1
nphi = ssht.sampling_dh_nphi(L)
ntheta = ssht.sampling_dh_ntheta(L)

phis = [ssht.sampling_dh_p2phi(p, L) for p in 1:nphi];
thetas = [ssht.sampling_dh_t2theta(t, L) for t in 1:ntheta];

flm = zeros(Complex{Float64}, L^2);
flm[ssht.sampling_elm2ind(1, +1)] = 1;

f⁺¹ = ssht.core_dh_inverse_sov(flm, L, +1);

f⁻¹ = ssht.core_dh_inverse_sov(flm, L, -1);

################################################################################

using GLMakie

fig = Figure(; resolution=(800, 400));

Axis(fig[1, 1]; title="real(f⁺¹)")
Axis(fig[1, 3]; title="imag(f⁺¹)")
hm = heatmap!(fig[1, 1], phis, thetas, real.(f⁺¹); colormap=:magma)
Colorbar(fig[1, 2], hm)
hm = heatmap!(fig[1, 3], phis, thetas, imag.(f⁺¹); colormap=:magma)
Colorbar(fig[1, 4], hm)
rowsize!(fig.layout, 1, Aspect(1, 1 / 2))

Axis(fig[2, 1]; title="real(f⁻¹)")
Axis(fig[2, 3]; title="imag(f⁻¹)")
hm = heatmap!(fig[2, 1], phis, thetas, real.(f⁻¹); colormap=:magma)
Colorbar(fig[2, 2], hm)
hm = heatmap!(fig[2, 3], phis, thetas, imag.(f⁻¹); colormap=:magma)
Colorbar(fig[2, 4], hm)
rowsize!(fig.layout, 2, Aspect(1, 1 / 2))

display(fig)
