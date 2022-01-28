module AbstractSphericalHarmonics

# using FastSphericalHarmonics
using ssht

export ash_grid_size, ash_nmodes
export ash_ntheta, ash_nphi, ash_thetas, ash_phis, ash_point_coord, ash_point_delta, ash_grid_as_phi_theta
export ash_mode_index
export ash_transform!, ash_transform, ash_evaluate!, ash_evaluate
export ash_eth!, ash_eth, ash_ethbar!, ash_ethbar

end
