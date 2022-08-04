export measure_state_cylinder

"""
Evaluates the pressure induced at `config.ss` by the regularized point vortices stored in `state`,
and an optional freestream of amplitude `config.U`.
The pressure is computed from the unsteady Bernoulli equation.
"""
measure_state_cylinder(state, t, config::VortexConfig) =
pressure(config.ss, cylinder_state_to_lagrange(state, config); ϵ = config.δ, walltype = Cylinder)
