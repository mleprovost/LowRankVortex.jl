export pressure!, pressure,
       pressure_FD!, pressure_FD,
       pressure_AD,
       measure_state

"""
Evaluates the pressure induced at `config.ss` by the regularized point vortices stored in `state`,
and an optional freestream of amplitude `config.U`.
The pressure is computed from the unsteady Bernoulli equation.
"""
function measure_state(state, t, config::VortexConfig; withfreestream::Bool=false)
    if withfreestream == false
        return pressure(config.ss, state_to_lagrange(state, config), t)
    else
        freestream = Freestream(config.U)
        return pressure(config.ss, state_to_lagrange(state, config), freestream, t)
    end
end

"""
Evaluates in-place the pressure induced at `target` by the point vortices `source`.
The pressure is computed from the unsteady Bernoulli equation.
"""
function pressure!(press, targetvels, sourcevels, target, source, t)
    source = deepcopy(source)

    reset_velocity!(sourcevels)
    reset_velocity!(targetvels)

    # Compute the self-induced velocity of the system
    self_induce_velocity!(sourcevels, source, t)

    # Compute the induced velocity on the target elements
    induce_velocity!(targetvels, target, source, t)

    #Only the vortices contribute to the unsteady term
    press .= -real.(convective_complexpotential(target, source, sourcevels)) -0.5*abs2.(targetvels)

    return press
end

"""
Evaluates the pressure induced at `target` by the point vortices `source`.
The pressure is computed from the unsteady Bernoulli equation.
"""
pressure(target, source, t) = pressure!(zeros(Float64, length(target)),
                                                        allocate_velocity(target),
                                                        allocate_velocity(source),
                                                        target, source, t)

# Version of the pressure calculation with freestream
"""
Evaluates in-place the pressure induced at `target` by the point vortices `source` and a freestream `freestream`.
The pressure is computed from the unsteady Bernoulli equation.
"""
function pressure!(press, targetvels, sourcevels, target, source, freestream, t)
    source = deepcopy(source)

    reset_velocity!(sourcevels)
    reset_velocity!(targetvels)

    # Compute the self-induced velocity of the system
    self_induce_velocity!(sourcevels, source, t)
    induce_velocity!(sourcevels, source, freestream, t)
    # Compute the induced velocity on the target elements
    induce_velocity!(targetvels, target, (source, freestream), t)

    #Only the vortices contribute to the unsteady term
    press .= -real.(convective_complexpotential(target, source, sourcevels))
    press .+= -0.5*abs2.(targetvels)

    return press
end

"""
Evaluates the pressure induced at `target` by the point vortices `source` and a freestream `freestream`.
The pressure is computed from the unsteady Bernoulli equation.
"""
pressure(target, source, freestream, t) = pressure!(zeros(Float64, length(target)),
                                                        allocate_velocity(target),
                                                        allocate_velocity(source),
                                                        target, source, freestream, t)

"""
Evaluates in-place the pressure induced at `target` by the point vortices `source` and a freestream `freestream`.
The pressure is computed from the unsteady Bernoulli equation.
The unsteady pressure term ?????/???t is computed by finite difference over ??t.
"""
function pressure_FD!(press, targetvels, target??, sourcevels, target, source, t, ??t)
    source = deepcopy(source)

    reset_velocity!(sourcevels)
    reset_velocity!(targetvels)

    # Compute the self-induced velocity of the system
    self_induce_velocity!(sourcevels, source, t)

    # Compute the induced velocity on the target elements
    induce_velocity!(targetvels, target, source, t)

    target?? .= real.(complexpotential(target, source))

    # advective term
    fill!(press, 0.0)
    press .= -0.5*abs2.(targetvels)

    # unsteady term
    advect!(source, source, sourcevels, ??t)
    press .+= (target?? - real.(complexpotential(target, source)))/??t

    return press
end

"""
Evaluates the pressure induced at `target` by the point vortices `source` and a freestream `freestream`.
The pressure is computed from the unsteady Bernoulli equation.
The unsteady pressure term ?????/???t is computed by finite difference over ??t.
"""
pressure_FD(target, source, t, ??t) = pressure_FD!(zeros(Float64, length(target)),
                                                        allocate_velocity(target),
                                                        zeros(Float64, length(target)),
                                                        allocate_velocity(source),
                                                        target, source, t, ??t)

"""
Evaluates the pressure induced at `target` by the point vortices `source`.
The pressure is computed from the unsteady Bernoulli equation.
This version can be used for automatic differentiation with respect to the positions and strengths of the singularities.
"""
function pressure_AD(target, source, t)

    source = deepcopy(source)

    sourcevels = zeros(Complex{Elements.property_type(eltype(source))},length(source))
    self_induce_velocity!(sourcevels, source, t)

    targetvels = zeros(Complex{Elements.property_type(eltype(source))},length(target))
    induce_velocity!(targetvels, target, source, t)

    press = zeros(Complex{Elements.property_type(eltype(source))}, length(target))

    # Unsteady term (only the vortices contribute)
    convective_complexpotential!(press, target, source, sourcevels)
    press .= -real(press)

    # Convective term
    press .-= 0.5*abs2.(targetvels)

    return press
end

"""
Evaluates the pressure induced at `target` by the point vortices `source` and a freestream `freestream`.
The pressure is computed from the unsteady Bernoulli equation.
This version can be used for automatic differentiation with respect to the positions and strengths of the singularities.
"""
function pressure_AD(target, source, freestream, t)

    source = deepcopy(source)

    sourcevels = zeros(Complex{Elements.property_type(eltype(source))},length(source))
    self_induce_velocity!(sourcevels, source, t)
    induce_velocity!(sourcevels, source, freestream, t)


    targetvels = zeros(Complex{Elements.property_type(eltype(source))},length(target))
    induce_velocity!(targetvels, target, (source, freestream), t)

    press = zeros(Complex{Elements.property_type(eltype(source))}, length(target))

    # Unsteady term (only the vortices contribute)
    convective_complexpotential!(press, target, source, sourcevels)
    press .= -real(press)

    # Convective term
    press .-= 0.5*abs2.(targetvels)

    return press
end
