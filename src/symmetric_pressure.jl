# Design optimized routines to compute pressure and its Jacobian if the vortices are placed according to the method of images, and we evaluate the pressure field about the x axis.

export measure_state_symmetric, symmetric_pressure#, analytical_symmetric_jacobian_strength

function measure_state_symmetric(state, t, config::VortexConfig; withfreestream::Bool = false)
	if withfreestream == false
    	return symmetric_pressure(real(config.ss), state_to_lagrange(state, config)[1], t)
	else
		freestream = Freestream(config.U)
		return symmetric_pressure(real(config.ss), state_to_lagrange(state, config)[1], freestream, t)
	end
end

# Symmetric pressure calculation for point vortices
function symmetric_pressure(target::Vector{Float64}, source::T, t) where T <: Vector{PotentialFlow.Points.Point{Float64, Float64}}

    Nv = size(source, 1)
    Ny = size(target, 1)

    press = zeros(Ny)

    # Quadratic term

    @inbounds for (J, bJ) in enumerate(source)
        tmpJ = bJ.S*imag(bJ.z)
        for (i, xi) in enumerate(target)
            diJ = inv(abs2(xi-bJ.z))
            press[i] += tmpJ*diJ
        end
    end
    press .= deepcopy((-0.5/π^2)*press.^2)

    sourcevels = zeros(ComplexF64, Nv)
    @inbounds for (J, bJ) in enumerate(source)
        zJ = bJ.z
        ΓJ = bJ.S
        for (K, bK) in enumerate(source)
            if K != J
                zK = bK.z
                ΓK = bK.S
                sourcevels[J] += (ΓK*imag(zK))/(π*(zJ - zK)*(zJ - conj(zK)))
            end
        end
        # Contribution of the mirrored vortex
        sourcevels[J] += ΓJ/(4*π*imag(zJ))
    end

    # Unsteady term
    @inbounds for (J, bJ) in enumerate(source)
        zJ = bJ.z
        ΓJ = bJ.S
        for (i, xi) in enumerate(target)
            press[i] += 2*real((-im*ΓJ)*inv(2*π*(xi - zJ))*conj(sourcevels[J]))
        end
    end

    return press
end

# Symmetric pressure calculation for point vortices with freestream
function symmetric_pressure(target::Vector{Float64}, source::T, freestream, t) where T <: Vector{PotentialFlow.Points.Point{Float64, Float64}}

    Nv = size(source, 1)
    Ny = size(target, 1)

	U = freestream.U

    press = zeros(Ny)

    # Quadratic term
    @inbounds for (J, bJ) in enumerate(source)
        tmpJ = bJ.S*imag(bJ.z)
        for (i, xi) in enumerate(target)
            diJ = inv(abs2(xi-bJ.z))
            press[i] += tmpJ*diJ
        end
    end

	press .=  -0.5*deepcopy(abs2.(1/π*press .+ conj(U)))

	sourcevels = zeros(ComplexF64, Nv)
    @inbounds for (J, bJ) in enumerate(source)
        zJ = bJ.z
        ΓJ = bJ.S
        for (K, bK) in enumerate(source)
            if K != J
                zK = bK.z
                ΓK = bK.S
                sourcevels[J] += (ΓK*imag(zK))/(π*(zJ - zK)*(zJ - conj(zK)))
            end
        end
        # Contribution of the mirrored vortex
        sourcevels[J] += ΓJ/(4*π*imag(zJ)) + conj(U)
    end

    # Unsteady term
    @inbounds for (J, bJ) in enumerate(source)
        zJ = bJ.z
        ΓJ = bJ.S
        for (i, xi) in enumerate(target)
            press[i] += 2*real((-im*ΓJ)*inv(2*π*(xi - zJ))*conj(sourcevels[J]))
        end
    end

    return press
end

# Symmetric pressure calculation for regularized vortices
function symmetric_pressure(target::Vector{Float64}, source::T, t) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}

	source = deepcopy(source)

    Nv = size(source, 1)
    Ny = size(target, 1)

	δ = source[1].δ

    press = zeros(Ny)

    # Quadratic term
    @inbounds for (J, bJ) in enumerate(source)
		zJ = bJ.z
        tmpJ = bJ.S*imag(zJ)
        for (i, xi) in enumerate(target)
            diJ = abs2(xi - zJ) + δ^2
            press[i] += tmpJ/diJ
		end
	end

	press .= deepcopy((-0.5/π^2)*press.^2)

    sourcevels = zeros(ComplexF64, Nv)
    @inbounds for (J, bJ) in enumerate(source)
        zJ = bJ.z
		yJ = imag(zJ)
        ΓJ = bJ.S
        for (K, bK) in enumerate(source)
            if K != J
                zK = bK.z
                ΓK = bK.S
                sourcevels[J] += ΓK*(conj(zJ - zK))/(abs2(zJ - zK) + δ^2)
				sourcevels[J] += -ΓK*(conj(zJ) - zK)/(abs2(zJ - conj(zK)) + δ^2)
            end
        end
		sourcevels[J] *= -im/(2*π)
        # Contribution of the mirrored vortex
		sourcevels[J] += (ΓJ*yJ/π)/(4*(yJ)^2 + δ^2)
    end

    # Unsteady term
    @inbounds for (J, bJ) in enumerate(source)
        zJ = bJ.z
        ΓJ = bJ.S
        for (i, xi) in enumerate(target)
            press[i] -= ΓJ/π*imag(sourcevels[J]/(xi - conj(zJ)))
        end
    end
    return press
end

# Symmetric pressure calculation for regularized vortices with freestream
function symmetric_pressure(target::Vector{Float64}, source::T, freestream, t) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}

	source = deepcopy(source)

    Nv = size(source, 1)
    Ny = size(target, 1)

	δ = source[1].δ
	U = freestream.U

    press = zeros(Ny)

    # Quadratic term
    @inbounds for (J, bJ) in enumerate(source)
		zJ = bJ.z
        tmpJ = bJ.S*imag(zJ)
        for (i, xi) in enumerate(target)
            diJ = abs2(xi - zJ) + δ^2
            press[i] += tmpJ/diJ
		end
	end

	press .=  -0.5*deepcopy(abs2.(1/π*press .+ conj(U)))

    sourcevels = zeros(ComplexF64, Nv)
    @inbounds for (J, bJ) in enumerate(source)
        zJ = bJ.z
		yJ = imag(zJ)
        ΓJ = bJ.S
        for (K, bK) in enumerate(source)
            if K != J
                zK = bK.z
                ΓK = bK.S
                sourcevels[J] += ΓK*(conj(zJ - zK))/(abs2(zJ - zK) + δ^2)
				sourcevels[J] += -ΓK*(conj(zJ) - zK)/(abs2(zJ - conj(zK)) + δ^2)
            end
        end
		sourcevels[J] *= -im/(2*π)
        # Contribution of the mirrored vortex
		sourcevels[J] += (ΓJ*yJ/π)/(4*(yJ)^2 + δ^2) + conj(U)
    end

    # Unsteady term
    @inbounds for (J, bJ) in enumerate(source)
        zJ = bJ.z
        ΓJ = bJ.S
        for (i, xi) in enumerate(target)
            press[i] -= ΓJ/π*imag(sourcevels[J]/(xi - conj(zJ)))
        end
    end
    return press
end
