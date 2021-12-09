# Design optimized routines to compute pressure and its Jacobian if the vortices are placed according to the method of images, and we evaluate the pressure field about the x axis.

export measure_state_symmetric, symmetric_pressure#, analytical_symmetric_jacobian_strength

function measure_state_symmetric(state, t, config::VortexConfig)
    return symmetric_pressure(real(config.ss), state_to_lagrange(state, config)[1], t)
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

    press .= (-0.5/π^2)*press.^2

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


# """
# Returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to the strength and conjugate strength of the singularities.
# Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to S = Q-iΓ.
# """
# function analytical_symmetric_jacobian_strength(target::Vector{Float64}, source::T, t) where T <:Vector{PotentialFlow.Points.Point{Float64, Float64}}
# 	Ny = size(target, 1)
#     Nx = size(source, 1)
#
# 	dpdΓ = zeros(Ny, Nx)
#
# 	zv = zeros(ComplexF64, Nx)
# 	zv .= getfield.(source, :z)
#
# 	yv = zeros(Nx)
# 	yv .= imag(zv)
#
# 	Γv = zeros(Nx)
# 	Γv .= getfield.(source, :S)
#
# 	# Evaluate ∂(-0.5v^2)/∂ΓL
#     for L = 1:Nx
#         zL = zv[L]
# 		yL = yv[L]
# 		ΓL = Γv[L]
#         for i=1:Ny
# 			xi = target[i]
# 			ΔiL = xi - zL
# 			inv2diL = inv(abs2(ΔiL))
# 			tmpiL = yL*inv2diL
# 			for J=1:Nx
# 				zJ = zv[J]
# 				yJ = yv[J]
# 				ΓJ = Γv[J]
#             	dpdΓ[i,L] -=  inv(π^2)*ΓJ*yJ/abs2(xi-zJ)*tmpiL
# 			end
#         end
#     end
#
# 	# Evaluate ∂(-∂ϕ/∂t)/∂ΓL
# 	for L = 1:Nx
# 		zL = zv[L]
# 		for i=1:Ny
# 			xi = target[i]
# 			for K=1:Nx
# 				if K != L
# 					zK = zv[K]
# 					yK = yv[K]
# 					ΓK = Γv[K]
# 					dpdΓ[i,L] +=  ΓK/(π^2)*yK*imag(inv((xi-zL)*(conj(zL)-zK)*(conj(zL - zK))))
# 				end
# 			end
# 		end
# 	end
#
# 	for L = 1:Nx
# 		ΓL = Γv[L]
# 		zL = zv[L]
# 		for i=1:Ny
# 			xi = target[i]
# 			for J=1:Nx
# 				zJ = zv[J]
# 				ΓJ = Γv[J]
# 				for K=1:Nx
# 					if K != J && K == L
# 						zK = zv[K]
# 						yK = yv[K]
# 						ΓK = Γv[K]
# 						dpdΓ[i,L] +=  ΓJ/(π^2)*yK*imag(inv((xi-zJ)*(conj(zJ)-zK)*(conj(zJ - zK))))
# 					end
# 				end
# 			end
# 			dpdΓ[i,L] += ΓL/(2π^2)*inv(abs2(xi-zL))
# 		end
# 	end
#
# 	return dpdΓ
# end
