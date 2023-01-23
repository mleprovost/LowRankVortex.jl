export analytical_jacobian_strength!,
       analytical_jacobian_strength,
       analytical_jacobian_position!,
       analytical_jacobian_position,
       analytical_jacobian_pressure


"""
In-place routine that returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to the strengths and conjugate strengths of the singularities.
Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to S = Q-iΓ.
"""
function analytical_jacobian_strength!(dpdS, dpdSstar, Css, Cts, Ctsblob, wtarget, target::Vector{ComplexF64}, source::T, freestream,
	                                   idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t;
	                                   iscauchystored::Bool = false, issourcefixed::Bool = false) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
	Ny = size(target, 1)
	Nx = size(source, 1)

	δ = source[1].δ

	U = freestream.U
	cst = inv(2*π)

	# @assert size(dpdS) == (Ny, Nx) && size(dpdSstar) == (Ny, Nx) && size(Css) == (Nx, Nx) && size(Cts) == (Ny, Nx) && size(wtarget) == (Ny,)

	fill!(dpdS, complex(0.0)); fill!(dpdSstar, complex(0.0))

	if iscauchystored == false
	  fill!(Css, complex(0.0));
	  fill!(Cts, complex(0.0));
	  fill!(Ctsblob, complex(0.0));

	  # Account for the presence of the free-stream
	  fill!(wtarget, conj(U))

	  # Construct the Cauchy matrix to store 1/(zJ-zK) in Css
	  @inbounds for K=1:Nx
		  zK = source[K].z
		  # exploit anti-symmetry of the velocity kernel
		  # Diagonal entries are equal to 0
		  for J=K+1:Nx
			  zJ = source[J].z
			  tmp = abs2(zJ - zK) + δ^2
			  Css[J,K] = conj(zJ - zK)/tmp
			  Css[K,J] = -Css[J,K]
		  end
	  end

	  # Construct the Cauchy matrix to store (z̄-z̄J)/(|z-zJ|^2 + δ^2) and the total induced velocity in Cts and wtarget
	  @inbounds for J=1:Nx
		  zJ = source[J].z
		  SJ = -im*source[J].S
		  for i=1:Ny
			  zi = target[i]
			  tmp = abs2(zi - zJ) + δ^2
			  Cts[i,J] = inv(zi - zJ)
			  Ctsblob[i,J] = conj(zi - zJ)/tmp
			  wtarget[i] += cst*SJ*Ctsblob[i,J]
		  end
	  end
	end

	# Evaluate ∂(-0.5v^2)/∂SL, ∂(-0.5v^2)/∂S̄L
	for L=1:Nx
	   zL = source[L].z
	   SL = -im*source[L].S
	   for i=1:Ny
	       wt = wtarget[i]
	       dpdS[i,L] = -0.5*conj(wt)*cst*Ctsblob[i,L]
	       # dpdSstar[i,L] = conj(dpdS[i,L])
	   end
	end

	# Evaluate ∂(-∂ϕ/∂t)/∂SL, ∂(-∂ϕ/∂t)/∂S̄L
	@inbounds for L=1:Nx
		SL = -im*source[L].S
		for K=1:Nx
			SK = -im*source[K].S
			for i=1:Ny
				term1 = 0.5*cst^2*conj(SK)*Cts[i,L]*conj(Css[L,K])
				dpdS[i,L] += term1
				# dpdSstar[i,L] += conj(term1)
			end
		end
		# add the freestream contribution
		if abs(U) > 100*eps()
			for i=1:Ny
				term3 = 0.5*cst*Cts[i,L]*U
				dpdS[i,L] += term3
				# dpdSstar[i,L] += conj(term3)
			end
		end

	    for J=1:Nx
	        if issourcefixed == true && typeof(source[J]) <: PotentialFlow.Blobs.Blob{ComplexF64, Float64}
	            continue
	        else
	        	SJ = -im*source[J].S
	        end

	  		for i=1:Ny
	           term2 = 0.5*cst^2*conj(SJ)*conj(Cts[i,J])*Css[J,L]
	           dpdS[i,L] += term2
	           # dpdSstar[i,L] += conj(term2)
	       end
	   end
	end
	dpdSstar .= conj(dpdS)
	nothing
end


"""
Returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to the strengths and conjugate strengths of the singularities.
Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to S = Q-iΓ.
"""
function analytical_jacobian_strength!(dpdS, dpdSstar, Css, Cts, wtarget, target::Vector{ComplexF64}, source::T, freestream,
	                                   idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t;
                                       iscauchystored::Bool = false) where T <: Vector{PotentialFlow.Points.Point{Float64, Float64}}
	Ny = size(target, 1)
    Nx = size(source, 1)

    U = freestream.U
    cst = inv(2*π)

    # @assert size(dpdS) == (Ny, Nx) && size(dpdSstar) == (Ny, Nx) && size(Css) == (Nx, Nx) && size(Cts) == (Ny, Nx) && size(wtarget) == (Ny,)

    fill!(dpdS, complex(0.0)); fill!(dpdSstar, complex(0.0))

    if iscauchystored == false

        fill!(Css, complex(0.0)); fill!(Cts, complex(0.0))

        # Account for the presence of the free-stream
        fill!(wtarget, conj(U))

        # Construct the Cauchy matrix to store 1/(zJ-zK) in Css
        @inbounds for K=1:Nx
            zK = source[K].z
            # exploit anti-symmetry of the velocity kernel
            for J=K+1:Nx
                zJ = source[J].z
                Css[J,K] = inv(zJ - zK)
                Css[K,J] = -Css[J,K]
            end
        end

        # Construct the Cauchy matrix to store 1/(z-zJ) and the total induced velocity in Cts and wtarget
        for J=1:Nx
            zJ = source[J].z
            SJ = -im*source[J].S
            for i=1:Ny
                zi = target[i]
                Cts[i,J] = inv(zi - zJ)
                wtarget[i] += Cts[i,J]*cst*SJ
            end
        end
    end

    # Evaluate ∂(-0.5v^2)/∂SL, ∂(-0.5v^2)/∂S̄L
    for L in idx #1:Nx
        zL = source[L].z
        SL = -im*source[L].S
        for i=1:Ny
            wt = wtarget[i]
            dpdS[i,L] = -0.5*conj(wt)*cst*Cts[i,L]
            # dpdSstar[i,L] = conj(dpdS[i,L])
        end
    end

    # Evaluate ∂(-∂ϕ/∂t)/∂SL, ∂(-∂ϕ/∂t)/∂S̄L
    @inbounds for L in idx #1:Nx
		SL = -im*source[L].S
		for K=1:Nx
			SK = -im*source[K].S
			for i=1:Ny
				term1 = 0.5*cst^2*conj(SK)*Cts[i,L]*conj(Css[L,K])
				dpdS[i,L] += term1
				# dpdSstar[i,L] += conj(term1)
			end
		end
		# add the freestream contribution
		if abs(U)>100*eps()
			for i=1:Ny
				term3 = 0.5*cst*Cts[i,L]*U
				dpdS[i,L] += term3
				# dpdSstar[i,L] += conj(term3)
			end
		end

	    for J=1:Nx
	        SJ = -im*source[J].S
       		for i=1:Ny
                term2 = 0.5*cst^2*conj(SJ)*conj(Cts[i,J])*Css[J,L]
                dpdS[i,L] += term2
                # dpdSstar[i,L] += conj(term2)
            end
        end
    end
	dpdSstar .= conj(dpdS)
    nothing
end

analytical_jacobian_strength!(dpdS, dpdSstar, Css, Cts, Ctsblob, wtarget, target::Vector{ComplexF64}, source, freestream, t;
	                          iscauchystored::Bool = false) =
analytical_jacobian_strength!(dpdS, dpdSstar, Css, Cts, Ctsblob, wtarget, target, source, freestream, 1:length(source), t;
							  iscauchystored = iscauchystored)


"""
Routine that returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to the strengths and conjugate strengths of the points or blobs.
Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to S = Q-iΓ.
"""
function analytical_jacobian_strength(target::Vector{ComplexF64}, source, freestream,
	                                  idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t;
                                      iscauchystored::Bool = false)
	Ny = size(target, 1)
	Nx = size(source, 1)
	Css = zeros(ComplexF64, Nx, Nx)
	Cts = zeros(ComplexF64, Ny, Nx)
	wtarget = zeros(ComplexF64, Ny)

	dpdS = zeros(ComplexF64, Ny, Nx)
	dpdSstar = zeros(ComplexF64, Ny, Nx)
	if typeof(source)<:Vector{PotentialFlow.Points.Point{Float64, Float64}}
		analytical_jacobian_strength!(dpdS, dpdSstar, Css, Cts, wtarget, target, source, freestream, idx, t;
	                                  iscauchystored = iscauchystored)
	elseif typeof(source)<:Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
		Ctsblob = zeros(ComplexF64, Ny, Nx)
		analytical_jacobian_strength!(dpdS, dpdSstar, Css, Cts, Ctsblob, wtarget, target, source, freestream, idx, t;
									  iscauchystored = iscauchystored)
	end

	return dpdS, dpdSstar
end

analytical_jacobian_strength(target::Vector{ComplexF64}, source, freestream, t;
                                      iscauchystored::Bool = false) =
analytical_jacobian_strength(target, source, freestream, 1:length(source), t;
                                      iscauchystored = iscauchystored)

# Version for point singularities
"""
Returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to the positions and conjugate positions of the singularities.
Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to Q-iΓ.
"""
function analytical_jacobian_position!(dpdz, dpdzstar, Css, Cts, wtarget, target::Vector{ComplexF64}, source::T, freestream,
	                                   idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t;
                                       iscauchystored::Bool = false) where T <: Vector{PotentialFlow.Points.Point{Float64, Float64}}
    Nx = size(source, 1)
	Ny = size(target, 1)

    U = freestream.U
    cst = inv(2*π)

    # @assert size(dpdz) == (Ny, Nx) && size(dpdzstar) == (Ny, Nx) && size(Css) == (Nx, Nx) && size(Cts) == (Ny, Nx) && size(wtarget) == (Ny,)

    fill!(dpdz, complex(0.0)); fill!(dpdzstar, complex(0.0))

    if iscauchystored == false
        fill!(Css, complex(0.0)); fill!(Cts, complex(0.0))

        # Account for the presence of the free-stream
        fill!(wtarget, conj(U))

        # Construct the Cauchy matrix to store 1/(zJ-zK) in Css
        @inbounds for K=1:Nx
            zK = source[K].z
            # exploit anti-symmetry of the velocity kernel
            for J=K+1:Nx
                zJ = source[J].z
                Css[J,K] = inv(zJ - zK)
                Css[K,J] = -Css[J,K]
            end
        end

        # Construct the Cauchy matrix to store 1/(z-zJ) and the total induced velocity in Cts and wtarget
        @inbounds for J=1:Nx
            zJ = source[J].z
            SJ = -im*source[J].S
            for i=1:Ny
                zi = target[i]
                Cts[i,J] = inv(zi - zJ)
                wtarget[i] += Cts[i,J]*cst*SJ
            end
        end
    end
	# @show norm(Cts)
	# @show norm(real(Css)), norm(imag(Css)), norm(Css), norm(real(wtarget))


    # Evaluate ∂(-0.5v^2)/∂zL, ∂(-0.5v^2)/∂z̄L
    @inbounds for L in idx #1:Nx
        zL = source[L].z
        SL = -im*source[L].S
        for i=1:Ny
            wt = wtarget[i] #+ conj(U)
            dpdz[i,L] = -0.5*conj(wt)*SL*cst*Cts[i,L]^2
            # dpdzstar[i,L] = conj(dpdz[i,L])
        end
    end

    # Evaluate ∂(-∂ϕ/∂t)/∂zL, ∂(-∂ϕ/∂t)/∂z̄L
	@inbounds for L in idx #1:Nx
		SL = -im*source[L].S
		# @show SL
		for K=1:Nx
			SK = -im*source[K].S
			for i=1:Ny
				term1 = 0.5*cst^2*SL*conj(SK)*(Cts[i,L])^2*conj(Css[L,K])
				term1 += 0.5*cst^2*conj(SL)*SK*conj(Cts[i,L])*(-Css[L,K]^2)
				dpdz[i,L] += term1
				# dpdzstar[i,L] += conj(term1)
			end
		end
		if abs(U) > 100*eps()
			for i=1:Ny
				term3 = 0.5*cst*SL*Cts[i,L]^2*U
				dpdz[i,L] += term3
				# dpdzstar[i,L] += conj(term3)
			end
		end

		for J=1:Nx
            SJ = -im*source[J].S
			for i=1:Ny
                term2 = 0.5*cst^2*conj(SJ)*SL*conj(Cts[i,J])*(Css[J,L])^2
                dpdz[i,L] += term2
                # dpdzstar[i,L] += conj(term2)
            end
        end
	end
	dpdzstar .= conj(dpdz)
    nothing
end

# Version for regularized point singularities (aka blobs)
"""
Returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to the positions and conjugate positions of the blobs.
Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to Q-iΓ.
"""
function analytical_jacobian_position!(dpdz, dpdzstar, Css, ∂Css, Cts, Ctsblob, ∂Ctsblob, wtarget, target::Vector{ComplexF64}, source::T, freestream,
									   idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t;
                                       iscauchystored::Bool = false) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
	Ny = size(target, 1)
	Nx = size(source, 1)

	δ = source[1].δ

	U = freestream.U
	cst = inv(2*π)

	# @assert size(dpdz) == (Ny, Nx) && size(dpdzstar) == (Ny, Nx) && size(Css) == (Nx, Nx) && size(Cts) == (Ny, Nx) && size(wtarget) == (Ny,)

	fill!(dpdz, complex(0.0)); fill!(dpdzstar, complex(0.0))

	if iscauchystored == false
	   fill!(Css, complex(0.0));
	   fill!(∂Css, 0.0);
	   fill!(Cts, complex(0.0));
	   fill!(Ctsblob, complex(0.0));
	   fill!(∂Ctsblob, 0.0)


	   # Account for the presence of the free-stream
	   fill!(wtarget, conj(U))

	   # Construct the Cauchy matrix to store 1/(zJ-zK) in Css
	   @inbounds for K=1:Nx
	       zK = source[K].z
	       # exploit anti-symmetry of the velocity kernel
		   # Diagonal entries are equal to 0
	       for J=K+1:Nx
	           zJ = source[J].z
			   tmp = abs2(zJ - zK) + δ^2
	           Css[J,K] = conj(zJ - zK)/tmp
			   Css[K,J] = -Css[J,K]
			   ∂Css[J,K] = -(δ/tmp)^2
			   ∂Css[K,J] =  ∂Css[J,K]
	       end
	   end

	   # Construct the Cauchy matrix to store (z̄-z̄J)/(|z-zJ|^2 + δ^2) and the total induced velocity in Cts and wtarget
	   @inbounds for J=1:Nx
	       zJ = source[J].z
	       SJ = -im*source[J].S
	       for i=1:Ny
	           zi = target[i]
			   tmp = abs2(zi - zJ) + δ^2
			   Cts[i,J] = inv(zi - zJ)
	           Ctsblob[i,J] = conj(zi - zJ)/tmp
			   ∂Ctsblob[i,J] = -(δ/tmp)^2
	           wtarget[i] += cst*SJ*Ctsblob[i,J]
	       end
	   end
	end
	# @show 0.5*(wtarget .- conj(U))*U + 0.5*conj(wtarget .- conj(U))*conj(U)

	# Evaluate ∂(-0.5v^2)/∂zL, ∂(-0.5v^2)/∂z̄L
	@inbounds for L in idx #1:Nx
	   zL = source[L].z
	   SL = -im*source[L].S
	   for i=1:Ny
	       wt = wtarget[i]
		   dpdz[i,L] = -0.5*cst*conj(SL)*∂Ctsblob[i,L]*wt
	       dpdz[i,L] += -0.5*conj(wt)*cst*SL*Ctsblob[i,L]^2
	       # dpdzstar[i,L] = conj(dpdz[i,L])
	   end
	end

	# Evaluate ∂(-∂ϕ/∂t)/∂zL, ∂(-∂ϕ/∂t)/∂z̄L
	@inbounds for L in idx #1:Nx
		SL = -im*source[L].S
		for K=1:Nx
			SK = -im*source[K].S
			for i=1:Ny
				termJL = 0.5*cst^2*SL*conj(SK)*Cts[i,L]^2*conj(Css[L,K])
				termJL += 0.5*cst^2*SL*conj(SK)*Cts[i,L]*(-∂Css[L,K])
				termJL += 0.5*cst^2*conj(SL)*SK*conj(Cts[i,L])*(-Css[L,K]^2)
				dpdz[i,L] += termJL
				# dpdzstar[i,L] += conj(termJL)
			end
		end
		if abs(U) > 100*eps()
			for i=1:Ny
				termU = 0.5*cst*SL*Cts[i,L]^2*U
				dpdz[i,L] += termU

				# dpdzstar[i,L] += conj(termU)
			end
		end

		for J=1:Nx
		SJ = -im*source[J].S
			for i=1:Ny
	           termKL = 0.5*cst^2*SJ*conj(SL)*Cts[i,J]*∂Css[J,L]
			   termKL += 0.5*cst^2*conj(SJ)*SL*conj(Cts[i,J])*Css[J,L]^2
	           dpdz[i,L] += termKL

	           # dpdzstar[i,L] += conj(termKL)
		   end
	   end
	end
	dpdzstar .= conj(dpdz)
	nothing
end

analytical_jacobian_position!(dpdz, dpdzstar, Css, Cts, wtarget, target::Vector{ComplexF64}, source::T, freestream,
	                                   idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t;
                                       iscauchystored::Bool = false) where T =
analytical_jacobian_position!(dpdz, dpdzstar, Css, Cts, wtarget, target, source, freestream,
	                                  1:length(source), t;
                                      iscauchystored = iscauchystored)


"""
Returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to the positions and conjugate positions of the singularities/blobs.
Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to Q-iΓ.
"""
function analytical_jacobian_position(target::Vector{ComplexF64}, source, freestream,
									  idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t;
                                      iscauchystored::Bool = false)
    Ny = size(target, 1)
    Nx = size(source, 1)
    Css = zeros(ComplexF64, Nx, Nx)
    Cts = zeros(ComplexF64, Ny, Nx)
    wtarget = zeros(ComplexF64, Ny)

    dpdz = zeros(ComplexF64, Ny, Nx)
    dpdzstar = zeros(ComplexF64, Ny, Nx)

	if typeof(source)<:Vector{PotentialFlow.Points.Point{Float64, Float64}}
	    analytical_jacobian_position!(dpdz, dpdzstar, Css, Cts, wtarget, target, source, freestream, idx, t;
	                                  iscauchystored = iscauchystored)
	elseif typeof(source)<:Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
		∂Css = zeros(Nx, Nx)
		Ctsblob = zeros(ComplexF64, Ny, Nx)
		∂Ctsblob = zeros(Ny, Nx)
		analytical_jacobian_position!(dpdz, dpdzstar, Css, ∂Css, Cts, Ctsblob, ∂Ctsblob, wtarget, target, source, freestream, idx, t;
									  iscauchystored = iscauchystored)
	end

    return dpdz, dpdzstar
end

analytical_jacobian_position(target::Vector{ComplexF64}, source, freestream, t;
                                      iscauchystored::Bool = false) =
analytical_jacobian_position(target, source, freestream, 1:length(source), t;
                                      iscauchystored = iscauchystored)


analytical_jacobian_pressure(target, source, freestream, t) = analytical_jacobian_pressure(target, source, freestream, 1:length(source), t)


"""
Returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to
the positions, conjugate positions, strengths and conjugate strengths  of the singularities.
Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to Q-iΓ.
"""
function analytical_jacobian_pressure(target, source::T, freestream, idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t) where T <: Vector{PotentialFlow.Points.Point{Float64, Float64}}
	Nv = size(source, 1)
	Nx = 3*Nv
	Ny = size(target, 1)

	J = zeros(Ny, Nx)

	wtarget = zeros(ComplexF64, Ny)

	dpd = zeros(ComplexF64, Ny, Nv)
	dpdstar = zeros(ComplexF64, Ny, Nv)

	Css = zeros(ComplexF64, Nv, Nv)
	Cts = zeros(ComplexF64, Ny, Nv)

	analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, target, source, freestream, idx, t)
	return J
end


"""
Returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to
the positions, conjugate positions, strengths and conjugate strengths  of the blobs.
Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to Q-iΓ.
"""
function analytical_jacobian_pressure(target, source::T, freestream, idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
	Nv = size(source, 1)
	Nx = 3*Nv
	Ny = size(target, 1)

	J = zeros(Ny, Nx)

	wtarget = zeros(ComplexF64, Ny)

	dpd = zeros(ComplexF64, Ny, Nv)
	dpdstar = zeros(ComplexF64, Ny, Nv)

	Css = zeros(ComplexF64, Nv, Nv)
	Cts = zeros(ComplexF64, Ny, Nv)

	∂Css = zeros(Nv, Nv)
	Ctsblob = zeros(ComplexF64, Ny, Nv)
	∂Ctsblob = zeros(Ny, Nv)

	analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob, target, source, freestream, idx, t)
	return J
end


# DEPENDS ON STATE ARRANGEMENT

# In-place version for point vortices
function analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, target, source::T, freestream, idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t) where T <: Vector{PotentialFlow.Points.Point{Float64, Float64}}
	Nv = size(source, 1)
	Nx = 3*Nv
	Ny = size(target, 1)

	@assert size(J) == (Ny, Nx)
	@assert size(wtarget) == (Ny,)
	@assert size(dpd) == (Ny, Nv)
	@assert size(dpdstar) == (Ny, Nv)
	@assert size(Css) == (Nv, Nv)
	@assert size(Cts) == (Ny, Nv)

	fill!(J, 0.0)
	fill!(wtarget, 0.0*im)
	fill!(dpd, 0.0*im)
	fill!(dpdstar, 0.0*im)
	fill!(Css, 0.0*im)
	fill!(Cts, 0.0*im)

	cst = inv(2*π)

	src = vcat(source...)

	# Account for the presence of the free-stream
	U = freestream.U
	fill!(wtarget, conj(U))

	# Construct the Cauchy matrix to store 1/(zJ-zK) in Css
	@inbounds for K=1:Nv
		zK = src[K].z
		# exploit anti-symmetry of the velocity kernel
		# Diagonal entries are equal to 0
		for J=K+1:Nv
			zJ = src[J].z
			Css[J,K] = inv(zJ - zK)
			Css[K,J] = -Css[J,K]
		end
	end

	# Construct the Cauchy matrix to store (z̄-z̄J)/(|z-zJ|^2 + δ^2) and the total induced velocity in Cts and wtarget
	@inbounds for J=1:Nv
		zJ = src[J].z
		SJ = -im*src[J].S
		for i=1:Ny
			zi = target[i]
			Cts[i,J] = inv(zi - zJ)
			wtarget[i] += cst*SJ*Cts[i,J]
		end
	end

	analytical_jacobian_position!(dpd, dpdstar, Css, Cts, wtarget, target, src, freestream, idx, t;
								  iscauchystored = true)

	# Fill dpdpx and dpdy
	# J[:, 1:3:3*(Nv-1)+1] .= 2*real.(dpd[:,1:Nv])
	# J[:, 2:3:3*(Nv-1)+2] .= -2*imag.(dpd[:,1:Nv])
	@inbounds for L in idx
		J[:, 3*(L-1)+1] .= 2*real.(view(dpd,:,L))
		J[:, 3*(L-1)+2] .= -2*imag.(view(dpd,:,L))
	end

	analytical_jacobian_strength!(dpd, dpdstar, Css, Cts, wtarget, target, src, freestream, idx, t;
								  iscauchystored = true)

	@inbounds for L in idx
  		J[:, 3*(L-1)+3] .=  2*imag.(view(dpd,:,L))
	end
	# Vortices
	# J[:, 3:3:3*(Nv-1)+3] .= 2imag.(dpd[:,1:Nv])

	return J
end

# DEPENDS ON STATE ARRANGEMENT

# In-place version for regularized vortices
function analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob, target, source::T, freestream, idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
    Nv = size(source, 1)
    Nx = 3*Nv
    Ny = size(target, 1)

	@assert size(J) == (Ny, Nx)
	@assert size(wtarget) == (Ny,)
	@assert size(dpd) == (Ny, Nv)
	@assert size(dpdstar) == (Ny, Nv)
	@assert size(Css) == (Nv, Nv)
	@assert size(Cts) == (Ny, Nv)
	@assert size(∂Css) == (Nv, Nv)
	@assert typeof(∂Css) <: Matrix{Float64}
	@assert size(Ctsblob) == (Ny, Nv)
	@assert typeof(Ctsblob) <: Matrix{ComplexF64}
	@assert size(∂Ctsblob) == (Ny, Nv)
	@assert typeof(∂Ctsblob) <: Matrix{Float64}

	fill!(J, 0.0)
	fill!(wtarget, 0.0*im)
	fill!(dpd, 0.0*im)
	fill!(dpdstar, 0.0*im)
	fill!(Css, 0.0*im)
	fill!(Cts, 0.0*im)
	fill!(∂Css, 0.0)
	fill!(Ctsblob, 0.0*im)
	fill!(∂Ctsblob, 0.0)

    cst = inv(2*π)

    src = vcat(source...)
	δ = src[1].δ

	# Account for the presence of the free-stream
	U = freestream.U
	fill!(wtarget, conj(freestream.U))

	# Construct the Cauchy matrix to store 1/(zJ-zK) in Css
	@inbounds for K=1:Nv
		zK = src[K].z
		# exploit anti-symmetry of the velocity kernel
		# Diagonal entries are equal to 0
		for J=K+1:Nv
			zJ = src[J].z
			tmp = abs2(zJ - zK) + δ^2
			Css[J,K] = conj(zJ - zK)/tmp
			Css[K,J] = -Css[J,K]
			∂Css[J,K] = -(δ/tmp)^2
			∂Css[K,J] =  ∂Css[J,K]
		end
	end

	# Construct the Cauchy matrix to store (z̄-z̄J)/(|z-zJ|^2 + δ^2) and the total induced velocity in Cts and wtarget
	@inbounds for J=1:Nv
		zJ = src[J].z
		SJ = -im*src[J].S
		for i=1:Ny
			zi = target[i]
			tmp = abs2(zi - zJ) + δ^2
			Cts[i,J] = inv(zi - zJ)
			Ctsblob[i,J] = conj(zi - zJ)/tmp
			∂Ctsblob[i,J] = -(δ/tmp)^2
			wtarget[i] += cst*SJ*Ctsblob[i,J]
		end
	end

	analytical_jacobian_position!(dpd, dpdstar, Css, ∂Css, Cts, Ctsblob, ∂Ctsblob, wtarget, target, src, freestream, idx, t;
								  iscauchystored = true)

	# Fill dpdpx and dpdy
    # J[:, 1:3:3*(Nv-1)+1] .= 2*real.(dpd[:,1:Nv])
    # J[:, 2:3:3*(Nv-1)+2] .= -2*imag.(dpd[:,1:Nv])
	@inbounds for L in idx
		J[:, 3*(L-1)+1] .= 2*real.(view(dpd,:,L))
		J[:, 3*(L-1)+2] .= -2*imag.(view(dpd,:,L))
	end

	analytical_jacobian_strength!(dpd, dpdstar, Css, Cts, Ctsblob, wtarget, target, src, freestream, idx, t;
								  iscauchystored = true)

	@inbounds for L in idx
		J[:, 3*(L-1)+3] .=  2*imag.(view(dpd,:,L))
	end

    # Vortices
    # J[:, 3:3:3*(Nv-1)+3] .= 2imag.(dpd[:,1:Nv])
    return J
end


# function analytical_jacobian_pressure(target, source::T, freestream, t; issourcefixed::Bool=false) where T <: Union{Vector{PotentialFlow.Points.Point{Float64, Float64}}, Vector{PotentialFlow.Points.Point{ComplexF64, Float64}}, Vector{PotentialFlow.Points.Point{S, Float64} where S<:Number}}
#
#
#     Nv = size(source[1], 1)
#     Ns = size(source[2], 1)
#     Nsing =  Nv + Ns
#     Nx = 3*Nv + Ns
#     Ny = size(target, 1)
#
#     J = zeros(Ny, Nx)
#     cst = inv(2*π)
#
#     wtarget = fill(conj(freestream.U), Ny)
#
#     dpd = zeros(ComplexF64, Ny, Nsing)
#     dpdstar = zeros(ComplexF64, Ny, Nsing)
#
#     Css = zeros(ComplexF64, Nsing, Nsing)
#     Cts = zeros(ComplexF64, Ny, Nsing)
#
#     src = vcat(source...)
#
#     # Construct the Cauchy matrix to store 1/(zJ-zK) in Css
#     @inbounds for K=1:Nsing
#         zK = src[K].z
#         # exploit anti-symmetry of the velocity kernel
#         for J=K+1:Nsing
#             zJ = src[J].z
#             Css[J,K] = inv(zJ - zK)
#             Css[K,J] = -Css[J,K]
#         end
#     end
#
#     # Construct the Cauchy matrix to store 1/(z-zJ) and the total induced velocity in Cts and wtarget
#     @avx for J=1:Nsing
#         zJ = src[J].z
#         SJ = -im*src[J].S
#         for i=1:Ny
#             zi = target[i]
#             Cts[i,J] = inv(zi - zJ)
#             wtarget[i] += Cts[i,J]*cst*SJ
#         end
#     end
#
#     analytical_jacobian_position!(dpd, dpdstar, Css, Cts, wtarget, target, src, freestream, t; iscauchystored = true, issourcefixed = issourcefixed)
#
# 	# Fill dpdpx and dpdy
#     J[:, 1:3:3*(Nv-1)+1] .= 2*real.(dpd[:,1:Nv])
#     J[:, 2:3:3*(Nv-1)+2] .= -2*imag.(dpd[:,1:Nv])
#
# 	analytical_jacobian_strength!(dpd, dpdstar, Css, Cts, wtarget, target, src, freestream, t; iscauchystored = true, issourcefixed = issourcefixed)
#
#     # Vortices
#     J[:, 3:3:3*(Nv-1)+3] .= 2imag.(dpd[:,1:Nv])
#     # # Sources
#     J[:, 3*Nv+1:3*Nv+Ns] .= 2*real.(dpd[:,Nv+1:Nv+Ns])
#     return J
# end
