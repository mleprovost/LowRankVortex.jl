export symmetric_analytical_jacobian_strength!,
       symmetric_analytical_jacobian_strength,
       symmetric_analytical_jacobian_position!,
       symmetric_analytical_jacobian_position,
       symmetric_analytical_jacobian_pressure

# Version for point singularities
"""
Returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to the position and conjugate position of the singularities.
Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to Q-iΓ.
"""
function symmetric_analytical_jacobian_position!(dpdz, dpdzstar, Css, Cts, wtarget, target::Vector{Float64}, source::T,
	                                   idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t;
                                       iscauchystored::Bool = false) where T <: Vector{PotentialFlow.Points.Point{Float64, Float64}}
	Nx = size(source, 1)
	@assert mod(Nx, 2) == 0
	halfNx = Nx÷2
	Ny = size(target, 1)

	cst = inv(2*π)

	# @assert size(dpdz) == (Ny, Nx) && size(dpdzstar) == (Ny, Nx) && size(Css) == (Nx, Nx) && size(Cts) == (Ny, Nx) && size(wtarget) == (Ny,)

	fill!(dpdz, complex(0.0)); fill!(dpdzstar, complex(0.0))

	if iscauchystored == false
	   fill!(Css, complex(0.0)); fill!(Cts, complex(0.0))

	   # the induced velocity is real along the real ais for a symmetric vorticity distribution
	   fill!(wtarget, 0.0)

	   # Construct the Cauchy matrix to store 1/(zJ-zK) in Css
	   for K=1:halfNx
	       zK = source[K].z
	       # exploit anti-symmetry of the velocity kernel
	       for J=K:halfNx
	           zJ = source[J].z
	           if K == J
	               tmpAJK = 0.0*im
				   tmpBJK = -0.5*im/imag(zJ)
				   tmpCJK = -tmpBJK
	               tmpDJK = 0.0*im
	           else
	               tmpAJK = inv(zJ - zK)
				   tmpBJK = inv(zJ - conj(zK))
				   tmpCJK = conj(tmpBJK)
				   tmpDJK = conj(tmpAJK)
	           end


	           # 	    | K | K̄ |
	           # | J |    A   B
	           # | J̄ |    C   D
	           # Fill top diagonal block
	           Css[J,K] =  tmpAJK
	           Css[K,J] = -tmpAJK

	           # Top right off-diagonal block
	           Css[J,halfNx+K] =  tmpBJK
	           Css[K,halfNx+J] = -tmpCJK

	           # Lower right off-diagonal block
	           Css[halfNx+J,K] =  tmpCJK
	           Css[halfNx+K,J] = -tmpBJK

	           # Lower diagonal block
	           Css[halfNx+J,halfNx+K] =  tmpDJK
	           Css[halfNx+K,halfNx+J] = -tmpDJK
	       end
	   end

		# Construct the Cauchy matrix to store 1/(z-zJ) and the total induced velocity in Cts and wtarget
			@inbounds for J=1:halfNx
				zJ = source[J].z
				SJ = -im*source[J].S
				for i=1:Ny
					xi = target[i]
					tmpiJ = inv(xi - zJ)
					Cts[i,J] = tmpiJ
					Cts[i,halfNx+J] = conj(tmpiJ) #as xi is real
					wtarget[i] +=  real(Cts[i,J]*SJ)# + Cts[i,halfNx+J]*conj(SJ))
			end
		end
		wtarget .*= 2*cst
	end

	# Evaluate ∂(-0.5v^2)/∂zL, ∂(-0.5v^2)/∂z̄L
	@inbounds for L in idx #1:Nx
	   zL = source[L].z
	   SL = -im*source[L].S
	   for i=1:Ny
	       wt = wtarget[i]
		   # wt is real for a symmetric vortex configuration evaluated on the real axis.
	       dpdz[i,L] = -0.5*wt*SL*cst*Cts[i,L]^2
	       # dpdzstar[i,L] = conj(dpdz[i,L])
	   end
	end

	# Evaluate ∂(-∂ϕ/∂t)/∂zL, ∂(-∂ϕ/∂t)/∂z̄L
	@inbounds for L in idx #1:Nx
		SL = -im*source[L].S
		for K=1:Nx
			SK = -im*source[K].S
			for i=1:Ny
				term1 = 0.5*cst^2*SL*conj(SK)*(Cts[i,L])^2*conj(Css[L,K])
				term1 += 0.5*cst^2*conj(SL)*SK*conj(Cts[i,L])*(-Css[L,K]^2)
				dpdz[i,L] += term1
				# dpdzstar[i,L] += conj(term1)
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


###############################################################################################
##################################### BLOBS ###################################################
###############################################################################################

# Version for regularized point singularities (aka blobs)
"""
Returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to the position and conjugate position of the singularities.
Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to Q-iΓ.
"""
function symmetric_analytical_jacobian_position!(dpdz, dpdzstar, Css, ∂Css, Cts, Ctsblob, ∂Ctsblob, wtarget, target::Vector{Float64}, source::T,
									             idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t;
                                                 iscauchystored::Bool = false) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
	Ny = size(target, 1)
	Nx = size(source, 1)

	halfNx = Nx÷2

	δ = source[1].δ

	cst = inv(2*π)

	# @assert size(dpdz) == (Ny, Nx) && size(dpdzstar) == (Ny, Nx) && size(Css) == (Nx, Nx) && size(Cts) == (Ny, Nx) && size(wtarget) == (Ny,)

	fill!(dpdz, complex(0.0)); fill!(dpdzstar, complex(0.0))

	if iscauchystored == false
	   fill!(Css, complex(0.0));
	   fill!(∂Css, 0.0);
	   fill!(Cts, complex(0.0));
	   fill!(Ctsblob, complex(0.0));
	   fill!(∂Ctsblob, 0.0)

	   fill!(wtarget, 0.0)

	   # Construct the Cauchy matrix to store 1/(zJ-zK) in Css
	   for K=1:halfNx
		   zK = source[K].z
		   # exploit anti-symmetry of the velocity kernel
		   for J=K:halfNx
			   zJ = source[J].z

			   ΔdiagJK = conj(zJ - zK)
			   ΔoffJK =  conj(zJ) - zK

			   tmpdiagJK  = abs2(ΔdiagJK) + δ^2
			   tmpoffJK  = abs2(ΔoffJK) + δ^2
			   tmpAJK = ΔdiagJK/tmpdiagJK
			   tmpBJK = ΔoffJK/tmpoffJK
			   tmpCJK = conj(tmpBJK)
			   tmpDJK = conj(tmpAJK)

			   if J == K
				   tmp∂AJK = 0.0
				   tmp∂DJK = 0.0
			   else
				   tmp∂AJK = -(δ/tmpdiagJK)^2
				   tmp∂DJK = tmp∂AJK
		   	   end

			   tmp∂BJK = -(δ/tmpoffJK)^2
			   tmp∂CJK = tmp∂BJK


			   # 	    | K | K̄ |
			   # | J |    A   B
			   # | J̄ |    C   D
			   # Fill top diagonal block
			   Css[J,K] =  tmpAJK
			   Css[K,J] = -tmpAJK

			   ∂Css[J,K] =  tmp∂AJK
			   ∂Css[K,J] =  tmp∂AJK

			   # Top right off-diagonal block
			   Css[J,halfNx+K] =  tmpBJK
			   Css[K,halfNx+J] = -tmpCJK

			   ∂Css[J,halfNx+K] =  tmp∂BJK
			   ∂Css[K,halfNx+J] =  tmp∂BJK

			   # Lower right off-diagonal block
			   Css[halfNx+J,K] =  tmpCJK
			   Css[halfNx+K,J] = -tmpBJK

			   ∂Css[halfNx+J,K] =  tmp∂CJK
			   ∂Css[halfNx+K,J] =  tmp∂CJK

			   # Lower diagonal block
			   Css[halfNx+J,halfNx+K] =  tmpDJK
			   Css[halfNx+K,halfNx+J] = -tmpDJK

			   ∂Css[halfNx+J,halfNx+K] =  tmp∂DJK
			   ∂Css[halfNx+K,halfNx+J] =  tmp∂DJK
		   end
	   end

	   # Construct the Cauchy matrix to store (z̄-z̄J)/(|z-zJ|^2 + δ^2) and the total induced velocity in Cts and wtarget
	   @inbounds for J=1:halfNx
	       zJ = source[J].z
	       ΓJ = source[J].S
	       for i=1:Ny
	           xi = target[i]
			   ΔiJ = xi - zJ
			   tmpregiJ = abs2(ΔiJ) + δ^2
			   tmp1iJ = inv(ΔiJ)
			   tmp2iJ = conj(ΔiJ)/tmpregiJ

			   Cts[i,J] = tmp1iJ
			   Cts[i,halfNx+J] = conj(Cts[i,J])

	           Ctsblob[i,J] = tmp2iJ
			   Ctsblob[i,halfNx+J] = conj(Ctsblob[i,J])

			   ∂Ctsblob[i,J] = -(δ/tmpregiJ)^2
			   ∂Ctsblob[i,halfNx+J] = ∂Ctsblob[i,J]

	           wtarget[i] += ΓJ*imag(Ctsblob[i,J])
	       end
	   end
	   wtarget .*= (1/π)
	end

	# Evaluate ∂(-∂ϕ/∂t)/∂zL, ∂(-∂ϕ/∂t)/∂z̄L
	@inbounds for L in idx #1:Nx
		SL = -im*source[L].S
		term1 = 0.0*im
		term2 = 0.0*im
		term3 = 0.0*im
		term4 = 0.0*im
		tmp = 0.0*im
		tmp2 = 0.0*im

		# Terms in δJL
		for K=1:Nx
			SK = -im*source[K].S
			tmp = SK*Css[L,K]
			tmp2 = conj(SK*∂Css[L,K])
			term1 += tmp
			term2 -= tmp2
			term3 -= tmp*Css[L,K]
			# if K !=L
			# 	for i=1:Ny
			# 		# We have ∂Css[K,L] = ∂Css[L,K]
			# 		dpdz[i,L] += SK*conj(SL)*Cts[i,K]*∂Css[K,L]
			# 	end
			# end
		end

		for i=1:Ny
			tmp = SL*Cts[i,L]
			dpdz[i,L] += tmp*(Cts[i,L]*conj(term1)+term2)
			dpdz[i,L] += conj(tmp)*term3
		end

		# Terms in δKL

		for J=1:Nx
			SJ = -im*source[J].S
			for i=1:Ny
	           termKL = SJ*conj(SL)*Cts[i,J]*∂Css[J,L]
			   termKL += conj(SJ)*SL*conj(Cts[i,J])*Css[J,L]^2
	           dpdz[i,L] += termKL

	           # dpdzstar[i,L] += conj(termKL)
		   end
	   end
	end

	Γ = getfield.(source, :S)
	# dpdz[:,idx] .+= conj(Cts) .* (im*Γ') .*  (Css .^2 .* -im*Γ')

	dpdz[:,idx] .*=  0.5*cst^2

	# Evaluate ∂(-0.5v^2)/∂zL, ∂(-0.5v^2)/∂z̄L
	@inbounds for L in idx #1:Nx
	   zL = source[L].z
	   SL = -im*source[L].S
	   for i=1:Ny
		   wt = wtarget[i]
		   # dpdz[i,L] = -0.5*cst*conj(SL)*∂Ctsblob[i,L]*wt
		   # dpdz[i,L] += -0.5*conj(wt)*cst*SL*Ctsblob[i,L]^2
		   # Exploit the fact that w(xi) is real
		   dpdz[i,L] -= 0.5*wt*cst*SL*(Ctsblob[i,L]^2-∂Ctsblob[i,L])
		   # dpdzstar[i,L] = conj(dpdz[i,L])
	   end
	end


	dpdzstar .= conj(dpdz)
	nothing
end

function symmetric_analytical_jacobian_position!(dpdz, dpdzstar, Css, Cts, wtarget, target::Vector{Float64}, source::T, t;
                                                 iscauchystored::Bool = false) where T <:Vector{PotentialFlow.Points.Point{Float64, Float64}}

	symmetric_analytical_jacobian_position!(dpdz, dpdzstar, Css, Cts, wtarget, target, source,
	                                        1:length(source), t; iscauchystored = iscauchystored)
end

function symmetric_analytical_jacobian_position!(dpdz, dpdzstar, Css, ∂Css, Cts, Ctsblob, ∂Ctsblob,
	                                             wtarget, target::Vector{Float64}, source::T, t;
                                                 iscauchystored::Bool = false) where T <:Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}

	symmetric_analytical_jacobian_position!(dpdz, dpdzstar, Css, ∂Css, Cts, Ctsblob, ∂Ctsblob,
		                                    wtarget, target, source,
	                                        1:length(source), t; iscauchystored = iscauchystored)
end


function symmetric_analytical_jacobian_position(target::Vector{Float64}, source,
									            idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t)
   Ny = size(target, 1)
   Nx = size(source, 1)
   # Create allocations for the Cauchy matrices
   Css = zeros(ComplexF64, Nx, Nx)
   Cts = zeros(ComplexF64, Ny, Nx)
   wtarget = zeros(Ny)

   dpdz = zeros(ComplexF64, Ny, Nx)
   dpdzstar = zeros(ComplexF64, Ny, Nx)

	if typeof(source)<:Vector{PotentialFlow.Points.Point{Float64, Float64}}
	    symmetric_analytical_jacobian_position!(dpdz, dpdzstar, Css, Cts, wtarget, target, source, idx, t;
	                                  iscauchystored = false)
	elseif typeof(source)<:Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
		∂Css = zeros(Nx, Nx)
		Ctsblob = zeros(ComplexF64, Ny, Nx)
		∂Ctsblob = zeros(Ny, Nx)
		symmetric_analytical_jacobian_position!(dpdz, dpdzstar, Css, ∂Css, Cts, Ctsblob, ∂Ctsblob, wtarget, target, source, idx, t;
									  iscauchystored = false)
	end

   return dpdz, dpdzstar
end

symmetric_analytical_jacobian_position(target::Vector{Float64}, source, t) = symmetric_analytical_jacobian_position(target, source, 1:length(source), t)

"""
Returns the Jacobian of the pressure computed from the unsteady Bernoulli equation with respect to the strength and conjugate strength of the singularities.
Note that we multiply the strength `point.S` of a singularity by -i to move from the convention Γ+iQ (used in PotentialFlow.jl) to S = Q-iΓ.
"""
function symmetric_analytical_jacobian_strength!(dpdS, dpdSstar, Css, Cts, wtarget, target::Vector{Float64}, source::T,
	                                   idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t;
                                       iscauchystored::Bool = false) where T <: Vector{PotentialFlow.Points.Point{Float64, Float64}}
	Ny = size(target, 1)
    Nx = size(source, 1)

	halfNx = Nx÷2

    cst = inv(2*π)

    # @assert size(dpdS) == (Ny, Nx) && size(dpdSstar) == (Ny, Nx) && size(Css) == (Nx, Nx) && size(Cts) == (Ny, Nx) && size(wtarget) == (Ny,)

    fill!(dpdS, complex(0.0)); fill!(dpdSstar, complex(0.0))

    if iscauchystored == false

        fill!(Css, complex(0.0)); fill!(Cts, complex(0.0))

        # Account for the presence of the free-stream
        fill!(wtarget, 0.0)

        # Construct the Cauchy matrix to store 1/(zJ-zK) in Css
		for K=1:halfNx
 	       zK = source[K].z
 	       # exploit anti-symmetry of the velocity kernel
 	       for J=K:halfNx
 	           zJ = source[J].z
 	           if K == J
 	               tmpAJK = 0.0*im
 				   tmpBJK = -0.5*im/imag(zJ)
 				   tmpCJK = -tmpBJK
 	               tmpDJK = 0.0*im
 	           else
 	               tmpAJK = inv(zJ - zK)
 				   tmpBJK = inv(zJ - conj(zK))
 				   tmpCJK = conj(tmpBJK)
 				   tmpDJK = conj(tmpAJK)
 	           end


 	           # 	    | K | K̄ |
 	           # | J |    A   B
 	           # | J̄ |    C   D
 	           # Fill top diagonal block
 	           Css[J,K] =  tmpAJK
 	           Css[K,J] = -tmpAJK

 	           # Top right off-diagonal block
 	           Css[J,halfNx+K] =  tmpBJK
 	           Css[K,halfNx+J] = -tmpCJK

 	           # Lower right off-diagonal block
 	           Css[halfNx+J,K] =  tmpCJK
 	           Css[halfNx+K,J] = -tmpBJK

 	           # Lower diagonal block
 	           Css[halfNx+J,halfNx+K] =  tmpDJK
 	           Css[halfNx+K,halfNx+J] = -tmpDJK
 	       end
 	   end

 		# Construct the Cauchy matrix to store 1/(z-zJ) and the total induced velocity in Cts and wtarget
 			@inbounds for J=1:halfNx
 				zJ = source[J].z
 				SJ = -im*source[J].S
 				for i=1:Ny
 					xi = target[i]
 					tmpiJ = inv(xi - zJ)
 					Cts[i,J] = tmpiJ
 					Cts[i,halfNx+J] = conj(tmpiJ) #as xi is real
 					wtarget[i] +=  real(Cts[i,J]*SJ)# + Cts[i,halfNx+J]*conj(SJ))
 			end
 		end
 		wtarget .*= 2*cst
    end

    # Evaluate ∂(-0.5v^2)/∂SL, ∂(-0.5v^2)/∂S̄L
    @avx for L in idx #1:Nx
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
			if K != L
				SK = -im*source[K].S
				for i=1:Ny
					term1 = 0.5*cst^2*conj(SK)*Cts[i,L]*conj(Css[L,K])
					dpdS[i,L] += term1
					# dpdSstar[i,L] += conj(term1)
				end
			end
		end

	    for J=1:Nx
			if J != L
		        SJ = -im*source[J].S
	       		for i=1:Ny
	                term2 = 0.5*cst^2*conj(SJ)*conj(Cts[i,J])*Css[J,L]
	                dpdS[i,L] += term2
	                # dpdSstar[i,L] += conj(term2)
	            end
			end
        end
    end
	dpdSstar .= conj(dpdS)
    nothing
end



###############################################################################################
##################################### BLOBS ###################################################
###############################################################################################


function symmetric_analytical_jacobian_strength!(dpdS, dpdSstar, Css, Cts, Ctsblob, wtarget, target::Vector{Float64}, source::T,
	                                   idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t;
	                       iscauchystored::Bool = false, issourcefixed::Bool = false) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
	Ny = size(target, 1)
	Nx = size(source, 1)
	@assert mod(Nx, 2) == 0
	halfNx = Nx÷2

	δ = source[1].δ

	cst = inv(2*π)

	# @assert size(dpdS) == (Ny, Nx) && size(dpdSstar) == (Ny, Nx) && size(Css) == (Nx, Nx) && size(Cts) == (Ny, Nx) && size(wtarget) == (Ny,)

	fill!(dpdS, complex(0.0)); fill!(dpdSstar, complex(0.0))

	if iscauchystored == false
		fill!(Css, complex(0.0));
		fill!(Cts, complex(0.0));
		fill!(Ctsblob, complex(0.0));

		fill!(wtarget, 0.0)

		# Construct the Cauchy matrix to store 1/(zJ-zK) in Css
		@inbounds for K=1:halfNx
			# exploit anti-symmetry of the velocity kernel
			for J=K:halfNx
				zK = source[K].z
				zJ = source[J].z

				ΔdiagJK = conj(zJ - zK)
				ΔoffJK =  conj(zJ) - zK

				tmpdiagJK  = abs2(ΔdiagJK) + δ^2
				tmpoffJK  = abs2(ΔoffJK) + δ^2
				tmpAJK = ΔdiagJK/tmpdiagJK
				tmpBJK = ΔoffJK/tmpoffJK
				tmpCJK = conj(tmpBJK)
				tmpDJK = conj(tmpAJK)

				# 	    | K | K̄ |
				# | J |    A   B
				# | J̄ |    C   D
				# Fill top diagonal block
				Css[J,K] =  tmpAJK
				Css[K,J] = -tmpAJK

				# Top right off-diagonal block
				Css[J,halfNx+K] =  tmpBJK
				Css[K,halfNx+J] = -tmpCJK

				# Lower right off-diagonal block
				Css[halfNx+J,K] =  tmpCJK
				Css[halfNx+K,J] = -tmpBJK

				# Lower diagonal block
				Css[halfNx+J,halfNx+K] =  tmpDJK
				Css[halfNx+K,halfNx+J] = -tmpDJK
			end

			# zv .= getfield.(source, :z)
			# Γv .= getfield.(source, :S)
		end

		# Construct the Cauchy matrix to store (z̄-z̄J)/(|z-zJ|^2 + δ^2) and the total induced velocity in Cts and wtarget
		@inbounds for J=1:halfNx
			zJ = source[J].z
			yJ = imag(zJ)
			ΓJ = source[J].S
			for i=1:Ny
				xi = target[i]
				ΔiJ = xi - zJ
				tmpregiJ = abs2(ΔiJ) + δ^2
				tmp1iJ = inv(ΔiJ)
				tmp2iJ = conj(ΔiJ)/tmpregiJ

				Cts[i,J] = tmp1iJ
				Cts[i,halfNx+J] = conj(Cts[i,J])

				Ctsblob[i,J] = tmp2iJ
				Ctsblob[i,halfNx+J] = conj(Ctsblob[i,J])

				wtarget[i] += ΓJ*imag(Ctsblob[i,J])
			end
		end
		wtarget .*= (1/π)
	end

	# Evaluate ∂(-∂ϕ/∂t)/∂SL, ∂(-∂ϕ/∂t)/∂S̄L
	@inbounds for L in idx
		term1 = 0.0*im
		for K=1:Nx
			# if K != L
				# Css[K.L] = 0
				SK = -im*source[K].S
				term1 += conj(SK * Css[L,K])
			# end
		end
		for i=1:Ny
			dpdS[i,L] += Cts[i,L]*term1
		end
	end

	# @inbounds dpdS[:,idx] .+= Cts[:,idx] .* conj(Css[idx,:] * (-im*Γ))'

	Γ = getfield.(source, :S)

	@inbounds dpdS[:,idx] .+= conj(Cts[:,idx])*(im*Γ .* Css[:,idx])

	dpdS .*= 0.5*cst^2

	# Evaluate ∂(-0.5v^2)/∂SL, ∂(-0.5v^2)/∂S̄L
	@inbounds dpdS[:,idx] .+= -0.5*cst*(wtarget .* Ctsblob[:,idx])
	# @inbounds for i=1:Ny
	# 	dpdS[i,:] .= -0.5*cst*wtarget[i]
	# 	for L in idx
	# 		dpdS[i,L] *= Ctsblob[i,L]
	# 		# dpdSstar[i,L] = conj(dpdS[i,L])
	# 	end
	# end

	dpdSstar .= conj(dpdS)
	nothing
end

symmetric_analytical_jacobian_strength!(dpdS, dpdSstar, Css, Cts, Ctsblob, wtarget, target::Vector{Float64}, source, t;
	                          iscauchystored::Bool = false) =
symmetric_analytical_jacobian_strength!(dpdS, dpdSstar, Css, Cts, Ctsblob, wtarget, target, source, 1:length(source), t;
							  iscauchystored = iscauchystored)

symmetric_analytical_jacobian_strength(target::Vector{Float64}, source, t) = symmetric_analytical_jacobian_strength(target, source, 1:length(source), t)

function symmetric_analytical_jacobian_strength(target::Vector{Float64}, source,
	                                  idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t)
	Ny = size(target, 1)
	Nx = size(source, 1)
	Css = zeros(ComplexF64, Nx, Nx)
	Cts = zeros(ComplexF64, Ny, Nx)
	wtarget = zeros(Ny)

	dpdS = zeros(ComplexF64, Ny, Nx)
	dpdSstar = zeros(ComplexF64, Ny, Nx)
	if typeof(source)<:Vector{PotentialFlow.Points.Point{Float64, Float64}}
		symmetric_analytical_jacobian_strength!(dpdS, dpdSstar, Css, Cts, wtarget, target, source, idx, t;
	                                            iscauchystored = false)
	elseif typeof(source)<:Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
		Ctsblob = zeros(ComplexF64, Ny, Nx)
		symmetric_analytical_jacobian_strength!(dpdS, dpdSstar, Css, Cts, Ctsblob, wtarget, target, source, idx, t;
									            iscauchystored = false)
	end

	return dpdS, dpdSstar
end


symmetric_analytical_jacobian_pressure(target, source, t) = symmetric_analytical_jacobian_pressure(target, source, 1:length(source), t)

# Version with allocations for point vortices
function symmetric_analytical_jacobian_pressure(target, source::T, idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t) where T <: Vector{PotentialFlow.Points.Point{Float64, Float64}}
	Nv = size(source, 1)
	Nx = 3*Nv
	Ny = size(target, 1)

	J = zeros(Ny, Nx)

	wtarget = zeros(Ny)

	dpd = zeros(ComplexF64, Ny, Nv)
	dpdstar = zeros(ComplexF64, Ny, Nv)

	Css = zeros(ComplexF64, Nv, Nv)
	Cts = zeros(ComplexF64, Ny, Nv)

	symmetric_analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, target, source, idx, t)
	return J
end


# Version with allocations for regularized vortices
function symmetric_analytical_jacobian_pressure(target, source::T, idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
	Nv = size(source, 1)
	Nx = 3*Nv
	Ny = size(target, 1)

	J = zeros(Ny, Nx)

	wtarget = zeros(Ny)

	dpdz = zeros(ComplexF64, Ny, Nv)
	dpdS = zeros(ComplexF64, Ny, Nv)

	Css = zeros(ComplexF64, Nv, Nv)
	Cts = zeros(ComplexF64, Ny, Nv)

	∂Css = zeros(Nv, Nv)
	Ctsblob = zeros(ComplexF64, Ny, Nv)
	∂Ctsblob = zeros(Ny, Nv)

	symmetric_analytical_jacobian_pressure!(J, wtarget, dpdz, dpdS, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob, target, source, idx, t)
	return J
end


###############################################################################################
##################################### BLOBS ###################################################
###############################################################################################

# In-place version for regularized vortices
function symmetric_analytical_jacobian_pressure!(J, wtarget, dpdz, dpdS, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob, target, source::T, idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
	Ny = size(target, 1)
	Nx = size(source, 1)
	@assert mod(Nx, 2) == 0
	halfNx = Nx÷2

	δ = source[1].δ

	cst = inv(2*π)

	# @assert size(dpdS) == (Ny, Nx) && size(dpdSstar) == (Ny, Nx) && size(Css) == (Nx, Nx) && size(Cts) == (Ny, Nx) && size(wtarget) == (Ny,)

	fill!(dpdz, complex(0.0)); fill!(dpdS, complex(0.0))

	fill!(Css, complex(0.0));
	fill!(Cts, complex(0.0));
	fill!(∂Css, complex(0.0));
	fill!(Cts, complex(0.0));
	fill!(Ctsblob, complex(0.0));
	fill!(∂Ctsblob, complex(0.0));
	fill!(wtarget, 0.0)

	# Construct the Cauchy matrix to store 1/(zJ-zK) in Css
	@inbounds for K=1:halfNx
		zK = source[K].z
		# exploit anti-symmetry of the velocity kernel
		for J=K:halfNx
			zJ = source[J].z

			ΔdiagJK = conj(zJ - zK)
			ΔoffJK =  conj(zJ) - zK

			tmpdiagJK  = abs2(ΔdiagJK) + δ^2
			tmpoffJK  = abs2(ΔoffJK) + δ^2
			tmpAJK = ΔdiagJK/tmpdiagJK
			tmpBJK = ΔoffJK/tmpoffJK
			tmpCJK = conj(tmpBJK)
			tmpDJK = conj(tmpAJK)

			if J == K
				tmp∂AJK = 0.0
				tmp∂DJK = 0.0
			else
				tmp∂AJK = -(δ/tmpdiagJK)^2
				tmp∂DJK = tmp∂AJK
			end

			tmp∂BJK = -(δ/tmpoffJK)^2
			tmp∂CJK = tmp∂BJK

			# 	    | K | K̄ |
			# | J |    A   B
			# | J̄ |    C   D
			# Fill top diagonal block
			Css[J,K] =  tmpAJK
			Css[K,J] = -tmpAJK

		   ∂Css[J,K] =  tmp∂AJK
		   ∂Css[K,J] =  tmp∂AJK

			# Top right off-diagonal block
			Css[J,halfNx+K] =  tmpBJK
			Css[K,halfNx+J] = -tmpCJK

		   ∂Css[J,halfNx+K] =  tmp∂BJK
		   ∂Css[K,halfNx+J] =  tmp∂BJK

			# Lower right off-diagonal block
			Css[halfNx+J,K] =  tmpCJK
			Css[halfNx+K,J] = -tmpBJK

		   ∂Css[halfNx+J,K] =  tmp∂CJK
		   ∂Css[halfNx+K,J] =  tmp∂CJK

			# Lower diagonal block
			Css[halfNx+J,halfNx+K] =  tmpDJK
			Css[halfNx+K,halfNx+J] = -tmpDJK

			∂Css[halfNx+J,halfNx+K] =  tmp∂DJK
			∂Css[halfNx+K,halfNx+J] =  tmp∂DJK
		end

		# zv .= getfield.(source, :z)
		# Γv .= getfield.(source, :S)
	end

	# Construct the Cauchy matrix to store (z̄-z̄J)/(|z-zJ|^2 + δ^2) and the total induced velocity in Cts and wtarget
	@inbounds for J=1:halfNx
		zJ = source[J].z
		ΓJ = source[J].S
		for i=1:Ny
			xi = target[i]
			ΔiJ = xi - zJ
			tmpregiJ = abs2(ΔiJ) + δ^2
			tmp1iJ = inv(ΔiJ)
			tmp2iJ = conj(ΔiJ)/tmpregiJ

			Cts[i,J] = tmp1iJ
			Cts[i,halfNx+J] = conj(Cts[i,J])

			Ctsblob[i,J] = tmp2iJ
			Ctsblob[i,halfNx+J] = conj(Ctsblob[i,J])

			∂Ctsblob[i,J] = -(δ/tmpregiJ)^2
			∂Ctsblob[i,halfNx+J] = ∂Ctsblob[i,J]

			wtarget[i] += ΓJ*imag(Ctsblob[i,J])
		end
	end
	wtarget .*= (1/π)

	# Evaluate ∂(-∂ϕ/∂t)/∂zL, ∂(-∂ϕ/∂t)/∂z̄L
	@inbounds for L in idx #1:Nx
		SL = -im*source[L].S
		term1 = 0.0*im
		term2 = 0.0*im
		term3 = 0.0*im
		term4 = 0.0*im
		tmp = 0.0*im
		tmp2 = 0.0*im

		# Terms in δJL
		for K=1:Nx
			SK = -im*source[K].S
			tmp = SK*Css[L,K]
			tmp2 = conj(SK*∂Css[L,K])
			term1 += tmp
			term2 -= tmp2
			term3 -= tmp*Css[L,K]
		end

		for i=1:Ny
			tmp3 = Cts[i,L]*conj(term1)
			dpdS[i,L] += tmp3
			tmp4 = SL*Cts[i,L]
			dpdz[i,L] += tmp4*(tmp3+term2)
			dpdz[i,L] += conj(tmp4)*term3
		end

		# Terms in δKL
		for J=1:Nx
			SJ = -im*source[J].S
			for i=1:Ny
				tmp5 = SJ*Cts[i,J]
				termKL = tmp5*conj(SL)*∂Css[J,L]
				termKL += conj(tmp5)*SL*Css[J,L]^2
				dpdz[i,L] += termKL
		   end
	   end
	end

	Γ = getfield.(source, :S)
	# dpdz[:,idx] .+= conj(Cts) .* (im*Γ') .*  (Css .^2 .* -im*Γ')

	dpdz[:,idx] .*=  0.5*cst^2

	# Evaluate ∂(-0.5v^2)/∂zL, ∂(-0.5v^2)/∂z̄L
	@inbounds for L in idx #1:Nx
	   zL = source[L].z
	   SL = -im*source[L].S
	   for i=1:Ny
		   wt = wtarget[i]
		   # Exploit the fact that w(xi) is real
		   dpdz[i,L] -= 0.5*wt*cst*SL*(Ctsblob[i,L]^2-∂Ctsblob[i,L])
	   end
	end


	# Evaluate ∂(-∂ϕ/∂t)/∂SL, ∂(-∂ϕ/∂t)/∂S̄L
	# @inbounds dpdS[:,idx] .+= Cts[:,idx] .* conj(Css[idx,:] * (-im*Γ))'
	# @show size((im*Γ .* Css[:,idx]))
	# @show size(conj(Cts[:,idx]))



	# @inbounds for i=1:Ny
	# 	for J=1:Nx
	# 		tmp = Cts[i,J]*im*Γ[J]
	# 		for L in idx
	# 			dpdS[i,L] += conj(tmp)*Css[J,L]
	# 		end
	# 	end
	# end
	@inbounds dpdS[:,idx] .+= conj(Cts.*(-im*Γ'))*view(Css,:,idx)

	# @inbounds dpdS[:,idx] .+= (conj(Cts)*(im*Γ .* Css))[:,idx]

	dpdS .*= 0.5*cst^2

	# Evaluate ∂(-0.5v^2)/∂SL, ∂(-0.5v^2)/∂S̄L
	@inbounds dpdS[:,idx] .+= -0.5*cst*(wtarget .* Ctsblob[:,idx])

	@inbounds for L in idx
		J[:, 3*(L-1)+1] .= 2*real.(view(dpdz,:,L))
		J[:, 3*(L-1)+2] .= -2*imag.(view(dpdz,:,L))
		J[:, 3*(L-1)+3] .=  2*imag.(view(dpdS,:,L))
	end

	return J
	# J[:, 1:3:3*(Nv-1)+1] .= 2*real.(dpd[:,1:Nv])

end


# # In-place version for regularized vortices
# function symmetric_analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, ∂Css, Ctsblob, ∂Ctsblob, target, source::T, idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}
#     Nv = size(source, 1)
#     Nx = 3*Nv
#     Ny = size(target, 1)
#
# 	@assert size(J) == (Ny, Nx)
# 	@assert size(wtarget) == (Ny,)
# 	@assert size(dpd) == (Ny, Nv)
# 	@assert size(dpdstar) == (Ny, Nv)
# 	@assert size(Css) == (Nv, Nv)
# 	@assert size(Cts) == (Ny, Nv)
# 	@assert size(∂Css) == (Nv, Nv)
# 	@assert typeof(∂Css) <: Matrix{Float64}
# 	@assert size(Ctsblob) == (Ny, Nv)
# 	@assert typeof(Ctsblob) <: Matrix{ComplexF64}
# 	@assert size(∂Ctsblob) == (Ny, Nv)
# 	@assert typeof(∂Ctsblob) <: Matrix{Float64}
#
# 	fill!(J, 0.0)
# 	fill!(wtarget, 0.0)
# 	fill!(dpd, 0.0*im)
# 	fill!(dpdstar, 0.0*im)
# 	fill!(Css, 0.0*im)
# 	fill!(Cts, 0.0*im)
# 	fill!(∂Css, 0.0)
# 	fill!(Ctsblob, 0.0*im)
# 	fill!(∂Ctsblob, 0.0)
#
#     cst = inv(2*π)
#
#     src = vcat(source...)
# 	δ = src[1].δ
#
# 	# Construct the Cauchy matrix to store 1/(zJ-zK) in Css
# 	@inbounds for K=1:Nv
# 		zK = src[K].z
# 		# exploit anti-symmetry of the velocity kernel
# 		# Diagonal entries are equal to 0
# 		for J=K+1:Nv
# 			zJ = src[J].z
# 			tmp = abs2(zJ - zK) + δ^2
# 			Css[J,K] = conj(zJ - zK)/tmp
# 			Css[K,J] = -Css[J,K]
# 			∂Css[J,K] = -(δ/tmp)^2
# 			∂Css[K,J] =  ∂Css[J,K]
# 		end
# 	end
#
# 	# Construct the Cauchy matrix to store (z̄-z̄J)/(|z-zJ|^2 + δ^2) and the total induced velocity in Cts and wtarget
# 	@inbounds for J=1:Nv
# 		zJ = src[J].z
# 		SJ = -im*src[J].S
# 		for i=1:Ny
# 			zi = target[i]
# 			tmp = abs2(zi - zJ) + δ^2
# 			Cts[i,J] = inv(zi - zJ)
# 			Ctsblob[i,J] = conj(zi - zJ)/tmp
# 			∂Ctsblob[i,J] = -(δ/tmp)^2
# 			wtarget[i] += cst*SJ*Ctsblob[i,J]
# 		end
# 	end
#
# 	symmetric_analytical_jacobian_position!(dpd, dpdstar, Css, ∂Css, Cts, Ctsblob, ∂Ctsblob, wtarget, target, src, idx, t;
# 								  iscauchystored = true)
#
# 	# Fill dpdpx and dpdy
#     J[:, 1:3:3*(Nv-1)+1] .= 2*real.(dpd[:,1:Nv])
#     J[:, 2:3:3*(Nv-1)+2] .= -2*imag.(dpd[:,1:Nv])
#
# 	symmetric_analytical_jacobian_strength!(dpd, dpdstar, Css, Cts, Ctsblob, wtarget, target, src, idx, t;
# 								  iscauchystored = true)
#
#     # Vortices
#     J[:, 3:3:3*(Nv-1)+3] .= 2imag.(dpd[:,1:Nv])
#     return J
# end


# In-place version for point vortices
function symmetric_analytical_jacobian_pressure!(J, wtarget, dpd, dpdstar, Css, Cts, target, source::T, idx::Union{Int64, Vector{Int64}, UnitRange{Int64}}, t) where T <: Vector{PotentialFlow.Points.Point{Float64, Float64}}
	Nv = size(source, 1)
	Nx = 3*Nv
	Ny = size(target, 1)

	@assert size(J) == (Ny, Nx)
	@assert size(wtarget) == (Ny,)
	@assert size(dpd) == (Ny, Nv)
	@assert size(dpdstar) == (Ny, Nv)
	@assert size(Css) == (Nv, Nv)
	@assert size(Cts) == (Ny, Nv)

	fill!(J, 0.0)# Construct the Cauchy matrix to store 1/(zJ-zK) in Css
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
	fill!(wtarget, 0.0)
	fill!(dpd, 0.0*im)
	fill!(dpdstar, 0.0*im)
	fill!(Css, 0.0*im)
	fill!(Cts, 0.0*im)

	cst = inv(2*π)

	src = vcat(source...)

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

	symmetric_analytical_jacobian_position!(dpd, dpdstar, Css, Cts, wtarget, target, src, idx, t;
								  iscauchystored = true)

	# Fill dpdpx and dpdy
	J[:, 1:3:3*(Nv-1)+1] .= 2*real.(dpd[:,1:Nv])
	J[:, 2:3:3*(Nv-1)+2] .= -2*imag.(dpd[:,1:Nv])

	symmetric_analytical_jacobian_strength!(dpd, dpdstar, Css, Cts, wtarget, target, src, idx, t;
								  iscauchystored = true)

	# Vortices
	J[:, 3:3:3*(Nv-1)+3] .= 2imag.(dpd[:,1:Nv])

	return J
end
