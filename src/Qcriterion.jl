export velocity_gradient!, velocity_gradient, Qcriterion_vortex!, Qcriterion_vortex, Qcriterion_vortexfield!, Qcriterion_vortexfield

velocity_gradient(target::ComplexF64, source) = velocity_gradient!(zeros(2,2), target, source)

function velocity_gradient!(∇v::Matrix{Float64}, target::ComplexF64, source::T) where T <: Vector{PotentialFlow.Points.Point{Float64, Float64}}

    Nx = size(source, 1)

    cst = -im/(2*π)
    dwdz = 0.0*im
    dwdzstar = 0.0*im

    for J = 1:Nx
        zJ = source[J].z
        ΓJ = source[J].S
        dwdz += cst*ΓJ*(-1)/(target - zJ)^2
    end

    # Populate ∇v = [∂u/∂x ∂u/∂y
    #                ∂v/∂x ∂v/∂y]

    # ∂u/∂x
    ∇v[1,1] = real(dwdz + dwdzstar)

    # ∂u/∂y
    ∇v[1,2] = real(im*(dwdz - dwdzstar))

    # ∂v/∂x
    ∇v[2,1] = -imag(dwdz + dwdzstar)

    # ∂v/∂y
    ∇v[2,2] = -imag(im*(dwdz - dwdzstar))
    return ∇v
end

function velocity_gradient!(∇v::Matrix{Float64}, target::ComplexF64, source::T) where T <: Vector{PotentialFlow.Blobs.Blob{Float64, Float64}}

    Nx = size(source, 1)

    cst = im/(2*π)
    δ = source[1].δ
    dwdz = 0.0*im
    dwdzstar = 0.0*im

    for J = 1:Nx
        zJ = source[J].z
        ΓJ = source[J].S
        # tmp = (-im*ΓJ/(2*π))*1/(abs2(target - zJ) + δ^2)^2
        dwdz += (-im*ΓJ)/(2*π)*(-(conj(target) - conj(zJ))^2)/(abs2(target - zJ) + δ^2)^2
        dwdzstar += (-im*ΓJ)/(2*π)*(δ^2)/(abs2(target - zJ) + δ^2)^2
    end

    # Populate ∇v = [∂u/∂x ∂u/∂y
    #                ∂v/∂x ∂v/∂y]

    # ∂u/∂x
    ∇v[1,1] = real(dwdz + dwdzstar)

    # ∂u/∂y
    ∇v[1,2] = real(im*(dwdz - dwdzstar))

    # ∂v/∂x
    ∇v[2,1] = -imag(dwdz + dwdzstar)

    # ∂v/∂y
    ∇v[2,2] = -imag(im*(dwdz - dwdzstar))
    return ∇v
end

Qcriterion_vortex(target::ComplexF64, source) = Qcriterion_vortex!(zeros(2,2), target, source)

function Qcriterion_vortex!(∇v::Matrix{Float64}, target::ComplexF64, source)

    velocity_gradient!(∇v, target, source)

    # Symmetric part of the velocity gradient
    S = Symmetric(∇v)

    # Anti-symmetric part of the velocity gradient
    Ω = ∇v - S

    Q = 0.5*(norm(Ω)^2 - norm(S)^2)

    return Q
end

Qcriterion_vortexfield(xg, yg, source; ismasked::Bool =false) = Qcriterion_vortexfield!(zeros(length(xg), length(yg)), xg, yg, source, ismasked = ismasked)

function Qcriterion_vortexfield!(Q, xg, yg, source; ismasked::Bool =false)
    fill!(Q, 0.0)
    ∇v = zeros(2,2)

    @showprogress for (j, yj) in enumerate(yg)
        for (i, xi) in enumerate(xg)
            qij = Qcriterion_vortex!(∇v, xi + im*yj, source)
            if ismasked == true && qij > 0.0
                Q[i,j] = qij
            end
        end
    end

    return Q
end
