export analytical_pressure, analytical_dpdzv, analytical_dpdΓv, analytical_force, analytical_dfdζv,
        analytical_dfdΓv


# A few helper routines
strength(v::Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}) = v.S
strength(v::Vector{T}) where {T<:Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}} = map(vj -> strength(vj),v)


analytical_pressure(ζ,v::Vector{T},config::VortexConfig{Body};kwargs...) where {T<:Element} = analytical_pressure(ζ,v,config.body;kwargs...)
analytical_pressure(ζ,v::Vector{T},config::VortexConfig;kwargs...) where {T<:Element} = analytical_pressure(ζ,v;kwargs...)

analytical_pressure(z,v::Vector{T},config::VortexConfig{WT}) where {T<:Element, WT<:ImageType} =
    analytical_pressure(z,v;ϵ=config.δ,walltype=config.body)



### For conformally mapped bodies

# This function expects to get the evaluation points and vortex elements in the circle plane
function analytical_pressure(ζ,v_ζ::Vector{T},b::Bodies.ConformalBody;kwargs...) where {T<:Element}

  return Bodies.pressure(ζ,v_ζ,b;kwargs...)

end

function analytical_dpdζv(ζ,l::Integer,v_ζ::Vector{T},b::Bodies.ConformalBody;kwargs...) where {T<:Element}

  return Bodies.dpdζv(ζ,l,v_ζ,b;kwargs...)

end

function analytical_dpdzv(ζ,l::Integer,v::Vector{T},b::Bodies.ConformalBody;kwargs...) where {T<:Element}

  # map elements to circle plane
  v_ζ = Elements.inverse_conftransform(v,b)
  return Bodies.dpdzv(ζ,l,v_ζ,b;kwargs...)

end

function analytical_dpdΓv(ζ,l::Integer,v_ζ::Vector{T},b::Bodies.ConformalBody;kwargs...) where {T<:Element}

  return Bodies.dpdΓv(ζ,l,v_ζ,b;kwargs...)

end

### FORCE ###

function analytical_force(v::Vector{T},config::VortexConfig{Body};kwargs...) where {T<:Element}
    fx, fy, mr = Bodies.force(v,config.body;kwargs...)
    return [fx,fy,mr]
end


function analytical_dfdζv(l::Integer,v_ζ::Vector{T},b::Bodies.ConformalBody;kwargs...) where {T<:Element}

  dfx, dfy, dmr = Bodies.dfdζv(l,v_ζ,b;kwargs...)

  return [dfx, dfy, dmr]

end

function analytical_dfdΓv(l::Integer,v_ζ::Vector{T},b::Bodies.ConformalBody;kwargs...) where {T<:Element}

  dfx, dfy, dmr = Bodies.dfdΓv(l,v_ζ,b;kwargs...)

  return [dfx, dfy, dmr]

end

### PRESSURE ROUTINES FOR FULL-SPACE OR FLAT WALL CASES

### Define the pressure and its gradients

function analytical_pressure(z,v::Vector{T};kwargs...) where {T<:Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}}
        out = 0.0
        for (j,vj) in enumerate(v)
            zj,Γj  = Elements.position(vj), strength(vj)
            out -= 0.5*Γj^2*P(z,zj;kwargs...)
            for vk in v[1:j-1]
                zk,Γk  = Elements.position(vk), strength(vk)
                out -= Γj*Γk*Π(z,zj,zk;kwargs...)
            end
        end
        return out
end

# Change of pressure with respect to change of strength of vortex l (specified by its index)
function analytical_dpdΓv(z,l::Integer,v::Vector{T};kwargs...) where {T<:Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}}
        zl,Γl  = Elements.position(v[l]), strength(v[l])
        out = -Γl*P(z,zl;kwargs...)
        for (k,vk) in enumerate(v)
            k == l && continue
            zk,Γk  = Elements.position(vk), strength(vk)
            out -= Γk*Π(z,zl,zk;kwargs...)
        end
        return out
end

# Change of pressure with respect to change of position of vortex l (specified by its index)
function analytical_dpdzv(z,l::Integer,v::Vector{T};kwargs...) where {T<:Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}}
        zl,Γl  = Elements.position(v[l]), strength(v[l])
        out = -0.5*Γl*dPdzv(z,zl;kwargs...)
        for (k,vk) in enumerate(v)
            k == l && continue
            zk,Γk  = Elements.position(vk), strength(vk)
            out -= Γk*dΠdzvl(z,zl,zk;kwargs...)
        end
        return Γl*out
end

analytical_pressure(z::AbstractArray,v::Vector{T};kwargs...) where {T<:Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}} = map(zj -> analytical_pressure(zj,v;kwargs...),z)
analytical_dpdzv(z::AbstractArray,l::Integer,v::Vector{T};kwargs...) where {T<:Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}} = map(zj -> analytical_dpdzv(zj,l,v;kwargs...),z)
analytical_dpdΓv(z::AbstractArray,l::Integer,v::Vector{T};kwargs...) where {T<:Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}} = map(zj -> analytical_dpdΓv(zj,l,v;kwargs...),z)


analytical_pressure(z,v,::Nothing;kwargs...) = analytical_pressure(z,v;kwargs...)

# Define the functions that comprise the pressure and its gradients
const EPSILON_DEFAULT = 0.01

Fvd(z,zv;ϵ=EPSILON_DEFAULT,kwargs...) = _Fvd(z,zv,ϵ)
Fvi(z,zv;walltype=NoWall) = _Fvi(z,zv,walltype)
Fv(z,zv;ϵ=EPSILON_DEFAULT,walltype=NoWall) = _Fvd(z,zv,ϵ) + _Fvi(z,zv,walltype)
_Fvd(z,zv,ϵ) = complexpotential(z,Vortex.Blob(zv,1.0,ϵ))
_Fvi(z,zv,::Type{NoWall}) = complex(0.0)
_Fvi(z,zv,::Type{FlatWall}) = -_Fvd(z,conj(zv),0.0)
_Fvi(z,zv,::Type{Cylinder}) = -0.5im/π*(-log(z-1/conj(zv)) + log(z))

wvd(z,zv;ϵ=EPSILON_DEFAULT,kwargs...) = _wvd(z,zv,ϵ)
wvi(z,zv;walltype=NoWall,kwargs...) = _wvi(z,zv,walltype)
wv(z,zv;ϵ=EPSILON_DEFAULT,walltype=NoWall) = _wvd(z,zv,ϵ) + _wvi(z,zv,walltype)
_wvd(z,zv,ϵ) = conj(induce_velocity(z,Vortex.Blob(zv,1.0,ϵ),0.0))
_wvi(z,zv,::Type{NoWall}) = complex(0.0)
_wvi(z,zv,::Type{FlatWall}) = -_wvd(z,conj(zv),0.0)
_wvi(z,zv,::Type{Cylinder}) = -0.5im/π*(-1/(z-1/conj(zv)) + 1/z)

dwvddz(z,zv;ϵ=EPSILON_DEFAULT,kwargs...) = _dwvddz(z,zv,ϵ)
dwvddzstar(z,zv;ϵ=EPSILON_DEFAULT,kwargs...) = _dwvddzstar(z,zv,ϵ)
dwvidz(z,zv;walltype=NoWall,kwargs...) = _dwvidz(z,zv,walltype)
dwvdz(z,zv;ϵ=EPSILON_DEFAULT,walltype=NoWall) = _dwvddz(z,zv,ϵ) + _dwvidz(z,zv,walltype)
_dwvddz(z,zv,ϵ) = 0.5im*conj(z-zv)^2/π/(abs2(z-zv) + ϵ^2)^2
_dwvddzstar(z,zv,ϵ) = -0.5im*ϵ^2/π/(abs2(z-zv) + ϵ^2)^2
_dwvidz(z,zv,::Type{NoWall}) = complex(0.0)
_dwvidz(z,zv,::Type{FlatWall}) = -_dwvddz(z,conj(zv),0.0)
_dwvidz(z,zv,::Type{Cylinder}) = -0.5im/π*(1/(z-1/conj(zv))^2-1/z^2)


dwvddzv(z,zv;ϵ=EPSILON_DEFAULT,kwargs...) = _dwvddzv(z,zv,ϵ)
dwvddzvstar(z,zv;ϵ=EPSILON_DEFAULT,kwargs...) = _dwvddzvstar(z,zv,ϵ)
dwvidzvstar(z,zv;walltype=NoWall,kwargs...) = _dwvidzvstar(z,zv,walltype)
dwvdzvstar(z,zv;ϵ=EPSILON_DEFAULT,walltype=NoWall) = _dwvddzvstar(z,zv,ϵ) + _dwvidzvstar(z,zv,walltype)
_dwvddzv(z,zv,ϵ) = -_dwvddz(z,zv,ϵ)
_dwvddzvstar(z,zv,ϵ) = -_dwvddzstar(z,zv,ϵ)
_dwvidzvstar(z,zv,::Type{NoWall}) = complex(0.0)
_dwvidzvstar(z,zv,::Type{FlatWall}) = -_dwvddzv(z,conj(zv),0.0)
_dwvidzvstar(z,zv,::Type{Cylinder}) = -0.5im/π/conj(zv)^2/(z-1/conj(zv))^2


dFddzv(z,zv;ϵ=EPSILON_DEFAULT,kwargs...) = _dFddzv(z,zv,ϵ)
dFidzvstar(z,zv;walltype=NoWall,kwargs...) = _dFidzvstar(z,zv,walltype)
dFdzv(z,zv;ϵ=EPSILON_DEFAULT,walltype=NoWall) = _dFddzv(z,zv,ϵ) + conj(_dFidzvstar(z,zv,walltype))
_dFddzv(z,zv,ϵ) = -_wvd(z,zv,ϵ)
_dFidzvstar(z,zv,::Type{NoWall}) = complex(0.0)
_dFidzvstar(z,zv,::Type{FlatWall}) = -_dFddzv(z,conj(zv),0.0)
_dFidzvstar(z,zv,::Type{Cylinder}) = 0.5im/π/conj(zv)^2/(z-1/conj(zv))

d2Fddzv2(z,zv;ϵ=EPSILON_DEFAULT,kwargs...) = _d2Fddzv2(z,zv,ϵ)
d2Fddzvdvzstar(z,zv;ϵ=EPSILON_DEFAULT,kwargs...) = _d2Fddzvdvzstar(z,zv,ϵ)
d2Fidzvstar2(z,zv;walltype=NoWall,kwargs...) = _d2Fidzvstar2(z,zv,walltype)
d2Fdzv2(z,zv;ϵ=EPSILON_DEFAULT,walltype=NoWall) = _d2Fddzv2(z,zv,ϵ) + conj(_d2Fidzvstar2(z,zv,walltype))
_d2Fddzv2(z,zv,ϵ) = _dwvddz(z,zv,ϵ)
_d2Fddzvdvzstar(z,zv,ϵ) = _dwvddzstar(z,zv,ϵ)
_d2Fidzvstar2(z,zv,::Type{NoWall}) = complex(0.0)
_d2Fidzvstar2(z,zv,::Type{FlatWall}) = -_d2Fddzv2(z,conj(zv),0.0)
_d2Fidzvstar2(z,zv,::Type{Cylinder}) = -0.5im/π/conj(zv)^3/(z-1/conj(zv))*(2 + 1/conj(zv)/(z-1/conj(zv)))


P(z,zv;kwargs...) = abs2(wv(z,zv;kwargs...)) + 2real(dFdzv(z,zv;kwargs...)*conj(wvi(zv,zv;kwargs...)))
Π(z,zvj,zvk;kwargs...) = real(wv(z,zvj;kwargs...)*conj(wv(z,zvk;kwargs...))) +
                           real(dFdzv(z,zvj;kwargs...)*conj(wv(zvj,zvk;kwargs...))) +
                           real(dFdzv(z,zvk;kwargs...)*conj(wv(zvk,zvj;kwargs...)))

dPdzv(z,zv;kwargs...) = (dwvddzv(z,zv;kwargs...)*conj(wv(z,zv;kwargs...)) + wv(z,zv;kwargs...)*conj(dwvdzvstar(z,zv;kwargs...))) +
                        (d2Fdzv2(z,zv;kwargs...)*conj(wvi(zv,zv;kwargs...)) + conj(d2Fddzvdvzstar(z,zv;kwargs...))*wvi(zv,zv;kwargs...) + dFdzv(z,zv;kwargs...)*conj(dwvidzvstar(zv,zv;kwargs...))) +
                        (conj(dFdzv(z,zv;kwargs...))*dwvidz(zv,zv;kwargs...))

dΠdzvl(z,zvl,zvk;kwargs...) = 0.5*(dwvddzv(z,zvl;kwargs...)*conj(wv(z,zvk;kwargs...)) + conj(dwvddzvstar(z,zvl;kwargs...)+dwvidzvstar(z,zvl;kwargs...))*wv(z,zvk;kwargs...)) +
                               0.5*(d2Fdzv2(z,zvl;kwargs...)*conj(wv(zvl,zvk;kwargs...)) + dFdzv(z,zvl;kwargs...)*conj(dwvddzstar(zvl,zvk;kwargs...)) + conj(d2Fddzvdvzstar(z,zvl;kwargs...))*wv(zvl,zvk;kwargs...)  + conj(dFdzv(z,zvl;kwargs...))*dwvdz(zvl,zvk;kwargs...)) +
                               0.5*(dFdzv(z,zvk;kwargs...)*conj(dwvdzvstar(zvk,zvl;kwargs...)) + conj(dFdzv(z,zvk;kwargs...))*dwvddzv(zvk,zvl;kwargs...))

for f in [:F,:w]

   vd = Symbol(f,"vd")
   vi = Symbol(f,"vi")

   @eval function $f(z,v::Vector{T};kwargs...) where {T<:Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}}
       out = complex(0)
       for vj in v
           zj = Elements.position(vj)
           # out += strength(vj)*($vd(z,zj;kwargs...) + $vi(z,zj;kwargs...))
           out += strength(vj)*($vd(z,zj;kwargs...) + $vi(z,zj))
       end
       return out
   end

   @eval $f(z::AbstractArray,v::Vector{T};kwargs...) where {T<:Union{PotentialFlow.Points.Point,PotentialFlow.Blobs.Blob}} = map(zj -> $f(zj,v;kwargs...),z)
end
