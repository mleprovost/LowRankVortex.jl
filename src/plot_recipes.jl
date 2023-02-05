#using RecipesBase
using ColorTypes
#using MakieCore
using LaTeXStrings
using CairoMakie

export color_palette
export draw_ellipse
export data_histogram, show_vortices, show_vortices!, show_vortex_samples,
        show_vortex_samples!, plot_vorticity, plot_vorticity!, vortex_ellipses,
        vortex_ellipses!, plot_pressure_field, plot_pressure_field!, plot_sensor_data,
        plot_sensor_data!, plot_sensors!

"""
A palette of colors for plotting
"""
const color_palette = [colorant"firebrick";
                       colorant"seagreen4";
                       colorant"goldenrod1";
                       colorant"skyblue1";
                       colorant"slateblue";
                       colorant"maroon3";
                       colorant"orangered2";
                       colorant"grey70";
                       colorant"dodgerblue4"]

#const color_palette2 = cgrad(:tab20, 10, categorical = true)

"""
    data_histogram(x::Vector[;bins=80,xlims=(-2,2)])

Create a histogram of the data in vector `x`.
"""
function data_histogram(x::Vector{T};bins=80,xlims = (-2,2), kwargs...) where {T<:Real}
    f = Figure()
    ax1 = f[1, 1] = Axis(f;kwargs...)
    hist!(ax1,x,bins=bins)
    xlims!(ax1, xlims...)
    f
end


function show_vortices!(ax,x::Vector,obs::AbstractObservationOperator)
    vort_array = state_to_vortex_states(x,obs.config)
    scatter!(ax,vort_array[1,:],vort_array[2,:])
end

"""
    show_vortices(x::Vector,obs::AbstractObservationOperator)

Given state vector `x`, create a plot that depicts the vortex positions.
"""
function show_vortices(x::Vector,obs::AbstractObservationOperator;kwargs...)
    f = Figure()
    ax = f[1, 1] = Axis(f;aspect=DataAspect(),xlabel=L"x",ylabel=L"y",kwargs...)
    show_vortices!(ax,x,obs)
    f
end


function show_vortex_samples!(ax,x_samples::Array,obs::AbstractObservationOperator;nskip=1,kwargs...)
    vort_array = states_to_vortex_states(x_samples[:,1:nskip:end],obs.config)
    scatter!(ax,vort_array[1,:],vort_array[2,:];markersize=1,label="sample",kwargs...)

end

"""
    show_vortex_samples(x_samples,obs[;nskip=1])

Plot the samples of an ensemble of states as a scatter plot of vortex positions. Note that
`x_samples` must be of size Nstate x Ne. Use `nskip` to plot every `nskip` state.
"""
function show_vortex_samples(x_samples::Array,obs::AbstractObservationOperator;nskip=1,xlims=(-2,2),ylims=(-2,2),kwargs...)
    f = Figure()
    ax = f[1,1] = Axis(f;aspect=DataAspect(),xlabel=L"x",ylabel=L"y")
    show_vortex_samples!(ax,x_samples,obs;nskip=nskip,kwargs...)
    xlims!(ax,xlims...)
    ylims!(ax,ylims...)
    f
end


function plot_vorticity!(ax,μ,Σ,obs::AbstractObservationOperator;xlims = (-2.5,2.5),Nx = 201, ylims = (-2.5,2.5), Ny = 201,kwargs...)
    xg = range(xlims...,length=Nx)
    yg = range(ylims...,length=Ny)
    w = [vorticity(x,y,μ,Σ,obs.config) for x in xg, y in yg]
    contour!(ax,xg,yg,w;kwargs...)
    xlims!(ax,xlims...)
    ylims!(ax,ylims...)
end

"""
        plot_vorticity(μ,Σ,obs::AbstractObservationOperator[;xlims=(-2.5,2.5),Nx = 201,ylims=(-2.5,2.5),Ny = 201])

For a given mean state `μ` and state covariance `Σ`, calculate the expected value of the vorticity
field on a grid. The optional arguments allow one to specify the dimensions of the grid.
"""
function plot_vorticity(μ,Σ,obs::AbstractObservationOperator; kwargs...)
    f = Figure()
    ax = f[1,1] = Axis(f;aspect=DataAspect(),xlabel=L"x",ylabel=L"y")
    plot_vorticity!(ax,μ,Σ,obs;kwargs...)
    f
end

function vortex_ellipses!(ax,μ::Vector{T},Σ::Matrix{T},obs::AbstractObservationOperator; kwargs...) where {T<:Real}
    for j = 1:obs.config.Nv
        xidj, yidj, Γidj = get_vortex_ids(j,obs.config)
        μxj = μ[[xidj,yidj]]
        Σxxj = Σ[xidj:yidj,xidj:yidj]
        xell, yell = draw_ellipse(μxj,Σxxj)
        lines!(ax,xell,yell;color=:red,marker=:none,kwargs...)
    end
end

function vortex_ellipses!(ax,μ::AbstractMatrix{T},Σ::Matrix{Matrix{T}},obs::AbstractObservationOperator; kwargs...) where {T<:Real}
  for c in 1:size(μ,2)
      vortex_ellipses!(ax,μ[:,c],Σ[c],obs;kwargs...)
  end
end


"""
    vortex_ellipses(μ,Σ,obs::AbstractObservationOperator)

Given state mean `μ` and state covariance `Σ`, plot ellipses of uncertainty at each of the
mean vortex locations in `μ`.
"""
function vortex_ellipses(μ,Σ,obs::AbstractObservationOperator; kwargs...)
    f = Figure()
    ax = f[1,1] = Axis(f;aspect=DataAspect(),xlabel=L"x",ylabel=L"y")
    vortex_ellipses!(ax,μ,Σ,obs;kwargs...)
    f
end

function plot_pressure_field!(ax,x::Vector,obs::PressureObservations;xlims=(-2.5,2.5),Nx=201,ylims=(-2.5,2.5),Ny=201,kwargs...)
    xg = range(xlims...,length=Nx)
    yg = range(ylims...,length=Ny)
    zg = xg .+ im*yg'
    vort = state_to_lagrange(x,obs.config;isblob=true)
    p = analytical_pressure(zg,vort,obs.config)
    plot_pressure_field!(ax,xg,yg,p,obs;kwargs...)
end

"""
        plot_pressure_field(x::Vector,obs::PressureOperations[;xlims=(-2.5,2.5),Nx = 201,ylims=(-2.5,2.5),Ny = 201])

For a given state `x`, calculate the pressure field on a grid. The optional arguments allow one to specify the dimensions of the grid.
"""
function plot_pressure_field(x::Vector,obs::PressureObservations; kwargs...)
    f = Figure()
    ax = f[1,1] = Axis(f;aspect=DataAspect(),xlabel=L"x",ylabel=L"y")
    plot_pressure_field!(ax,x,obs;kwargs...)
    f
end

function plot_pressure_field!(ax,xg::AbstractVector,yg::AbstractVector,p::Matrix,obs::AbstractObservationOperator; levels = range(-0.5,0.01,length=21), kwargs...)
    contour!(ax,xg,yg,p,levels=levels,colormap=:RdBu)
    plot_sensors!(ax,obs)
end

function plot_sensor_data!(ax,ystar::Vector,x::Vector,t::Real,obs::AbstractObservationOperator; sensor_noise=zero(ystar))
    plot_sensor_data!(ax,ystar,obs;sensor_noise=sensor_noise)
    y_est = observations(x,t,obs)
    scatter!(ax,y_est,markersize=15,color=:transparent,strokewidth=1,label="estimate")
end

function plot_sensor_data!(ax,ystar::Vector,obs::AbstractObservationOperator; sensor_noise=zero(ystar))
    scatter!(ax,ystar,markersize=10,color=:black,label="truth")
    errorbars!(ax,1:length(ystar),ystar,sensor_noise)
end

"""
    plot_sensor_data(ystar::Vector,obs::AbstractObservationOperator[; sensor_noise=zero(ystar)])

Plot the sensor data in `ystar`.

  plot_sensor_data(ystar::Vector,x::Vector,t,obs::AbstractObservationOperator[; sensor_noise=zero(ystar)])

Compute the sensor data associated with state vector `x` at time `t` and plot it with the sensor data in `ystar`.
"""
function plot_sensor_data(a...; sensor_noise=zero(ystar))
    f = Figure()
    ax = f[1,1] = Axis(f;aspect=DataAspect(),xlabel="Sensor no.",ylabel="Sensor value")
    plot_sensor_data!(ax,a...;sensor_noise=sensor_noise)
    f
end

function plot_sensors!(ax,obs::PressureObservations;kwargs...)
    scatter!(ax,real.(obs.sens),imag.(obs.sens);marker=:rect,color=:black,kwargs...)
end

function draw_ellipse(μ::Vector,Σ::AbstractMatrix)
    θ = range(0,2π,length=100)
    xc, yc = cos.(θ), sin.(θ)
    sqrtΣ = sqrt(Σ)
    xell = μ[1] .+ sqrtΣ[1,1]*xc .+ sqrtΣ[1,2]*yc
    yell = μ[2] .+ sqrtΣ[2,1]*xc .+ sqrtΣ[2,2]*yc
    return xell,yell
end

#=

setmarkersize(x::Float64) = 5 + x*4


function trajectory_theme()
    CairoMakie.Theme(
        fontsize=16,
        Axis=(xlabel=L"x", ylabel=L"y",limits=((-2,2),(-1,2)),aspect=1),
        resolution=(500,400)
    )
end


CairoMakie.@recipe(Trajectory, x, y) do scene
    CairoMakie.Attributes(
        color = :black
    )
end

CairoMakie.@recipe(Trajectory3d, x, y, z) do scene
    CairoMakie.Attributes(
        color = :black
    )
end

function CairoMakie.convert_arguments(P::Union{Type{<:LowRankVortex.Trajectory},Type{<:LowRankVortex.Trajectory3d}}, solhist::Vector{<:Any}, obs::AbstractObservationOperator)
    config = obs.config
    Nv = config.Nv
    x = [Float64[] for j in 1:Nv]
    y = [Float64[] for j in 1:Nv]
    Γ = [Float64[] for j in 1:Nv]
    for sol in solhist
      xt =  state_to_vortex_states(mean(sol.Xf),config)
      for j in 1:Nv
        push!(x[j],xt[1,j])
        push!(y[j],xt[2,j])
        push!(Γ[j],xt[3,j])
      end
    end


    return x,y,Γ
end




function CairoMakie.plot!(trajectory::Trajectory)
    x,y = trajectory[:x], trajectory[:y]
    for j in eachindex(x.val)
      xj, yj = x.val[j], y.val[j]
      l1 = lines!(trajectory,xj,yj,color=trajectory[:color])
      l2 = scatter!(trajectory,[xj[1]],[yj[1]])
      l2.color=:transparent
      l2.strokewidth=1
      l2.markersize=10
      l2.strokecolor=l1.attributes[:color]
      l3 = scatter!(trajectory,[xj[end]],[yj[end]],color=l1.attributes[:color])
    end
    trajectory
end



function CairoMakie.plot!(trajectory::Trajectory3d)
    x,y,z = trajectory[:x], trajectory[:y], trajectory[:z]
    l1 = lines!(trajectory,x,y,z,color=trajectory[:color])
    l2 = scatter!(trajectory,[x.val[1]],[y.val[1]],[z.val[1]])
    l2.color=:transparent
    l2.strokewidth=1
    l2.markersize=10
    l2.strokecolor=l1.attributes[:color]
    l3 = scatter!(trajectory,[x.val[end]],[y.val[end]],[z.val[end]],color=l1.attributes[:color])
    trajectory
end

=#

#=
@userplot Filtertrajectory

@recipe function f(h::Filtertrajectory;trajcolor=nothing)

  solhist, obs, vort_true = h.args

  config = obs.config
  Nv = config.Nv

  truez = Elements.position(vort_true)
  trueΓ = LowRankVortex.strength(vort_true)

  true_center = vortexmoment(1,truez,trueΓ)

  ratio := 1
  @series begin
    seriestype := :scatter
    markersize --> 10
    markershape := :star
    markerstrokecolor --> :black
    markercolor --> :grey
    label := "true"
    real(truez), imag(truez)
  end

  bflag = false
  if typeof(config.body) == Bodies.ConformalBody
    bflag = true
    @series begin
      label := "body"
      config.body
    end
  end


  if hasfield(typeof(obs),:sens)
    sens = physical_space_sensors(obs)

    @series begin
      seriestype := :scatter
      markersize --> 5
      markercolor --> :gray
      markerstrokecolor --> :gray
      label := "sensor"
      real.(sens), imag.(sens)
    end
  end

  #col = theme_palette(:auto)
  for j in 1:Nv
    #xj = map(x -> mean(x.Xf)[2j-1],solhist)
    #yj = map(x -> mean(x.Xf)[2j],solhist)
    zj = map(x -> state_to_positions_and_strengths(mean(x.Xf),config)[1][j],solhist)

    #rj = map(x -> 1.0 + exp(mean(x.Xf)[j]),solhist)
    #rΘj = map(x -> mean(x.Xf)[Nv+j],solhist)
    #ζj = rj.*exp.(im*rΘj./rj)
    zj = bflag ? Elements.conftransform(zj,config.body) : zj
    xj, yj = real(zj), imag(zj)

    this_trajcolor = isnothing(trajcolor) ? color_palette[j] : trajcolor
    @series begin
      seriestype := :path
      label := "traj of "*string(j)
      color --> this_trajcolor
      xj, yj
    end
    @series begin
      seriestype := :scatter
      markershape := :circle
      label := "prior of "*string(j)
      markersize --> 5
      markerstrokecolor --> this_trajcolor
      markercolor --> :white
      [xj[1]], [yj[1]]
    end
    @series begin
      seriestype := :scatter
      markershape := :circle
      markerstrokewidth := 0
      #markercolor := :none
      markersize --> 5
      label := "post of "*string(j)
      markercolor --> this_trajcolor
      [xj[end]], [yj[end]]
    end

  end

end

@userplot Filterstepplot

@recipe function f(h::Filterstepplot;arrows_on = true,ubarscale=1,vbarscale=1)
  jdex, solhist, xtrue = h.args
  Nv = size(solhist[1].X,1) ÷ 3

  Nv_true = length(xtrue) ÷ 3

  truex = xtrue[1:2:2Nv_true]
  truey = xtrue[2:2:2Nv_true]
  trueΓ = xtrue[2Nv_true+1:3Nv_true]

  sol = solhist[jdex]
  Y̆mean = mean(sol.Y̆)
  ucoeff_j = Y̆mean
  vcoeff_j = sol.ΣX̆Y̆*(sol.ΣY̆\Y̆mean)
  dx_j = sqrt(sol.Σx)*sol.V[:,1:sol.rx]*vcoeff_j

  Xfmean = mean(sol.Xf)
  xjf = Xfmean[1:2:2Nv]
  yjf = Xfmean[2:2:2Nv]

  Xamean = mean(sol.X)
  xja = Xamean[1:2:2Nv]
  yja = Xamean[2:2:2Nv]

  l = @layout [a{0.25w} b c{0.25w}]

  layout := l
  size --> (700,700)
  @series begin
    subplot := 1
    xlims := (0.5,length(ucoeff_j)+0.5)
    ylims := (-500*ubarscale,500*ubarscale)
    seriestype := :bar
    xlabel := "u mode"
    legend := :false
    barwidths --> 0.5
    ucoeff_j
  end



  @series begin
    subplot := 2
    seriestype := :scatter
    markersize := setmarkersize.(dx_j[2Nv+1:3Nv])
    markershape := :star
    markercolor := :gray
    xja, yja
  end



  if arrows_on
    @series begin
      subplot := 2
      seriestype := :scatter
      markersize := 5
      markershape := :diamond
      markercolor := color_palette[1]
      xjf, yjf
    end
    @series begin
      subplot := 2
      seriestype := :quiver
        quiver := (10*dx_j[1:2:2Nv],10*dx_j[2:2:2Nv])
        color := :black
        xjf, yjf
      end
  end

  @series begin
    subplot := 2
    seriestype := :scatter
    ratio := 1
    legend := :false
    markersize := 3
    markercolor := :black
    truex, truey
  end

  @series begin
    subplot := 3
    seriestype := :bar
    xlims := (0.5,length(vcoeff_j)+0.5)
    ylims := (-100*vbarscale,100*vbarscale)
    xlabel := "v mode"
    legend := :false
    barwidths --> 0.5
    vcoeff_j
  end

end

@userplot Showmode

@recipe function f(h::Showmode)
    jdex, mode, solhist, config = h.args

    bflag = false
    if typeof(config.body) == Bodies.ConformalBody
      bflag = true
    end

    Nv = size(solhist[1].X,1) ÷ 3

    sol = solhist[jdex]
    V = sol.V
    U = sol.U
    lenU = size(U,1)
    Y̆mean = mean(sol.Y̆)

    Xfmean = mean(sol.Xf)
    #vortf = state_to_lagrange_reordered(Xfmean,config)
    #vortf_z = Elements.conftransform(vortf,config.body)
    #xjf, yjf = reim(Elements.position(vortf_z))
    #Γf = LowRankVortex.strength(vortf_z)
    zf, Γf = state_to_positions_and_strengths(Xfmean,config)
    zf = bflag ? Elements.conftransform(zf,config.body) : zf

    xjf, yjf = reim(zf)

    Umode = U[:,mode]
    Vmode = V[:,mode]

    Xf_plus = Xfmean .+ Vmode
    zplus, Γplus = state_to_positions_and_strengths(Xf_plus,config)
    zplus = bflag ? Elements.conftransform(zplus,config.body) : zplus
    xplus, yplus = reim(zplus)

    dx = xplus .- xjf
    dy = yplus .- yjf
    dΓ = Γplus .- Γf

    #xjf = Xfmean[1:2:2Nv]
    #yjf = Xfmean[2:2:2Nv]

    layout := @layout [a{0.5w} b{0.5w}]

    @series begin
      subplot := 2
      seriestype := :scatter
      markercolor := :white
      markersize --> 5
      xjf, yjf
    end



      @series begin
        subplot := 2
        seriestype := :scatter
        #markersize := setmarkersize.(V[2Nv+1:3Nv,mode])
        markersize --> 5
        markercolor := :RdBu_4
        marker_z := dΓ
        xplus, yplus
      end

      @series begin
        subplot := 2
        seriestype := :quiver
        quiver := (dx,dy)
        ratio := 1
        legend := :false
        clims := (-1,1)
        color := :black
        xjf, yjf
        end

      @series begin
        subplot := 1
        legend := :false
        xlabel --> "sensor no."
        xlims := (1,lenU)
        ylims := (-5/sqrt(lenU),5/sqrt(lenU))
        Umode
      end


end
=#
