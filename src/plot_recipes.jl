#using RecipesBase
using ColorTypes
using MakieCore
using LaTeXStrings

export color_palette


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


_physical_space_sensors(sens,config) = sens
_physical_space_sensors(sens,config::VortexConfig{Bodies.ConformalBody}) = Bodies.conftransform(sens,config.body)
setmarkersize(x::Float64) = 5 + x*4


function trajectory_theme()
    MakieCore.Theme(
        fontsize=16,
        Axis=(xlabel=L"x", ylabel=L"y",limits=((-2,2),(-1,2)),aspect=1),
        resolution=(500,400)
    )
end


MakieCore.@recipe(Trajectory, x, y) do scene
    MakieCore.Attributes(
        color = :black
    )
end

MakieCore.@recipe(Trajectory3d, x, y, z) do scene
    MakieCore.Attributes(
        color = :black
    )
end

function MakieCore.convert_arguments(P::Union{Type{<:LowRankVortex.Trajectory},Type{<:LowRankVortex.Trajectory3d}}, solhist::Vector{<:Any}, obs::AbstractObservationOperator)
    config = obs.config
    Nv = config.Nv
    x = []
    y = []
    Γ = []
    for j in 1:Nv
      zj = map(x -> state_to_positions_and_strengths(mean(x.Xf),config)[1][j],solhist)
      xj, yj = real(zj), imag(zj)
      push!(x,copy(xj))
      push!(y,copy(yj))
      Γj = map(x -> state_to_positions_and_strengths(mean(x.Xf),config)[2][j],solhist)
      push!(Γ,copy(Γj))
    end
    return x,y,Γ
end


function MakieCore.plot!(trajectory::Trajectory)
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



function MakieCore.plot!(trajectory::Trajectory3d)
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
    sens = _physical_space_sensors(obs.sens,config)

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
