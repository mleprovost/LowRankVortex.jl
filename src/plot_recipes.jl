using RecipesBase
using ColorTypes

export color_palette

setmarkersize(x::Float64) = 5 + x*4

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

@userplot Filtertrajectory

@recipe function f(h::Filtertrajectory)

  if length(h.args) > 3
    solhist, sens, vort_true, b = h.args
  else
    solhist, sens, vort_true = h.args
  end
  Nv = size(solhist[1].X,1) ÷ 3
  truez = Elements.position(vort_true)
  trueΓ = LowRankVortex.strength(vort_true)

  true_center = vortexmoment(1,truez,trueΓ)

  ratio := 1
  @series begin
    seriestype := :scatter
    markersize --> 5
    markerstrokecolor --> :black
    markercolor --> :white
    label := "true"
    real(truez), imag(truez)
  end

  if length(h.args) > 3
    @series begin
      label := "body"
      b
    end
  end

  #=
  @series begin
    seriestype := :scatter
    markersize --> 5
    markershape --> :utriangle
    markerstrokecolor --> :black
    markercolor --> :black
    label := "true centroid"
    [real(true_center)], [imag(true_center)]
  end
  =#

  @series begin
    seriestype := :scatter
    markersize --> 5
    markercolor --> :blue
    label := "sensor"
    real.(sens), imag.(sens)
  end

  #col = theme_palette(:auto)
  for j in 1:Nv
    #xj = map(x -> mean(x.Xf)[2j-1],solhist)
    #yj = map(x -> mean(x.Xf)[2j],solhist)
    rj = map(x -> 1.0 + exp(mean(x.Xf)[j]),solhist)
    rΘj = map(x -> mean(x.Xf)[Nv+j],solhist)
    ζj = rj.*exp.(im*rΘj./rj)
    zj = Elements.conftransform(ζj,b)
    xj, yj = real(zj), imag(zj)
    @series begin
      seriestype := :path
      label := "traj of "*string(j)
      color := color_palette[j]
      xj, yj
    end
    @series begin
      seriestype := :scatter
      markershape := :diamond
      label := "prior of "*string(j)
      markercolor := color_palette[j]
      [xj[1]], [yj[1]]
    end
    @series begin
      seriestype := :scatter
      markershape := :star
      label := "post of "*string(j)
      markercolor := color_palette[j]
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
    Nv = size(solhist[1].X,1) ÷ 3

    sol = solhist[jdex]
    V = sol.V
    U = sol.U
    lenU = size(U,1)
    Y̆mean = mean(sol.Y̆)

    Xfmean = mean(sol.Xf)
    vortf = state_to_lagrange_reordered(Xfmean,config)
    vortf_z = Elements.conftransform(vortf,config.body)
    xjf, yjf = reim(Elements.position(vortf_z))
    Γf = LowRankVortex.strength(vortf_z)

    Umode = U[:,mode]
    Vmode = V[:,mode]

    Xf_plus = Xfmean .+ Vmode
    vort_plus = state_to_lagrange_reordered(Xf_plus,config)
    vort_plus_z = Elements.conftransform(vort_plus,config.body)
    xplus, yplus = reim(Elements.position(vort_plus_z))
    Γplus = LowRankVortex.strength(vort_plus_z)

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
      markersize := 5
      xjf, yjf
    end



      @series begin
        subplot := 2
        seriestype := :scatter
        #markersize := setmarkersize.(V[2Nv+1:3Nv,mode])
        markersize := 5
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

    #=
    p1 = scatter(real.(zv_j).+V[1:2:2Nv,mode],imag.(zv_j).+V[2:2:2Nv,mode],markersize=setmarkersize.(V[2Nv+1:3Nv,mode]),legend=false,markercolor=:gray)
scatter!(p1,real.(zv_j),imag.(zv_j),markersize=5)
quiver!(p1,real.(zv_j),imag.(zv_j),quiver=(V[1:2:2Nv,mode],V[2:2:2Nv,mode]),ratio=1,legend=:false,color=:black,xlim=(-1.5,1.5),ylim=(-1.5,1.5),size=(300,300))
p2 = plot(U[:,mode],legend=false,size=(800,250))
p3 = plot(p1,p2)
    =#

end
