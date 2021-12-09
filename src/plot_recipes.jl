export color_palette

const color_palette = [colorant"firebrick";
                colorant"orangered2";
                    colorant"goldenrod1";
             colorant"skyblue1";
                 colorant"slateblue";
                colorant"maroon3";
             colorant"grey70";
                 colorant"dodgerblue4";
              colorant"seagreen4"]


# import Colors: colormap
# import PlotUtils: cgrad
#
# function routine_plot(state, config::VortexConfig, X::StepRangeLen, Y::StepRangeLen)
#     source = state_to_lagrange(state, config.zs, config)
#     plt = streamlines(X, Y, source, colorbar = false)
#     plot!(plt, source, markersize = 12, markerstrokealpha = 0, color = cgrad(reverse(colormap("RdBu")[10:end-10])),
#           clim = (-1.0, 1.0), legend = :outerleft, label = ["Vortices" "Sources"])
#     hline!(plt, [0.0], label = "", color = :black)
#     scatter!(plt, real.(sensors), imag.(sensors), label = "Sensors", color = :orangered2)
#     plt
# end
