export generate_data_twin_experiment, SyntheticData, save_synthetic_data, load_synthetic_data

# This is taken from TransportBasedInference.jl, but defined here
# since we don't call that package anymore.
"""
    SyntheticData
A structure to store the synthetic data in a twin-experiment
## Fields
- `tt` : time history
- `Δt` : time step
- `x0` : the initial condition
- `xt` : history of the state
- `yt` : history of the observations
"""
struct SyntheticData
	tt::Array{Float64,1}
	Δt::Float64
	x0::Array{Float64,1}
	xt::Array{Float64,2}
	yt::Array{Float64,2}
end


"""
    generate_data_twin_experiment(x0,t0,tf,Δt,fdata::AbstractForecastOperator,odata::AbstractObservationOperator,Σx,Σϵ)

Starting with initial state `x0`, this routine advances the system with the forecast model
and evaluates the observation data at each time step in the range t0:Δt:tf.
It returns the data history as a `SyntheticData` structure.
"""
function generate_data_twin_experiment(x0, t0, tf, Δt, fdata::AbstractForecastOperator{Nx}, odata::AbstractObservationOperator{Nx,Ny}, Σx, Σϵ) where {Nx,Ny}

    # 0.1 Generate initial state and inflate it
    # 0.2 Evaluate initial observations and noise them up
    # For each time step
    #   1. Advance the system, using same routines as forecast!
    #   2. Inflate the state
    #   3. Evaluate the observations
    #   4. Noise them up


    n0 = ceil(Int64, t0/Δt) + 1
    J = ceil(Int64, (tf-t0)/Δt)
    Acycle = n0:n0+J-1

    xt = zeros(length(x0), J+1)
    x = deepcopy(x0)
    y = zeros(Ny)
    yt = zeros(Ny, J+1)
    tt = zeros(J+1)

    tt[1] = t0
    xt[:,1] .= x

    additive_inflation!(x,Σx)

    y .= observations(x,t0,odata)
    additive_inflation!(y,Σϵ)

    yt[:,1] .= deepcopy(y)

    for i=1:length(Acycle)

        x .= forecast(x,tt[i],Δt,fdata)

        additive_inflation!(x,Σx)

        xt[:,i+1] .= deepcopy(x)

        tt[i+1] = t0+i*Δt

        y .= observations(x,tt[i+1],odata)

        additive_inflation!(y,Σϵ)

        # Apply observation noise
        yt[:,i+1] .= deepcopy(y)

    end

    # Store initial condition and the history of the state and observation variables into `data`.
    data = SyntheticData(tt, Δt, x0, xt, yt)

    # Save the fields of data into `data_twin_experiment.jld` in the folder `path`
    #save(path*"data_twin_experiment.jld", "tt", tt, "x0", x0, "xt", xt, "yt", yt)

    return data
end

function save_synthetic_data(data::SyntheticData,path::String)
  save(path*"data_twin_experiment.jld", "tt", data.tt, "Δt", data.Δt, "x0", data.x0, "xt", data.xt, "yt", data.yt)
end

function load_synthetic_data(path::String)
  tt, Δt, x0, xt, yt = load(path*"data_twin_experiment.jld", "tt", "Δt", "x0", "xt", "yt")
  return SyntheticData(tt, Δt, x0, xt, yt)
end
