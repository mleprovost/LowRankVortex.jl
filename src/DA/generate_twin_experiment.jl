export generate_data_twin_experiment, SyntheticData, save_synthetic_data, load_synthetic_data,
      create_truth_function

# This is taken from TransportBasedInference.jl, but defined here
# since we don't call that package anymore.
"""
    SyntheticData
A structure to store the synthetic data in a twin-experiment
## Fields
- `sens` : Sensor positions where measurements are evaluated (might be empty)
- `tt` : time history
- `Δt` : time step
- `x0` : the initial condition
- `xt` : history of the state
- `yt` : history of the observations
"""
struct SyntheticData{ST}
  sens :: ST
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

  @unpack sens = odata
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
    data = SyntheticData(sens, tt, Δt, x0, xt, yt)
    
    return data
end

function save_synthetic_data(data::SyntheticData,path::String)
  xsens = isnothing(data.sens) ? nothing : real(data.sens)
  ysens = isnothing(data.sens) ? nothing : imag(data.sens)
  save(path*"data_twin_experiment.jld", "xsens", xsens, "ysens", ysens, "tt", data.tt, "Δt", data.Δt, "x0", data.x0, "xt", data.xt, "yt", data.yt)
end

function load_synthetic_data(path::String)
  xsens, ysens, tt, Δt, x0, xt, yt = load(path*"data_twin_experiment.jld", "xsens", "ysens", "tt", "Δt", "x0", "xt", "yt")
  zsens = isnothing(xsens) ? nothing : xsens.+im*ysens
  return SyntheticData(zsens, tt, Δt, x0, xt, yt)
end

"""
    create_truth_function(data::SyntheticData,odata::AbstractObservationOperator,odata_truth::AbstractObservationOperator)

Creates a function of time, `ytrue(t)`, that will return the true observation
data at a given time. It builds this function by interpolating spatially
and temporally from the synthetic sensor data in `data`. Note that this
expects that the sensors are evenly spaced in the x direction and can
be parameterized by this coordinate. If there are no physical locations for
the sensors, then this only interpolates over time.
"""
function create_truth_function(data::SyntheticData,odata::AbstractObservationOperator{Nx,Ny,true}) where {Nx,Ny}
  @unpack sens, tt, Δt, yt = data
  trange = tt[1]:Δt:tt[end]
  y_itp = CubicSplineInterpolation((LinRange(real(sens[1]), real(sens[end]), length(sens)),trange), yt, extrapolation_bc =  Line())
  ytrue(t) = y_itp(real.(odata.sens), t)
  return ytrue
end

function create_truth_function(data::SyntheticData,odata::AbstractObservationOperator{Nx,Ny,false}) where {Nx,Ny}
  @unpack tt, Δt, yt = data
  trange = tt[1]:Δt:tt[end]
  y_itp = CubicSplineInterpolation((trange,), yt, extrapolation_bc =  Line())
  ytrue(t) = y_itp(t)
  return ytrue
end
