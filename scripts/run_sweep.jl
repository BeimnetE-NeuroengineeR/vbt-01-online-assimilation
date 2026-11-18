push!(LOAD_PATH, joinpath(@__DIR__, "../src"))
using Statistics, Random, CSV, DataFrames, Plots, LinearAlgebra
include("../src/jansen_rit.jl")
include("../src/enkf.jl")

using .JansenRitSimulator
using .EnsembleKalmanFilter

function run_experiment()
    @info "Initializing Project 1: Online Data-Assimilating Virtual Brain Twin..."
    mkpath("results")
    mkpath("figures")

    # True parameter drift: Linear drift in excitatory gain A(t)
    drift_rate = 0.002
    true_drift(t) = drift_rate * t
    
    # 1. Simulate true system with drifting parameter
    tspan = (0.0, 500.0)
    sol = simulate_jansen_rit(tspan, A_val=3.25, drift_func=true_drift)
    
    # Extract synthetic EEG observation (pyramidal - inhibitory)
    times = sol.t
    obs_signal = sol[1, :] .- sol[5, :] + 0.05 * randn(length(sol.t))
    
    @info "Running Online EnKF Parameter Tracking..."
    # Initialize EnKF state: [y0, dy0, A_estimate]
    x0 = [obs_signal[1], 0.0, 3.25]
    P0 = Matrix{Float64}(I, 3, 3) .* 0.1
    enkf = EnsembleKF(x0, P0, 40)
    
    estimated_A = Float64[]
    true_A = Float64[]
    
    for (i, t) in enumerate(times)
        push!(true_A, 3.25 + true_drift(t))
        # Assimilate observation at time t
        H = [1.0, 0.0, 0.0]
        update_enkf!(enkf, obs_signal[i], H, 0.05)
        # Record mean ensemble estimate for A
        push!(estimated_A, mean(enkf.ensemble[3, :]))
    end
    
    # Compute RMSE for EnKF vs Static Baseline (mean of prior)
    static_baseline_A = 3.25
    rmse_enkf = sqrt(mean((estimated_A .- true_A).^2))
    rmse_static = sqrt(mean((static_baseline_A .- true_A).^2))
    
    @info "Results: EnKF RMSE = $rmse_enkf | Static Baseline RMSE = $rmse_static"
    
    # Save metrics CSV
    df = DataFrame(Time=times, True_A=true_A, Estimated_A=estimated_A)
    CSV.write("results/tracking_metrics.csv", df)
    
    # Generate and save main result figure
    p = plot(times, true_A, label="True Drifting A(t)", lw=2, color=:black, linestyle=:dash)
    plot!(p, times, estimated_A, label="Online EnKF Estimate", lw=2, color=:blue)
    hline!([static_baseline_A], label="Static Baseline", lw=1.5, color=:red, linestyle=:dot)
    xlabel!("Time (ms)")
    ylabel!("Excitatory Synaptic Gain A")
    title!("Project 1: Online Parameter Assimilation (EnKF)")
    savefig(p, "figures/main_result.png")
    @info "Saved verification figure to figures/main_result.png!"
end

run_experiment()
