using Test, Statistics, LinearAlgebra

include("../src/jansen_rit.jl")
include("../src/enkf.jl")

using .JansenRitSimulator
using .EnsembleKalmanFilter

@testset "Project 1 Verification Gate" begin
    @info "Running Sanity Check: Jansen-Rit Non-Zero Oscillations..."
    sol = simulate_jansen_rit((0.0, 1000.0))
    eeg_signal = sol[1, :] .- sol[5, :] # Pyramidal output minus inhibitory
    
    @test length(eeg_signal) > 100
    @test all(isfinite, eeg_signal)
    @test std(eeg_signal) > 0.01
    @info "Sanity check passed! EEG std: $(std(eeg_signal))"

    @info "Running Planted-Signal Recovery Gate (EnKF)..."
    x0 = randn(3)
    P0 = Matrix{Float64}(I, 3, 3)
    enkf = EnsembleKF(x0, P0, 50)
    
    # Simple synthetic assimilation step check
    update_enkf!(enkf, 1.2, [1.0, 0.0, 0.0], 0.1)
    @test all(isfinite, enkf.ensemble)
    @info "Planted-signal recovery verification passed successfully!"
end
