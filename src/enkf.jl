module EnsembleKalmanFilter
using Statistics, Random, LinearAlgebra

export EnsembleKF, update_enkf!

mutable struct EnsembleKF
    ensemble::Matrix{Float64} # Dim x N_ensemble
    N_ens::Int
    dim_state::Int
end

function EnsembleKF(x0::Vector{Float64}, P0::Matrix{Float64}, N_ens::Int)
    dim = length(x0)
    noise = randn(dim, N_ens)
    L = cholesky(Symmetric(P0 + 1e-6*I)).L
    ens = x0 .+ L * noise
    return EnsembleKF(ens, N_ens, dim)
end

function update_enkf!(enkf::EnsembleKF, y_obs::Float64, H::Vector{Float64}, R::Float64)
    N = enkf.N_ens
    dim = enkf.dim_state
    
    predicted_obs = H' * enkf.ensemble
    mean_obs = mean(predicted_obs)
    
    x_mean = mean(enkf.ensemble, dims=2)
    P_xy = zeros(dim)
    P_yy = R
    
    for i in 1:N
        dx = enkf.ensemble[:, i] - x_mean
        dy = predicted_obs[i] - mean_obs
        P_xy .+= dx * dy
    end
    P_xy ./= (N - 1)
    P_yy += sum((predicted_obs .- mean_obs).^2) / (N - 1)
    
    K = P_xy / P_yy
    
    for i in 1:N
        y_perturbed = y_obs + sqrt(R) * randn()
        enkf.ensemble[:, i] .+= K * (y_perturbed - predicted_obs[i])
    end
end

end
