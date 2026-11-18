module JansenRitSimulator
using DifferentialEquations, Random, Statistics

export jansen_rit_ode!, simulate_jansen_rit

function jansen_rit_ode!(du, u, p, t)
    # u = [y0, y1, y2, y3, y4, y5] (states and derivatives)
    # p = [A, B, a, b, v0, r, nu_max, g]
    A, B, a, b, v0, r, nu_max, g = p
    
    y0, dy0, y1, dy1, y2, dy2, y3, dy3, y4, dy4, y5, dy5 = u
    
    # Firing rate sigmoidal transformation
    S(v) = nu_max / (1.0 + exp(r * (v0 - v)))
    
    # Post-synaptic potentials driven by pulse densities
    # Pyramidal to excitatory/inhibitory interneurons & vice versa
    e0 = 2.5 # mV threshold/scaling
    
    # Differential equations for Jansen-Rit (6 blocks -> 12 state variables)
    du[1] = dy0
    du[2] = A * a * S(y1 - y2) - 2a * dy0 - a^2 * y0
    
    du[3] = dy1
    du[4] = A * a * (g * S(y0) + p_input(t)) - 2a * dy1 - a^2 * y1
    
    du[5] = dy2
    du[6] = B * b * S(y3) - 2b * dy2 - b^2 * y2
    
    du[7] = dy3
    du[8] = B * b * S(y0) - 2b * dy3 - b^2 * y3
    
    du[9] = dy4
    du[10] = A * a * S(y0) - 2a * dy4 - a^2 * y4
    
    du[11] = dy5
    du[12] = B * b * S(y4) - 2b * dy5 - b^2 * y5
end

p_input(t) = 220.0 + 20.0 * sin(2π * 10.0 * t / 1000.0) # External driving input

function simulate_jansen_rit(tspan; A_val=3.25, drift_func=t->0.0)
    u0 = zeros(12)
    u0[1] = 1.0
    
    # Time-varying parameter wrapper
    p_params(t) = [A_val + drift_func(t), 22.0, 100.0, 50.0, 6.0, 0.56, 5.0, 1.0]
    
    prob = ODEProblem(jansen_rit_ode!, u0, tspan, p_params(0.0))
    sol = solve(prob, Tsit5(), dt=0.5, adaptive=true)
    return sol
end

end
