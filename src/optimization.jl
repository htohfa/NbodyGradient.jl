
module Optimization

using NbodyGradient
using .Elements
using .Simulation
using Optim
using Zygote

function fit_orbits(observations, initial_guess::Simulation; max_iter=100)
    # Define loss function
    function loss(sim_params)
        sim = update_simulation(initial_guess, sim_params)
        residuals = compute_residuals(sim, observations)
        return sum(abs2, residuals)
    end
    
    # Automatic differentiation
    grad_loss(params) = Zygote.gradient(loss, params)[1]
    
    # Optimization
    result = optimize(loss, grad_loss, initial_guess_to_params(initial_guess),
                     LBFGS(), Optim.Options(iterations=max_iter))
    
    return update_simulation(initial_guess, result.minimizer)
end

end # module