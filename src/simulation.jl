
module Simulation

using NbodyGradient
using .Elements
using .Integrators

struct Particle
    mass::Float64
    elements::Elements.KeplerianElements{Float64}
end

struct Simulation
    particles::Vector{Particle}
    integrator::Integrators.AbstractIntegrator
    μ::Float64
    t::Float64
end

function add_particle!(sim::Simulation, mass, elements)
    push!(sim.particles, Particle(mass, elements))
    update_μ!(sim)
end

function update_μ!(sim::Simulation)
    sim.μ = sum(p.mass for p in sim.particles)
end

function run_simulation(sim::Simulation, t_end; save_every=0.0)
    # Convert elements to Cartesian
    u0 = elements_to_cartesian(sim)
    
    # Run integration
    sol = Integrators.integrate(sim.integrator, u0, (sim.t, t_end), sim.μ)
    
    # Process results
    results = process_solution(sol, sim, save_every)
    
    # Update simulation time
    sim.t = t_end
    
    return results
end

end # module