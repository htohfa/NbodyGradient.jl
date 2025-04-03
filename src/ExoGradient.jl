module ExoGradient

using Reexport, ForwardDiff, StaticArrays, LinearAlgebra
using NbodyGradient: StateTransitionMatrix
@reexport using NbodyGradient

include("elements.jl")
include("integrators.jl")
include("simulation.jl")
include("optimization.jl")

export KeplerianElements, cartesian_to_keplerian, keplerian_gradients
export AdaptiveWH15, FixedStepWH, Simulation, add_particle!, run_simulation
export fit_orbits, optimize_elements

end