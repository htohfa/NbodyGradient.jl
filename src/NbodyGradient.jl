"""
    NbodyGradient

An N-body integrator that computes derivatives with respect to initial conditions for TTVs, RV, Photodynamics, and more.
"""
module NbodyGradient

using LinearAlgebra, DelimitedFiles
using FileIO, JLD2

# Constants used by most functions
const NDIM = 3
const YEAR = 365.242
const GNEWT = 39.4845/(YEAR*YEAR)
const third = 1.0/3.0
const alpha0 = 0.0

# Types
export Elements, ElementsIC, CartesianIC, InitialConditions
export State, dState
export Integrator
export Jacobian, dTime
export CartesianOutput, ElementsOutput
export TransitTiming, TransitParameters, TransitSnapshot

# Integrator methods
export ahl21!, dh17!

# Utility functions
export available_systems, get_default_ICs

# Source code
include("PreAllocArrays.jl")
include("ics/InitialConditions.jl")
include("integrator/Integrator.jl")
include("utils.jl")
include("outputs/Outputs.jl")
include("transits/Transits.jl")

# Include the orbital elements file
include("orbital_elements.jl")
using .OrbitalElements
export KeplerianElements, KeplerianGradient, cartesian_to_keplerian, keplerian_gradients

# Function to integrate the system
function integrate_system(u0, tspan, param; save_orbital_elements=false, compute_gradients=false)
    # ... existing integration code ...
    
    solution = []  # Ensure solution is defined
    state_transition_matrices = []  # Ensure state_transition_matrices is defined
    
    if save_orbital_elements
        orbital_elements = []
        orbital_element_gradients = []
        
        for i in eachindex(solution)
            state = solution[i]
            x = state[1:3]
            v = state[4:6]
            
            # Compute orbital elements
            elements = cartesian_to_keplerian(x, v, param.μ)
            push!(orbital_elements, elements)
            
            # Compute gradients if requested
            if compute_gradients
                STM = state_transition_matrices[i]
                grad = keplerian_gradients(x, v, param.μ, 
                                         STM.dxdx0, STM.dvdx0, 
                                         STM.dxdv0, STM.dvdv0)
                push!(orbital_element_gradients, grad)
            end
        end
        
        return solution, orbital_elements, orbital_element_gradients
    else
        return solution
    end
end

end  # Close the module