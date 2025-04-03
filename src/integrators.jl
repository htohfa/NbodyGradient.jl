# src/integrators.jl
module Integrators

using NbodyGradient
using DiffEqBase, OrdinaryDiffEq

struct AdaptiveWH15 <: NbodyGradient.AbstractIntegrator
    atol::Float64
    rtol::Float64
    dt_initial::Float64
end

function integrate(integrator::AdaptiveWH15, u0, tspan, param)
    prob = ODEProblem(NbodyGradient.nbody_rhs!, u0, tspan, param)
    solve(prob, WisdomHolman15(integrator.dt_initial), 
          abstol=integrator.atol, reltol=integrator.rtol,
          adaptive=true)
end

struct FixedStepWH <: NbodyGradient.AbstractIntegrator
    dt::Float64
end

function integrate(integrator::FixedStepWH, u0, tspan, param)
    prob = ODEProblem(NbodyGradient.nbody_rhs!, u0, tspan, param)
    solve(prob, WisdomHolman15(integrator.dt), adaptive=false)
end

end # module