module OrbitalElements

using LinearAlgebra
using StaticArrays

export cartesian_to_keplerian, keplerian_gradients, KeplerianGradient

"""
    KeplerianElements
Struct holding the six classical orbital elements.
"""
struct KeplerianElements{T}
    a::T  # Semi-major axis
    e::T  # Eccentricity
    i::T  # Inclination (radians)
    Ω::T  # Longitude of ascending node (radians)
    ω::T  # Argument of periapsis (radians)
    M::T  # Mean anomaly (radians)
end

"""
    KeplerianGradient
Struct holding gradients of orbital elements with respect to initial conditions.
"""
struct KeplerianGradient{T}
    da_dx0::SVector{3,T}  # Derivative of a w.r.t. initial position
    da_dv0::SVector{3,T}  # Derivative of a w.r.t. initial velocity
    de_dx0::SVector{3,T}  # Derivative of e w.r.t. initial position
    de_dv0::SVector{3,T}  # Derivative of e w.r.t. initial velocity
    di_dx0::SVector{3,T}  # Derivative of i w.r.t. initial position
    di_dv0::SVector{3,T}  # Derivative of i w.r.t. initial velocity
    dΩ_dx0::SVector{3,T} # Derivative of Ω w.r.t. initial position
    dΩ_dv0::SVector{3,T} # Derivative of Ω w.r.t. initial velocity
    dω_dx0::SVector{3,T} # Derivative of ω w.r.t. initial position
    dω_dv0::SVector{3,T} # Derivative of ω w.r.t. initial velocity
    dM_dx0::SVector{3,T} # Derivative of M w.r.t. initial position
    dM_dv0::SVector{3,T} # Derivative of M w.r.t. initial velocity
end

"""
    cartesian_to_keplerian(x, v, μ)

Convert Cartesian coordinates to Keplerian orbital elements.

# Arguments
- `x::AbstractVector{T}`: Position vector (3D)
- `v::AbstractVector{T}`: Velocity vector (3D)
- `μ::T`: Standard gravitational parameter (G*(m1 + m2))

# Returns
- `KeplerianElements{T}`: Orbital elements (a, e, i, Ω, ω, M)
"""
function cartesian_to_keplerian(x::AbstractVector{T}, v::AbstractVector{T}, μ::T) where T
    r = norm(x)
    v2 = dot(v, v)
    h = cross(x, v)
    h_norm = norm(h)
    
    # Specific angular momentum
    ξ = v2/2 - μ/r
    
    # Semi-major axis
    a = -μ/(2ξ)
    
    # Eccentricity vector and magnitude
    e_vec = cross(v, h)/μ - x/r
    e = norm(e_vec)
    
    # Inclination
    i = acos(h[3]/h_norm)
    
    # Node line vector
    n = cross([0, 0, 1], h)
    n_norm = norm(n)
    
    # Longitude of ascending node
    Ω = n_norm != 0 ? atan(n[2], n[1]) : zero(T)
    
    # Argument of periapsis
    ω = if n_norm != 0 && e > sqrt(eps(T))
        cos_ω = dot(n, e_vec)/(n_norm*e)
        cos_ω = clamp(cos_ω, -one(T), one(T))
        sgn = e_vec[3] >= 0 ? 1 : -1
        sgn * acos(cos_ω)
    else
        zero(T)
    end
    
    # True anomaly
    ν = if e > sqrt(eps(T))
        cos_ν = dot(e_vec, x)/(e*r)
        cos_ν = clamp(cos_ν, -one(T), one(T))
        sgn = dot(x, v) >= 0 ? 1 : -1
        sgn * acos(cos_ν)
    else
        # Circular orbit - use angle from node line
        if n_norm != 0
            cos_θ = dot(n, x)/(n_norm*r)
            cos_θ = clamp(cos_θ, -one(T), one(T))
            sgn = x[3] >= 0 ? 1 : -1
            sgn * acos(cos_θ)
        else
            # Equatorial circular orbit
            atan(x[2], x[1])
        end
    end
    
    # Eccentric anomaly
    E = if e < 1
        2 * atan(sqrt((1-e)/(1+e)) * tan(ν/2))
    else
        # Hyperbolic case
        asinh(sqrt((e-1)/(e+1)) * tan(ν/2))
    end
    
    # Mean anomaly
    M = if e < 1
        E - e*sin(E)
    else
        # Hyperbolic case
        e*sinh(E) - E
    end
    
    return KeplerianElements{T}(a, e, i, Ω, ω, M)
end

"""
    keplerian_gradients(x, v, μ, dxdx0, dvdx0, dxdv0, dvdv0)

Compute gradients of Keplerian elements with respect to initial conditions.

# Arguments
- `x`, `v`: Current position and velocity
- `μ`: Standard gravitational parameter
- `dxdx0`, `dvdx0`, `dxdv0`, `dvdv0`: State transition matrix components

# Returns
- `KeplerianGradient{T}`: Gradients of all orbital elements
"""
function keplerian_gradients(x::AbstractVector{T}, v::AbstractVector{T}, μ::T,
                            dxdx0::AbstractMatrix{T}, dvdx0::AbstractMatrix{T},
                            dxdv0::AbstractMatrix{T}, dvdv0::AbstractMatrix{T}) where T
    
    r = norm(x)
    r3 = r^3
    v2 = dot(v, v)
    h = cross(x, v)
    h_norm = norm(h)
    h_norm3 = h_norm^3
    
    # Common partial derivatives
    ∂r∂x = x ./ r
    ∂v2∂v = 2v
    ∂h∂x = [-v[3] v[2]; v[3] -v[1]; -v[2] v[1]]
    ∂h∂v = [x[3] -x[2]; -x[3] x[1]; x[2] -x[1]]
    
    # Energy gradient (ξ = v²/2 - μ/r)
    ∂ξ∂x = μ ./ r3 .* x
    ∂ξ∂v = v
    
    # Semi-major axis gradient (a = -μ/(2ξ))
    ξ = v2/2 - μ/r
    ∂a∂ξ = μ/(2ξ^2)
    ∂a∂x = ∂a∂ξ .* ∂ξ∂x
    ∂a∂v = ∂a∂ξ .* ∂ξ∂v
    
    # Eccentricity vector gradient (e_vec = (v×h)/μ - x/r)
    ∂evec∂x = (cross(v, ∂h∂x) ./ μ .- (I(3) ./ r .- x * ∂r∂x' ./ r2))
    ∂evec∂v = (cross(v, ∂h∂v) .+ cross(I(3), h)) ./ μ
    
    e_vec = cross(v, h)/μ - x/r
    e = norm(e_vec)
    ∂e∂evec = e > eps(T) ? e_vec ./ e : zeros(T, 3)
    
    ∂e∂x = ∂e∂evec' * ∂evec∂x
    ∂e∂v = ∂e∂evec' * ∂evec∂v
    
    # Inclination gradient (i = acos(h_z/|h|))
    ∂i∂h = if h_norm > eps(T)
        h_z = h[3]
        [-h_z*h[1], -h_z*h[2], h_norm^2 - h_z^2] ./ (h_norm3 * sqrt(1 - (h_z/h_norm)^2))
    else
        zeros(T, 3)
    end
    
    ∂i∂x = ∂i∂h' * ∂h∂x
    ∂i∂v = ∂i∂h' * ∂h∂v
    
    # Node line gradient (n = k × h)
    ∂n∂h = [0 0 0; 0 0 -1; 0 1 0]
    
    n = cross([0, 0, 1], h)
    n_norm = norm(n)
    
    # Longitude of ascending node gradient (Ω = atan(n_y/n_x))
    ∂Ω∂n = if n_norm > eps(T)
        [-n[2], n[1], 0] ./ (n[1]^2 + n[2]^2)
    else
        zeros(T, 3)
    end
    
    ∂Ω∂x = ∂Ω∂n' * ∂n∂h * ∂h∂x
    ∂Ω∂v = ∂Ω∂n' * ∂n∂h * ∂h∂v
    
    # Argument of periapsis gradient (ω = acos(n·e/(|n||e|)))
    if n_norm > eps(T) && e > eps(T)
        n_dot_e = dot(n, e_vec)
        ∂ω∂n = (e_vec .* n_norm .- n .* n_dot_e ./ n_norm) ./ (e * n_norm^2 * sqrt(1 - (n_dot_e/(e*n_norm))^2))
        ∂ω∂evec = (n .* n_norm .- e_vec .* n_dot_e ./ e) ./ (e^2 * n_norm * sqrt(1 - (n_dot_e/(e*n_norm))^2))
        
        ∂ω∂x = ∂ω∂n' * ∂n∂h * ∂h∂x .+ ∂ω∂evec' * ∂evec∂x
        ∂ω∂v = ∂ω∂n' * ∂n∂h * ∂h∂v .+ ∂ω∂evec' * ∂evec∂v
    else
        ∂ω∂x = zeros(T, 3)
        ∂ω∂v = zeros(T, 3)
    end
    
    # Mean anomaly gradient (M = E - e sin E)
    # First need to compute eccentric anomaly E
    ν = acos(clamp(dot(e_vec, x)/(e*r), -one(T), one(T)))
    E = 2 * atan(sqrt((1-e)/(1+e)) * tan(ν/2))
    
    ∂E∂e = (-sin(ν))/(e^2 - 2e*cos(ν) + 1)
    ∂E∂ν = (1 - e^2)/(e^2 - 2e*cos(ν) + 1)
    
    ∂ν∂x = if e > eps(T)
        # Derivative of ν = acos(e·r/(e r))
        term1 = ∂evec∂x' * x ./ (e*r)
        term2 = e_vec' * (I(3)/r - x*∂r∂x'/r2) ./ (e*r)
        term3 = (dot(e_vec, x) * (∂e∂x'*x + e_vec'*I(3)))/(e^2*r)
        (-term1 .+ term2 .- term3) ./ sqrt(1 - (dot(e_vec, x)/(e*r))^2)
    else
        zeros(T, 3)
    end
    
    ∂ν∂v = if e > eps(T)
        term1 = ∂evec∂v' * x ./ (e*r)
        term2 = (dot(e_vec, x) * ∂e∂v' * x)/(e^2*r)
        (-term1 .+ term2) ./ sqrt(1 - (dot(e_vec, x)/(e*r))^2)
    else
        zeros(T, 3)
    end
    
    ∂M∂E = 1 - e*cos(E)
    ∂M∂e = -sin(E)
    
    ∂M∂x = ∂M∂E * (∂E∂e * ∂e∂x .+ ∂E∂ν * ∂ν∂x) .+ ∂M∂e * ∂e∂x
    ∂M∂v = ∂M∂E * (∂E∂e * ∂e∂v .+ ∂E∂ν * ∂ν∂v) .+ ∂M∂e * ∂e∂v
    
    # Combine with state transition matrix to get derivatives w.r.t. initial conditions
    da_dx0 = ∂a∂x' * dxdx0 .+ ∂a∂v' * dvdx0
    da_dv0 = ∂a∂x' * dxdv0 .+ ∂a∂v' * dvdv0
    
    de_dx0 = ∂e∂x' * dxdx0 .+ ∂e∂v' * dvdx0
    de_dv0 = ∂e∂x' * dxdv0 .+ ∂e∂v' * dvdv0
    
    di_dx0 = ∂i∂x' * dxdx0 .+ ∂i∂v' * dvdx0
    di_dv0 = ∂i∂x' * dxdv0 .+ ∂i∂v' * dvdv0
    
    dΩ_dx0 = ∂Ω∂x' * dxdx0 .+ ∂Ω∂v' * dvdx0
    dΩ_dv0 = ∂Ω∂x' * dxdv0 .+ ∂Ω∂v' * dvdv0
    
    dω_dx0 = ∂ω∂x' * dxdx0 .+ ∂ω∂v' * dvdx0
    dω_dv0 = ∂ω∂x' * dxdv0 .+ ∂ω∂v' * dvdv0
    
    dM_dx0 = ∂M∂x' * dxdx0 .+ ∂M∂v' * dvdx0
    dM_dv0 = ∂M∂x' * dxdv0 .+ ∂M∂v' * dvdv0
    
    return KeplerianGradient{T}(
        SVector{3,T}(da_dx0), SVector{3,T}(da_dv0),
        SVector{3,T}(de_dx0), SVector{3,T}(de_dv0),
        SVector{3,T}(di_dx0), SVector{3,T}(di_dv0),
        SVector{3,T}(dΩ_dx0), SVector{3,T}(dΩ_dv0),
        SVector{3,T}(dω_dx0), SVector{3,T}(dω_dv0),
        SVector{3,T}(dM_dx0), SVector{3,T}(dM_dv0)
    )
end

end # module