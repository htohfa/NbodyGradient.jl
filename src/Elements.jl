module Elements

using LinearAlgebra, StaticArrays, ForwardDiff
using NbodyGradient: StateTransitionMatrix

export KeplerianElements, cartesian_to_keplerian, keplerian_gradients

struct KeplerianElements{T}
    a::T      # Semi-major axis
    e::T      # Eccentricity
    i::T      # Inclination (radians)
    Ω::T      # Longitude of ascending node (radians)
    ω::T      # Argument of periapsis (radians)
    M::T      # Mean anomaly (radians)
    f::T      # True anomaly (radians)
end

# Base constructor with default true anomaly
KeplerianElements(a, e, i, Ω, ω, M) = KeplerianElements(a, e, i, Ω, ω, M, zero(M))

# Helper functions
function clamp_cos(x)
    clamp(x, -1, 1)
end

function safe_acos(x)
    acos(clamp_cos(x))
end

function safe_atan(y, x)
    isapprox(x, 0, atol=1e-12) && isapprox(y, 0, atol=1e-12) ? 0.0 : atan(y, x)
end

"""
    cartesian_to_keplerian(x, v, μ)

Convert Cartesian coordinates to Keplerian orbital elements with full AD support.

# Arguments
- `x`: Position vector (3D)
- `v`: Velocity vector (3D)
- `μ`: Standard gravitational parameter (G*(m1 + m2))

# Returns
- `KeplerianElements`: Orbital elements (a, e, i, Ω, ω, M, f)
"""
function cartesian_to_keplerian(x::AbstractVector{T}, v::AbstractVector{T}, μ::T) where T
    r = norm(x)
    v² = dot(v, v)
    h = cross(x, v)
    h_norm = norm(h)
    
    # Specific orbital energy
    ξ = v²/2 - μ/r
    
    # Semi-major axis (handles hyperbolic cases)
    a = if ξ ≠ 0
        -μ/(2ξ)
    else
        T(Inf)  # Parabolic case
    end
    
    # Eccentricity vector and magnitude
    e_vec = if h_norm ≠ 0
        cross(v, h)/μ - x/r
    else
        zeros(T, 3)  # Rectilinear case
    end
    e = norm(e_vec)
    
    # Inclination
    i = if h_norm ≠ 0
        safe_acos(h[3]/h_norm)
    else
        zero(T)  # Undefined for rectilinear motion
    end
    
    # Node vector
    n = if !isapprox(h[1], 0, atol=1e-12) || !isapprox(h[2], 0, atol=1e-12)
        cross([0, 0, 1], h)
    else
        zeros(T, 3)  # Equatorial orbit
    end
    n_norm = norm(n)
    
    # Longitude of ascending node
    Ω = if n_norm ≠ 0
        safe_atan(n[2], n[1])
    else
        zero(T)  # Undefined for equatorial orbits
    end
    
    # Argument of periapsis
    ω = if n_norm ≠ 0 && e ≠ 0
        cos_ω = dot(n, e_vec)/(n_norm*e)
        sign(e_vec[3]) * safe_acos(clamp_cos(cos_ω))
    else
        zero(T)
    end
    
    # True anomaly
    f = if e ≠ 0
        cos_f = dot(e_vec, x)/(e*r)
        sign(dot(x, v)) * safe_acos(clamp_cos(cos_f))
    else
        # Circular orbit - use argument of latitude
        if n_norm ≠ 0
            cos_θ = dot(n, x)/(n_norm*r)
            sign(x[3]) * safe_acos(clamp_cos(cos_θ))
        else
            # Equatorial circular orbit
            safe_atan(x[2], x[1])
        end
    end
    
    # Mean anomaly (handles hyperbolic cases)
    M = if e < 1
        E = 2 * atan(sqrt((1-e)/(1+e)) * tan(f/2))  # Eccentric anomaly
        E - e*sin(E)
    elseif e > 1
        H = 2 * atanh(sqrt((e-1)/(e+1)) * tan(f/2))  # Hyperbolic anomaly
        e*sinh(H) - H
    else
        # Parabolic case - use Barker's equation
        D = tan(f/2)  # Parabolic eccentric anomaly
        D + D^3/3
    end
    
    KeplerianElements{T}(a, e, i, Ω, ω, M, f)
end

"""
    keplerian_gradients(x, v, μ, stm)

Compute gradients of all Keplerian elements with respect to initial conditions.

# Arguments
- `x`: Current position vector (3D)
- `v`: Current velocity vector (3D)
- `μ`: Standard gravitational parameter
- `stm`: StateTransitionMatrix from NbodyGradient

# Returns
Named tuple with all partial derivatives:
- da_dx0, da_dv0: Semi-major axis derivatives
- de_dx0, de_dv0: Eccentricity derivatives
- di_dx0, di_dv0: Inclination derivatives
- dΩ_dx0, dΩ_dv0: Longitude of node derivatives
- dω_dx0, dω_dv0: Argument of periapsis derivatives
- dM_dx0, dM_dv0: Mean anomaly derivatives
- df_dx0, df_dv0: True anomaly derivatives
"""
function keplerian_gradients(x::AbstractVector{T}, v::AbstractVector{T}, μ::T,
                            stm::StateTransitionMatrix{T}) where T
    
    # Convert to Dual numbers to compute partials
    x_dual = Dual.(x, (1,0,0))
    v_dual = Dual.(v, (0,1,0))
    
    # Compute elements with dual numbers
    elements = cartesian_to_keplerian(x_dual, v_dual, μ)
    
    # Extract partial derivatives for each element
    ∂a∂x = ForwardDiff.partials.(elements.a, 1)
    ∂a∂v = ForwardDiff.partials.(elements.a, 2)
    
    ∂e∂x = ForwardDiff.partials.(elements.e, 1)
    ∂e∂v = ForwardDiff.partials.(elements.e, 2)
    
    ∂i∂x = ForwardDiff.partials.(elements.i, 1)
    ∂i∂v = ForwardDiff.partials.(elements.i, 2)
    
    ∂Ω∂x = ForwardDiff.partials.(elements.Ω, 1)
    ∂Ω∂v = ForwardDiff.partials.(elements.Ω, 2)
    
    ∂ω∂x = ForwardDiff.partials.(elements.ω, 1)
    ∂ω∂v = ForwardDiff.partials.(elements.ω, 2)
    
    ∂M∂x = ForwardDiff.partials.(elements.M, 1)
    ∂M∂v = ForwardDiff.partials.(elements.M, 2)
    
    ∂f∂x = ForwardDiff.partials.(elements.f, 1)
    ∂f∂v = ForwardDiff.partials.(elements.f, 2)
    
    # Apply chain rule through state transition matrix
    da_dx0 = ∂a∂x' * stm.dxdx0 + ∂a∂v' * stm.dvdx0
    da_dv0 = ∂a∂x' * stm.dxdv0 + ∂a∂v' * stm.dvdv0
    
    de_dx0 = ∂e∂x' * stm.dxdx0 + ∂e∂v' * stm.dvdx0
    de_dv0 = ∂e∂x' * stm.dxdv0 + ∂e∂v' * stm.dvdv0
    
    di_dx0 = ∂i∂x' * stm.dxdx0 + ∂i∂v' * stm.dvdx0
    di_dv0 = ∂i∂x' * stm.dxdv0 + ∂i∂v' * stm.dvdv0
    
    dΩ_dx0 = ∂Ω∂x' * stm.dxdx0 + ∂Ω∂v' * stm.dvdx0
    dΩ_dv0 = ∂Ω∂x' * stm.dxdv0 + ∂Ω∂v' * stm.dvdv0
    
    dω_dx0 = ∂ω∂x' * stm.dxdx0 + ∂ω∂v' * stm.dvdx0
    dω_dv0 = ∂ω∂x' * stm.dxdv0 + ∂ω∂v' * stm.dvdv0
    
    dM_dx0 = ∂M∂x' * stm.dxdx0 + ∂M∂v' * stm.dvdx0
    dM_dv0 = ∂M∂x' * stm.dxdv0 + ∂M∂v' * stm.dvdv0
    
    df_dx0 = ∂f∂x' * stm.dxdx0 + ∂f∂v' * stm.dvdx0
    df_dv0 = ∂f∂x' * stm.dxdv0 + ∂f∂v' * stm.dvdv0
    
    # Return as named tuple for easy access
    return (a=(dx0=da_dx0, dv0=da_dv0),
            e=(dx0=de_dx0, dv0=de_dv0),
            i=(dx0=di_dx0, dv0=di_dv0),
            Ω=(dx0=dΩ_dx0, dv0=dΩ_dv0),
            ω=(dx0=dω_dx0, dv0=dω_dv0),
            M=(dx0=dM_dx0, dv0=dM_dv0),
            f=(dx0=df_dx0, dv0=df_dv0))
end

"""
    keplerian_to_cartesian(elements::KeplerianElements{T}, μ::T) where T

Convert Keplerian orbital elements to Cartesian coordinates.

# Arguments
- `elements`: KeplerianElements struct containing (a, e, i, Ω, ω, M, f)
- `μ`: Standard gravitational parameter (G*(m1 + m2))

# Returns
- `x`: Position vector in inertial frame (3D)
- `v`: Velocity vector in inertial frame (3D)
"""
function keplerian_to_cartesian(elements::KeplerianElements{T}, μ::T) where T
    # Unpack elements
    a, e, i, Ω, ω, M, f = elements.a, elements.e, elements.i, elements.Ω, elements.ω, elements.M, elements.f
    
    # Handle special cases for anomaly
    true_anomaly = if isnan(f)
        # Compute true anomaly from mean anomaly if not provided
        if e < 1 - eps(T)  # Elliptical case
            # Solve Kepler's equation M = E - e sin(E) for E
            E = solve_keplers_equation(M, e)
            # Convert to true anomaly
            2 * atan(sqrt((1 + e)/(1 - e)) * tan(E/2))
        elseif e > 1 + eps(T)  # Hyperbolic case
            # Solve M = e sinh(H) - H for H
            H = solve_hyperbolic_kepler(M, e)
            # Convert to true anomaly
            2 * atan(sqrt((e + 1)/(e - 1)) * tanh(H/2))
        else  # Parabolic case
            # Solve Barker's equation
            D = solve_barkers_equation(M)
            # True anomaly for parabolic case
            2 * atan(D)
        end
    else
        f  # Use provided true anomaly
    end
    
    # Compute perifocal coordinates (PQW frame)
    r = if e ≈ 1.0
        # Parabolic case special formula
        p = a * (1 - e^2)
        p / (1 + e * cos(true_anomaly))
    else
        # Standard elliptical/hyperbolic case
        a * (1 - e^2) / (1 + e * cos(true_anomaly))
    end
    
    # Position in perifocal frame
    x_pqw = r * [cos(true_anomaly), sin(true_anomaly), zero(T)]
    
    # Velocity in perifocal frame
    h = sqrt(μ * a * (1 - e^2))
    v_pqw = if e ≈ 1.0
        # Parabolic case
        sqrt(μ/2/p) * [-sin(true_anomaly), e + cos(true_anomaly), zero(T)]
    else
        (μ/h) * [-sin(true_anomaly), e + cos(true_anomaly), zero(T)]
    end
    
    # Rotation matrices
    R_Ω = [cos(Ω) -sin(Ω) 0;
           sin(Ω)  cos(Ω) 0;
           0       0      1]
    
    R_i = [1   0       0;
           0   cos(i) -sin(i);
           0   sin(i)  cos(i)]
    
    R_ω = [cos(ω) -sin(ω) 0;
           sin(ω)  cos(ω) 0;
           0       0      1]
    
    # Combined rotation matrix (PQW → IJK)
    R = R_Ω * R_i * R_ω
    
    # Transform to inertial frame
    x = R * x_pqw
    v = R * v_pqw
    
    return (x, v)
end

# Helper functions for Kepler's equation solutions
function solve_keplers_equation(M::T, e::T, tol=1e-12, maxiter=100) where T
    # Initial guess
    E = M + 0.85 * e * sign(sin(M))
    
    # Newton-Raphson iteration
    for _ in 1:maxiter
        f = E - e * sin(E) - M
        f_prime = 1 - e * cos(E)
        ΔE = f / f_prime
        E -= ΔE
        abs(ΔE) < tol && break
    end
    return E
end

function solve_hyperbolic_kepler(M::T, e::T, tol=1e-12, maxiter=100) where T
    # Initial guess
    H = M
    
    # Newton-Raphson iteration
    for _ in 1:maxiter
        f = e * sinh(H) - H - M
        f_prime = e * cosh(H) - 1
        ΔH = f / f_prime
        H -= ΔH
        abs(ΔH) < tol && break
    end
    return H
end

function solve_barkers_equation(M::T, tol=1e-12, maxiter=100) where T
    # Initial guess
    D = M^(1/3)
    
    # Newton-Raphson iteration
    for _ in 1:maxiter
        f = D + D^3/3 - M
        f_prime = 1 + D^2
        ΔD = f / f_prime
        D -= ΔD
        abs(ΔD) < tol && break
    end
    return D
end

"""
    keplerian_to_cartesian_derivatives(elements::KeplerianElements{T}, μ::T)

Compute derivatives of Cartesian coordinates with respect to Keplerian elements.

# Returns
- (dx_da, dx_de, dx_di, dx_dΩ, dx_dω, dx_dM)
- (dv_da, dv_de, dv_di, dv_dΩ, dv_dω, dv_dM)
"""
function keplerian_to_cartesian_derivatives(elements::KeplerianElements{T}, μ::T) where T
    # Convert to dual numbers to compute partials
    a_dual = Dual(elements.a, (1,0,0,0,0,0))
    e_dual = Dual(elements.e, (0,1,0,0,0,0))
    i_dual = Dual(elements.i, (0,0,1,0,0,0))
    Ω_dual = Dual(elements.Ω, (0,0,0,1,0,0))
    ω_dual = Dual(elements.ω, (0,0,0,0,1,0))
    M_dual = Dual(elements.M, (0,0,0,0,0,1))
    
    dual_elements = KeplerianElements(
        a_dual, e_dual, i_dual, Ω_dual, ω_dual, M_dual, elements.f
    )
    
    x_dual, v_dual = keplerian_to_cartesian(dual_elements, μ)
    
    # Extract partial derivatives for position
    dx_da = ForwardDiff.partials.(x_dual, 1)
    dx_de = ForwardDiff.partials.(x_dual, 2)
    dx_di = ForwardDiff.partials.(x_dual, 3)
    dx_dΩ = ForwardDiff.partials.(x_dual, 4)
    dx_dω = ForwardDiff.partials.(x_dual, 5)
    dx_dM = ForwardDiff.partials.(x_dual, 6)
    
    # Extract partial derivatives for velocity
    dv_da = ForwardDiff.partials.(v_dual, 1)
    dv_de = ForwardDiff.partials.(v_dual, 2)
    dv_di = ForwardDiff.partials.(v_dual, 3)
    dv_dΩ = ForwardDiff.partials.(v_dual, 4)
    dv_dω = ForwardDiff.partials.(v_dual, 5)
    dv_dM = ForwardDiff.partials.(v_dual, 6)
    
    return (
        (dx_da, dx_de, dx_di, dx_dΩ, dx_dω, dx_dM),
        (dv_da, dv_de, dv_di, dv_dΩ, dv_dω, dv_dM)
    )
end

end # module Elements