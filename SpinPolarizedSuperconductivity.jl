# Copyright (c) 2025 Max Geier, Massachusetts Institute of Technology, MA, USA
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================

""" 
Functions required to compute superconductivity from screening of
electron repulsion in a two-dimensional spin-polarized electron gas
in Random Phase Approximation.
"""


using Hankel
using SpecialFunctions
using Plots
using LinearAlgebra
using Interpolations
using LaTeXStrings
using Base.Threads
using DelimitedFiles
using FileIO
using Measures

print("Included SpinPolarizedSuperconductivity.jl. \n")


struct ModelParameters
    n::Float64 # density
    EF::Float64 # Fermi energy
    kF::Float64 # largest Fermi momentum
    kFs::Vector{Float64} # Collection of Fermi momenta for annular FS
    BandBottom::Float64 # Bottom of band
    m::Float64 # effective mass of quadratic term; m = 1.6625774286228414 [meV ps^2 / nm^2] s.t. hbar^2 k0^2 / 2 m = 5 meV
    D::Float64 # band splitting from displacement field
    m2::Float64 # mass of an additional quadratic term to obtain an annular dispersion
    α::Float64 # prefactor of k^4 term
    vF_G::Float64 # Fermi velocity of Graphene 10^3 nm / ps
    k_D::Float64 # Parameter two-band model of graphene
    which_dispersion::String
    which_geometry::String
    which_interaction::String
    alphaD::Float64 # Ratio between applied displacement field and parameter D in fitted dispersion "ABCA"
    rKeldysh::Float64 # Length scale of the Keldysh potential [nm]
    eps2d::Float64 # relative dielectric constant of multilayer graphene sheet
    d_gates::Float64 # distance to gates d = 30 - 40 nm
    epsilon0::Float64 # e^2 / meV / nm
    hbar::Float64 # reduced Planck action quantum # meV ps
    hbarc::Float64
    c::Float64 # Speed of light nm / ps
    kBoltzmann::Float64 # Boltzmann constant # meV / K
    m_bare_electron::Float64 # bare electron mass
    nu_bare_electron::Float64 # density of states corresponding to bare electron mass [1/meV 1/nm^2]
    a::Float64 # Lattice constant a = 0.245 nm
    L_Torus::Int64 # Number of unit cells in the two directions of the torus = number of momentum space grid points in each direction
    A::Float64 # Total area of torus
    unit_cell_area::Float64 # Area of graphene unit cell
    stamp::String # Stamp to identify dataset from filename

    function ModelParameters(
            n::Float64=1.0, 
            m::Float64=1.0, 
            D::Float64=60.0, 
            m2::Float64=0.0, 
            α::Float64=1.0, 
            vF_G::Float64=1e3, 
            k_D::Float64=0.0,
            which_dispersion::String="ABCA_k4", 
            which_geometry::String="flat", 
            which_interaction::String="Keldysh",
            alphaD::Float64=1.0,
            rKeldysh::Float64=1.0, 
            eps2d::Float64=5.0, 
            d_gates::Float64=40.0, 
            L_Torus::Int64=21)
        epsilon0 = 4π*55.26349406e-6 # e^2 / meV / nm
        c = 299792.458 # nm / ps
        hbar = 6.582119569e-1 # meV ps
        kBoltzmann = 8.617333262e-2 # meV / K
        hbarc =  hbar*c # meV nm
        m_bare_electron = 5.10998950e8 / c^2 # meV ps^2 / nm^2 
        nu_bare_electron = m_bare_electron/hbar^2/(2π)
        a = 0.245 # nm 
        unit_cell_area = sqrt(3)*a^2/2
        A = L_Torus^2*unit_cell_area
        if which_dispersion[end-2:end] == "_k2"
            DPars = DispersionParameters(m, D, m2, α, vF_G, k_D, which_dispersion, which_geometry)
            # @show DPars
            kmax_findEF = 0.6
            EF, kFs, BandBottom = find_EF(DPars, n, [-10.0, 0.0, 6.0, 36.0, 100.0], kmax_findEF) # This line gives an error when E_F is larger than the third argument
            print("\n finding from density: EF, kFs = "*string((EF, kFs)))
            if length(kFs) == 0 || abs(maximum([Dispersion(kF, DPars) for kF in kFs]) - EF)/EF >= 1e-4
                kmax_findEF = kmax_findEF + 0.25
                EF, kFs, BandBottom = find_EF(DPars, n, [1.0, 6.0, 36.0, 100.0], kmax_findEF)
            end
            kF = maximum(abs.(kFs))
            if length(kFs) == 2
                n_from_kFs = (kFs[1]^2 - kFs[2]^2)/4π
            else
                n_from_kFs = kFs[1]^2/4π
            end
            # print("Verifying density: [n from kFs, n] = " * string([n_from_kFs, n]))
            if abs(n_from_kFs - n) >= 1e-6
                print("\n ModelParameters: n_from_kFs does not agree with n!")
                print([n_from_kFs, n])
            end
        else
            print("using kF for single Fermi surface for dispersion " * which_dispersion * "\n")
            kF = sqrt(4π*n)
            DPars = DispersionParameters(m, D, m2, α, vF_G, k_D, which_dispersion, which_geometry)
            EF = Dispersion(kF, DPars)
            kFs = [kF]
            BandBottom = 0.0
        end
        
        if which_dispersion == "ABCA_linear_alpha" || which_dispersion == "ABCA_linear_alpha_k2"
            stamp = "data/"*join([string(el) for el in [which_dispersion, n, D, m2, α, vF_G, rKeldysh, eps2d, d_gates, L_Torus]], "_")
        elseif which_dispersion[1:9] == "ABCA_sqrt"
            stamp = "data/"*join([string(el) for el in [which_dispersion, n, m, D, m2, rKeldysh, eps2d, d_gates, L_Torus]], "_")
        elseif which_dispersion[1:9] == "ABCA_2band_k2"
            stamp = "data/"*join([string(el) for el in [which_dispersion, n, m, D, k_D, rKeldysh, eps2d, d_gates, L_Torus]], "_")
        else
            stamp = "data/"*join([string(el) for el in [which_dispersion, n, m, D, m2, α, vF_G, rKeldysh, eps2d, d_gates, L_Torus]], "_")
        end

        new(n, EF, kF, kFs, BandBottom, m, D, m2, α, vF_G, k_D, which_dispersion, which_geometry, which_interaction, alphaD, rKeldysh, eps2d, d_gates, epsilon0, hbar, hbarc, c, kBoltzmann, m_bare_electron, nu_bare_electron, a, L_Torus, A, unit_cell_area, stamp)
    end
end

# -------------------------------------------------------------------------
# ----------------------- Dispersion and parameters -----------------------
# -------------------------------------------------------------------------
struct DispersionParameters
    m::Float64
    D::Float64
    m2::Float64
    α::Float64
    vF_G::Float64
    k_D::Float64
    hbar::Float64
    which_dispersion::String
    which_geometry::String
    function DispersionParameters(m::Float64, D::Float64, m2::Float64, α::Float64, vF_G::Float64, k_D::Float64, which_dispersion::String, which_geometry::String)
        hbar = 6.582119569e-1 # meV ps
        new(m, D, m2, α, vF_G, k_D, hbar, which_dispersion, which_geometry)
    end
end

function find_EF(Pars::Union{ModelParameters,DispersionParameters}, n::Float64, EF_maxs::Vector{Float64}, k_edge::Float64)
    nks = 5000
    # dk = k_edge/nks
    ks = collect(range(0.0, k_edge, nks))
    xi_k = Dispersion.(ks, Ref(Pars)) .- Pars.D
    # @show ks[1], ks[10], ks[end]
    # @show xi_k[1], xi_k[10], xi_k[end]

    BandBottom = minimum(xi_k)
    
    EF = 0.0
    for EF_max in EF_maxs
        n_E = Vector{Float64}()
        EF_range = collect(range(minimum(xi_k), EF_max, 2001))
        for EF_i in EF_range
            kFs_i = find_sign_changes(ks, xi_k .- EF_i)
            if length(kFs_i) == 0
                # print("find_EF: Warning: No Fermi surface detected. \n")
                push!(n_E, 0.0)
            elseif length(kFs_i) == 1
                push!(n_E, (kFs_i[1]^2)/(4π))
            elseif length(kFs_i) == 2
                push!(n_E, abs(kFs_i[2]^2 - kFs_i[1]^2)/(4π))
            else
                print("find_EF: Warning: more than two Fermi surfaces detected. \n")
                push!(n_E, 0.0)
            end
        end
        if n > maximum(n_E)
            continue
        else
            EF = find_largest_zero_crossing(EF_range, n_E .- n)
            break
        end
    end
    if EF === nothing
        print("\n find_EF: WARNING: EF not found below "*string(maximum(EF_maxs)))
    end
    kFs = sort(find_sign_changes(ks, xi_k .- EF), rev=true)
    return EF, kFs, BandBottom
end

function find_hat_density(Pars::Union{ModelParameters,DispersionParameters})::Float64
    nks = 10000
    k_edge = 2.0
    ks = collect(range(0.0, k_edge, nks))
    xi_k = Dispersion.(ks, Ref(Pars)) .- Pars.D
    kF = find_largest_zero_crossing(ks, xi_k)
    if kF === nothing
        return 0.0
    else
        return (kF^2)/(4π)
    end
end

function Dispersion(k::Float64, Pars::Union{ModelParameters,DispersionParameters})::Float64
    if Pars.which_dispersion == "quadratic"
        return Pars.hbar^2*k^2/2/Pars.m
    elseif Pars.which_dispersion == "ABCA_Slizovskiy_k2"
        return Pars.D*sqrt(1 + (k/Pars.k_D)^8) + Pars.hbar^2*k^2/2/Pars.m2
    else
        print("Dispersion: WARNING: Invalid value passed to Pars.which_dispersion")
        # exit()
    end
end

function Dispersion(k::Vector{Float64}, Pars::Union{ModelParameters,DispersionParameters})::Float64
    return Dispersion(norm(k), Pars)
end

function density(kF::Float64)::Float64
   return kF^2/4π 
end

function DOS(E::Float64, Pars::Union{ModelParameters,DispersionParameters})::Float64
    if Pars.which_dispersion == "quadratic"
        return Pars.m/Pars.hbar^2/2π
    else
        print("DOS: No analytical expressing for DOS given for which_dispersion = "*Pars.which_dispersion*", resorting to numerical calculation. \n")
        return DOS_EF(Pars)
    end
end

function DOS_EF(Pars::Union{ModelParameters,DispersionParameters})::Float64
    dk = 1e-6
    dEdks = [(Dispersion(kFi+dk/2, Pars) - Dispersion(kFi-dk/2, Pars))/dk for kFi in Pars.kFs]
    nu = sum(Pars.kFs./abs.(dEdks)/(2π))
    return nu
end

# ----------------------------------------------------------------------------------
# -------------- Functions for form factor for wavefunction geometry ---------------
# ----------------------------------------------------------------------------------
function TrMetricTensor(k::Vector{Float64}, Pars::Union{ModelParameters,DispersionParameters})::Float64
    knorm = norm(k)
    if Pars.which_geometry == "flat"
        return 0
    elseif Pars.which_geometry == "ABCA_Slizovskiy_k2"
        N_layers = 4
        return N_layers^2/4/Pars.unit_cell_area * (Pars.k_D^(2*N_layers) * knorm^(2*N_layers - 2))/(Pars.k_D^(2*N_layers) + knorm^(2*N_layers))^2
    else
        print("TrMetricTensor: Pars.which_geometry: "*Pars.which_geometry*" not implemented")
        return 0
    end
end

function TrMetricTensor(k::Float64, Pars::Union{ModelParameters,DispersionParameters})::Float64
    return TrMetricTensor([k, 0], Pars)
end

function BerryCurvature(k::Vector{Float64}, Pars::Union{ModelParameters,DispersionParameters})::Float64
    knorm = norm(k)
    if Pars.which_geometry == "flat"
        return 0
    elseif Pars.which_geometry == "ABCA_Slizovskiy_k2"
        N_layers = 4
        return -N_layers^2/2/Pars.A_uc * (Pars.k_D^(N_layers) * knorm^(2*N_layers - 2))/(Pars.k_D^(2*N_layers) + knorm^(2*N_layers))^(3/2)
    else
        print("BerryCurvature: Pars.which_geometry: "*Pars.which_geometry*" not implemented")
        return 0
    end
end

function BerryCurvature(k::Float64, Pars::Union{ModelParameters,DispersionParameters})::Float64
    return BerryCurvature([k, 0], Pars)
end

function FormFactor(Pars::Union{ModelParameters,DispersionParameters},k1::Vector{Float64}, k2::Vector{Float64})::ComplexF64
    if Pars.which_geometry == "flat"
        return 1.
    elseif Pars.which_geometry == "ABCA_Slizovskiy_k2"
        kappa1 = norm(k1)/Pars.k_D
        gamma1 = atan(k1[2], k1[1])
        kappa2 = norm(k2)/Pars.k_D
        gamma2 = atan(k2[2], k2[1])
        sqrt_k1 = sqrt(1 + kappa1^8)
        sqrt_k2 = sqrt(1 + kappa2^8)
        
        sin_t1 = sqrt((sqrt_k1 - 1)/2/sqrt_k1)
        cos_t1 = sqrt((sqrt_k1 + 1)/2/sqrt_k1)
        sin_t2 = sqrt((sqrt_k2 - 1)/2/sqrt_k2)
        cos_t2 = sqrt((sqrt_k2 + 1)/2/sqrt_k2)
        return (1*exp(-1*im*gamma2 + 1*im*gamma1))*sin_t1*sin_t2 + cos_t1*cos_t2 
    else
        print("FormFactor: Pars.which_geometry: "*Pars.which_geometry*" not implemented")
        return 1.
    end
end

# ----------------------------------------------------------------------------------
# --------------------------- Susceptibility calculations --------------------------
# ----------------------------------------------------------------------------------
function NoninteractingSusceptibilityNumerical(q::Vector{Float64}, 
    Mesh::Vector{Vector{Float64}}, Pars::ModelParameters)::Float64
    if norm(q)/Pars.kF <= 4e-2
        return -DOS_EF(Pars)
    else
        chi0 = 0.0
        # dk = 2*Pars.kF/(Pars.L_Torus-1)/2/π
        dk = norm(Mesh[2] .- Mesh[1])/2π
        mu = (Pars.EF + Pars.D)
        if Pars.which_dispersion[end-2:end] == "_k2"
            fFD(x) = x < 0 ? 1.0 : 0.0 # Fermi-Dirac distribution
            for k in Mesh
                Ek  = Dispersion(k, Pars)
                Ekq = Dispersion(k + q, Pars)

                if abs(Ek - Ekq) >= 1e-16
                    chi0 = chi0 + (fFD(Ek - mu) - fFD(Ekq - mu))/(Ek - Ekq)
                end
                # @show chi0, Ek, Ekq
            end
            # @show chi0
            return chi0*dk^2
        else
            for k in Mesh
                if norm(k) <= Pars.kF
                    Ek  = Dispersion(k, Pars)
                    Ekq = Dispersion(k + q, Pars)
                    if abs(Ek - Ekq) >= 1e-12
                        chi0 = chi0 + 1/(Ek - Ekq)
                    end
                end
            end
            return 2*chi0*dk^2
        end
    end
end

function NoninteractingSusceptibilityAnalytical(q::Vector{Float64}, Pars::ModelParameters)::Float64
    qnorm = norm(q)
    if Pars.which_dispersion == "quadratic"
        DOS = Pars.m/Pars.hbar^2/2π
        z = norm(q)/2/Pars.kF
        function heaviside(t)
            0.5 * (sign(t) + 1)
        end
        if z<=1 
            return -DOS
        else
            return -DOS*(1 - sqrt(1-z^(-2)))
        end
    else
        print("\n NoninteractingSusceptibilityAnalytical: WARNING: which_dispersion not implemented! \n")
    end
end

function NoninteractingSusceptibilityFormFactor(q::Vector{Float64}, 
    Mesh::Vector{Vector{Float64}}, Pars::ModelParameters)::Float64
    chi0 = 0.0
    # dk = 2*Pars.kF/(Pars.L_Torus-1)/2/π
    dk = norm(Mesh[2] .- Mesh[1])/2π
    mu = (Pars.EF + Pars.D)
    if Pars.which_dispersion[end-2:end] == "_k2"
        fFD(x) = x < 0 ? 1.0 : 0.0
        for k in Mesh
            Ek  = Dispersion(k, Pars)
            Ekq = Dispersion(k + q, Pars)

            if abs(Ek - Ekq) >= 1e-16
                chi0 = chi0 + abs(FormFactor(Pars, k, k+q))^2 * (fFD(Ek - mu) - fFD(Ekq - mu))/(Ek - Ekq)
            end
            # @show chi0, Ek, Ekq
        end
        # @show chi0
        return chi0*dk^2
    else
        for k in Mesh
            if norm(k) <= Pars.kF
                Ek  = Dispersion(k, Pars)
                Ekq = Dispersion(k + q, Pars)
                if abs(Ek - Ekq) >= 1e-12
                    chi0 = chi0 + abs(FormFactor(Pars, k, k+q))^2/(Ek - Ekq)
                end
            end
        end
        return 2*chi0*dk^2
    end
end

# ----------------------------------------------------------------------------------
# ----------------------------- Interaction definition -----------------------------
# ----------------------------------------------------------------------------------
function Coulomb2d(q::Float64, Pars::ModelParameters)::Float64
    if isapprox(q, 0.0)
        return 0
    else
        return 2*π/Pars.epsilon0/Pars.eps2d/abs(q)
    end
end

function Coulomb2d_RealSpace(r::Float64, Pars::ModelParameters)::Float64
    if isapprox(r, 0.0)
        return 0
    else
        return 1/Pars.epsilon0/Pars.eps2d/abs(r)
    end
end

function Keldysh2d(q::Float64, Pars::ModelParameters)::Float64
    q0_term = 2*π/Pars.epsilon0/Pars.eps2d*Pars.d_gates
    if isapprox(q, 0.0)
        return q0_term
    else
        return 2*π/Pars.epsilon0/Pars.eps2d/abs(q)/(1 + Pars.rKeldysh*abs(q))*tanh(abs(q)*Pars.d_gates)
    end
end

# ----------------------------------------------------------------------------------
# ------------------------------------- RPA ----------------------------------------
# ----------------------------------------------------------------------------------
function InteractionRPA(Interaction::Function, Susceptibility::Float64, q::Float64, Pars::ModelParameters)::Float64
    """
    RPA expression for screened interaction.
    """
    return Interaction(q, Pars)/(1 - Interaction(q, Pars)*Susceptibility)
end

function InteractionRPAFormFactor(Interaction::Function, k1::Vector{Float64}, k2::Vector{Float64}, Susceptibility::Float64, Pars::ModelParameters)::Complex64
    """
    Includes the form factor in the Coulomb interaction
    """
    q = norm(k1 - k2)
    return FormFactor(k1, k2, Pars)*FormFactor(-k1, -k2, Pars)*Interaction(q, Pars)/(1 - Interaction(q, Pars)*Susceptibility)
end

# ----------------------------------------------------------------------------------
# ------------------- Decomposition into angular harmonics -------------------------
# ----------------------------------------------------------------------------------
function FermiSurfaceAngularDecomposition(V::Function, ms::Vector{Int64}, kF::Float64, 
    ThetaResolution::Int64=101)::Vector{Float64}
    AngularCoefficients = Vector{Float64}()
    Thetas = range(0, 2π, ThetaResolution)[1:end-1]
    for m in ms
        push!(AngularCoefficients, sum(V.(2*kF*sin.(Thetas/2)) .* cos.(m*Thetas))/ThetaResolution*2π)
    end
    return AngularCoefficients
end

function AngularDecompositionRadial(k1::Float64, k2::Float64, V::Function, m::Int64, 
    ThetaResolution::Int64=101)::Float64
    Thetas = range(0, 2π, ThetaResolution)[1:end-1]
    return sum(V.(sqrt.(abs.(k1^2 .+ k2^2 .- 2*k1*k2*cos.(Thetas)))) .* cos.(m*Thetas))/ThetaResolution*2π
end

function AngularDecompositionRadial_FormFactor(k1::Float64, k2::Float64, V::Function, m::Int64, 
    ThetaResolution::Int64, Pars::ModelParameters)::Float64
    Thetas = range(0, 2π, ThetaResolution)[1:end-1]
    FormFactors = [FormFactor(Pars, [k1, 0], [k2*cos(theta), k2*sin(theta)])^2 for theta in Thetas]
    return real(sum( FormFactors .* V.(sqrt.(abs.(k1^2 .+ k2^2 .- 2*k1*k2*cos.(Thetas)))) .* exp.(im*m*Thetas)))/ThetaResolution*2π
end

function AngularDecompositionRadial_FormFactorComplex(k1::Float64, k2::Float64, V::Function, m::Int64, 
    ThetaResolution::Int64, Pars::ModelParameters)::ComplexF64
    Thetas = range(0, 2π, ThetaResolution)[1:end-1]
    FormFactors = [FormFactor(Pars, [k1, 0], [k2*cos(theta), k2*sin(theta)])^2 for theta in Thetas]
    return sum( FormFactors .* V.(sqrt.(abs.(k1^2 .+ k2^2 .- 2*k1*k2*cos.(Thetas)))) .* exp.(im*m*Thetas))/ThetaResolution*2π
end

# ----------------------------------------------------------------------------------
# ---------- Computes susceptibility and renormalized interaction ------------------
# ----------------------------------------------------------------------------------
function createSquareMesh(kF::Float64, Pars::ModelParameters)::Vector{Vector{Float64}}
    xs = range(-kF, kF, Pars.L_Torus)
    ys = range(-kF, kF, Pars.L_Torus)
    Mesh = Vector{Vector{Float64}}()
    for x in xs
        for y in ys
            if norm([x,y]) < kF
                push!(Mesh, [x,y])
            end
        end
    end
    return Mesh
end

function createSquareMesh_Extended(kMax::Float64, Pars::ModelParameters, qmax::Int64)::Tuple{Vector{Vector{Float64}}, Vector{Float64}}
    xs = range(-kMax, kMax, Pars.L_Torus)
    ys = range(-kMax, kMax, Pars.L_Torus)
    qs = collect(range(0.0, qmax*kMax, Int((Pars.L_Torus-1)/2)*qmax+1))
    Mesh = Vector{Vector{Float64}}()
    for x in xs
        for y in ys
            push!(Mesh, [x,y])
        end
    end
    return Mesh, qs
end

function get_chi0q_Vq_RPA(Pars::ModelParameters, qmax::Int64)::Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}
    """ Computes susceptibilty chi and renormalized potential Vq in RPA without form factor."""
    # Create Mesh
    # Mesh = createSquareMesh(Pars.kF, Pars)
    # qs = collect(range(0.0, qmax*Pars.kF, Int((Pars.L_Torus-1)/2)*qmax+1))
    Mesh, qs = createSquareMesh_Extended(1*Pars.kF, Pars, qmax)
    # @show Mesh
    # @show qs
    if Pars.which_interaction == "Keldysh"
        interaction_fn = Keldysh2d
    elseif Pars.which_interaction == "KeldyshSq0"
        interaction_fn = Keldysh2dSq0
    elseif Pars.which_interaction == "Coulomb"
        interaction_fn = Coulomb2d
    elseif Pars.which_interaction == "Gaussian"
        interaction_fn = Gaussian2d
    elseif Pars.which_interaction == "DeltaFn"
        interaction_fn = DeltaFn
    else
        Print("Invalid interaction passed to Pars.which_interaction")
    end

    # Calculate charge susceptibility and Coulomb potential in RPA
    chi0s = Vector{Float64}()
    VqRPAs = Vector{Float64}()
    if Pars.which_dispersion == "quadratic"
        for q in qs
            chi_q = NoninteractingSusceptibilityAnalytical([q, 0], Pars)
            push!(chi0s, chi_q)
            push!(VqRPAs, InteractionRPA(interaction_fn, chi_q, q, Pars))
        end
    else
        for q in qs
            chi_q = NoninteractingSusceptibilityNumerical([q, 0], Mesh, Pars)
            push!(chi0s, chi_q)
            push!(VqRPAs, InteractionRPA(interaction_fn, chi_q, q, Pars))
        end
    end
    return (qs, chi0s, VqRPAs)
end

function get_chi0q_Vq_FormFactor(Pars::ModelParameters, qmax::Int64)::Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}
    """ Computes susceptibilty chi and renormalized potential Vq in RPA with susceptibility with form factor."""
    # Create Mesh
    Mesh, qs = createSquareMesh_Extended(1*Pars.kF, Pars, qmax)

    if Pars.which_interaction == "Keldysh"
        interaction_fn = Keldysh2d
    elseif Pars.which_interaction == "Coulomb"
        interaction_fn = Coulomb2d
    else
        Print("Invalid interaction passed to Pars.which_interaction")
    end

    # Calculate charge susceptibility and Coulomb potential in RPA
    chi0s = Vector{Float64}()
    VqRPAs = Vector{Float64}()
    for q in qs
        chi_q = NoninteractingSusceptibilityFormFactor([q, 0], Mesh, Pars)
        push!(chi0s, chi_q)
        push!(VqRPAs, InteractionRPA(interaction_fn, chi_q, q, Pars))
        # push!(VqRPAs, InteractionRPAFormFactor(interaction_fn, [q, 0.0], [0.0, 0.0], chi_q, Pars))
    end
    return (qs, chi0s, VqRPAs)
end

# ----------------------------------------------------------------------------------
# --------------- Helper functions for self-consistency equations ------------------
# ----------------------------------------------------------------------------------
function g_SmallDelta(k::Float64, kbT::Float64, Pars::ModelParameters)::Float64
    """
    Temperature dependence in the self-consistency equation for the pairing potential around Tc.
    """
    Ek = abs(Dispersion(k, Pars) - Dispersion(Pars.kF, Pars))
    if isapprox(Ek,0.0)
        return 1/2/kbT
    else
        return tanh(Ek/2/kbT)/Ek
    end
end

function g_ZeroTemperature(k::Float64, gapk::Float64, Pars::ModelParameters)::Float64
    """
    Non-linear factor in the self-consistency equation for the pairing potential.
    """
    xi_k = abs(Dispersion(k, Pars) - Dispersion(Pars.kF, Pars))
    return 1/sqrt( (xi_k)^2 + gapk^2 )
end

# ----------------------------------------------------------------------------------
# ------------------------- Compute gas parameter r_s ------------------------------
# ----------------------------------------------------------------------------------
function get_GasParameters(ParsList::Vector{ModelParameters}, qs::Vector{Float64}, check_Hankel::Bool)::Vector{Vector{Float64}}
    """ 
    Computes gas parameter r_s from ratio of (unscreened) total interaction energy 
    at mean interparticle distance over total kinetic energy. 
    """
    GasParameters = Vector{Vector{Float64}}()
    # For mean interaction energy density, get real space Hankel transform of Keldysh potential
    HankelQDHT = QDHT(qs[end], length(qs))
    rhos = HankelQDHT.k/2/π
    # rhos = HankelQDHT.k
    Vq_RK = Keldysh2d.(qs, Ref(ParsList[1]))
    fr_RK = HankelQDHT * Vq_RK[1:end]

    if check_Hankel
        pVq = plot(rhos, smooth_vector(fr_RK,2), label=L"Hankel $V_{RK}(r)$")
        r_1 = 1/sqrt(π * ParsList[1].n) 
        plot!(pVq, rhos, [1/ParsList[1].epsilon0/ParsList[1].eps2d/(rho) for rho in rhos], 
        label="Coulomb direct", ylims=[-10, 100], xlims=[0, 50], xlabel=L"r [nm]", ylabel=L"$V_{Coulomb}$ [meV]")
        vline!(pVq, [r_1], legend=nothing)
        display(pVq)
    end

    for Pars in ParsList
        # Get kinetic energy density
        if Pars.which_dispersion == "quadratic"
            KineticEnergyDensity = Pars.EF*Pars.n/2
        else
            if length(Pars.kFs) == 1
                ks = collect(range(0.0, Pars.kF, 1000))
            elseif length(Pars.kFs) == 2
                ks = collect(range(Pars.kFs[2], Pars.kFs[1], 1000))
            else
                ks = collect(range(0.0, Pars.kF, 1000))
            end
            dk = abs(ks[2]-ks[1])
            ks = ks[2:end].-dk/2
            Eks = [Dispersion(k, Pars) - Pars.D for k in ks]
            BandBottom = minimum(Eks)
            KineticEnergyDensity = 0.0
            for cntk in eachindex(Eks)
                if Eks[cntk] <= Pars.EF
                    KineticEnergyDensity = KineticEnergyDensity + (Eks[cntk] - BandBottom)*ks[cntk]*2π*dk
                end
            end
            KineticEnergyDensity = KineticEnergyDensity/(2π)^2
        end

        # Get interaction energy density
        r_1 = 1/sqrt(π * Pars.n) # [nm]
        InteractionEnergyDensity_Keldysh = LinearInterpolation(rhos, smooth_vector(fr_RK,2))(r_1)*Pars.n
        r_s = InteractionEnergyDensity_Keldysh / KineticEnergyDensity
        # @show KineticEnergyDensity, InteractionEnergyDensity_Keldysh, r_s, r_1
        push!(GasParameters, [KineticEnergyDensity, InteractionEnergyDensity_Keldysh, r_s])
    end
    return GasParameters
end

# -------------------------------------------------------------------------
# ------------------------------- UTILITIES -------------------------------
# -------------------------------------------------------------------------
function InverseHankelTransform(qs::Vector{Float64}, Fs::Vector{Float64}, rs::Vector{Float64})::Vector{Float64}
    """ 
    Computes the inverse Hankel transform for a function F(q) defined over q,
    where values for the vector qs are given in Fs. 
    The inverse Hankel transform HT(F(q))(r) is evaluated for all rs. """
    dq = qs[2] - qs[1]
    HankelF = (dq/2π) * [sum(qs .* besselj0.(qs * ri) .* Fs) for ri in rs]
    return HankelF
end

function get_density_S(Pars::ModelParameters, ks::Vector{Float64}, eta_k::Vector{Float64})::Float64
    """ Density in the superconducting state accounting for shifts due to the pairing potential. """
    Mesh = createSquareMesh(Pars.kF*2, Pars)
    # xi_ks = Vector{Float64}()
    itp_eta = LinearInterpolation(ks, eta_k)
    density_S = 0.0
    for k in Mesh
        xi_k = Dispersion(k, Pars) - Pars.D - Pars.EF
        E_k = sqrt(xi_k^2 + itp_eta(norm(k))^2)
        density_S = density_S + (1 - xi_k/E_k)
    end
    dk = norm(Mesh[2] .- Mesh[1])/2π
    density_S = density_S*dk^2/2
    return density_S
end

function find_largest_zero_crossing(x::Vector{Float64}, y::Vector{Float64})
    largest_x = nothing
    n = length(x)
    
    for i in 1:n-1
        if y[i] * y[i+1] < 0  # Check for sign change
            # Linear interpolation to find zero crossing
            slope = (y[i+1] - y[i]) / (x[i+1] - x[i])
            zero_crossing_x = x[i] - y[i] / slope
            
            if largest_x === nothing || zero_crossing_x > largest_x
                largest_x = zero_crossing_x
            end
        end
    end
    return largest_x
end

function find_sign_changes(xs::Vector{Float64}, ys::Vector{Float64})
    sign_change_xs = Vector{Float64}()  # Initialize an empty array to store x-values where sign changes occur
    
    for i in 1:length(ys) - 1
        if ys[i] * ys[i + 1] < 0  # Check for sign change
            # Perform linear interpolation to find the exact x where y crosses zero
            slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
            zero_crossing_x = xs[i] - ys[i] / slope
            
            push!(sign_change_xs, zero_crossing_x)  # Add the x-value to the list
        end
    end
    
    return sign_change_xs
end

function sum_xs_below_threshold(xs::Vector{Float64}, ys::Vector{Float64}, threshold::Float64)
    total_sum = 0.0  # Initialize the sum
    
    # Iterate over the xs and ys
    for (x, y) in zip(xs, ys)
        if y < threshold
            total_sum += x  # Add the corresponding x to the sum
        end
    end
    
    return total_sum
end

function weighted_binning(xs::Vector{Float64}, ys::Vector{Float64}, bin_edges::Vector{Float64})
    # Initialize bin weights
    bin_counts = zeros(Float64, length(bin_edges) - 1)
    
    # Iterate through xs and ys
    for (x, y) in zip(xs, ys)
        # Find the appropriate bin for x
        for i in eachindex(bin_counts)
            if bin_edges[i] <= x < bin_edges[i + 1]
                bin_counts[i] += y
                break
            end
        end
    end
    
    return bin_counts
end

function smooth_vector(v::Vector{T}, N::Int) where T
    len = length(v)
    smoothed = similar(v)  # Create a vector to store the smoothed result
    
    for i in 1:len
        # Define the range of neighbors to consider, ensuring it doesn't go out of bounds
        left = max(1, i - N)
        right = min(len, i + N)
        
        # Compute the mean of the neighbors (including the current point)
        smoothed[i] = sum(v[left:right])/(2*N+1)
    end
    
    return smoothed
end

# -------------------------------------------------------------------------
# ------ MAIN FUNCTIONS THAT COMPUTE INTERACTION AND SELF-CONSISTENCY -----
# -------------------------------------------------------------------------
function main_Susceptibility(Pars::ModelParameters, get_radial_decomposition::Bool, do_save::Bool, qmax::Int64)
    # kF, m, D, kD_relk0, rKeldysh, eps2d, d_gates, α, L
    n = 5e11*1e-14 # 1/nm^2
    # kF = sqrt(4π*n)
    # kF = 0.8
    print("kF = " * string(Pars.kF) * " [1/nm] \n")
    # Pars = ModelParameters(kF, m, D, 6e-3, which_dispersion, rK, 1.0, 40.0, 0.0, 101)
    EF = Dispersion([Pars.kF, 0], Pars)
    print("E_F = " * string(EF) * " [meV] \n")
    print("DOS = " * string(DOS(EF, Pars)) * " [meV^-1 nm^-2] \n")

    if Pars.which_interaction == "Keldysh"
        interaction_fn = Keldysh2d
    elseif Pars.which_interaction == "Coulomb"
        interaction_fn = Coulomb2d
    else
        Print("Invalid interaction passed to Pars.which_interaction")
    end

    if get_radial_decomposition && qmax < 32
        qmax = 32
    end
    # qmax = 32
    # Get susceptibility chi0(q) and interaction V(q) in RPA
    (qs, chi0s, VqRPAs) = get_chi0q_Vq_RPA(Pars, qmax)

    # Angular decomposition of interaction potential for BCS limit
    # Interpolate V(q)
    interp_V = linear_interpolation(qs, VqRPAs)
    function InterpolatedV(q::Float64)::Float64
        return interp_V(q)
    end 
    # Get angular decomposition
    ms = collect(-11:2:11)
    # AngularCoefficients = FermiSurfaceAngularDecomposition(InterpolatedV, ms, Pars.kF, 361)

    # Plot susceptibility, renormalized interaction, and angular decomposition
    pchi = plot(qs./(2*Pars.kF), chi0s, xlabel="q/2k_F", ylabel="χ^0(q)",legend=false) #, title = "kF="*string(round(Pars.kF,digits=4))
    hline!(pchi, [-DOS(EF, Pars)])
    pVq = plot(qs./(2*Pars.kF), VqRPAs, xlabel="q/2k_F", ylabel="V(q)", label="RPA", xlims=[0,10], ylims=[0,maximum(VqRPAs)*2])
    Vq_RK = interaction_fn.(qs, Ref(Pars))
    plot!(pVq, qs./(2*Pars.kF), Vq_RK, label="bare interaction", linestyle=:dash, color=:gray)
    # pVms = plot(ms, AngularCoefficients, xlabel="m", ylabel="V_m", legend=false)
    # ploti = plot(pchi, pVq, pVms, layout = (3, 1))  # 3 rows, 1 column
    # display(ploti)

    # KineticEnergyDensity, InteractionEnergyDensity_RPA, InteractionEnergyDensity_Keldysh = get_GasParameters(Pars, qs, VqRPAs)
    # GasParameters = get_GasParameters([Pars], qs)
    # print("Gas parameters: \n")
    # print(GasParameters)

    if do_save
        save_struct_if_not_exists(Pars, Pars.stamp, "Pars.txt")
        save_data_to_dat(qs, Pars.stamp, "RPA_qs.dat")
        save_data_to_dat(chi0s, Pars.stamp, "RPA_chi0s.dat")
        save_data_to_dat(VqRPAs, Pars.stamp, "RPA_VqRPAs.dat")
        save_data_to_dat(Vq_RK, Pars.stamp, "RPA_VqRKs.dat")
        save_data_to_dat([Pars.n, Pars.kF, Pars.rKeldysh], Pars.stamp, "RPA_n_maxkF_rK.dat")
    end

    # Plot Hankel transformed V(rho)
    rhos = collect(range(0.0, 1*2π/Pars.kF, 800)) # use rationals of Fermi wavelength for smooth plot
    fr = InverseHankelTransform(qs, VqRPAs, rhos)
    fr_RK = InverseHankelTransform(qs, Vq_RK, rhos)
    plotrho = plot(rhos, fr, xlabel="ρ [nm]", ylabel="V(ρ)", label=nothing, xlims=[0,10], ylims=[minimum(fr),-1*minimum(fr)])

    plot!(plotrho, rhos, fr_RK, label=nothing, linestyle=:dash, color=:gray)

    plotFriedel = plot(pchi, pVq, plotrho, layout = (3, 1))  # 3 rows, 1 column
    display(plotFriedel) # , ylims=[-0.5,2.2]
    if do_save
        save_data_to_dat(rhos, Pars.stamp, "RPA_Hankel_rs.dat")
        save_data_to_dat(fr, Pars.stamp, "RPA_Hankel_VrRPAs.dat")
        save_data_to_dat(fr_RK, Pars.stamp, "RPA_Hankel_VrRKs.dat")
    end
    


    if get_radial_decomposition
        # Calculate angular decomposition with radial dependence
        k1Mesh = range(0, 10.0*Pars.kF, 101)
        k2Mesh = range(0, 10.0*Pars.kF, 101)
        ms = [-1,1,3]
        plots = []
        AngularCoefficientsRadial = [[AngularDecompositionRadial(k1, k2, InterpolatedV, mi, 361) for k1 in k1Mesh, k2 in k2Mesh] for mi in ms]
        # @show AngularCoefficientsRadial
        max_val = maximum([maximum(abs.(M)) for M in AngularCoefficientsRadial])
        clims = (-max_val, max_val)
        for (cnt, data) in enumerate(AngularCoefficientsRadial)
            # @show data
            if cnt == length(ms)
                push!(plots, heatmap(k1Mesh./Pars.kF, k2Mesh./Pars.kF, data, color=:seismic, clims=clims, 
                colorbar = true, colorbar_title=L"$V_l(k_1, k_2)$ [meV nm$^2$]", title = "l = "*string(ms[cnt]), grid=:on, xlabel=L"k_1/k_F", ylabel=L"k_2/k_F"))
            else
                push!(plots, heatmap(k1Mesh./Pars.kF, k2Mesh./Pars.kF, data, color=:seismic, clims=clims, 
                colorbar = false, colorbar_title=L"$V_l(k_1, k_2)$ [meV nm$^2$]", title = "l = "*string(ms[cnt]), grid=:on, xlabel=L"k_1/k_F", ylabel=L"k_2/k_F"))
            end
            vline!(1.0:1.0, c=:black, legend = false)
            hline!(1.0:1.0, c=:black, legend = false)
            # @show AngularCoefficientsRadial
        end
        combined_plot = plot(plots..., layout = (1, length(ms)), size=(800, 250), grid=:on,
        left_margin=5mm, bottom_margin=5mm)
        # cf_radial = heatmap(AngularCoefficientsRadial, c=:seismic )
        display(combined_plot)

        if do_save
            save_data_to_dat(collect(k1Mesh), Pars.stamp, "RPA_Vm_kk_k1s.dat")
            save_data_to_dat(collect(k2Mesh), Pars.stamp, "RPA_Vm_kk_k2s.dat")
            for (cntm, m) in enumerate(ms)
                save_data_to_dat(AngularCoefficientsRadial[cntm], Pars.stamp, "RPA_Vm_kk_m"*string(m)*"_data.dat")
            end
        end
    end
end

function main_Susceptibility_FormFactor(Pars::ModelParameters, get_radial_decomposition::Bool, do_save::Bool, qmax::Int64)
    # Get q mesh and 
    (qs, chi0s, VqRPAs) = get_chi0q_Vq_FormFactor(Pars, qmax)

    if Pars.which_interaction == "Keldysh"
        interaction_fn = Keldysh2d
    elseif Pars.which_interaction == "Coulomb"
        interaction_fn = Coulomb2d
    else
        Print("Invalid interaction passed to Pars.which_interaction")
    end

    # Interpolate V(q)
    interp_V = linear_interpolation(qs, VqRPAs)
    function InterpolatedV(q::Float64)::Float64
        return interp_V(q)
    end 

    EF = Dispersion([Pars.kF, 0], Pars)
    # Plot susceptibility, renormalized interaction
    pchi = plot(qs./(2*Pars.kF), chi0s, xlabel="q/2k_F", ylabel="χ^0(q)", title = "kF="*string(Pars.kF),legend=false)
    hline!(pchi, [-DOS(EF, Pars)])
    pVq = plot(qs./(2*Pars.kF), VqRPAs, xlabel="q/2k_F", ylabel="V(q)", legend=false)
    ploti = plot(pchi, pVq, layout = (2, 1))  # 3 rows, 1 column
    display(ploti)

    # # Plot V_k(k, k_F)
    ms = [-1, 1]
    k1Mesh = range(0, 6.0*Pars.kF, 401)
    pVms = plot(k1Mesh./(2*Pars.kF), interaction_fn.(k1Mesh, Ref(Pars)), 
        xlabel="q/2k_F", ylabel="V_m(k, k_F)", title = "kF="*string(Pars.kF),label="Keldysh",ylims=[-800,800])
    for m in ms
        # @show m
        VmComplex = AngularDecompositionRadial_FormFactorComplex.(Ref(Pars.kF), k1Mesh, Ref(InterpolatedV), Ref(m), Ref(121), Ref(Pars))
        plot!(pVms, k1Mesh./(2*Pars.kF), real.(VmComplex), 
        xlabel="q/2k_F", ylabel="V_m(k, k_F)", title = "kF="*string(Pars.kF),label="Re V_"*string(m))
        plot!(pVms, k1Mesh./(2*Pars.kF), imag.(VmComplex)*100, 
        xlabel="q/2k_F", ylabel="V_m(k, k_F)", title = "kF="*string(Pars.kF),label="Im V_"*string(m))
    end
    display(pVms)

    if get_radial_decomposition
        # Calculate angular decomposition with radial dependence
        k1Mesh = range(0, 8.0*Pars.kF, 101)
        k2Mesh = range(0, 8.0*Pars.kF, 101)
        ms = [-1,1]
        plots = []
        AngularCoefficientsRadial = [[AngularDecompositionRadial_FormFactor(k1, k2, InterpolatedV, mi, 121, Pars) for k1 in k1Mesh, k2 in k2Mesh] for mi in ms]
        # @show AngularCoefficientsRadial
        max_val = maximum([maximum(abs.(M)) for M in AngularCoefficientsRadial])
        clims = (-max_val, max_val)
        for (cnt, data) in enumerate(AngularCoefficientsRadial)
            # @show data
            push!(plots, heatmap(k1Mesh./Pars.kF, k2Mesh./Pars.kF, data, color=:seismic, clims=clims, 
            colorbar = false, title = "m = "*string(ms[cnt]), grid=:on, xlabel=L"k_1/k_F", ylabel=L"k_2/k_F"))
            vline!(1.0:1.0, c=:black, legend = false)
            hline!(1.0:1.0, c=:black, legend = false)
            # @show AngularCoefficientsRadial
        end
        combined_plot = plot(plots..., layout = (1, length(ms)), colorbar = true, c=:seismic, size=(1000, 400), grid=:on)
        # cf_radial = heatmap(AngularCoefficientsRadial, c=:seismic )
        display(combined_plot)
    end
end

function main_direct_integration_iterative(
    ParsList::Vector{ModelParameters}, l::Int64, kbTList::Vector{Float64},
    qmaxList::Vector{Int64}, res_ks_integration::Int64, res_theta_integration::Int64, 
    iterations_radial_Tc::Vector{Int64}, iterations_radial_gap::Vector{Int64}, 
    ansatz::String, reference_ansaetze::Vector{Vector{Float64}}, convergence_tolerance::Float64, include_FormFactor::Bool)
    """
    This function computes Tc and the zero-temperature gap by direct integration by iterating \eta_m(k).
    For the raadial profile at Tc, various Ansaetze are available:
    VmkFk: angular component of the interaction V_m(k_F,k)
    previous: Uses VmkFk for the first calculation, and then uses \eta_m(k) from the previous parameter set for the next
    For the gap at zero temperature, the functional form of the gap at Tc is used as Ansatz, which works efficiently.
    """
    print_warnings = false
    Tcs_k_iter = Vector{Vector{Float64}}()
    Tcs_T0Gap = Vector{Float64}()
    eta_m_k_iter = Vector{Vector{Vector{Float64}}}()
    norm_eta_m_k_iter = Vector{Vector{Float64}}()
    Gaps_k_iter = Vector{Vector{Float64}}()
    f_m_k_iter = Vector{Vector{Vector{Float64}}}()
    norm_f_m_k_iter = Vector{Vector{Float64}}()
    ks_integration_k = Vector{Vector{Float64}}()
    VmkFk = Vector{Vector{Float64}}()
    Convergence_TcGap_k = Vector{Vector{Int64}}()
    
    # GasParameters = Vector{Tuple{Float64, Float64, Float64}}()
    # dT = kbTList[2] - kbTList[1]

    for (cntP, Pars) in enumerate(ParsList)
        LastWarning_Tc = 0
        LastWarning_Gap = 0
        qmax = qmaxList[cntP]
        # Get RPA interaction V(q), interpolate and wrap as a function
        if include_FormFactor
            (qs, chi0s, VqRPAs) = get_chi0q_Vq_FormFactor(Pars, qmax)
        else
            (qs, chi0s, VqRPAs) = get_chi0q_Vq_RPA(Pars, qmax)
        end
        interp_V = linear_interpolation(qs, VqRPAs)
        function InterpolatedV(q::Float64)::Float64
            return interp_V(q)
        end 

        # get integration range
        # ks_integration = round.(LinRange(0.0, Pars.kF*(qmax/2)*0.9999,res_ks_integration), digits=14)
        ks_integration = qs[1:res_ks_integration:Int(ceil(length(qs)/2))-1]
        push!(ks_integration_k, ks_integration)
        ks_range = ks_integration[end]-ks_integration[1]

        # # INITIALIZATION eta_m^(0)(k). 
        # eta_m is normalized so that 1 = \int_0^\infty dk eta_m(k) = sum(eta_m.(ks_integration))/length(ks_integration)*ks_range
        
        # functional form of gap for Tc calculation
        eta_m_iter = Vector{Vector{Float64}}()
        norm_eta_m_iter = Vector{Float64}()

        # functional form of gap for zero-temperature gap calculation
        f_m_iter = Vector{Vector{Float64}}()
        norm_f_m_iter = Vector{Float64}()

        if ansatz == "VmkFk"
            if include_FormFactor
                push!(f_m_iter, AngularDecompositionRadial_FormFactor.(Ref(Pars.kF), ks_integration, Ref(InterpolatedV), Ref(l), Ref(res_theta_integration), Ref(Pars)))
            else
                push!(f_m_iter, AngularDecompositionRadial.(Ref(Pars.kF), ks_integration, Ref(InterpolatedV), Ref(l), Ref(res_theta_integration)))
            end
        elseif ansatz == "reference"
            push!(f_m_iter, reference_ansaetze[cntP])
        elseif ansatz == "previous"
            if cntP == 1
                if include_FormFactor
                    push!(f_m_iter, AngularDecompositionRadial_FormFactor.(Ref(Pars.kF), ks_integration, Ref(InterpolatedV), Ref(l), Ref(res_theta_integration), Ref(Pars)))
                else
                    push!(f_m_iter, AngularDecompositionRadial.(Ref(Pars.kF), ks_integration, Ref(InterpolatedV), Ref(l), Ref(res_theta_integration)))
                end
            else
                push!(f_m_iter, f_m_iter[end][end]/norm_f_m_iter[end][end])
            end
        else
            print("WARNING: Invalid value passed to ansatz, aborting...")
            # exit()
        end
        push!(norm_f_m_iter, sum(f_m_iter[1])/length(ks_integration)*ks_range)

        # Get angular components of V as a function of magnitude of both scattering momenta
        # Vmkk = [AngularDecompositionRadial.(Ref(ki), ks_integration, Ref(InterpolatedV), Ref(l), Ref(res_theta_integration)) for ki in ks_integration]
        # Using parallelization, equivalent to the above
        Vmkk = Vector{Vector{Float64}}(undef, length(ks_integration))
        for i in 1:length(ks_integration)
            Vmkk[i] = Vector{Float64}(undef, length(ks_integration))
        end
        if include_FormFactor
            @threads for cnti in eachindex(ks_integration)
                for cntj in cnti:length(ks_integration)
                    value = AngularDecompositionRadial_FormFactor.(ks_integration[cnti], ks_integration[cntj], InterpolatedV, l, res_theta_integration, Ref(Pars))
                    Vmkk[cntj][cnti] = value
                    Vmkk[cnti][cntj] = value
                end
            end
            push!(VmkFk, AngularDecompositionRadial_FormFactor.(Ref(Pars.kF), ks_integration, Ref(InterpolatedV), Ref(l), Ref(res_theta_integration), Ref(Pars)))
        else
            @threads for cnti in eachindex(ks_integration)
                for cntj in cnti:length(ks_integration)
                    value = AngularDecompositionRadial.(ks_integration[cnti], ks_integration[cntj], InterpolatedV, l, res_theta_integration)
                    Vmkk[cntj][cnti] = value
                    Vmkk[cnti][cntj] = value
                end
            end
            push!(VmkFk, AngularDecompositionRadial.(Ref(Pars.kF), ks_integration, Ref(InterpolatedV), Ref(l), Ref(res_theta_integration)))
        end

        Gaps_iter = Vector{Float64}()
        converged_gap = false
        for iter in 1:iterations_radial_gap[cntP]
            if converged_gap
                push!(Gaps_iter,Gaps_iter[end])
                if iter < iterations_radial_gap[cntP]
                    push!(f_m_iter, f_m_iter[end])
                    push!(norm_f_m_iter, norm_f_m_iter[end])
                end
                continue
            end
            # 1. Find Tc with eta_m^(iter)
            # 1.1. Get RHS as a function of gap
            gapList_i = gapList/maximum(f_m_iter[iter])
            # @show maximum(f_m_iter[iter]),norm_f_m_iter[iter]
            RHS_Gap = Vector{Float64}(undef, length(gapList))
            @threads for cntG in eachindex(gapList)
                RHS_Gap[cntG] = -1/(2*(2π)^2) / norm_f_m_iter[iter] * ks_range/length(ks_integration) *
                sum(ks_integration .*                                              # k'
                [sum(Vmk) * ks_range/length(ks_integration) for Vmk in Vmkk] .*     # \int dk V_l(k, k')
                g_ZeroTemperature.(ks_integration, gapList_i[cntG].*f_m_iter[iter], Ref(Pars)) .*               # g(k')
                f_m_iter[iter] )                         # eta_m^(iter)
            end
            # 1.2 Find gap by solving for 1 = RHS(T)
            if 1 < minimum(RHS_Gap)
                gap = gapList_i[end]
                LastWarning_Gap = copy(iter)
                if print_warnings
                    print("WARNING: gap larger than gapList[end]; at (kF, n, iter) = " * string((round(Pars.kF,digits=3), round(100*Pars.n,digits=4), iter)) * "\n")
                end
            elseif 1 > maximum(RHS_Gap)
                gap = gapList_i[1]
                LastWarning_Gap = copy(iter)
                if print_warnings
                    print("WARNING: gap smaller than gapList[1]; at (kF, n, iter) = " * string((round(Pars.kF,digits=3), round(100*Pars.n,digits=4), iter)) * "\n")
                end
            else
                gap = find_largest_zero_crossing(gapList_i, RHS_Gap .- (1.0))
            end
            # @show RHS_Gap
            push!(Gaps_iter, gap)

            # Check for convergence
            if length(Gaps_iter) >= 4
                gap_iter = f_m_iter[end] / norm_f_m_iter[end] * Gaps_iter[end]
                gap_iter_max = maximum(gap_iter)
                gaps_distance_rel = [
                    maximum(abs.(f_m_iter[end-3] / norm_f_m_iter[end-3] * Gaps_iter[end-3] .- gap_iter))/gap_iter_max,
                    maximum(abs.(f_m_iter[end-2] / norm_f_m_iter[end-2] * Gaps_iter[end-2] .- gap_iter))/gap_iter_max,   
                    maximum(abs.(f_m_iter[end-1] / norm_f_m_iter[end-1] * Gaps_iter[end-1] .- gap_iter))/gap_iter_max]
                if maximum(gaps_distance_rel) <= convergence_tolerance
                    converged_gap = true
                    print("(cntP, n)="*string((cntP,round(Pars.n*100,digits=4)))*"; gap converged at iter = "*string(iter)*", gaps_distance_rel = "*string([round(gapi,sigdigits=3) for gapi in gaps_distance_rel])*"\n")
                end
            end

            # 2. Get next iteration \eta_m^(j+1)
            if iter < iterations_radial_gap[cntP]
                new_f_m = Vector{Float64}(undef, length(ks_integration))
                @threads for cntk in eachindex(ks_integration)
                    new_f_m[cntk] = -1/(2*(2π)^2) / norm_f_m_iter[iter] * ks_range/length(ks_integration) * 
                    sum(ks_integration .*                                          # k'
                    Vmkk[cntk] .*                                                   # \int dk V_l(k, k')
                    g_ZeroTemperature.(ks_integration, Gaps_iter[iter].*f_m_iter[iter], Ref(Pars)) .*       # g(k', kbT_c^(iter))
                    f_m_iter[iter] )       # eta_m^(iter)
                end
                push!(f_m_iter, new_f_m)
                f_m_iter[end] = f_m_iter[end] / (sum(f_m_iter[end])/length(ks_integration)*ks_range)
                push!(norm_f_m_iter, sum(f_m_iter[iter])/length(ks_integration)*ks_range)
            end
        end
        push!(Gaps_k_iter, Gaps_iter)
        push!(f_m_k_iter, f_m_iter)
        push!(norm_f_m_k_iter, norm_f_m_iter)

        # # Get T_c for zero-temerpature gap
        RHS_kbT_T0Gap = Vector{Float64}(undef, length(kbTList))
        @threads for cntT in eachindex(kbTList)
            RHS_kbT_T0Gap[cntT] = -1/(2*(2π)^2) / norm_f_m_iter[end] * ks_range/length(ks_integration) * 
            sum(ks_integration .*                                              # k'
            [sum(Vmk) * ks_range/length(ks_integration) for Vmk in Vmkk] .*     # \int dk V_l(k, k')
            g_SmallDelta.(ks_integration, Ref(kbTList[cntT]), Ref(Pars)) .*               # g(k')
            f_m_iter[end] )
        end
        # 1.2 Find critical temperature by solving for 1 = RHS(T)
        if 1 < minimum(RHS_kbT_T0Gap)
            Tc_T0Gap = kbTList[end]
        elseif 1 > maximum(RHS_kbT_T0Gap)
            Tc_T0Gap = kbTList[1]
        else
            Tc_T0Gap = find_largest_zero_crossing(kbTList, RHS_kbT_T0Gap .- (1.0))
        end
        push!(Tcs_T0Gap, Tc_T0Gap)

        # # CALCULATE Tcs and functional form of gap for Tcs
        push!(eta_m_iter, f_m_iter[end] ./ norm_f_m_iter[end])
        eta_m_iter[end] = eta_m_iter[end] / (sum(eta_m_iter[end])/length(ks_integration)*ks_range)
        push!(norm_eta_m_iter, sum(eta_m_iter[end])/length(ks_integration)*ks_range)

        Tcs_iter = Vector{Float64}()
        converged_Tc = false
        for iter in 1:iterations_radial_Tc[cntP]
            if converged_Tc
                push!(Tcs_iter, Tcs_iter[end])
                if iter < iterations_radial_Tc[cntP]
                    push!(eta_m_iter, eta_m_iter[end])
                    push!(norm_eta_m_iter, norm_eta_m_iter[end])
                end
                continue
            end
            # 1. Find Tc with eta_m^(iter)
            # 1.1. Get RHS as a function of temperature
            RHS_kbT = Vector{Float64}(undef, length(kbTList))
            @threads for cntT in eachindex(kbTList)
                RHS_kbT[cntT] = -1/(2*(2π)^2) / norm_eta_m_iter[iter] * ks_range/length(ks_integration) * 
                sum(ks_integration .*                                              # k'
                [sum(Vmk) * ks_range/length(ks_integration) for Vmk in Vmkk] .*     # \int dk V_l(k, k')
                g_SmallDelta.(ks_integration, Ref(kbTList[cntT]), Ref(Pars)) .*               # g(k')
                eta_m_iter[iter] )
            end
            # RHS_kbT = -1/(2*(2π)^2) / norm_eta_m_iter[iter] * ks_range/length(ks_integration) *
            #     [sum(ks_integration .*                                              # k'
            #     [sum(Vmk) * ks_range/length(ks_integration) for Vmk in Vmkk] .*     # \int dk V_l(k, k')
            #     g_SmallDelta.(ks_integration, Ref(kbT), Ref(Pars)) .*               # g(k')
            #     eta_m_iter[iter] ) for kbT in kbTList]                              # eta_m^(iter)
            # 1.2 Find critical temperature by solving for 1 = RHS(T)
            if 1 <= minimum(RHS_kbT)
                Tc = kbTList[end]
                if print_warnings
                    print("WARNING: Tc larger than kbTList[end]; at (kF, n, iter) = " * string((round(Pars.kF,digits=3), round(Pars.n*100,digits=4), iter)) * "\n")
                end
                LastWarning_Tc = copy(iter)
            elseif 1 >= maximum(RHS_kbT)
                Tc = kbTList[1]
                if print_warnings
                    print("WARNING: Tc smaller than kbTList[1]; at (kF, n, iter) = " * string((round(Pars.kF,digits=3), round(Pars.n*100,digits=4), iter)) * "\n")
                end
                LastWarning_Tc = copy(iter)
            else
                Tc = find_largest_zero_crossing(kbTList, RHS_kbT .- (1.0))
            end
            push!(Tcs_iter, Tc)
            if length(Tcs_iter) >= 4
                Tcs_rel = [Tcs_iter[end-3]/Tcs_iter[end]-1, Tcs_iter[end-2]/Tcs_iter[end]-1, Tcs_iter[end-1]/Tcs_iter[end]-1]
                if maximum(abs.(Tcs_rel)) <= convergence_tolerance
                    converged_Tc = true
                    print("(cntP, n)="*string((cntP,round(Pars.n*100,digits=4)))*"; Tc converged at iter = "*string(iter)*", Tcs_iter[end-3:end] = "*string([round(Tc,sigdigits=3) for Tc in Tcs_iter[end-3:end]])*"\n")
                end
            end

            # 2. Get next iteration \eta_m^(j+1)
            if iter < iterations_radial_Tc[cntP]
                new_eta_m = Vector{Float64}(undef, length(ks_integration))
                @threads for cntk in 1:length(ks_integration)
                    new_eta_m[cntk] = -1/(2*(2π)^2) / norm_eta_m_iter[iter] * ks_range/length(ks_integration) * sum(ks_integration .*                                          # k'
                    Vmkk[cntk] .*                                                   # \int dk V_l(k, k')
                    g_SmallDelta.(ks_integration, Tcs_iter[iter], Ref(Pars)) .*       # g(k', kbT_c^(iter))
                    eta_m_iter[iter] )
                end
                push!(eta_m_iter, new_eta_m)
                # push!(eta_m_iter, 
                #     -1/(2*(2π)^2) / norm_eta_m_iter[iter] * ks_range/length(ks_integration) * 
                #     [sum(ks_integration .*                                          # k'
                #     Vmkk[cntk] .*                                                   # \int dk V_l(k, k')
                #     g_SmallDelta.(ks_integration, Tcs_iter[iter], Ref(Pars)) .*       # g(k', kbT_c^(iter))
                #     eta_m_iter[iter] ) for cntk in 1:length(ks_integration)] )        # eta_m^(iter)
                push!(norm_eta_m_iter, sum(eta_m_iter[iter])/length(ks_integration)*ks_range)
            end
        end
        push!(Tcs_k_iter, Tcs_iter)
        push!(eta_m_k_iter, eta_m_iter)
        push!(norm_eta_m_k_iter, norm_eta_m_iter)


        # # Check if k range is large enough
        if abs(f_m_iter[end][end])/maximum(abs.(f_m_iter[end])) >= 0.05
            print("(cntP, n)="*string((cntP,round(Pars.n*100,digits=4)))*"; WARNING: qmax likely too small; eta(k_max)/max(eta)="
            *string(round(abs(f_m_iter[end][end])/maximum(abs.(f_m_iter[end])) ,digits=4))*"\n")
        end

        # Verify convergence
        Convergence_TcGap = [1, 1]
        if LastWarning_Tc >= iterations_radial_Tc[cntP] - 5 || !(converged_Tc)
            Convergence_TcGap[1] = 0
            print("(cntP, n)="*string((cntP,round(Pars.n*100,digits=4)))*"; NO CONVERGENCE for T_c \n")
            if length(Tcs_iter) > 10
                @show Tcs_iter[end-10:end]
            else
                @show Tcs_iter
            end
            # @show Tc_T0Gap
        end
        if LastWarning_Gap >= iterations_radial_gap[cntP] - 1 || !(converged_gap)
            Convergence_TcGap[2] = 0
            print("(cntP, n)="*string((cntP,round(Pars.n*100,digits=4)))*"; NO CONVERGENCE for gap \n")
        end
        push!(Convergence_TcGap_k, Convergence_TcGap)
        # @show Convergence_TcGap
    end

    return ks_integration_k, Tcs_k_iter, eta_m_k_iter, norm_eta_m_k_iter, Gaps_k_iter, f_m_k_iter, norm_f_m_k_iter, VmkFk, Convergence_TcGap_k, Tcs_T0Gap
end

# -------------------------------------------------------------------------
# --------------------------------- PLOTS ---------------------------------
# -------------------------------------------------------------------------
function plot_Dispersion(Pars::ModelParameters, n::Float64, k_edge::Float64, 
    E_bin_edges::Vector{Float64}, EFs::Vector{Float64})
    # Find kFs
    nks_DOS = 10000
    dk_DOS = k_edge/nks_DOS
    ks_DOS_xi = collect(range(dk_DOS/2, k_edge+dk_DOS/2, nks_DOS+1))
    xi_k_centers = Dispersion.(ks_DOS_xi, Ref(Pars)) .- Pars.D

    # find EF
    mu, kFs, BandBottom = find_EF(Pars, n, [1.0, 6.0, 36.0], k_edge)
    # kFs = find_sign_changes(ks_DOS_xi, xi_k_centers .- mu)

    # plot dispersion
    ks_BS = range(-k_edge, k_edge, 201)
    xi_k = Dispersion.(abs.(ks_BS), Ref(Pars)) .- Pars.D
    plot_BS = plot(ks_BS, xi_k, 
        label = L"$\xi_k$", xlabel = L"$k$ [1/nm]", ylabel = L"$E - D$ [meV]", color=:black, 
        ylim = [minimum(xi_k), 5.0])
    vline!(plot_BS, [-0.08/Pars.a, 0.08/Pars.a], color=:black, linestyle=:dash, label=nothing)
    # vline!(plot_BS, [0.35], color=:black, linestyle=:dash, label=nothing)
    # hline!(plot_BS, [-10.6], color=:black, linestyle=:dash, label=nothing)
    vline!(plot_BS, vcat(-kFs, kFs), color=:blue, linestyle=:dash, label=nothing)
    hline!(plot_BS, [mu], color=:blue, linestyle=:dash, label=nothing, 
        title=L"m="*string(round(Pars.m,sigdigits=3))
        *L"; m_2="*string(round(Pars.m2,sigdigits=3))
        *L"; D="*string(round(Pars.D,sigdigits=3))
        *L"; $k_D=$"*string(round(Pars.k_D,sigdigits=3))
        *L"; $\alpha_D=$"*string(round(Pars.alphaD,sigdigits=3)),
        titlefont = font(12,"Computer Modern"))

    # Find DOS
    # E_bin_edges = collect(range(0.0, 0.1, 401))
    dE_DOS = E_bin_edges[2] - E_bin_edges[1]
    E_bin_centers = E_bin_edges[2:end] .- dE_DOS/2
    # @show dk_DOS, (ks_DOS_xi[2] - ks_DOS_xi[1])
    Ns_k = [2π*ki.*(dk_DOS) for ki in ks_DOS_xi]
    # NEs_xi = [π*((ki+dk_DOS/2)^2 - (ki-dk_DOS/2)^2) for ki in ks_DOS_xi]
    Ns_xi = weighted_binning(xi_k_centers, Ns_k, E_bin_edges)/(2π)^2
    # DOS_xi = (Ns_xi[2:end]-Ns_xi[1:end-1])/dE_DOS
    # E_DOS = E_bin_centers[2:end] .- dE_DOS/2
    DOS_xi = Ns_xi/dE_DOS
    pDOS = plot(E_bin_centers, DOS_xi / Pars.nu_bare_electron , color=:gray, linestyle=:dash,
        ylim = [0.0, DOS_EF(Pars)/ Pars.nu_bare_electron*3.0], label=L"\nu_0", xlabel=L"$E - D$ [meV]", ylabel=L"$\nu / \nu_{\rm e}$")
    hline!(pDOS, [DOS_EF(Pars)/ Pars.nu_bare_electron], lc=:red, linestyle=:dash, label=L"\nu_0(E_F)")
    # Find density as a function of EF
    # EFs = collect(range(1e-4, 0.1, 101))
    ns_num = Vector{Float64}()
    ns_analytical = Vector{Float64}()
    for EF in EFs
        kFs_i = sort(find_sign_changes(ks_DOS_xi, xi_k_centers .- EF), rev=true)
        push!(ns_num, sum_xs_below_threshold(Ns_xi, xi_k_centers, EF) * dE_DOS /(4*π^2))
        if length(kFs_i) == 0
            # print("Warning: No Fermi surface detected.")
            push!(ns_analytical, 0.0)
        elseif length(kFs_i) == 1
            push!(ns_analytical, (kFs_i[1]^2)/(4π))
        elseif length(kFs_i) == 2
            push!(ns_analytical, abs(kFs_i[1]^2 - kFs_i[2]^2)/(4π))
        else
            print("Warning: more than two Fermi surfaces detected.")
            push!(ns_analytical, 0.0)
        end
    end
    pn = plot(EFs, ns_num .* 100, label = nothing, xlabel=L"$\mu - D$ [meV]", ylabel=L"$n$ [$10^{12}$ cm$^{-2}$]", color=:black) # L"$n$ numerical"
    plot!(pn, EFs, ns_analytical .* 100, label = nothing, color=:black, linestyle=:dash) # L"$n$ analytical"
    hline!(pn, [n*100], color=:blue, linestyle=:dash, label=nothing)
    vline!(pn, [mu], color=:blue, linestyle=:dash, label=nothing)

    combined_plot = plot(plot_BS, pDOS, pn, layout = grid(3, 1, heights=[0.4, 0.3, 0.3]), figsize = (600, 800))
    display(combined_plot)
end

function plot_Tcs(ParsList::Vector{ModelParameters}, x_axis::Vector{Float64}, x_label::Union{LaTeXString,String}, y_unit::Union{LaTeXString,String},
    Tcs_k_iter::Vector{Vector{Float64}}, 
    ylims::Vector{Float64}, converged_Tcs::Vector{Int64}, plot_iters::Bool)
    iterations_radial_min = minimum([length(Tcs) for Tcs in Tcs_k_iter])
    if y_unit == "[K]"
        conversion_factor = 11.60452
    else
        conversion_factor = 1
    end
    pTcn = plot(x_axis, [Tcs[1]*conversion_factor for Tcs in Tcs_k_iter], 
        xlabel = x_label, ylabel = "T_c "*y_unit, label="ansatz", linewidth=1) # "iter="*string(1)
    if plot_iters
        gradient = cgrad([:gold, :red, :purple, :black])
        for iter in 2:iterations_radial_min-1
            color = gradient.colors[round(Int, (iter-1) * (length(gradient.colors) - 1) / (iterations_radial_min - 1)) + 1]
            plot!(pTcn, x_axis, [Tcs[iter]*conversion_factor for Tcs in Tcs_k_iter], 
            xlabel = x_label, ylabel = "T_c "*y_unit, label=nothing, linewidth=0.5, 
            color=color) # "iter="*string(iter)
        end
    end
    if iterations_radial_min >= 2
        plot!(pTcn, x_axis, [Tcs_k_iter[cntP][end]*conversion_factor*converged_Tcs[cntP] - 1000*(1-converged_Tcs[cntP]) for cntP in eachindex(Tcs_k_iter)],
            ylabel = "T_c "*y_unit, label=string(iterations_radial_min)*" iters", linewidth=2, color=:black, 
            ylims=ylims, 
            foreground_color_legend = nothing, background_color_legend = nothing) # "iter="*string(iterations_radial_min)
    end
    return pTcn
end

function plot_gaps(ParsList::Vector{ModelParameters}, x_axis::Vector{Float64}, x_label::Union{LaTeXString,String}, y_unit::Union{LaTeXString,String},
    ks_integration_k::Vector{Vector{Float64}}, Gaps_k_iter::Vector{Vector{Float64}}, 
    f_m_k_iter::Vector{Vector{Vector{Float64}}}, norm_f_m_k_iter::Vector{Vector{Float64}}, 
    ylims::Vector{Float64}, converged_gaps::Vector{Int64})

    # iterations_radial = length(norm_eta_m_k_iter[1])
    iterations_radial_min = minimum([length(norm_f) for norm_f in norm_f_m_k_iter])
    max_FSs = maximum([length(Pars.kFs) for Pars in ParsList])
    Gaps_Pars_kFs = Vector{Vector{Float64}}()
    Gaps_Pars_max = Vector{Float64}()
    for cntP in eachindex(ParsList)
        Gaps_kFs = Vector{Float64}()
        for cntFS in range(1,max_FSs)
            if cntFS <= length(ParsList[cntP].kFs)
                ind_kF = findmin(abs.(ks_integration_k[cntP] .- ParsList[cntP].kFs[cntFS]))[2]
                # Gap_at_kF = [abs(Gaps_k_iter[cntP][iter]*f_m_k_iter[cntP][iter][ind_kF]/norm_f_m_k_iter[cntP][iter]) for iter in 1:iterations_radial_min]
                Gap_at_kF = abs(Gaps_k_iter[cntP][end]*f_m_k_iter[cntP][end][ind_kF]/norm_f_m_k_iter[cntP][end])
                push!(Gaps_kFs, Gap_at_kF)
            else
                push!(Gaps_kFs, 0.0)
            end
        end
        push!(Gaps_Pars_kFs, Gaps_kFs)
        push!(Gaps_Pars_max, maximum(abs.(Gaps_k_iter[cntP][end]*f_m_k_iter[cntP][end]/norm_f_m_k_iter[cntP][end])))
    end
    pGapn = plot(x_axis, [Gaps_Pars_kFs[cntP][1]*converged_gaps[cntP] - 1000*(1-converged_gaps[cntP]) for cntP in eachindex(Gaps_Pars_kFs)], 
            xlabel = x_label, ylabel = L"||$\eta_m$|| "*y_unit, label=L"$\eta_m(k_{F,1})$", linewidth=2, ylims=(0.0, maximum(Gaps_Pars_max)*1.5)) # "iter="*string(1)
    for cntkF in range(2,max_FSs)
        plot!(pGapn, x_axis, [Gaps_Pars_kFs[cntP][cntkF]*converged_gaps[cntP] - 1000*(1-converged_gaps[cntP])  for cntP in eachindex(Gaps_Pars_kFs)], 
            xlabel = x_label, ylabel = L"||$\eta_m$|| "*y_unit, label=L"$\eta_m(k_{F,"*string(cntkF)*L"})$", linewidth=2) # "iter="*string(1)
    end
    plot!(pGapn, x_axis, [Gaps_Pars_max[cntP]*converged_gaps[cntP] - 1000*(1-converged_gaps[cntP]) for cntP in eachindex(Gaps_Pars_max)], 
        xlabel = x_label, ylabel = L"||$\eta_m$|| "*y_unit, label=L"max$(|\eta_m(k)|)$", linewidth=2, 
        foreground_color_legend = nothing, background_color_legend = nothing) # "iter="*string(1)
    return pGapn, Gaps_Pars_kFs, Gaps_Pars_max
end

function plot_GapTcRatio(ParsList::Vector{ModelParameters}, x_axis::Vector{Float64}, x_label::Union{LaTeXString,String}, 
    Tcs_k_iter::Vector{Vector{Float64}}, 
    Gaps_Pars_kFs::Vector{Vector{Float64}}, Gaps_Pars_max::Vector{Float64}, ylims::Vector{Float64},
    converged_bothTcGap::Vector{Int64})
    max_FSs = maximum([length(Pars.kFs) for Pars in ParsList])
    pRatio = plot(x_axis, [Gaps[1] for Gaps in Gaps_Pars_kFs] ./ [Tcs[end] for Tcs in Tcs_k_iter] .* converged_bothTcGap .- 1000*(1 .- converged_bothTcGap), 
        xlabel = x_label, ylabel = L"$|\eta_m / k_B T_c|$", label=L"$\eta_m(k_{F,1}) / k_B T_c$", linewidth=2, ylims=ylims) # "iter="*string(1)
    for cntkF in range(2,max_FSs)
        plot!(pRatio, x_axis, [Gaps[cntkF] for Gaps in Gaps_Pars_kFs] ./ [Tcs[end] for Tcs in Tcs_k_iter] .* converged_bothTcGap .- 1000*(1 .- converged_bothTcGap), 
            xlabel = x_label, ylabel = L"$|\eta_m / k_B T_c|$", label=L"$\eta_m(k_{F,"*string(cntkF)*L"})/ k_B T_c$", linewidth=2, ylims=ylims) # "iter="*string(1)
    end
    plot!(pRatio, x_axis, Gaps_Pars_max ./ [Tcs[end] for Tcs in Tcs_k_iter] .* converged_bothTcGap .- 1000*(1 .- converged_bothTcGap), 
        xlabel = x_label, ylabel = L"$|\eta_m / k_B T_c|$", label=L"max$(|\eta_m(k)|)/ k_B T_c$", linewidth=2, ylims=ylims, 
        foreground_color_legend = nothing, background_color_legend = nothing) # "iter="*string(1)
    return pRatio
end

function plot_etas(ParsList::Vector{ModelParameters}, ks_integration_k::Vector{Vector{Float64}}, 
    eta_m_k_iter::Vector{Vector{Vector{Float64}}}, norm_eta_m_k_iter::Vector{Vector{Float64}}, VmkFk::Vector{Vector{Float64}},
    show_iters::Bool, which_Pars::Vector{Int64})
    plots_eta = []
    ks_ret = Vector{Float64}()
    etas_ret = Vector{Float64}()
    VmkFk_ret = Vector{Float64}()
    for (num, cntP) in enumerate(which_Pars)
        iterations_radial = length(norm_eta_m_k_iter[cntP])
        gradient = cgrad([:gold, :red, :purple, :black])
        push!(plots_eta, plot(ks_integration_k[cntP]/ParsList[cntP].kF, [eta_m_k_iter[cntP][1][cntk]/norm_eta_m_k_iter[cntP][1] for cntk in 1:length(ks_integration_k[cntP])], 
            label=nothing,color=:gold))
        if show_iters
            for iter in 2:iterations_radial-1
                color = gradient.colors[round(Int, (iter-1) * (length(gradient.colors) - 1) / (iterations_radial - 1)) + 1]
                plot!(plots_eta[num], ks_integration_k[cntP]/ParsList[cntP].kF, [eta_m_k_iter[cntP][iter][cntk]/norm_eta_m_k_iter[cntP][iter] for cntk in 1:length(ks_integration_k[cntP])], 
                label=nothing,color=color,linewidth=0.5)
            end
        end
        etas = [eta_m_k_iter[cntP][end][cntk]/norm_eta_m_k_iter[cntP][end] for cntk in 1:length(ks_integration_k[cntP])]
        plot!(plots_eta[num], ks_integration_k[cntP]/ParsList[cntP].kF, etas, 
        xlabel=L"$k/k_F$",label=L"$n=$"*string(round(ParsList[cntP].n*100,digits=3)), linewidth=2, color=:black, ylims = [-1.5*maximum(abs.(etas)), 1.5*maximum(abs.(etas))])
        plot!(twinx(plots_eta[num]), ks_integration_k[cntP]/ParsList[cntP].kF, VmkFk[cntP],
        color=:red, label=L"V(k_F, k)", linestyle=:dash, ylims = (-1.5*maximum(abs.(VmkFk[cntP])), 1.5*maximum(abs.(VmkFk[cntP]))),legend=:bottomright, 
        foreground_color_legend = nothing, background_color_legend = nothing)
    
        if num == 1
            ks_ret = copy(ks_integration_k[cntP])
            etas_ret = copy(etas)
            VmkFk_ret = copy(VmkFk[cntP])
        end
    end
    combined_plot = plot(plots_eta..., layout = (Int(ceil(length(plots_eta)//2)), 2), size=(800,800))
    # display(combined_plot)
    return ks_ret, etas_ret, VmkFk_ret, combined_plot
end

function plot_QPDispersion(ParsList::Vector{ModelParameters}, ks_integration_k::Vector{Vector{Float64}}, Gaps_k_iter::Vector{Vector{Float64}}, 
    f_m_k_iter::Vector{Vector{Vector{Float64}}}, norm_f_m_k_iter::Vector{Vector{Float64}},
    which_Pars::Vector{Int64}, ylim_max::Float64, k_res::Int64)
    plots_QPDispersion = []
    ks_ret = Vector{Float64}()
    Eks_ret = Vector{Float64}()
    xi_ks_ret = Vector{Float64}()
    spectral_gap = Vector{Float64}()
    
    for (num, cntP) in enumerate(which_Pars)
        ks = ks_integration_k[cntP]
        xi_k = Dispersion.(ks, Ref(ParsList[cntP])) .- Dispersion(ParsList[cntP].kF, ParsList[cntP])
        etaks_end = [Gaps_k_iter[cntP][end]*f_m_k_iter[cntP][end][cntk]/norm_f_m_k_iter[cntP][end] for cntk in 1:length(ks_integration_k[cntP])]
        Eks_end = sqrt.((xi_k).^2 .+ etaks_end.^2)   
        max_dE = maximum(abs.(Eks_end - xi_k))
        xlim_max = maximum([ks[argmin(abs.(Eks_end .- 2*max_dE))]/ParsList[cntP].kF, 2.0])
        if xlim_max > maximum(ks)
            xlim_max = maximum(ks)
        end

        ks_plot = range(0, xlim_max, k_res)
        xi_k_plot = Dispersion.(ks_plot, Ref(ParsList[cntP])) .- Dispersion(ParsList[cntP].kF, ParsList[cntP])
        itp = LinearInterpolation(ks, etaks_end)
        etaks_plot = itp.(ks_plot)
        Eks_plot = sqrt.((xi_k_plot).^2 .+ etaks_plot.^2)   
        
        push!(plots_QPDispersion, plot(ks_plot/ParsList[cntP].kF, abs.(xi_k_plot),
        label=L"$|\xi_k - \mu|$, n="*string(round(ParsList[cntP].n,digits=4)),color=:gray,linestyle=:dash))
        
        if cntP >= length(ParsList)-2
            plot!(plots_QPDispersion[num], ks_plot/ParsList[cntP].kF, Eks_plot, xlim=[0,xlim_max],ylim=[0, ylim_max],
                label=L"$E_k$",color=:black, xlabel=L"$k/k_F$")
        else
            plot!(plots_QPDispersion[num], ks_plot/ParsList[cntP].kF, Eks_plot, xlim=[0,xlim_max],ylim=[0, ylim_max],
            label=L"$E_k$",color=:black)
        end
        push!(spectral_gap, minimum(abs.(Eks_plot)))
        # @show num
        if num == 1
            ks_ret = copy(ks_plot)
            Eks_ret = copy(Eks_plot)
            xi_ks_ret = copy(xi_k_plot)
        end
    end
    combined_plot_QPDispersion = plot(plots_QPDispersion..., layout = (Int(ceil(length(plots_QPDispersion)//2)), 2), size=(800,800))
    # display(combined_plot_QPDispersion)
    return ks_ret, Eks_ret, xi_ks_ret, combined_plot_QPDispersion, spectral_gap
end

function plot_QPDOS(ParsList::Vector{ModelParameters}, ks_integration_k::Vector{Vector{Float64}}, Gaps_k_iter::Vector{Vector{Float64}}, 
    f_m_k_iter::Vector{Vector{Vector{Float64}}}, norm_f_m_k_iter::Vector{Vector{Float64}},
    which_Pars::Vector{Int64}, E_bin_edges::Vector{Float64}, nks_DOS::Int64)
    plots_QPDOS = []
    Es_ret = Vector{Float64}()
    S_DOSs_ret = Vector{Float64}()
    N_DOSs_ret = Vector{Float64}()

    E_bin_size = (E_bin_edges[2] - E_bin_edges[1])
    E_bin_centers = E_bin_edges[2:end] .- E_bin_size/2

    for (num, cntP) in enumerate(which_Pars)
        ks = ks_integration_k[cntP]
        xi_k = Dispersion.(ks, Ref(ParsList[cntP])) .- Dispersion(ParsList[cntP].kF, ParsList[cntP])
        etaks_end = [Gaps_k_iter[cntP][end]*f_m_k_iter[cntP][end][cntk]/norm_f_m_k_iter[cntP][end] for cntk in 1:length(ks_integration_k[cntP])]
        Eks_end = sqrt.((xi_k).^2 .+ etaks_end.^2)   
        max_dE = maximum(abs.(Eks_end - xi_k))
        xlim_max = maximum([ks[argmin(abs.(Eks_end .- 2*max_dE))]/ParsList[cntP].kF, 2.0])
        if xlim_max > maximum(ks)
            xlim_max = maximum(ks)
        end

        ks_plot = collect(range(0, xlim_max, 10000))
        xi_k_plot = Dispersion.(ks_plot, Ref(ParsList[cntP])) .- Dispersion(ParsList[cntP].kF, ParsList[cntP])
        itp = LinearInterpolation(ks, etaks_end)
        etaks_plot = itp.(ks_plot)
        Eks_plot = sqrt.((xi_k_plot).^2 .+ etaks_plot.^2)   

        k_edge = find_largest_zero_crossing(ks_plot, Eks_plot .- E_bin_edges[end])
        dk_DOS = k_edge/(nks_DOS-1)
        ind_k_edge = argmin(abs.(ks_plot .- (k_edge+dk_DOS)))+2

        itp = linear_interpolation(ks_plot[1:ind_k_edge], Eks_plot[1:ind_k_edge])
        function InterpolatedEk(k::Float64)::Float64
            return itp(k)
        end 

        ks_DOS = range(dk_DOS/2, k_edge+dk_DOS/2, nks_DOS+1)
        Es_DOS = InterpolatedEk.(ks_DOS)
        NEs = [2π*ks_DOS[cntk].*dk_DOS for cntk in eachindex(ks_DOS)]
        DOS_p = weighted_binning(Es_DOS, NEs, E_bin_edges)./E_bin_size./(2π)^2 # Additional factor 1/2 for S DOS

        ks_DOS_xi = 2*collect(range(dk_DOS, k_edge+dk_DOS, nks_DOS))
        # xi_k_centers = abs.(Dispersion.(ks_DOS_xi, Ref(ParsList[cntP])) .- Dispersion(ParsList[cntP].kF, ParsList[cntP]))
        xi_k_centers = Dispersion.(ks_DOS_xi, Ref(ParsList[cntP])) .- ParsList[cntP].D
        NEs_xi = [2π*ki.*(2*dk_DOS) for ki in ks_DOS_xi]
        DOS_xi = weighted_binning(xi_k_centers, NEs_xi, E_bin_edges)./E_bin_size./(2π)^2

        if cntP >= length(ParsList)-1
            push!(plots_QPDOS, plot(E_bin_centers, DOS_p / ParsList[cntP].nu_bare_electron, xlim=[E_bin_edges[1],E_bin_edges[end]],
                label=L"$\nu / \nu_{\rm bare}$; n="*string(round(ParsList[cntP].n,digits=4)),color=:black, xlabel=L"$E$"))
        else
            push!(plots_QPDOS, plot(E_bin_centers, DOS_p / ParsList[cntP].nu_bare_electron, xlim=[E_bin_edges[1],E_bin_edges[end]],
                label=L"$\nu/ \nu_{\rm bare}$; n="*string(round(ParsList[cntP].n,digits=4)),color=:black))
        end
        plot!(plots_QPDOS[num], E_bin_centers, DOS_xi / ParsList[cntP].nu_bare_electron, color=:gray, linestyle=:dash,
        ylim = [0.0, maximum(DOS_p)], label=L"\nu_0 / \nu_{\rm bare}") # DOS.(ParsList[cntP].D .+ E_bin_centers, Ref(ParsList[cntP]))
    
        if num == 1
            Es_ret = copy(E_bin_centers)
            S_DOSs_ret = copy(DOS_p)
            N_DOSs_ret = copy(DOS_xi)
        end
    end
    combined_plots_QPDOS = plot(plots_QPDOS..., layout = (Int(ceil(length(plots_QPDOS)//2)), 2), size=(800,800))
    # display(combined_plots_QPDOS)
    return Es_ret, S_DOSs_ret, N_DOSs_ret, combined_plots_QPDOS
end

# -------------------------------------------------------------------------
# -------------------------- Load and save data ---------------------------
# -------------------------------------------------------------------------
function save_data_to_dat(data, folder_name::String, file_name::String)
    # Check if the folder exists, if not, create it
    if !isdir(folder_name)
        mkpath(folder_name)
    end

    # Define the full file path
    file_path = joinpath(folder_name, file_name)

    # Check the type of data and save accordingly
    if isa(data, Vector{<:Number})  # Single vector
        writedlm(file_path, data)
    elseif isa(data, Vector{<:Vector{<:Number}})  # Vector of vectors
        # Convert vector of vectors to a 2D array before saving
        max_length = maximum(length.(data))  # Handle vectors of different lengths
        padded_data = hcat([vcat(v, fill(NaN, max_length - length(v))) for v in data]...)
        writedlm(file_path, padded_data)
    elseif isa(data, AbstractArray{<:Number})  # Multidimensional array
        writedlm(file_path, data)
    else
        error("Unsupported data type")
    end
    
    # println("Data saved to $file_path")
end

function load_data_from_dat(folder_name::String, file_name::String)
    # Define the full file path
    file_path = joinpath(folder_name, file_name)

    # Check if the file exists
    if !isfile(file_path)
        error("File not found at $file_path")
    end

    # Load the data
    data = readdlm(file_path, Any)

    # Determine if it's a single vector, vector of vectors, or array
    if isa(data, Vector)
        if isa(data[1], Vector)
            # Vector of vectors
            data_out = [Vector{eltype(data[1])}(row) for row in data]
        else
            # Single vector
            data_out = Vector{eltype(data)}(data)
        end
    elseif isa(data, AbstractArray)
        # Multidimensional array
        data_out = data
    else
        error("Unknown data format in file")
    end

    return data_out
end

function load_vector_from_dat(folder_name::String, file_name::String)
    # Define the full file path
    file_path = joinpath(folder_name, file_name)

    # Check if the file exists
    if !isfile(file_path)
        error("File not found at $file_path")
    end

    # Load the data
    # @show folder_name, file_name, file_path
    data = readdlm(file_path, Any)
    data_out = Vector()
    for el in data
        push!(data_out, el)
    end
    # print(data_out)
    # @show data_out

    # Determine if it's a single vector, vector of vectors, or array
    # if isa(data, Vector)
    #     # Single vector
    #     data_out = Vector{eltype(data)}(data)
    # else
    #     error("Unknown data format in file")
    # end
    # @show data_out
    return data_out
end

function save_struct_to_dat_if_not_exists(struct_data::T, folder_name::String, file_name::String) where T
    # Check if the folder exists, if not, create it
    if !isdir(folder_name)
        mkpath(folder_name)
    end

    # Define the full file path
    file_path = joinpath(folder_name, file_name)

    # Check if the file already exists
    if isfile(file_path)
        println("File already exists at $file_path. No data saved.")
    else
        # Save the struct to the file
        open(file_path, "w") do io
            serialize(io, struct_data)
        end
        # println("Struct saved to $file_path")
    end
end

function save_struct_if_not_exists(struct_data::T, folder_name::String, file_name::String) where T
    # Check if the folder exists, if not, create it
    if !isdir(folder_name)
        mkpath(folder_name)
    end

    # Define the full file path
    file_path = joinpath(folder_name, file_name)

    # Check if the file already exists
    if isfile(file_path)
        println("File already exists at $file_path. No data saved.")
    else
        # Open the file for writing
        open(file_path, "w") do io
            # Iterate over the fields of the struct
            for field in fieldnames(T)
                value = getfield(struct_data, field)
                # Write the field name and value to the file
                println(io, "$field = $value")
            end
        end
        # println("Struct saved to $file_path")
    end
end
