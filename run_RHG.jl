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
Set up calculation and for superconductivity from repulsion in 
isospin-polarized rhombohedral graphene. 
"""


include("SpinPolarizedSuperconductivity.jl")

# System parameters
me = 0.005685630103608402 # electron mass in meV ps^2 / nm^2

# dispersion
which_dispersion_i = "ABCA_Slizovskiy_k2" # Refers to the dispersion derived from the effective two-band model in [Koshino et al., Phys. Rev. B 80, 165409 (2009)], [Slizovskiy et al., Nat. Commun. Phys. 2, 164 (2019)]
D_i = 60.0 # displacement field entering RHG dispersion
k_D = 0.38 # parameter determining the (k/k_D)^8 term in the dispersion
m2i = -0.005*(60.0 - 40.1)/(D_i - 40.1) # factor of the quadratic term in the dispersion
default_density = 0.5e-2 # default density to use in susceptibility calculations
include_FormFactor = false # whether to include form factors due to the wavefunction geometry
if include_FormFactor
    which_geometry = "ABCA_Slizovskiy_k2" # use form factors derived from the effective two-band model [Koshino et al., Phys. Rev. B 80, 165409 (2009)], [Slizovskiy et al., Nat. Commun. Phys. 2, 164 (2019)]
else
    which_geometry = "flat" # no form factors
end
mi, alphaDi = 0.0, 0.0 # unused for "ABCA_Slizovskiy_k2" dispersion

# interaction
which_interaction = "Keldysh"
eps_hBN = 5.0 # dielectric constant of surrounding medium
rK      = 3.0 # Keldysh parameter for dielectric screening
d_gates = 30.0 # distance to metallic gates on both sides of the 2DEG

# which pairing angular momentum channel to solve for
angular_momentum = 1

# momentum discretization
L_torus = 129 # Momentum discretization. Check L_torus for convergence. Best values are 2^n + 1
qmaxi = 50 # Cutoff of momentum discretization. Check qmaxi for convergence. qmaxi sets the maximum of the range of ks over which the pairing potential is evaluated
res_kx = 1 # kx resolution (default: 1)

# set up default parameter structures
DPars0 = DispersionParameters(me, D_i, m2i, 0.0, 0.0, k_D, which_dispersion_i, which_geometry)
Pars0 = ModelParameters(default_density, DPars0.m, DPars0.D, DPars0.m2, DPars0.α, DPars0.vF_G, DPars0.k_D, which_dispersion_i, which_geometry, which_interaction, 0.0, 0.0, eps_hBN, d_gates, L_torus)

# set up discretization of kbT and gap magnitude for the self-consistency solution
logrange(x1, x2, n) = collect((10^y for y in range(log10(x1), log10(x2), length=n)))
kbTList = 1*logrange(0.0001,1.0,2001)
gapList = 1*logrange(0.0001,1.0,2001)

# set the range of densities for which the system is evaluated
nList = range(0.1*1e-2,0.8*1e-2,40) 

# maximum number of iterations to perform for self consistency in Tc and gap
iters_rad_Tc, iters_rad_gap = fill(24, length(nList)), fill(16, length(nList))

# whether to plot susceptiblity and RPA screened V(q) and its angular harmonics
plot_susceptibility = true

# whether to save computed results
do_save = true 



# # ---------------------------------------------------------------------------------
# # ----------------------------- Starting calculation ------------------------------
# # ---------------------------------------------------------------------------------
print("\n starting [rK, mi, Di, k_Di, alphaDi] = "*string([rK, mi, D_i, k_D, alphaDi])*"\n")
DPars = DispersionParameters(mi, D_i, m2i, alphaDi, 0.0, k_D, which_dispersion_i, which_geometry)
initialial_guess = "VmkFk" # initial guess for gap functional form

hat_density = find_hat_density(DPars)
print("hat density: " * string(hat_density*100) * " [10^12 cm^-2]")
print("Conversion factor to [eV^-1 A_uc^-1]: " * string(1e3*0.246^2*sqrt(3)/2))

# # -------------------------------- Plot dispersion --------------------------------
Pars = ModelParameters(0.5e-2, DPars.m, DPars.D, DPars.m2, DPars.α, DPars.vF_G, DPars.k_D, DPars.which_dispersion, which_geometry, which_interaction, alphaDi, rK, eps_hBN, d_gates, L_torus)
plot_Dispersion(Pars, 0.5*1e-2, 0.6, collect(range(-2.0, 0.5, 401)), collect(range(-2.0, 0.5, 401)))

# # Verify density of states
print("\n Bare electron DOS: \n "*string(Pars.nu_bare_electron)*"\n")

# # ------------------ Compute susceptibility, V(q), and angular harmonics of V(q) ------------------
if plot_susceptibility
    main_Susceptibility(Pars, true, do_save, qmaxi)
end

ParsList = [ModelParameters(n, DPars.m, DPars.D, DPars.m2, DPars.α, DPars.vF_G, DPars.k_D, DPars.which_dispersion, which_geometry, which_interaction, 
    alphaDi, rK, eps_hBN, d_gates, L_torus)
    for n in nList]
kFList = [Pars.kF for Pars in ParsList]

# Calculate DOS at EF of normal state
DOS_EFs = [DOS_EF(Pars) for Pars in ParsList]
ThomasFermiLengths = [Pars.epsilon0*eps_hBN/DOS for DOS in DOS_EFs]
print("Thomas Fermi lengths: \n")
print(ThomasFermiLengths)

# # ------------------------------------ Get Gas Parameters ------------------------------------
print("\n Calculating gas parameters ...")
# High resolution for smooth curves, however requires significant compute (a few mins on single CPU core)
# GasParameters = [get_GasParameters([ParsList[cntP]], collect(range(0.0, 16*ParsList[cntP].kF,8000)),false)[1] for cntP in eachindex(ParsList)] 
# Small resolution, Verify convergence in doubt, curves may be not smooth
print("Warning: Using small resolution for gas parameter calculation, may result in noisy curves.")
GasParameters = [get_GasParameters([ParsList[cntP]], collect(range(0.0, 2*ParsList[cntP].kF,1000)),false)[1] for cntP in eachindex(ParsList)] 
print(" done. \n")
if do_save
    save_data_to_dat(nList, Pars.stamp, "TcsDelta0_ns.dat")
    save_data_to_dat([el[1] for el in GasParameters], Pars.stamp, "TcsDelta0_GasParameters_KineticEnergyDensity.dat")
    save_data_to_dat([el[2] for el in GasParameters], Pars.stamp, "TcsDelta0_GasParameters_InteractionEnergyDensity.dat")
    save_data_to_dat([el[3] for el in GasParameters], Pars.stamp, "TcsDelta0_GasParameters_r_s.dat")
    r_ss = [el[3] for el in GasParameters]
    density_rScrit = Vector{Float64}()
    for rScrit in 20.0:5:60
        if maximum(r_ss) <= rScrit
            push!(density_rScrit, 0.0)
        elseif minimum(r_ss) >= rScrit
            push!(density_rScrit, 1000.0)
        else
            push!(density_rScrit, find_largest_zero_crossing(collect(nList), r_ss .- rScrit))
        end
    end
    save_data_to_dat(density_rScrit, Pars.stamp, "TcsDelta0_GasParameters_n_rS20-5-60.dat")
end
pGas = plot(nList*100, [abs(el[3]) for el in GasParameters], label=L"$r_s$", 
    lc=:green, lw=2.0, y_foreground_color_text=:green, yguidefontcolor=:green, ylims=[0, 100], 
    xlims = (minimum(nList)*100, maximum(nList)*100))
vline!(pGas, density_rScrit*100, label=nothing, color=:gray)
vline!(pGas, [hat_density*100], label=L"$n_{top}$", color=:gray)
display(pGas)
@show [el[3] for el in GasParameters]

qmaxList = [qmaxi for kF in kFList]
plot_Pars = Vector(1:Int(ceil(length(kFList)/8)):length(kFList))

# Main calculation: Get Tcs and gaps
convergence_tolerance = 0.05
ks_integration_k, Tcs_k_iter, eta_m_k_iter, norm_eta_m_k_iter, Gaps_k_iter, f_m_k_iter, norm_f_m_k_iter, VmkFk, Convergence_TcGap, Tcs_T0Gap = 
    main_direct_integration_iterative(ParsList, angular_momentum, kbTList, 
    qmaxList, res_kx, 121, iters_rad_Tc, iters_rad_gap, initialial_guess, [[0.0]], convergence_tolerance, include_FormFactor)
converged_Tcs = [el[1] for el in Convergence_TcGap]
converged_Gaps = [el[2] for el in Convergence_TcGap]

# Calculate plots
x_data = [Pars.n*100 for Pars in ParsList]
x_label = L"n [$10^{12} $cm$^{-2}$]"
# plot T_c
pTcs = plot_Tcs(ParsList, x_data, x_label, "[K]", Tcs_k_iter, [0.0, maximum([Tcs[end] for Tcs in Tcs_k_iter]*11.60452*1.1)], converged_Tcs, false)
@show maximum([Tcs[end] for Tcs in Tcs_k_iter]*11.60452)
plot!(pTcs, [hat_density*100], seriestype="vline", linewidth=1, color=:gray, label=L"n_{\rm hat}")
plot!(pTcs, [Pars.n*100 for Pars in ParsList], Tcs_T0Gap, linewidth=1, color=:green, linestyle=:dash, label=L"$T_c$ for $\eta_m(T=0))$")
# plot superconducting gap at zero temperature
ks_eta0, etas_eta0, VmkFk_eta0, pEtas_Tc = plot_etas(ParsList, ks_integration_k, eta_m_k_iter, norm_eta_m_k_iter, VmkFk, true, plot_Pars)
pGaps, Gaps_Pars_kFs, Gaps_Pars_max = plot_gaps(ParsList, x_data, x_label, "[Ha]", ks_integration_k, Gaps_k_iter, f_m_k_iter, norm_f_m_k_iter, [0.0, 0.2], converged_Gaps)
eta_gaps_k_iter = [Gaps_k_iter[cntP] .* f_m_k_iter[cntP]  for cntP in eachindex(Gaps_k_iter)]
ks_eta0, etas_eta0, VmkFk_eta0, pEtas_zeroT = plot_etas(ParsList, ks_integration_k, eta_gaps_k_iter, norm_f_m_k_iter, VmkFk, true, plot_Pars)
# plot dispersion
ks_disp, Eks_disp, xi_ks_disp, pDispersion, spectral_gap_plots = plot_QPDispersion(ParsList, ks_integration_k, Gaps_k_iter, f_m_k_iter, norm_f_m_k_iter, plot_Pars, 0.5, 4000)
# plot ratio gap / k_B T_c
pRatio = plot_GapTcRatio(ParsList, x_data, x_label, Tcs_k_iter, Gaps_Pars_kFs, Gaps_Pars_max, [0.0, 8.0], [el[1]*el[2] for el in Convergence_TcGap])

# plot gas parameters in Tc panel
plot!(twinx(pTcs), nList*100, [abs(el[3]) for el in GasParameters], label=L"$r_s$", 
    lc=:green, lw=2.0, y_foreground_color_text=:green, yguidefontcolor=:green, ylims=[0, 100])

# plot DOS in Gaps panel
plot!(twinx(pGaps), [Pars.n*100 for Pars in ParsList], DOS_EFs/Pars.nu_bare_electron, 
    lc=:red, ylims=[0.0, maximum(DOS_EFs/Pars.nu_bare_electron)], ylabel=L"$\nu_F / \nu_{\rm bare}$", label=nothing, y_foreground_color_text=:red, yguidefontcolor=:red)

# Get spectral_gaps
ks_disp, Eks_disp, xi_ks_disp, pDispersion_all, spectral_gaps = plot_QPDispersion(
    ParsList, ks_integration_k, Gaps_k_iter, f_m_k_iter, norm_f_m_k_iter, [cnt for cnt in eachindex(ParsList)], 0.5, 8000)
plot!(pGaps, nList*100, spectral_gaps, lc=:purple, label=L"min($E$)")

combined_plot = plot(pTcs, pGaps, pRatio, layout = grid(3, 1, heights=[0.5, 0.25, 0.25]), link=:x, size=(600, 800))
display(pEtas_zeroT)
display(combined_plot)

# Plot n_S, the calculated electron density density in the superconducting state, relative to the input density from which EF was calculated
densities_S = [get_density_S(ParsList[cntP], ks_integration_k[cntP], eta_gaps_k_iter[cntP][end]) for cntP in eachindex(ParsList)]

# save data
if do_save
    indicator_m = "-l"*string(angular_momentum)
    save_struct_if_not_exists(Pars, Pars.stamp, "Pars.txt")
    save_data_to_dat(nList, Pars.stamp, "TcsDelta0_ns"*indicator_m*".dat")
    save_data_to_dat([hat_density, ParsList[1].BandBottom], Pars.stamp, "TcsDelta0_HatDensity_BandBottom.dat")
    save_data_to_dat([Pars.EF for Pars in ParsList], Pars.stamp, "TcsDelta0_EFs.dat")
    save_data_to_dat(DOS_EFs, Pars.stamp, "TcsDelta0_DOS_EFs.dat")
    save_data_to_dat(densities_S, Pars.stamp, "TcsDelta0_SuperconductingDensities"*indicator_m*".dat")
    save_data_to_dat([el[1] for el in GasParameters], Pars.stamp, "TcsDelta0_GasParameters_KineticEnergyDensity.dat")
    save_data_to_dat([el[2] for el in GasParameters], Pars.stamp, "TcsDelta0_GasParameters_InteractionEnergyDensity.dat")
    save_data_to_dat([el[3] for el in GasParameters], Pars.stamp, "TcsDelta0_GasParameters_r_s.dat")
    save_data_to_dat(converged_Tcs, Pars.stamp, "TcsDelta0_Convergence_Tcs"*indicator_m*".dat")
    save_data_to_dat(converged_Gaps, Pars.stamp, "TcsDelta0_Convergence_Gaps"*indicator_m*".dat")
    save_data_to_dat([minimum(Dispersion.(range(0, Pars.kF, 1000), Ref(Pars))) .- Pars.D for Pars in ParsList], Pars.stamp, "TcsDelta0_BandBottoms.dat")  
    
    save_data_to_dat([Tcs[end] for Tcs in Tcs_k_iter], Pars.stamp, "TcsDelta0_kbTcs_InclNotConverged"*indicator_m*".dat")
    save_data_to_dat([Gaps[1] for Gaps in Gaps_Pars_kFs], Pars.stamp, "TcsDelta0_Delta0_k1_InclNotConverged"*indicator_m*".dat")
    max_FSs = maximum([length(Pars.kFs) for Pars in ParsList])
    if max_FSs == 2
        save_data_to_dat([Gaps[2] for Gaps in Gaps_Pars_kFs], Pars.stamp, "TcsDelta0_Delta0_k2_InclNotConverged"*indicator_m*".dat")
    end
    save_data_to_dat(Gaps_Pars_max, Pars.stamp, "TcsDelta0_maxDelta0_InclNotConverged"*indicator_m*".dat")

    save_data_to_dat([Tcs[end] for Tcs in Tcs_k_iter] .* converged_Tcs .- 1000 .* (1 .- converged_Tcs), Pars.stamp, "TcsDelta0_kbTcs"*indicator_m*".dat")
    save_data_to_dat([Gaps[1] for Gaps in Gaps_Pars_kFs] .* converged_Gaps .- 1000 .* (1 .- converged_Gaps), Pars.stamp, "TcsDelta0_Delta0_k1"*indicator_m*".dat")
    max_FSs = maximum([length(Pars.kFs) for Pars in ParsList])
    if max_FSs == 2
        save_data_to_dat([Gaps[2] for Gaps in Gaps_Pars_kFs] .* converged_Gaps .- 1000 .* (1 .- converged_Gaps), Pars.stamp, "TcsDelta0_Delta0_k2"*indicator_m*".dat")
    end
    save_data_to_dat(Gaps_Pars_max .* converged_Gaps .- 1000 .* (1 .- converged_Gaps), Pars.stamp, "TcsDelta0_maxDelta0"*indicator_m*".dat")
    save_data_to_dat(spectral_gaps .* converged_Gaps .- 1000 .* (1 .- converged_Gaps), Pars.stamp, "TcsDelta0_SpectralGaps"*indicator_m*".dat")
    
    save_Pars = [maximum([1,Int(round(length(nList)//2))-1])]
    print("saving eta, dispersion, DOS at density "*string(ParsList[save_Pars[1]].n*100)*"\n")
    ks_eta0, etas_eta0, VmkFk_eta0, pEta_ToSave = plot_etas(ParsList, ks_integration_k, eta_gaps_k_iter, norm_f_m_k_iter, VmkFk, true, save_Pars)
    ks_disp, Eks_disp, xi_ks_disp, pDispersion_ToSave = plot_QPDispersion(ParsList, ks_integration_k, Gaps_k_iter, f_m_k_iter, norm_f_m_k_iter, save_Pars, 0.5, 4000)
    Es_DOS, S_DOS, N_DOS, pDOS_ToSave = plot_QPDOS(ParsList,ks_integration_k, Gaps_k_iter, f_m_k_iter, norm_f_m_k_iter, save_Pars, collect(range(0,0.5,101)), 100000)

    save_struct_if_not_exists(ParsList[save_Pars[1]], Pars.stamp, "eta0_dispersion_DOS_Pars.txt")
    save_data_to_dat(ks_eta0, Pars.stamp, "eta0_ks.dat")
    save_data_to_dat(etas_eta0, Pars.stamp, "eta0_etaks"*indicator_m*".dat")
    save_data_to_dat(VmkFk_eta0, Pars.stamp, "eta0_VmkFks"*indicator_m*".dat")

    save_data_to_dat(ks_disp, Pars.stamp, "dispersion_ks.dat")
    save_data_to_dat(Eks_disp, Pars.stamp, "dispersion_Eks"*indicator_m*".dat")
    save_data_to_dat(xi_ks_disp, Pars.stamp, "dispersion_xi_ks.dat")

    save_data_to_dat(Es_DOS, Pars.stamp, "DOS_Es"*indicator_m*".dat")
    save_data_to_dat(S_DOS, Pars.stamp, "DOS_DOS_S"*indicator_m*".dat")
    save_data_to_dat(N_DOS, Pars.stamp, "DOS_DOS_N.dat")
end

