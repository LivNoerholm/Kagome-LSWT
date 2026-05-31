using Sunny, GLMakie, StaticArrays, LinearAlgebra, LsqFit, NonlinearSolve, Statistics, DelimitedFiles
# Import the planck noise functions
include("plancknoise.jl")

units = Units(:meV, :angstrom);

latvecs = lattice_vectors(7.3177, 7.3177, 17.534, 90, 90, 120)
fe_cryst = Crystal(latvecs, [[1/6, 5/6, 5/6]], 166)

moments = [1 => Moment(s=5/2, g=2)]  
sys_pn = System(fe_cryst, moments, :dipole; dims=(18,18,12))
sys_wn = System(fe_cryst, moments, :dipole; dims=(18,18,12))
D = [0, 0.218, -0.195]
set_exchange!(sys_pn, 3.3*Matrix(I, 3, 3) + dmvec(D), Bond(1, 2, [0, 0, 0]))
set_exchange!(sys_pn, 0.11, Bond(1, 2, [1, 0, 0]))
set_exchange!(sys_pn, 0.11, Bond(1, 2, [0, 1, 0]))
minimize_energy!(sys_pn)

set_exchange!(sys_wn, 3.3*Matrix(I, 3, 3) + dmvec(D), Bond(1, 2, [0, 0, 0]))
set_exchange!(sys_wn, 0.11, Bond(1, 2, [1, 0, 0]))
set_exchange!(sys_wn, 0.11, Bond(1, 2, [0, 1, 0]))
minimize_energy!(sys_wn)


Ts = range(0.1, 70, 140)

dt = 0.004
damping = 0.1
nsamples = 500
ndecorr = 50

buf_pn = zeros(nsamples)
Emeans_pn = zero(Ts*units.K)

buf_wn = zeros(nsamples)
Emeans_wn = zero(Ts*units.K)

@time for (n, T) in enumerate(Ts)
    # Set up both a Langevin integrator and LangevinPlanck integrator. The
    # Planck version requires the dimension of the sys_pntem's dipole field as an
    # added argument.
    integrator_pn = LangevinPlanck(dt; damping, kT = T*units.K, sysdims=size(sys_pn.dipoles))
    integrator_wn = Langevin(dt; damping, kT = T*units.K)
    # Warm up the Planck noise integrator
    for _ in 1:1000
        Sunny.step!(sys_pn, integrator_pn)
        Sunny.step!(sys_wn, integrator_wn)
    end
    println("Sampling at kT = $T K")
    # Collect samples
    for i in 1:nsamples

        for _ in 1:ndecorr
            Sunny.step!(sys_pn, integrator_pn)
        end
        for _ in 1:ndecorr
            Sunny.step!(sys_wn, integrator_wn)
        end    

        # Record energy
        buf_pn[i] = energy_per_site(sys_pn)
        buf_wn[i] = energy_per_site(sys_wn)
    end
    # Record mean energy of samples at given temperature
    Emeans_pn[n] = Statistics.mean(buf_pn)
    Emeans_wn[n] = Statistics.mean(buf_wn)
end

data = hcat(Ts, Emeans_pn, Emeans_wn)
writedlm("energies.csv", data, ',')

# Plot the results
fig = Figure(size=(800, 350))
ax1 = Axis(fig[1,1]; xlabel="T (K)", ylabel="Energy per site (K)")
ax2 = Axis(fig[1,2]; xlabel="T (K)", ylabel="C∝ΔE/ΔkT")

scatter!(ax1, Ts, Emeans_pn; label="Planck Noise")
scatter!(ax1, Ts, Emeans_wn; label="White Noise")

axislegend(ax1, position=:rb)

Ts_mid = (Ts[2:end] + Ts[1:end-1])/2
ΔTs = Ts[2:end] - Ts[1:end-1]
C_pn = (Emeans_pn[2:end] - Emeans_pn[1:end-1]) ./ ΔTs
C_wn = (Emeans_wn[2:end] - Emeans_wn[1:end-1]) ./ ΔTs

cvdata = hcat(Ts_mid, C_pn, C_wn)
writedlm("heat_capacity.csv", cvdata, ',')

scatter!(ax2, Ts_mid, C_pn; label="Planck Noise")
scatter!(ax2, Ts_mid, C_wn; label="White Noise")

axislegend(ax2, position=:rt)

fig