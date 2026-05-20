using Sunny, LinearAlgebra, Statistics, GLMakie

# Import the planck noise functions
include("plancknoise.jl")

# Set up simple BCC antiferromagnet systems, one for white nosie, one for Planck
# noise.
crystal = Sunny.bcc_crystal()

sys_pn = System(crystal, [1 => Moment(s=1, g=2)], :dipole; dims=(6, 6, 6))
set_exchange!(sys_pn, 1, Bond(1, 2, [0, 0, 0]))
minimize_energy!(sys_pn)

sys_wn = System(crystal, [1 => Moment(s=1, g=2)], :dipole; dims=(6, 6, 6))
set_exchange!(sys_wn, 1, Bond(1, 2, [0, 0, 0]))
minimize_energy!(sys_wn)


# Calculate the mean energy at a range of temperatures
kTs = range(0.1, 7.0, 60)

dt = 0.01
damping = 0.3
nsamples = 500
ndecorr = 20

buf_wn = zeros(nsamples)
buf_pn = zeros(nsamples)
Emeans_wn = zero(kTs)
Emeans_pn = zero(kTs)



@time for (n, kT) in enumerate(kTs)

    # Set up both a Langevin integrator and LangevinPlanck integrator. The
    # Planck version requires the dimension of the sys_pntem's dipole field as an
    # added argument.
    integrator_wn = Langevin(dt; damping, kT)
    integrator_pn = LangevinPlanck(dt; damping, kT, sysdims=size(sys_pn.dipoles))

    # Warm up the Planck noise integrator
    for _ in 1:500
        step!(sys_pn, integrator_pn)
    end

    # Collect samples
    for i in 1:nsamples

        # Decorrelate between samples
        for _ in 1:ndecorr
            step!(sys_wn, integrator_wn)
        end
        for _ in 1:ndecorr
            step!(sys_pn, integrator_pn)
        end

        # Record energy
        buf_wn[i] = energy_per_site(sys_wn)
        buf_pn[i] = energy_per_site(sys_pn)
    end

    # Record mean energy of samples at given temperature
    Emeans_wn[n] = mean(buf_wn)
    Emeans_pn[n] = mean(buf_pn)
end


# Plot the results
fig = Figure(size=(800, 350))
ax1 = Axis(fig[1,1]; xlabel="T (J)", ylabel="Energy per site (J)")
ax2 = Axis(fig[1,2]; xlabel="T (J)", ylabel="C∝ΔE/ΔkT")

scatter!(ax1, kTs, Emeans_wn; label="White Noise")
scatter!(ax1, kTs, Emeans_pn; label="Planck Noise")
axislegend(ax1, position=:rb)

kTs_mid = (kTs[2:end] + kTs[1:end-1])/2
ΔkTs = kTs[2:end] - kTs[1:end-1]
C_wn = (Emeans_wn[2:end] - Emeans_wn[1:end-1]) ./ ΔkTs
C_pn = (Emeans_pn[2:end] - Emeans_pn[1:end-1]) ./ ΔkTs
scatter!(ax2, kTs_mid, C_wn; label="White Noise")
scatter!(ax2, kTs_mid, C_pn; label="Planck Noise")
axislegend(ax2, position=:rt)

fig