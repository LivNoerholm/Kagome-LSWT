using Sunny, LinearAlgebra, Statistics, GLMakie

# Import the planck noise functions
include("plancknoise.jl")

latvecs = lattice_vectors(7.3177, 7.3177, 17.534, 90, 90, 120)
fe_cryst = Crystal(latvecs, [[1/6, 5/6, 5/6]], 166)

moments = [1 => Moment(s=5/2, g=2)]  
sys_pn = System(fe_cryst, moments, :dipole; dims=(18,18,12))
D = [0, 0.218, -0.195]
set_exchange!(sys_pn, 3.3*Matrix(I, 3, 3) + dmvec(D), Bond(1, 2, [0, 0, 0]))
set_exchange!(sys_pn, 0.11, Bond(1, 2, [1, 0, 0]))
set_exchange!(sys_pn, 0.11, Bond(1, 2, [0, 1, 0]))
minimize_energy!(sys_pn)

kTs = range(0.1, 70, 140)

dt = 0.004
damping = 0.1
nsamples = 500
ndecorr = 50

buf_pn = zeros(nsamples)
Emeans_pn = zero(kTs)

@time for (n, kT) in enumerate(kTs)
    # Set up both a Langevin integrator and LangevinPlanck integrator. The
    # Planck version requires the dimension of the sys_pntem's dipole field as an
    # added argument.
    integrator_pn = LangevinPlanck(dt; damping, kT, sysdims=size(sys_pn.dipoles))
    # Warm up the Planck noise integrator
    for _ in 1:1000
        step!(sys_pn, integrator_pn)
    end

    # Collect samples
    for i in 1:nsamples

        for _ in 1:ndecorr
            step!(sys_pn, integrator_pn)
        end

        # Record energy
        buf_pn[i] = energy_per_site(sys_pn)
    end
    # Record mean energy of samples at given temperature
    Emeans_pn[n] = mean(buf_pn)
end


# Plot the results
fig = Figure(size=(800, 350))
ax1 = Axis(fig[1,1]; xlabel="T (J)", ylabel="Energy per site (J)")
ax2 = Axis(fig[1,2]; xlabel="T (J)", ylabel="C∝ΔE/ΔkT")

scatter!(ax1, kTs, Emeans_pn; label="Planck Noise")
axislegend(ax1, position=:rb)

kTs_mid = (kTs[2:end] + kTs[1:end-1])/2
ΔkTs = kTs[2:end] - kTs[1:end-1]
C_pn = (Emeans_pn[2:end] - Emeans_pn[1:end-1]) ./ ΔkTs
scatter!(ax2, kTs_mid, C_pn; label="Planck Noise")
axislegend(ax2, position=:rt)

fig