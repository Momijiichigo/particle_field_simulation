using LinearAlgebra
using CUDA           # GPU computing
using GLMakie        # high‑performance 3‑D plotting
using ColorSchemes
using GeometryBasics
import ProgressMeter
using Markdown
using StaticArrays

const Nx, Ny, Nz = 64, 64, 64          # grid size
threads = (4, 4, 4)                                # 512 threads per block
blocks  = (cld(Nx,threads[1]),
           cld(Ny,threads[2]),
           cld(Nz,threads[3]))                     # as many blocks as needed
############ Physical & numerical constants #############
const dx  = 0.1f0                      # spacing
const dt  = 0.001f0                    # time step
const nsteps      = 2000                # total steps
const frame_every = 5                  # dump every n steps


# electron mass in GeV
const mₑ = 0.000511f0
const qₑ = 0.3029f0 # electron charge
# Pauli matrices

σx = [0 1;
      1 0]

σy = [0 -im;
       im 0]

σz = [1 0;
      0 -1]

σ0 = [1 0;
      0 1]

# Gamma matrices
# (first make empty 4x4 matrices and set upper-right and lower-left 2x2 blocks to σx, σy, σz)
γ₀ = zeros(ComplexF32, 4, 4)
γ₀[3:4, 1:2] .= σ0
γ₀[1:2, 3:4] .= σ0
# γ₀ = zeros(ComplexF32, 4, 4)
# γ₀[1:2, 1:2] .= σ0
# γ₀[3:4, 3:4] .= -σ0
const γ0 = SMatrix{4,4}(γ₀)
γ₁ = zeros(ComplexF32, 4, 4)
γ₁[3:4, 1:2] .= -σx
γ₁[1:2, 3:4] .= σx
const γ1 = SMatrix{4,4}(γ₁)
γ₂ = zeros(ComplexF32, 4, 4)
γ₂[3:4, 1:2] .= -σy
γ₂[1:2, 3:4] .= σy
const γ2 = SMatrix{4,4}(γ₂)
γ₃ = zeros(ComplexF32, 4, 4)
γ₃[3:4, 1:2] .= -σz
γ₃[1:2, 3:4] .= σz
const γ3 = SMatrix{4,4}(γ₃)
γ₅ = zeros(ComplexF32, 4, 4)
γ₅[1:2, 1:2] .= -σ0
γ₅[3:4, 3:4] .= σ0
γ5 = SMatrix{4,4}(γ₅)


ψ = CUDA.zeros(ComplexF32, Nx, Ny, Nz, 4) # 4-component Dirac spinor
A = CUDA.zeros(ComplexF32, Nx, Ny, Nz, 4) # 4-vector photon field Aᵘ
∂A = CUDA.zeros(ComplexF32, Nx, Ny, Nz, 4, 4) # 4x4 matrix of ∂ᵤAᵛ

function init_gaussian!(field, σ, A, x0=0f0, y0=0f0, z0=0f0)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    k = (blockIdx().z-1)*blockDim().z + threadIdx().z
    nx, ny, nz = size(field)
    if 1 ≤ i ≤ nx && 1 ≤ j ≤ ny && 1 ≤ k ≤ nz
        r2 = (i - nx/2 - x0)^2 + (j - ny/2 - y0)^2 + (k - nz/2 - z0)^2
        field[i,j,k] += A * exp(-r2 / (2σ^2))
    end
    return
end
σ = 1f0
# function init_field!(ψ, A, ∂A)
#     i = (blockIdx().x-1)*blockDim().x + threadIdx().x
#     j = (blockIdx().y-1)*blockDim().y + threadIdx().y
#     k = (blockIdx().z-1)*blockDim().z + threadIdx().z
#     # nx, ny, nz = size(A[:,:,:,1])
#     if 1 ≤ i ≤ Nx && 1 ≤ j ≤ Ny && 1 ≤ k ≤ Nz
#         # TODO: initialize ψ, A, ∂ₜA

#     end
#     return
# end

# ψ[Int64(Nx/2), Int64(Ny/2), Int64(Nz/2) - 4, :] .= [sqrt(mₑ), 0, sqrt(mₑ), 0]
# ψ[Int64(Nx/2), Int64(Ny/2), Int64(Nz/2) + 4, :] .= [sqrt(mₑ), 0, -sqrt(mₑ), 0]


@cuda threads=threads blocks=blocks init_gaussian!(@view(ψ[:,:,:,1]), σ, mₑ, 0, 0, 0)
@cuda threads=threads blocks=blocks init_gaussian!(@view(ψ[:,:,:,3]), σ, mₑ, 0, 0, 0)

# @cuda threads=threads blocks=blocks init_gaussian!(@view(ψ[:,:,:,1]), σ, sqrt(mₑ), 0, 0, -5)
# @cuda threads=threads blocks=blocks init_gaussian!(@view(ψ[:,:,:,3]), σ, -sqrt(mₑ), 0, 0, -5)

synchronize()

fig = Figure(size = (700, 500))
ax  = Axis3(fig[1,1]; 
    perspectiveness=0.5, 
    title="ψ field evolution",
    limits=(1, Nx, 1, Ny, 1, Nz),  # Set explicit limits
    aspect = (1.0, 1.0, 1.3),
)


# Convert 3D array to vector of points
function array_to_points(arr)
    points = Point3f[]
    for i in 1:size(arr,1), j in 1:size(arr,2), k in 1:size(arr,3)
        push!(points, Point3f(i, j, k))
    end
    return points
end


# Initialize points and colors
points = array_to_points(Array(ψ[:,:,:,1]))
# Create explicit black colors with zero alpha
init_colors = [RGBAf(0.0f0, 0.0f0, 0.0f0, 0.0f0) for _ in 1:length(points)]
markers = scatter!(ax, points; 
    markersize = 4,
    color = init_colors,
    transparency = true
)
function update∂ⱼA!(A, ∂A, dx)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    k = (blockIdx().z-1)*blockDim().z + threadIdx().z
    if 2 ≤ i ≤ Nx-1 && 2 ≤ j ≤ Ny-1 && 2 ≤ k ≤ Nz-1
    
        # To make it compilable for GPU,
        # I cannot use broadcast here
        @inbounds for μ in 1:4
            # ∂₁A
            ∂A[i,j,k,2,μ] = (A[i+1,j,k,μ] - A[i,j,k,μ]) / dx
            # ∂₂A
            ∂A[i,j,k,3,μ] = (A[i,j+1,k,μ] - A[i,j,k,μ]) / dx
            # ∂₃A
            ∂A[i,j,k,4,μ] = (A[i,j,k+1,μ] - A[i,j,k,μ]) / dx
        end
    end
    return
end
function trace4x4(A::T) where {T}
    @inbounds tr = A[1,1] + A[2,2] + A[3,3] + A[4,4]
    return tr
end
function update!(ψ, A, ∂A, dx::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    k = (blockIdx().z-1)*blockDim().z + threadIdx().z
    if 2 ≤ i ≤ Nx-1 && 2 ≤ j ≤ Ny-1 && 2 ≤ k ≤ Nz-1
        # ψ at i,j,k
        # arrays; not allocated - requirement for GPU kernel
        psi = @SVector [@inbounds ψ[i,j,k,l] for l = 1:4]
        ∂₁ψ = @SVector [@inbounds (ψ[i+1,j,k,l] - psi[l]) / dx for l = 1:4]
        ∂₂ψ = @SVector [@inbounds (ψ[i,j+1,k,l] - psi[l]) / dx for l = 1:4]
        ∂₃ψ = @SVector [@inbounds (ψ[i,j,k+1,l] - psi[l]) / dx for l = 1:4]

        dpsi = (
            # -imγᵗψ
            -im * mₑ * γ0 * psi
            # -iqγᵗγᵘψAᵤ
            # -im * qₑ * γ0 * γ0 * psi * A[i,j,k,1]
            # +im * qₑ * γ0 * γ1 * psi * A[i,j,k,2]
            # +im * qₑ * γ0 * γ2 * psi * A[i,j,k,3]
            # +im * qₑ * γ0 * γ3 * psi * A[i,j,k,4]
            # -γᵗγᵃ∂ₐψ
            -γ0 * γ1 * ∂₁ψ
            -γ0 * γ2 * ∂₂ψ
            -γ0 * γ3 * ∂₃ψ
        ) * dt
        @inbounds for l in 1:4
            # Update ψ
            ψ[i,j,k,l] += dpsi[l]
        end
        
        # ∂ⱼ∂ₜAʲ = (
        #     ∂A[i+1,j,k,1,2] - ∂A[i,j,k,1,2]
        #     + ∂A[i,j+1,k,1,3] - ∂A[i,j,k,1,3]
        #     + ∂A[i,j,k+1,1,4] - ∂A[i,j,k,1,4]
        # ) / dx
        # # Laplacian(A) = -qψ†ψ - ∂ⱼ∂ₜAʲ
        # lapAᵗ = -qₑ * psi' * psi - ∂ⱼ∂ₜAʲ

        # Aᵗ_new = (
        #     # Aᵗ of all neighbors
        #     A[i-1,j,k,1] + A[i+1,j,k,1] +
        #     A[i,j-1,k,2] + A[i,j+1,k,2] +
        #     A[i,j,k-1,3] + A[i,j,k+1,3]
        #     - lapAᵗ * dx^2
        # ) / 6.0

        
        # ∂ₜ²A¹ = (
        #     # qψ†γᵗγʲψ
        #     qₑ * (psi' * γ0 * γ1 * psi)
        #     # - ∂₁(∂ᵤAᵘ))
        #     - (trace4x4(@view ∂A[i+1,j,k,:,:]) - trace4x4(@view ∂A[i,j,k,:,:])) / dx
        #     # + lap A¹
        #     + (A[i-1,j,k,2] + A[i+1,j,k,2] +
        #        A[i,j-1,k,2] + A[i,j+1,k,2] +
        #        A[i,j,k-1,2] + A[i,j,k+1,2]
        #        - 6.0 * A[i,j,k,2]) / dx^2
        # )
        # ∂ₜ²A² = (
        #     # qψ†γᵗγʲψ
        #     qₑ * (psi' * γ0 * γ2 * psi)
        #     # - ∂₂(∂ᵤAᵘ))
        #     - (trace4x4(@view ∂A[i,j+1,k,:,:]) - trace4x4(@view ∂A[i,j,k,:,:])) / dx
        #     # + lap A²
        #     + (A[i-1,j,k,3] + A[i+1,j,k,3] +
        #        A[i,j-1,k,3] + A[i,j+1,k,3] +
        #        A[i,j,k-1,3] + A[i,j,k+1,3]
        #        - 6.0 * A[i,j,k,3]) / dx^2
        # )
        # ∂ₜ²A³ = (
        #     # qψ†γᵗγʲψ
        #     qₑ * (psi' * γ0 * γ3 * psi)
        #     # - ∂₃(∂ᵤAᵘ))
        #     - (trace4x4(@view ∂A[i,j,k+1,:,:]) - trace4x4(@view ∂A[i,j,k,:,:])) / dx
        #     # + lap A³
        #     + (A[i-1,j,k,4] + A[i+1,j,k,4] +
        #        A[i,j-1,k,4] + A[i,j+1,k,4] +
        #        A[i,j,k-1,4] + A[i,j,k+1,4]
        #        - 6.0 * A[i,j,k,4]) / dx^2
        # )

        # ∂ₜAᵗ = (Aᵗ_new - A[i,j,k,1]) / dt
        # # Update A
        # A[i,j,k,1] = Aᵗ_new
        # A[i,j,k,2] += ∂A[i,j,k,1,2] * dt
        # A[i,j,k,3] += ∂A[i,j,k,1,3] * dt
        # A[i,j,k,4] += ∂A[i,j,k,1,4] * dt
        
        # # Update ∂ₜAᵗ
        # ∂A[i,j,k,1,1] = ∂ₜAᵗ
        # # Update ∂ₜA
        # ∂A[i,j,k,1,2] += ∂ₜ²A¹ * dt
        # ∂A[i,j,k,1,3] += ∂ₜ²A² * dt
        # ∂A[i,j,k,1,4] += ∂ₜ²A³ * dt
    end
    return
end

function snapshot!(markers, ψh)
    # Create colors with varying transparency - explicitly use zeros for RGB (black)
    # get 4-component Dirac spinor at each point and calculate inner product
    cols = zeros(RGBAf, size(ψh,1), size(ψh,2), size(ψh,3))


    for i in 1:size(ψh,1), j in 1:size(ψh,2), k in 1:size(ψh,3)
        ψh_point = ψh[i,j,(size(ψh,3)-k+1),:]
        mass = real(ψh_point' * γ0 * ψh_point)
        intensity = abs(mass)
        

        alpha = min(1.0f0, 200*intensity / (2 * mₑ))
        cols[i,j,k] = RGBAf(mass > 0 ? 1.0f0 : 0.0f0, 0.0f0, mass < 0 ? 1.0f0 : 0.0f0, alpha)

    end
    
    # Try direct attribute access
    markers.color = vec(cols)
    return
end 


############ Time integration + recording ################
iter_num = ceil(Int,nsteps/frame_every)
p = ProgressMeter.Progress(iter_num)
GLMakie.record(fig, "qed3d.mp4", 1:iter_num) do frame
    for _ in 1:frame_every
        # @cuda threads=threads blocks=blocks update∂ⱼA!(A, ∂A, dx)
        @cuda threads=threads blocks=blocks update!(ψ, A, ∂A, dx)
    end
    # copy to CPU & refresh plot every few steps
    ψh = Array(ψ)
    snapshot!(markers, ψh)
    ProgressMeter.next!(p)
end
ProgressMeter.finish!(p)
