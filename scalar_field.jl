############################################################
# phi_gpu_animation.jl — 3‑D lattice evolution on the GPU  #
############################################################
using CUDA           # GPU computing
using GLMakie        # high‑performance 3‑D plotting
using ColorSchemes
using GeometryBasics

const Nx, Ny, Nz = 64, 64, 64          # grid size
threads = (4,4,4)                                # 512 threads per block
blocks  = (cld(Nx,threads[1]),
           cld(Ny,threads[2]),
           cld(Nz,threads[3]))                     # as many blocks as needed
############ Physical & numerical constants #############
const dx  = 0.1f0                      # spacing
const dt  = 0.001f0                    # time step
const nsteps      = 2000                # total steps
const frame_every = 5                  # dump every n steps

############ 2. Allocate GPU arrays ########################
ϕ   = CUDA.zeros(Float32, Nx, Ny, Nz)
dϕ  = CUDA.zeros(Float32, Nx, Ny, Nz)  # time derivative

############ 3. Initial condition (Gaussian pulse) #########
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
σ = 2f0
A = 10f0
# @cuda threads=threads blocks=blocks init_gaussian!(ϕ, σ, A, 0, 0, 0)
@cuda threads=threads blocks=blocks init_gaussian!(dϕ, σ, A * sqrt(2), 10, 0, 0)
@cuda threads=threads blocks=blocks init_gaussian!(dϕ, σ, - A * sqrt(2), - 10, 0, 0)

synchronize()

############ 4. Finite‑difference kernel ###################
function update!(dϕ, ϕ, dx2, m2)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    k = (blockIdx().z-1)*blockDim().z + threadIdx().z
    nx, ny, nz = size(ϕ)
    if 2 ≤ i ≤ nx-1 && 2 ≤ j ≤ ny-1 && 2 ≤ k ≤ nz-1
        lap = (ϕ[i-1,j,k] - 2f0*ϕ[i,j,k] + ϕ[i+1,j,k] +
               ϕ[i,j-1,k] - 2f0*ϕ[i,j,k] + ϕ[i,j+1,k] +
               ϕ[i,j,k-1] - 2f0*ϕ[i,j,k] + ϕ[i,j,k+1]) / dx2
        dϕ[i,j,k] += (lap - m2*ϕ[i,j,k]) * dt      # ∂ₜϕ update
    end
    return
end

############ 5. Makie set‑up ################################
fig = Figure(size = (900, 900))
ax  = Axis3(fig[1,1]; 
    perspectiveness=0.8, 
    title="ϕ field evolution",
    limits=(1, Nx, 1, Ny, 1, Nz)  # Set explicit limits
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
points = array_to_points(Array(ϕ))
# Create explicit black colors with zero alpha
init_colors = [RGBAf(0.0f0, 0.0f0, 0.0f0, 0.0f0) for _ in 1:length(points)]
markers = scatter!(ax, points; 
    markersize = 4,
    color = init_colors,
    transparency = true
)

# Set initial camera position

# ax.scene.camera.eyeposition[] = Vec3f(Nx/2, Ny/2, Nz*2)
# cam.lookat = Vec3f(Nx/2, Ny/2, Nz/2)

# Helper that converts GPU array → positions + colors
function snapshot!(markers, ϕh)
    # Create colors with varying transparency - explicitly use zeros for RGB (black)
    cols = [RGBAf(ϕh > 0 ? 0.0f0 : 1.0f0, 0.0f0, ϕh < 0 ? 0.0f0 : 1.0f0, abs(ϕh)) for ϕh in vec(ϕh)]
    # Try direct attribute access
    markers.color = cols
    return
end

############ 6. Time integration + recording ################
GLMakie.record(fig, "phi3d.gif", 1:ceil(Int,nsteps/frame_every)) do frame
    for _ in 1:frame_every
        # (a) compute ∂ₜϕ
        @cuda threads=threads blocks=blocks update!(dϕ, ϕ, dx^2, m^2)
        # (b) integrate ϕ ← ϕ + dϕ*dt
        @. ϕ  = ϕ + dϕ*dt        # broadcast runs on GPU automatically :contentReference[oaicite:3]{index=3}
    end
    # copy to CPU & refresh plot every few steps
    ϕh = Array(ϕ)
    snapshot!(markers, ϕh)
end
