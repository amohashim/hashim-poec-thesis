module SpatialCharacteristics

using Random
using LinearAlgebra
using Statistics
using Distances

export SpatialAutocorrelationMeasurement, generate_spatial_characteristics
export district_entropies, compute_mantel_spatial_correlation

struct SpatialAutocorrelationMeasurement

    aggregated_distance_corr::Float64
    aggregate_distance_corr_p_val::Float64
    characteristic_level_corr::Vector{Float64}
    char_level_corr_p_val::Vector{Float64}
    average_entropies::Vector{Float64} # averaged over districts, for each characteristic
    median_entropies::Vector{Float64}
    sd_entropies::Vector{Float64}

end

function generate_cluster_centers(K::Int, sigma_c_prime::Float64, rng::AbstractRNG; max_tries::Int=10000)
    centers = Matrix{Float64}(undef, K, 2)
    found = 0
    tries_count = 0
    while (found < K) && (tries_count < max_tries)
        candidate = rand(rng, 2)  # Generate a candidate point in the [0, 1]x[0, 1] space
        ok = true
        for i in 1:found
            if norm(candidate .- centers[i, :]) < sigma_c_prime
                ok = false
                break
            end
        end
        if ok
            found += 1
            centers[found, :] = candidate
        end
        tries_count += 1
    end
    if found < K
        for i in (found+1):K
            centers[i, :] = rand(rng, 2)
        end
    end
    return centers
end

function place_nodes(N::Int, U::Float64, centers::Matrix{Float64}, sigma_c::Float64, rng::AbstractRNG)
    coords = zeros(Float64, N, 2)
    is_urban = fill(false, N)
    K = size(centers, 1)
    n_urban = floor(Int, U * N)
    idx = collect(1:N)
    shuffle!(rng, idx)
    urban_idx = idx[1:n_urban]
    rural_idx = idx[n_urban+1:end]

    for i in urban_idx
        is_urban[i] = true
        attempts = 0
        while true
            c_idx = rand(rng, 1:K)
            meanx, meany = centers[c_idx, :]
            candx = randn(rng) * sigma_c + meanx
            candy = randn(rng) * sigma_c + meany
            if (0.0 <= candx <= 1.0) && (0.0 <= candy <= 1.0)
                coords[i, 1] = candx
                coords[i, 2] = candy
                break
            end
            attempts += 1
            if attempts > 1000
                coords[i, 1] = rand(rng)
                coords[i, 2] = rand(rng)
                break
            end
        end
    end
    for i in rural_idx
        coords[i, 1] = rand(rng)
        coords[i, 2] = rand(rng)
    end
    return coords, is_urban
end

function build_adjacency(coords::Matrix{Float64}; delta::Union{Nothing,Float64}=0.1, k::Union{Nothing,Int}=nothing)
    N = size(coords, 1)
    adjacency = zeros(Float64, N, N)
    # vectorized distance
    dmat = pairwise(Euclidean(), coords')
    if delta !== nothing
        @inbounds for i in 1:N
            for j in i+1:N
                if dmat[i, j] <= delta
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0
                end
            end
        end
    elseif k !== nothing
        @inbounds for i in 1:N
            row = dmat[i, :]
            idx_s = sortperm(row)
            nn = idx_s[2:(k+1)]
            for j in nn
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
            end
        end
    else
        error("build_adjacency: Must provide either delta or k.")
    end
    return adjacency
end

function build_node_distributions(
    coords::AbstractMatrix{Float64},
    statewide_probs::AbstractVector{Vector{Float64}},
    A_values::AbstractVector{Float64},
    rng::AbstractRNG;
    alpha::Float64=1.0,
    beta::Float64=1.0,
    center_and_scale::Bool=true
)
    N = size(coords, 1)
    J = length(statewide_probs)
    max_k = maximum(length.(statewide_probs))
    node_dists = fill(0.0, N, J, max_k)
    dmat = pairwise(Euclidean(), coords')

    @inbounds for j_idx in 1:J
        p_j = statewide_probs[j_idx]
        k_j = length(p_j)
        A_j = A_values[j_idx]
        phi_j = (A_j == 0) ? 1e12 : 1.0 / A_j

        # Build covariance matrix
        cov_mat = Array{Float64}(undef, N, N)
        @inbounds for i in 1:N
            for m in 1:N
                cov_mat[i, m] = (A_j == 0) ? ((i == m) ? 1e-9 : 0.0) : exp(-dmat[i, m] / phi_j)
            end
        end
        for i in 1:N
            cov_mat[i, i] += 1e-9
        end

        # Cholesky decomposition
        L = cholesky(Symmetric(cov_mat)).L

        # Sample latent fields
        Z_fields = zeros(Float64, N, k_j)
        @inbounds for r in 1:k_j
            z_r = randn(rng, N)
            Z_fields[:, r] .= L * z_r
        end

        # Center and scale if requested
        if center_and_scale
            @inbounds for r in 1:k_j
                col = Z_fields[:, r]
                m_c = mean(col)
                s_c = std(col) + 1e-12
                @inbounds for i in 1:N
                    col[i] = (col[i] - m_c) / s_c
                end
            end
        end

        # Compute log-baseline
        log_baseline = [alpha * log(x + 1e-12) for x in p_j]

        # Compute final distribution
        @inbounds for i in 1:N
            extended = Vector{Float64}(undef, k_j)
            @inbounds for r in 1:k_j
                extended[r] = beta * Z_fields[i, r] + log_baseline[r]
            end
            max_val = maximum(extended)
            exps = [exp(x - max_val) for x in extended]
            denom = sum(exps)
            @inbounds for r in 1:k_j
                node_dists[i, j_idx, r] = exps[r] / denom
            end
        end
    end

    return node_dists
end

"""
Array with the following dimensions:
N x C x d_max

coords is an N x 2 matrix

where N is the number of districts, C is the number of charactersitics, and d_max is the maximum
number of groups

"""
function generate_spatial_characteristics(n_metros::Int, spatial_dispersion::Float64,
    n_seats::Int, urbanization::Float64, urban_sprawl::Float64, a_vals::AbstractVector{Float64},
    statewide_distributions::Vector{Vector{Float64}}, rng::AbstractRNG
)

    centers = SpatialCharacteristics.generate_cluster_centers(n_metros, spatial_dispersion, rng)
    coords, is_urban = SpatialCharacteristics.place_nodes(n_seats, urbanization, centers,
        urban_sprawl, rng)
    node_dists = SpatialCharacteristics.build_node_distributions(coords, statewide_distributions,
        a_vals, rng)
    adjacency = build_adjacency(coords; delta=0.2)


    return node_dists, coords

end


function test_spatial_fourthpass()
    rng = MersenneTwister(42)
    K = 3
    sigma_c_prime = 0.1
    centers = generate_cluster_centers(K, sigma_c_prime, rng)
    coords, is_urban = place_nodes(100, 0.5, centers, 0.01, rng)
    adjacency = build_adjacency(coords; delta=0.2)
    statewide_probs = [[0.4, 0.3, 0.1, 0.2], [0.4, 0.4, 0.2], [0.5, 0.5]]
    A_values = [2.0, 2.0, 5.0]
    node_dists = build_node_distributions(coords, statewide_probs, A_values, rng)
    # println("Test done. adjacency size: ", size(adjacency))
    # println("Node dist example: ", node_dists[1,1,:])
end

test_spatial_fourthpass()

using Statistics
using Random

# 1) KL-divergence helper for Jensen-Shannon
@inline function _kl_divergence(p::AbstractVector{Float64},
    q::AbstractVector{Float64})::Float64
    s = 0.0
    @inbounds for k in 1:length(p)
        pk = p[k]
        if pk > 0
            qk = q[k]
            @assert qk > 0.0 "Encountered nonzero p[k] with zero M[k]."
            s += pk * log(pk / qk)
        end
    end
    return s
end

# 2) Jensen-Shannon Divergence for probability vectors
@inline function jensen_shannon(p::AbstractVector{Float64},
    q::AbstractVector{Float64})::Float64
    @assert length(p) == length(q)
    len = length(p)
    M = Vector{Float64}(undef, len)
    @inbounds for i in 1:len
        M[i] = 0.5 * (p[i] + q[i])
    end
    return 0.5 * _kl_divergence(p, M) + 0.5 * _kl_divergence(q, M)
end

# 3) Pairwise Euclidean distances on N×2 coords
function compute_spatial_distance(coords::Matrix{Float64})
    @assert size(coords, 2) == 2 "coords must be N×2"
    N = size(coords, 1)
    Dgeo = Matrix{Float64}(undef, N, N)
    @inbounds for i in 1:N
        Dgeo[i, i] = 0.0
        xi, yi = coords[i, 1], coords[i, 2]
        for j in i+1:N
            xj, yj = coords[j, 1], coords[j, 2]
            dx = xi - xj
            dy = yi - yj
            dist = sqrt(dx * dx + dy * dy)
            Dgeo[i, j] = dist
            Dgeo[j, i] = dist
        end
    end
    return Dgeo
end

# 4) Compute per-characteristic distance matrices and an aggregated matrix
function compute_demo_distances(
    district_dists::Array{Float64,3},
    n_groups::AbstractVector{Int},
    metric::Function=jensen_shannon
)
    @assert ndims(district_dists) == 3
    N, C, Gmax = size(district_dists)
    @assert length(n_groups) == C

    # We'll create a vector of NxN matrices for each characteristic
    dist_by_char = [Matrix{Float64}(undef, N, N) for _ in 1:C]
    dist_agg = Matrix{Float64}(undef, N, N)
    fill!(dist_agg, 0.0)

    @inbounds for c in 1:C
        Gc = n_groups[c]
        Dc = dist_by_char[c]

        # Build pairwise distances for this characteristic
        @inbounds for i in 1:N
            Dc[i, i] = 0.0
            pi = view(district_dists, i, c, 1:Gc)
            for j in i+1:N
                pj = view(district_dists, j, c, 1:Gc)
                dval = metric(pi, pj)
                Dc[i, j] = dval
                Dc[j, i] = dval
            end
        end

        # Accumulate into dist_agg
        @inbounds for i in 1:N
            for j in i+1:N
                dist_agg[i, j] += Dc[i, j]
                dist_agg[j, i] += Dc[i, j]
            end
        end
    end

    # Take average across characteristics
    invC = 1.0 / C
    @inbounds for i in 1:N
        for j in i:N
            dist_agg[i, j] *= invC
            dist_agg[j, i] = dist_agg[i, j]
        end
    end

    return dist_by_char, dist_agg
end

# 5) Mantel test for correlation between two NxN distance matrices
function mantel_test(
    D1::AbstractMatrix{<:Real},
    D2::AbstractMatrix{<:Real};
    permutations::Int,
    rng::AbstractRNG
)
    @assert size(D1) == size(D2)
    N = size(D1, 1)
    # Flatten upper triangle
    n_pairs = (N * (N - 1)) >>> 1  # N*(N-1)//2
    distvec1 = Vector{Float64}(undef, n_pairs)
    distvec2 = Vector{Float64}(undef, n_pairs)

    idx = 1
    @inbounds for i in 1:N-1
        for j in i+1:N
            distvec1[idx] = D1[i, j]
            distvec2[idx] = D2[i, j]
            idx += 1
        end
    end

    obs_r = cor(distvec1, distvec2)

    # Permutation test
    count_extreme = 0
    permdist = Vector{Float64}(undef, n_pairs)

    for _ in 1:permutations
        p = randperm(rng, N)
        idx = 1
        @inbounds for i in 1:N-1
            for j in i+1:N
                permdist[idx] = D1[p[i], p[j]]
                idx += 1
            end
        end
        this_r = cor(permdist, distvec2)
        if abs(this_r) >= abs(obs_r)
            count_extreme += 1
        end
    end

    p_value = (count_extreme + 1.0) / (permutations + 1.0)
    return obs_r, p_value
end

"""
    compute_mantel_spatial_correlation(
        district_dists::Array{Float64,3},
        n_groups::AbstractVector{Int},
        coords::Matrix{Float64},
        permutations::Int,
        rng::AbstractRNG
    ) -> NamedTuple

Runs the entire pipeline:
1. Compute NxN distances among demographic distributions (per characteristic + aggregated).
2. Compute NxN distances among districts (geographic).
3. Mantel test comparing dist_agg to Dgeo => (r_agg, p_agg).
4. Mantel test comparing dist_by_char[c] to Dgeo for each characteristic => (r_chars[c], p_chars[c]).

Returns a named tuple with:
  :r_agg, :p_agg,
  :r_chars, :p_chars,
  :dist_agg, :dist_by_char, :Dgeo
"""
function compute_mantel_spatial_correlation(
    district_dists::Array{Float64,3},
    n_groups::AbstractVector{Int},
    coords::Matrix{Float64},
    permutations::Int,
    rng::AbstractRNG
)
    # 1) Demographic distance matrices
    dist_by_char, dist_agg = compute_demo_distances(district_dists, n_groups)

    # 2) Geographic distance
    Dgeo = compute_spatial_distance(coords)

    # 3) Mantel test for aggregated
    r_agg, p_agg = mantel_test(dist_agg, Dgeo; permutations=permutations, rng=rng)

    # 4) Mantel test for each characteristic
    C = length(n_groups)
    r_chars = Vector{Float64}(undef, C)
    p_chars = Vector{Float64}(undef, C)
    for c in 1:C
        rc, pc = mantel_test(dist_by_char[c], Dgeo; permutations=permutations, rng=rng)
        r_chars[c] = rc
        p_chars[c] = pc
    end

    return (
        r_agg=r_agg,
        p_agg=p_agg,
        r_chars=r_chars,
        p_chars=p_chars,
        dist_agg=dist_agg,
        dist_by_char=dist_by_char,
        Dgeo=Dgeo
    )
end

function shannon_entropy(p::Vector{Float64})
    H = 0.0
    @inbounds for pk in p
        if pk > 0
            H -= pk * log(pk)
        end
    end
    return H
end

# Suppose 'distmat' is a D x K matrix where distmat[d, :] is the distribution
# for district d over K categories. We'll compute a vector of entropies.
function district_entropies(distmat::AbstractMatrix{Float64})
    D, K = size(distmat)
    ent = Vector{Float64}(undef, D)
    for d in 1:D
        p = distmat[d, :]
        ent[d] = shannon_entropy(p)
    end
    return ent
end

function compute_district_entropies_over_N(array::Array{Float64,3})
    N, C, G = size(array)
    # Compute the entropies for each district (N dimension)
    entropies = Vector{Float64}(undef, N)
    for n in 1:N
        slice = view(array, n, :, :)  # Extract C×G slice for district n
        entropies[n] = sum(district_entropies(slice))
    end
    return entropies
end


end