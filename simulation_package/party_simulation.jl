
############################################################
# Pass 5: Julia Detailed Implementation
############################################################

module PartySimulation

using LinearAlgebra
using Random
using MultivariateStats
using Clustering
using Statistics
using StatsBase
using Distributions
using Distances
using StaticArrays

using ..HelpfulFunctions
export structured_noise_parties,
    compute_engagement,
    remodel_issue_space,
    dimension_reduce_and_cluster

"""
build_covariances(ideal_point_variances::Vector{Vector{Float64}})

Given an `I`-vector of variance vectors, build an `I`-vector of diagonal covariance matrices,
where each matrix has dimension `d_i x d_i`. By default, each variance is scaled by `V_P`.
Returns a `Vector{Matrix{Float64}}` of length `I`.
"""
function build_covariances(ideal_point_variances::AbstractVector{Vector{Float64}}, n_issues::Int)

    covariances = Vector{Matrix{Float64}}(undef, n_issues)
    @inbounds for i in 1:n_issues
        varvec = ideal_point_variances[i]  # e.g. [σ²_1, σ²_2, ..., σ²_d_i]
        d_i = length(varvec)
        Sigma_i = zeros(Float64, d_i, d_i)  # allocate the d_i x d_i matrix
        # Fill the diagonal
        @inbounds for k in 1:d_i
            Sigma_i[k, k] = varvec[k] + abs(rand(Normal(0, 1)))
        end
        covariances[i] = Sigma_i
    end
    return SVector{n_issues,Matrix{Float64}}(covariances)
end

"""
    structured_noise_generate_party_ideal_points(
        num_parties::Int,
        ideal_point_means::AbstractVector{Vector{Float64}},
        covariances::AbstractVector{Matrix{Float64}},
        n_issues::Int,
        issue_dimensions:AbstractVector{Int}
        rng::AbstractRNG = Random.GLOBAL_RNG
    ) -> Vector{Matrix{Float64}}

Given:
- `num_parties` = P
- `ideal_point_means[i]` = µ_i (d_i-vector)
- `covariances[i]` = Σ_i (d_i x d_i)

Return an `I`-vector of `(P x d_i)` matrices, where row `p` in the `i`-th matrix
is the ideal point of party `p` for issue `i`.
"""
function structured_noise_generate_party_ideal_points(
    n_parties::Int,
    ideal_point_means::AbstractVector{Vector{Float64}},
    covariances::AbstractVector{Matrix{Float64}},
    n_issues::Int,
    issue_dimensions::AbstractVector{Int},
    rng::AbstractRNG
)::AbstractVector{Matrix{Float64}}

    party_positions = Vector{Matrix{Float64}}(undef, n_issues)

    @inbounds for i in 1:n_issues

        μ_i = ideal_point_means[i]      # d_i-vector
        Σ_i = covariances[i]           # d_i x d_i

        # Prepare output for this issue: P x d_i
        issue_mat = Matrix{Float64}(undef, n_parties, issue_dimensions[i])

        # We'll make a Normal distribution for each dimension if we want to do it dimension-wise,
        # but more elegantly, we can sample from a full multivariate normal using the Distributions.jl package:
        mvn = MvNormal(μ_i, Σ_i)

        @inbounds for p in 1:n_parties
            # draw a single sample from the MVN
            issue_mat[p, :] = rand(rng, mvn)
        end

        # Store
        party_positions[i] = issue_mat
    end

    return party_positions
end

function structured_noise_generate_parties(ideal_point_means::AbstractVector{Vector{Float64}},
    ideal_point_variances::AbstractVector{Vector{Float64}},
    n_parties::Int, n_issues::Int, issue_dimensions::AbstractVector{Int},
    rng::AbstractRNG)

    covariances = PartySimulation.build_covariances(ideal_point_variances, n_issues)
    party_ideal_points = PartySimulation.structured_noise_generate_party_ideal_points(n_parties,
        ideal_point_means, covariances, n_issues, issue_dimensions, rng
    )

    return party_ideal_points
end

"""
compute_engagement(pop_issues::Vector{Array{Float64,3}}, alpha::Float64, p::Float64)
-> (engagement, is_political_class)
each is (N,A)

Produces N x A matricies where rows are districts and columns are meaningless 
engagement is a matrix of floats, each entry represents an individuals' computed engagement
is_political_class is a bit matrix, each entry represents whether an individual is political 
"""
function compute_engagement(pop_issues::AbstractVector{Array{Float64,3}}, alpha::Float64,
    p::Float64, n_issues::Int, n_seats::Int, pop_per_seat::Int, rng::AbstractRNG; scale::Bool=true)

    engagement = zeros(Float64, n_seats, pop_per_seat)

    # Minkowski norm
    for n in 1:n_seats
        for a in 1:pop_per_seat
            sum_ = 0.0
            for i in 1:n_issues
                arr_i = pop_issues[i]
                d_i = size(arr_i, 3)
                mag = norm(view(arr_i, n, a, :))
                sum_ += ((mag / sqrt(d_i))^p)
            end
            E_na = sum_^(1 / p)
            engagement[n, a] = E_na
        end
    end

    if scale
        engagement = scale_utilities(engagement)
    end

    # Bernoulli
    is_political_class = falses(n_seats, pop_per_seat)
    for n in 1:n_seats
        for a in 1:pop_per_seat
            prob = alpha * engagement[n, a]
            prob = prob > 1.0 ? 1.0 : prob
            if rand(rng) < prob
                is_political_class[n, a] = true
            end
        end
    end
    return engagement, is_political_class
end

"""
remodel_issue_space(pop_issues) -> (super_issue, index_map)
super_issue: (N*A, sum(d_i))
index_map: Vector of (n,a) pairs

# NEED TO CHECK COMPUTATIONS HERE, MAKE SURE IT'S MAPPING INTO THE SALIENCE SPACE CORRECTLY
"""

function remodel_issue_space(pop_issues::AbstractVector{Array{Float64,3}}, n_issues::Int,
    n_seats::Int, pop_per_seat::Int, issue_dimensions::AbstractVector{Int})

    sum_d = sum(issue_dimensions)

    # Initialize super_issue matrix
    super_issue = zeros(Float64, n_seats * pop_per_seat, sum_d)

    # Populate super_issue
    for rowidx in 1:(n_seats*pop_per_seat)
        n = div(rowidx - 1, pop_per_seat) + 1  # Node index
        a = rem(rowidx - 1, pop_per_seat) + 1  # Agent index
        offset = 1
        for i in 1:n_issues
            arr_i = pop_issues[i]
            vec_i = arr_i[n, a, :]  # Extract the vector for node n, agent a, issue i
            mag_i = norm(vec_i)
            scale_i = sqrt(mag_i)
            d_i = size(arr_i, 3)
            @inbounds @simd for k in 1:d_i
                super_issue[rowidx, offset+k-1] = vec_i[k] * scale_i
            end
            offset += d_i
        end
    end

    return super_issue
end

function apply_pca(super_issue_points::Matrix{Float64}, n_issues::Int)
    println("Reducing points")
    X = super_issue_points'
    pca_model = fit(PCA, X; maxoutdim=n_issues)

    # Ensure the output has exactly M dimensions
    reduced_transposed = transform(pca_model, X)  # Reduced data in (M_actual × observations)
    actual_n_issues = size(reduced_transposed, 1)

    if actual_n_issues < n_issues # n issues
        # Add zero-padded dimensions if fewer than M components are returned
        padding = zeros(n_issues - actual_n_issues, size(reduced_transposed, 2))
        reduced_transposed = vcat(reduced_transposed, padding)
    end

    reduced_points = reduced_transposed'  # Transpose back: rows are observations
    return Matrix{Float64}(reduced_points)
end


function subsample_political_points(
    reduced_points::Matrix{Float64},
    flat_pc::Union{Vector{Bool},BitVector},
    max_sample_size::Int
)
    political_indices = findall(flat_pc)
    total_pop = length(flat_pc)
    num_political = length(political_indices)

    if num_political <= max_sample_size
        # If there are fewer points than the maximum sample size, return all political points
        sampled_indices = sample(1:total_pop, max_sample_size; replace=false)
        return reduced_points[sampled_indices, :], sampled_indices
    else
        # Randomly sample without replacement
        sampled_indices = sample(political_indices, max_sample_size; replace=false)
        return reduced_points[sampled_indices, :], sampled_indices
    end
end


function kmeans_model_selection(sampled_points::Matrix{Float64}, k_range::UnitRange{Int64})
    best_k = nothing
    best_score = -Inf
    best_labels = nothing
    best_centers = nothing

    for k in k_range
        try
            kmeans_result = kmeans(sampled_points', k; maxiter=300)
            cluster_labels = kmeans_result.assignments
            centers = kmeans_result.centers'
            silhouette_scores = silhouettes(cluster_labels, sampled_points'; metric=Euclidean())
            score = mean(silhouette_scores)

            if score > best_score
                best_score = score
                best_k = k
                best_labels = cluster_labels
                best_centers = centers
            end
        catch e
            @warn "Error during clustering or silhouette calculation for k = $k: $e"
        end
    end

    if best_k === nothing
        throw(ArgumentError("No valid clusters could be found within the specified range of k values."))
    end

    return best_labels, best_centers
end

function dimension_reduce_and_cluster_with_kmeans(
    super_issue_points::Matrix{Float64},
    is_political_class::Union{Matrix{Bool},BitMatrix},
    n_issues::Int,
    k_range::UnitRange{Int64}=2:10,
    max_sample_size::Int=5000,
    min_sample_size::Int=5000
)
    # flatten matrix into vector for indexing
    flat_pc = vec(is_political_class)

    # reducing points
    reduced_points = apply_pca(super_issue_points, n_issues)  # Ensure this function is defined!

    # subsampling
    # sampled_indices was meant for labeling clusters
    sampled_points, sampled_indices = subsample_political_points(
        reduced_points, flat_pc, max_sample_size
    )

    # for edge cases
    if size(sampled_points, 1) < min_sample_size
        println("Not enough points to cluster. Returning defaults.")
        full_labels = rand([-1, 1], size(reduced_points, 1))
        party_positions = Matrix{Float64}(undef, 0, n_issues)
        return reduced_points, full_labels, party_positions
    end

    # iterative k-means clustering
    # political_labels was meant for labeling clusters
    political_labels, party_positions = kmeans_model_selection(sampled_points, k_range)

    return reduced_points, party_positions
end

"""
    recover_3d_after_pca(reduced_points, N, A)

`reduced_points` must be size (N*A) x M, where M is the new dimension
(e.g., after PCA). Returns a (N, A, M) array.
"""
@inline function recover_3d_after_pca(reduced_points::Matrix{Float64}, n_issues::Int,
    n_seats::Int, pop_per_seat::Int)

    # Step 1: reshape to (A, N, M) so that dimension 1 = 'a' (the fastest index)
    tmp = reshape(reduced_points, (pop_per_seat, n_seats, n_issues))

    # Step 2: permute to get (N, A, M)
    return permutedims(tmp, (2, 1, 3))
end

end