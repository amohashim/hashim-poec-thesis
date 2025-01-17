
############################################################
# Pass 5: Julia Detailed Implementation
############################################################

module PartySimulation

using LinearAlgebra, Random, MultivariateStats, Clustering, Statistics, StatsBase
using Distributions, StaticArrays
# We'll use these packages if they're installed:
# ] add MultivariateStats, Clustering
# import MultivariateStats: fit, transform, PCA
# import Clustering: dbscan

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
function build_covariances(ideal_point_variances::Vector{Vector{Float64}}, n_issues::Int)

    covariances = Vector{Matrix{Float64}}(undef, n_issues)
    @inbounds for i in 1:n_issues
        varvec = ideal_point_variances[i]  # e.g. [σ²_1, σ²_2, ..., σ²_d_i]
        d_i = length(varvec)
        Sigma_i = zeros(Float64, d_i, d_i)  # allocate the d_i x d_i matrix
        # Fill the diagonal
        @inbounds for k in 1:d_i
            Sigma_i[k, k] = varvec[k]
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
    rng::AbstractRNG=Random.GLOBAL_RNG
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

"""
structured_noise_parties(pop_issues, cov_mats)

pop_issues: Vector{Array{Float64,3}}, length M
  pop_issues[i] has shape (N, A, d_i)
cov_mats: Vector{Vector{Vector{Float64}}}, length M
  cov_mats[i] is length P
  each cov_mats[i][p] is a d_i-length vector representing diag. variances
returns: structured_parties[i] => shape (P, d_i)
"""
function structured_noise_parties(pop_issues::AbstractVector{Array{Float64,3}},
    cov_mats::AbstractVector; rng=MersenneTwister(42))
    M = length(pop_issues)
    structured_parties = Vector{Matrix{Float64}}(undef, M)
    for i in 1:M
        arr = pop_issues[i]   # (N, A, d_i)
        (N, A, d_i) = size(arr)
        # flatten
        flatten_i = similar(arr, N * A, d_i)
        idx = 1
        for n in 1:N
            for a in 1:A
                @inbounds begin
                    @views flatten_i[idx, :] = arr[n, a, :]
                    idx += 1
                end
            end
        end
        # mean
        mean_i = vec(mean(flatten_i, dims=1))  # d_i
        # let P = length(cov_mats[i])
        P = length(cov_mats[i])
        parties_i = zeros(Float64, P, d_i)
        for p in 1:P
            diag_vars = cov_mats[i][p]  # d_i
            offset = Vector{Float64}(undef, d_i)
            for k in 1:d_i
                offset[k] = randn(rng) * sqrt(diag_vars[k])
            end
            parties_i[p, :] = mean_i .+ offset
        end
        structured_parties[i] = parties_i
    end
    return structured_parties
end

"""
compute_engagement(pop_issues::Vector{Array{Float64,3}}, alpha::Float64, p::Float64)
-> (engagement, is_political_class)
each is (N,A)

Produces N x A matricies where rows are districts and columns are meaningless 
engagement is a matrix of floats, each entry represents an individuals' computed engagement
is_political_class is a bit matrix, each entry represents whether an individual is political 
"""
function compute_engagement(pop_issues::Vector{Array{Float64,3}}, alpha::Float64, p::Float64; rng=MersenneTwister(42))
    M = length(pop_issues)
    # assume all have same N, A
    (N, A, dfirst) = size(pop_issues[1])
    engagement = zeros(Float64, N, A)
    for n in 1:N
        for a in 1:A
            sum_ = 0.0
            for i in 1:M
                arr_i = pop_issues[i]
                d_i = size(arr_i, 3)
                mag = norm(view(arr_i, n, a, :))
                sum_ += ((mag / sqrt(d_i))^p)
            end
            E_na = sum_^(1 / p)
            engagement[n, a] = E_na
        end
    end
    # Bernoulli
    is_political_class = falses(N, A)
    for n in 1:N
        for a in 1:A
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

function remodel_issue_space(pop_issues::Vector{Array{Float64,3}})
    M = length(pop_issues)
    # Extract N and A from the first issue
    (N, A, _) = size(pop_issues[1])

    # Ensure all issues have the same N and A
    for i in 2:M
        current_size = size(pop_issues[i])
        if current_size[1] != N || current_size[2] != A
            throw(ArgumentError("All pop_issues must have the same number of nodes (N) and agents per node (A)."))
        end
    end

    # Determine the total dimensionality after concatenation
    dims = [size(pop_issues[i], 3) for i in 1:M]
    sum_d = sum(dims)

    # Initialize super_issue matrix
    super_issue = zeros(Float64, N * A, sum_d)

    # Populate super_issue
    for rowidx in 1:(N*A)
        n = div(rowidx - 1, A) + 1  # Node index
        a = rem(rowidx - 1, A) + 1  # Agent index
        offset = 1
        for i in 1:M
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

function apply_pca(super_issue_points::Matrix{Float64}, M::Int64)
    println("Reducing points")
    X = super_issue_points'
    pca_model = fit(PCA, X; maxoutdim=M)

    # Ensure the output has exactly M dimensions
    reduced_transposed = transform(pca_model, X)  # Reduced data in (M_actual × observations)
    actual_M = size(reduced_transposed, 1)

    if actual_M < M
        # Add zero-padded dimensions if fewer than M components are returned
        padding = zeros(M - actual_M, size(reduced_transposed, 2))
        reduced_transposed = vcat(reduced_transposed, padding)
    end

    reduced_points = reduced_transposed'  # Transpose back: rows are observations
    return Matrix{Float64}(reduced_points)
end


function subsample_political_points(
    reduced_points::Matrix{Float64},
    flat_pc::Vector{Bool},
    max_sample_size::Int
)
    political_indices = findall(flat_pc)
    num_political = length(political_indices)

    if num_political <= max_sample_size
        # If there are fewer points than the maximum sample size, return all political points
        return reduced_points[political_indices, :], political_indices
    else
        # Randomly sample without replacement
        sampled_indices = sample(political_indices, max_sample_size; replace=false)
        return reduced_points[sampled_indices, :], sampled_indices
    end
end


function kmeans_model_selection(sampled_points::Matrix{Float64}, k_range::UnitRange{Int64}, M::Int64)
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
    is_political_class::Matrix{Bool},
    M::Int64;
    k_range::UnitRange{Int64}=2:10,
    max_sample_size::Int64=5000
)
    # flatten matrix into vector for indexing
    flat_pc = vec(is_political_class)

    # reducing points
    reduced_points = apply_pca(super_issue_points, M)  # Ensure this function is defined!

    # subsampling
    # sampled_indices was meant for labeling clusters
    sampled_points, sampled_indices = subsample_political_points(reduced_points, flat_pc, max_sample_size)

    # for edge cases
    if size(sampled_points, 1) < 2
        println("Not enough points to cluster. Returning defaults.")
        full_labels = fill(-1, size(reduced_points, 1))
        party_positions = Matrix{Float64}(undef, 0, M)
        return reduced_points, full_labels, party_positions
    end

    # iterative k-means clustering
    # political_labels was meant for labeling clusters
    political_labels, party_positions = kmeans_model_selection(sampled_points, k_range, M)

    return reduced_points, party_positions
end

module TestingClustering

using LinearAlgebra, Random, MultivariateStats, Clustering, Statistics, StatsBase, Test
using ..PartySimulation

function run_tests()
    # Example parameters
    N, A, d1, d2 = 100, 5000, 3, 3
    pop_issue1 = randn(N, A, d1)
    pop_issue2 = randn(N, A, d2)
    pop_issues = [pop_issue1, pop_issue2]
    alpha = 0.5
    p = 2.0
    (eng, pc) = compute_engagement(pop_issues, alpha, p)
    pc = Matrix{Bool}(pc)

    # Super issue space and political class
    super_issue_points = remodel_issue_space(pop_issues)
    # Select a few political class members

    # Parameters
    M = 2
    k_range = 2:10
    max_sample_size = 5000

    # Run the updated function
    reduced_points, full_labels, party_positions = dimension_reduce_and_cluster_with_kmeans(
        super_issue_points, pc, M; k_range=k_range, max_sample_size=max_sample_size
    )

    @testset "structured_noise_parties" begin

        # 2 issues
        N, A, d1, d2 = 100, 10000, 2, 2
        # We'll do random
        pop_issue1 = randn(N, A, d1)
        pop_issue2 = randn(N, A, d2)
        pop_issues = [pop_issue1, pop_issue2]

        P = 3
        cov_mats = [
            [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],  # for i=1
            [[0.2, 0.2], [0.2, 0.2], [0.2, 0.2]]   # for i=2
        ]
        parties = structured_noise_parties(pop_issues, cov_mats)
        # @test length(parties) == 2
        # @test size(parties[1]) == (2, 2)  # (P, d1)
        # @test size(parties[2]) == (2, 1)  # (P, d2)
    end

    @testset "compute_engagement" begin

        # 2 issues
        N, A, d1, d2 = 100, 10000, 2, 2

        pop_issue1 = randn(N, A, d1)
        pop_issue2 = randn(N, A, d2)
        pop_issues = [pop_issue1, pop_issue2]
        alpha = 0.5
        p = 2.0
        (eng, pc) = compute_engagement(pop_issues, alpha, p)
        @test size(eng) == (2, 3)
        @test size(pc) == (2, 3)
    end

    @testset "remodel_issue_space" begin
        # 2 issues
        N, A, d1, d2 = 100, 10000, 2, 2

        pop_issue1 = randn(N, A, d1)
        pop_issue2 = randn(N, A, d2)
        pop_issues = [pop_issue1, pop_issue2]
        super_issue = remodel_issue_space(pop_issues)
        @test size(super_issue) == (N * A, d1 + d2)  # e.g. (6,3)
        @test length(index_map) == N * A
    end

    @testset "dimension_reduce_and_cluster" begin

        N, A, d1, d2 = 100, 10000, 2, 2

        pop_issue1 = randn(N, A, d1)
        pop_issue2 = randn(N, A, d2)
        pop_issues = [pop_issue1, pop_issue2]
        super_issue = remodel_issue_space(pop_issues)
        alpha = 0.5
        p = 2.0
        (eng, pc) = compute_engagement(pop_issues, alpha, p)
        M = 2
        reduced_points, labels, party_positions = dimension_reduce_and_cluster(super_issue, pc, M; eps=0.5, min_samples=2)
        @test size(reduced_points) == (N * A, M)
        @test length(labels) == N * A
        @test size(party_positions, 2) == M
    end
end

function test_dimension_reduce_and_cluster_with_kmeans()
    @testset "dimension_reduce_and_cluster_with_kmeans Tests" begin
        # --- Test 1: Basic Functionality ---
        @testset "Basic Functionality" begin
            N, A, d1, d2 = 5, 10, 3, 3
            pop_issue1 = randn(N, A, d1)
            pop_issue2 = randn(N, A, d2)
            pop_issues = [pop_issue1, pop_issue2]
            alpha, p = 0.5, 2.0
            (eng, pc) = compute_engagement(pop_issues, alpha, p)
            pc = Matrix{Bool}(pc)

            # Super issue space
            super_issue_points = remodel_issue_space(pop_issues)

            # Parameters
            M = 2
            k_range = 2:4
            max_sample_size = 50

            # Run the function
            reduced_points, full_labels, party_positions = dimension_reduce_and_cluster_with_kmeans(
                super_issue_points, pc, M; k_range=k_range, max_sample_size=max_sample_size
            )

            # Basic assertions
            @test size(reduced_points, 1) == N * A  # Number of points matches input
            @test size(reduced_points, 2) == M      # Reduced to M dimensions
            @test typeof(full_labels) == Vector{Int64}  # Labels should be a vector of integers
            @test length(full_labels) == size(super_issue_points, 1)  # Match input size
            if size(party_positions, 1) > 0
                @test size(party_positions, 2) == M  # Correct dimensionality for centers
            end
        end

        # --- Test 2: Toy Example with Known Results ---
        @testset "Toy Example with Known Results" begin

            # 10 people, 2 dimensional super-issue
            super_issue_points = [
                1.0 1.0; 2.0 1.0; 1.0 2.0; 2.0 2.0;  # Cluster 1
                10.0 10.0; 11.0 10.0; 10.0 11.0; 11.0 11.0;  # Cluster 2
                5.0 5.0; 6.0 5.0  # Noise
            ]
            is_political_class = Bool[
                true true; true true;
                true true; true true;
                false false
            ]
            M = 2
            k_range = 2:3
            max_sample_size = 10

            reduced_points, full_labels, party_positions = dimension_reduce_and_cluster_with_kmeans(
                super_issue_points, is_political_class, M; k_range=k_range, max_sample_size=max_sample_size
            )

            # Validate reduced points
            @test size(reduced_points) == size(super_issue_points)

            # Validate labels (noise should be -1)
            @test full_labels[1:8] |> all(x -> x >= 0)  # Political points clustered
            @test full_labels[9:10] == [-1, -1]  # Non-political points labeled as noise

            # Validate cluster centers
            @test size(party_positions, 2) == M
        end

        # --- Test 3: Edge Case - No Political Class Members ---
        @testset "No Political Class Members" begin
            N, A, d1, d2 = 5, 10, 3, 3
            pop_issue1 = randn(N, A, d1)
            pop_issue2 = randn(N, A, d2)
            pop_issues = [pop_issue1, pop_issue2]
            alpha, p = 0.0, 2.0  # No political engagement
            (eng, pc) = compute_engagement(pop_issues, alpha, p)
            pc = Matrix{Bool}(pc)

            super_issue_points = remodel_issue_space(pop_issues)

            M = 2
            k_range = 2:4
            max_sample_size = 50

            reduced_points, full_labels, party_positions = dimension_reduce_and_cluster_with_kmeans(
                super_issue_points, pc, M; k_range=k_range, max_sample_size=max_sample_size
            )

            @test all(full_labels .== -1)  # All points should be noise
            @test size(party_positions) == (0, M)  # No clusters
        end

        # --- Test 4: Edge Case - All Political Class Members ---
        @testset "All Political Class Members" begin
            N, A, d1, d2 = 5, 10, 3, 3
            pop_issue1 = randn(N, A, d1)
            pop_issue2 = randn(N, A, d2)
            pop_issues = [pop_issue1, pop_issue2]
            alpha, p = 1.0, 2.0  # All are political
            (eng, pc) = compute_engagement(pop_issues, alpha, p)
            pc = Matrix{Bool}(pc)

            super_issue_points = remodel_issue_space(pop_issues)

            M = 2
            k_range = 2:4
            max_sample_size = 50

            reduced_points, full_labels, party_positions = dimension_reduce_and_cluster_with_kmeans(
                super_issue_points, pc, M; k_range=k_range, max_sample_size=max_sample_size
            )

            @test all(full_labels .>= 0)  # All points should be clustered
            @test size(party_positions, 2) == M  # Correct cluster dimensionality
        end

        # --- Test 5: High-Dimensional Data ---
        @testset "High Dimensional Data" begin
            N, A, d1, d2 = 100, 1000, 3, 3
            pop_issue1 = randn(N, A, d1)
            pop_issue2 = randn(N, A, d2)
            pop_issues = [pop_issue1, pop_issue2]
            alpha, p = 0.5, 2.0
            (eng, pc) = compute_engagement(pop_issues, alpha, p)
            pc = Matrix{Bool}(pc)

            super_issue_points = remodel_issue_space(pop_issues)

            M = 5
            k_range = 2:10
            max_sample_size = 10000

            reduced_points, full_labels, party_positions = dimension_reduce_and_cluster_with_kmeans(
                super_issue_points, pc, M; k_range=k_range, max_sample_size=max_sample_size
            )

            @test size(reduced_points) == (N * A, M)  # Reduced dimensionality
            if size(party_positions, 1) > 0
                @test size(party_positions, 2) == M  # Check party positions
            end
        end
    end
end

end
end
