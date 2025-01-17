module SimulationImplementation

using StatsBase  # Import StatsBase for Weights
using Random

function simulate_agent_demographics(node_dists::Array{Float64,3}, num_agents::Int64;
    rng::AbstractRNG=MersenneTwister(42))
    """
    Simuialtes agents' demographic characteristic

    Arguments:
        node_dists::Array{Float64, 3} - A 3D array of size (N, J, K),
            where:
            - N is the number of nodes
            - J is the number of characteristics
            - K is the maximum number of groups across all characteristics
        num_agents::Int - The number of agents to simulate per node
        rng::AbstractRNG - Random number generator (default: MersenneTwister)

    Returns:
        agents::Array{Int64, 3} - A 3D array of size (N, num_agents, J),
            where:
            - Each entry corresponds to the group of an agent for a specific characteristic.
    """
    N, J, K = size(node_dists)  # Dimensions: nodes (districts), characteristics, groups
    agents = Array{Int64}(undef, N, num_agents, J)  # Preallocate storage

    @inbounds for node in 1:N

        # samples Inverse Transform Sampling Method

        for characteristic in 1:J

            #  compute cumulative probabilities for the given node and characteristic
            cumulative_probs = cumsum(node_dists[node, characteristic, :])
            random_vals = rand(rng, num_agents)

            # samplng agents' groups
            agents[node, :, characteristic] .= searchsortedfirst.(Ref(cumulative_probs), random_vals)
        end
    end

    return agents
end

function test_spatial_fourthpass_optimized()
    rng = MersenneTwister(42)
    K = 3
    sigma_c_prime = 0.1
    centers = generate_cluster_centers(K, sigma_c_prime, rng)
    coords, is_urban = place_nodes(100, 0.5, centers, 0.01, rng)
    adjacency = build_adjacency(coords; delta=0.2)
    statewide_probs = [[0.4, 0.3, 0.1, 0.2], [0.4, 0.4, 0.2], [0.5, 0.5]]
    A_values = [2.0, 2.0, 5.0]
    node_dists = build_node_distributions(coords, statewide_probs, A_values, rng)

    # Simulate agents for 10,000 per node
    num_agents = 10_000
    agents = simulate_agents_optimized(node_dists, num_agents; rng=rng)

    println("First 5 agents in Node 1:")
    println(agents[1, 1:5, :])  # Print the first 5 agents in node 1
end
using Test

function run_tests()
    rng = MersenneTwister(123)

    # Input Parameters
    N = 10  # Number of nodes
    J = 3   # Number of characteristics
    K = 5   # Maximum number of groups per characteristic
    num_agents = 10_000

    # Generate random node distributions
    node_dists = rand(rng, Float64, N, J, K)
    for node in 1:N
        for characteristic in 1:J
            node_dists[node, characteristic, :] .= node_dists[node, characteristic, :] ./ sum(node_dists[node, characteristic, :])
        end
    end

    # Simulate agents
    agents = simulate_agents_optimized(node_dists, num_agents; rng=rng)

    @testset "Simulate Agents Tests" begin

        # Test 1: Basic Shape Test
        @test size(agents) == (N, num_agents, J)

        # Test 2: Range Test
        for node in 1:N
            for characteristic in 1:J
                sampled_groups = agents[node, :, characteristic]
                @test all(sampled_groups .>= 1) && all(sampled_groups .<= K)
            end
        end

        # Test 3: Probability Consistency Test
        tolerance = 0.01  # Allowable deviation
        for node in 1:N
            for characteristic in 1:J
                sampled_groups = agents[node, :, characteristic]
                group_counts = countmap(sampled_groups)  # Count occurrences of each group
                sampled_probs = [get(group_counts, g, 0) / num_agents for g in 1:K]
                input_probs = node_dists[node, characteristic, :]
                @test all(abs.(sampled_probs .- input_probs) .<= tolerance)
            end
        end

        # Test 4: Edge Case - Zero Probabilities
        zero_prob_node_dists = copy(node_dists)
        zero_prob_node_dists[1, 1, 2] = 0.0  # Set group 2 of node 1, characteristic 1 to zero
        zero_prob_node_dists[1, 1, :] .= zero_prob_node_dists[1, 1, :] ./ sum(zero_prob_node_dists[1, 1, :])
        zero_prob_agents = simulate_agents_optimized(zero_prob_node_dists, num_agents; rng=rng)
        sampled_groups = zero_prob_agents[1, :, 1]
        @test 2 ∉ sampled_groups  # Group 2 should never appear

        # Test 5: Edge Case - Single Group
        single_group_node_dists = ones(Float64, N, J, 1)  # Single group
        single_group_agents = simulate_agents_optimized(single_group_node_dists, num_agents; rng=rng)
        @test all(single_group_agents .== 1)  # All agents must belong to group 1

        # Test 6: Cumulative Probability Test
        for node in 1:N
            for characteristic in 1:J
                cumulative_probs = cumsum(node_dists[node, characteristic, :])
                @test all(diff(cumulative_probs) .>= 0)  # Non-decreasing
                @test abs(cumulative_probs[end] - 1.0) < 1e-8  # Sum to 1
            end
        end

    end
end
# run_tests()


end