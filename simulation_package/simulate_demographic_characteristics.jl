module ExogeneousDemographicCharacteristics

using Distributions, Random

export generate_state_level_probabilities

# Function to generate state-level probabilities for Low Homogeneity
function get_base_probabilities_low(n_categories::Int64)
    probs = ones(n_categories) / n_categories
    return probs
end

# Function to generate state-level probabilities for Medium Homogeneity (Nominal)
function get_base_probabilities_medium_nominal(n_categories::Int64)

    n_prominent = min(3, n_categories)  # Choose 2 or 3 categories
    prominent_categories = randperm(n_categories)[1:n_prominent]
    n_non_prominent = n_categories - n_prominent
    # Assign weights to prominent categories
    α_prominent = fill(2.0, n_prominent)
    ω_prominent = rand(Dirichlet(α_prominent))
    # Assign weights to non-prominent categories
    if n_non_prominent > 0
        α_non_prominent = ones(n_non_prominent)
        ω_non_prominent = rand(Dirichlet(α_non_prominent))
    else
        ω_non_prominent = Float64[]
    end
    # Combine and normalize

    probs = zeros(n_categories)
    idx = 1
    for cat in 1:n_categories
        if cat in prominent_categories
            probs[cat] = ω_prominent[findfirst(==(cat), prominent_categories)]
        else
            if n_non_prominent > 0
                probs[cat] = ω_non_prominent[idx]
                idx += 1
            end
        end
    end
    probs /= sum(probs)
    return probs

end

# Function to generate state-level probabilities for Medium Homogeneity (Ordinal)
function get_base_probabilities_medium_ordinal(n_categories::Int64)

    peak_category = rand(1:n_categories)
    distances = abs.(collect(1:n_categories) .- peak_category)
    base_ω = exp.(-distances / 2.0)
    noise = rand(n_categories) .* 0.4 .+ 0.8  # Uniform(0.8, 1.2)
    ω = base_ω .* noise
    probabilities = ω / sum(ω)

    return probabilities
end

# Function to generate state-level probabilities for High Homogeneity (Ordinal)
function get_base_probabilities_high(n_categories::Int64)

    peak_category = rand(1:n_categories)
    peak_prob = rand(Uniform(0.70, 0.85))
    non_peak_categories = setdiff(1:n_categories, [peak_category])
    remaining_prob = 1.0 - peak_prob
    n_non_peak = n_categories - 1

    if n_non_peak > 0
        α_non_peak = ones(n_non_peak)
        ω_non_peak = rand(Dirichlet(α_non_peak))
        non_peak_probs = ω_non_peak .* remaining_prob
    else
        non_peak_probs = Float64[]
    end
    probs = zeros(n_categories)
    probs[peak_category] = peak_prob
    idx = 1
    for cat in non_peak_categories
        probs[cat] = non_peak_probs[idx]
        idx += 1
    end
    return probs
end

# Function to generate state-level probabilities for Perfect Homogeneity
function get_base_probabilities_perfect(n_categories::Int64)
    peak_category = rand(1:n_categories)
    probs = zeros(n_categories)
    probs[peak_category] = 1.0
    return probs
end

# Main function to generate state-level probabilities based on homogeneity and characteristic type
function generate_state_level_probabilities(homogeneity::Symbol, characteristic_type::Symbol,
    n_categories::Int64)
    if homogeneity == :low
        return get_base_probabilities_low(n_categories)
    elseif homogeneity == :moderate
        if characteristic_type == :nominal
            return get_base_probabilities_medium_nominal(n_categories)
        elseif characteristic_type == :ordinal
            return get_base_probabilities_medium_ordinal(n_categories)
        else
            error("Invalid characteristic type: $characteristic_type")
        end
    elseif homogeneity == :high
        return get_base_probabilities_high(n_categories)
    elseif homogeneity == :perfect
        return get_base_probabilities_perfect(n_categories)
    else
        error("Invalid homogeneity level: $homogeneity")
    end
end

"""
Vector of vectors of equal length; each vector is the distribution for a given demographic
characteristic
"""
function generate_statewide_distributions(homogeneity::AbstractVector{Symbol},
    characteristic_type::AbstractVector{Symbol}, n_groups::AbstractVector{Int},
)::Vector{Vector{Float64}}

    max_groups = maximum(n_groups)
    begin
        statewide_distributions =
            ExogeneousDemographicCharacteristics.generate_state_level_probabilities.(
                homogeneity, characteristic_type, n_groups
            )
    end

    statewide_distributions = map(
        dist -> vcat(dist, zeros(max_groups - length(dist))), statewide_distributions
    )


    return statewide_distributions

end

end

module EndogeneousDemogprahicCharacteristics

using Random, StaticArrays

export precompute_salience_lookup_table, simulate_agents_with_salience

"""
Simuialtes `num_agent` demographic characteristic for all voters in a state (graph), using probabilities 
    pre-defined categorical distributions for each district (node)

## Arguments:
    `node_dists::Array{Float64, 3}` - A 3D array of size (N, J, K),
        where:
        - `N` is the number of nodes
        - `J` is the number of characteristics
        - `K` is the maximum number of groups across all characteristics
    num_agents::Int - The number of agents to simulate per node
    rng::AbstractRNG - Random number generator (default: MersenneTwister)

## Returns:
    `agents::Array{Int64, 3}` - A 3D array of size (N, num_agents, J),
        where:
        - Each entry corresponds to the group of an agent for a specific characteristic.
"""
function simulate_voter_demographics(node_dists::Array{Float64,3}, num_agents::Int64,
    rng::AbstractRNG)

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

# salience table for population of 10,000, 
function precompute_salience_lookup_table(salience_probs::SMatrix{J,4,Float64}) where {J}
    """
    Precomputes a static lookup table for salience levels based on salience probabilities.

    Arguments:
        salience_probs::Matrix{Float64} - A 2D array of size (J, 4)

    Returns:
        salience_lookup::SMatrix{J, 10, Int64} - A static lookup table of size (J, 10)
    """
    salience_cumsum_scaled = Int.(round.(Float64, 10 .* cumsum(salience_probs, dims=2)))  # (J, 4)

    salience_lookup = zeros(Int, J, 10)
    for j in 1:J
        for scaled_rand in 1:10
            if scaled_rand <= salience_cumsum_scaled[j, 1]
                salience_lookup[j, scaled_rand] = 0
            elseif scaled_rand <= salience_cumsum_scaled[j, 2]
                salience_lookup[j, scaled_rand] = 1
            elseif scaled_rand <= salience_cumsum_scaled[j, 3]
                salience_lookup[j, scaled_rand] = 2
            else
                salience_lookup[j, scaled_rand] = 3
            end
        end
    end

    return SMatrix{J,10,Int64}(salience_lookup)
end

"""
Simulates agents' demographic groups and salience levels using a static lookup table.

Arguments:
    node_dists::Array{Float64, 3} - A 3D array of size (N, J, K)
    salience_lookup::SMatrix{J, 10, Int} - A static lookup table of size (J, 10)
    num_agents::Int - Number of agents per node
    rng::AbstractRNG - Random number generator (default: MersenneTwister)

Returns:
    agents::Array{Int, 4} - A 4D array of size (N, num_agents, J, 3)

Here, agents[1,1,j,:] gives a vector with the following info:
    [What group in the characteritic you're in,
     How salient your identity is,
     Whether your identity is salient or not]
"""
function simulate_agents_with_salience(
    node_dists::Array{Float64,3},
    salience_lookup::SMatrix{J,10,Int64},
    num_agents::Int64,
    rng::AbstractRNG
) where {J}

    N, _, K = size(node_dists)
    agents = Array{Int}(undef, N, num_agents, J, 3)  # Preallocate storage

    # Preallocate temporary arrays
    random_vals_groups = rand(rng, Float64, num_agents)
    random_vals_salience = rand(rng, Float64, num_agents)
    group_indices = similar(random_vals_groups, Int)
    salience_levels = similar(random_vals_salience, Int)
    salience_indicators = similar(salience_levels, Int)

    @inbounds for node in 1:N
        for characteristic in 1:J
            # Compute cumulative probabilities for groups
            group_cumsum = cumsum(view(node_dists, node, characteristic, :))  # (K,)

            # Generate random values for all agents at once
            rand!(rng, random_vals_groups)
            rand!(rng, random_vals_salience)

            # Vectorized assignment for group indices using @simd
            @simd for agent in 1:num_agents
                group_indices[agent] = searchsortedfirst(group_cumsum, random_vals_groups[agent])
            end
            agents[node, :, characteristic, 1] .= group_indices

            # Vectorized assignment for salience levels using the lookup table
            @simd for agent in 1:num_agents
                # Scale random value to integer [1,10]
                scaled_rand = clamp(Int(round(Int, random_vals_salience[agent] * 10)), 1, 10)
                salience_levels[agent] = salience_lookup[characteristic, scaled_rand]
            end
            agents[node, :, characteristic, 2] .= salience_levels

            # Vectorized assignment for salience indicators
            @simd for agent in 1:num_agents
                salience_indicators[agent] = salience_levels[agent] > 0 ? 1 : 0
            end
            agents[node, :, characteristic, 3] .= salience_indicators
        end
    end

    return agents
end

end

module TestSimulateDems

using ..EndogeneousDemogprahicCharacteristics

using Random, StaticArrays

salience_probs = SMatrix{5,4,Float64}([
    0.1 0.2 0.5 0.2;  # Characteristic 1
    0.3 0.3 0.3 0.1;  # Characteristic 2
    0.2 0.3 0.4 0.1;  # Characteristic 3
    0.25 0.25 0.25 0.25; # Characteristic 4
    0.05 0.15 0.60 0.20  # Characteristic 5
])

salience_lookup_static = precompute_salience_lookup_table(salience_probs)

# Define simulation parameters
N, J, K = 100, 5, 4
num_agents = 10_000
rng = MersenneTwister(42)

# Generate node distributions
node_dists = rand(Float64, N, J, K)
for node in 1:N, char in 1:J
    node_dists[node, char, :] /= sum(node_dists[node, char, :])  # Normalize
end

simulate_agents_with_salience(node_dists, salience_lookup_static, num_agents, rng)

end

