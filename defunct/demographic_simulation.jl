module DemographicSimulation

using Distributions
using Graphs
using Random
using LinearAlgebra
using StaticArrays
using Parameters

export State, generate_state, assign_characteristics,
    apply_spatial_autocorrelation, apply_urban_heterogeneity,
    simulate_population

const CHARACTERISTIC_ORDER = SVector{7,Symbol}(
    [:age, :gender, :race, :income, :education, :religion, :immigrant]
)

struct ExogeneousParameters
    N_SEATS::Int64                 # Total number of nodes
    N_POP::Int64                   # Total population
    POP_DENSITY::Symbol            # Population Density
    SPATIAL_POL::Symbol            # Spatial Polarization
    SPATIAL_AUTOCOR::Symbol        # Spatial Autocorrelation
    URBAN_HETER::Symbol            # Urban Heterogeneity
    CHARACTERISTIC_ORDER::AbstractVector{Symbol} # Order in which `HOMOGENEITY` levels are defined
    HOMOGENEITY_LEVELS::AbstractVector{Symbol} # ordered by `CHARACTERISTIC_ORDER`
end

struct InitialGraphAttrs

    g::SimpleGraph

    n_clusters::Int64
    intra_cluster_spread::Float64
    epsilon_intra::Float64

    cluster_centers::Matrix{Float64}
    nodes_per_cluster::Vector{Int64}
    nodes::Matrix{Float64}
    cluster_assignments::Vector{Int64}

end

struct ConnectedStateGraph


end

struct ConnectedStateGraphWithDemographicDistributions

    # each dict in `thetas` maps a characteristic (e.g. :age) to the parameters of their 
    # distribution, for a given node (the vector is a vector of nodes)

    g::SimpleGraph # from ConnectedStateGraph
    params::ExogeneousParameters

    # Outter dict is :state or :node
    # Inner dict for :state is :characteristic => Vector(probabilities)
    # Inner dict for :node is :characteristic => Vector( Vector(probabilities) )
    # where each inner vector is the probability dist for node i, where i is the index (within
    # the outter vector) that the inner vector is located at

    distributions::Dict{
        Symbol,Dict{Symbol,Union{Vector{Vector{Float64}},Vector{Float64}}}
    }

    function ConnectedStateGraphWithDemographicDistributions(
        g::SimpleGraph, params::ExogeneousParameters
    )
        dictionary = Dict{Symbol,Dict{Symbol,Union{Vector{Vector{Float64}},Vector{Float64}}}}()

        new(g, params, dictionary)

    end
end

struct DemographicCharacteristic

    characteristic::Symbol      # name, e.g. :race
    n_categories::Int64
    homogeneity::Symbol         # low, moderate, high, or perfect
    scale_type::Symbol          # nominal or ordinal

end

# Data Structures
struct State
    S::Int                         # Total number of nodes
    N::Int                         # Total population
    D::String                      # Population Density
    P::String                      # Spatial Polarization
    H::String                      # Homogeneity Level
    A::String                      # Spatial Autocorrelation
    U::String                      # Urban Heterogeneity
    nodes::Matrix{Float64}         # Node positions (2 x S matrix)
    g::SimpleGraph                 # Graph of the state
    cluster_assignments::Vector{Int} # Cluster assignments for nodes
    theta::Dict{Symbol,Any}       # Node-level characteristic parameters
end

# Helper functions for parameter mappings
function get_n_clusters(params::ExogeneousParameters)

    @unpack SPATIAL_POL, N_SEATS = params

    if SPATIAL_POL == :low
        return N_SEATS
    elseif SPATIAL_POL == :moderate
        return floor(Int64, sqrt(N_SEATS))
    elseif SPATIAL_POL == :high
        return 1
    else
        error("Invalid value for SPATIAL_POL")
    end
end

function get_spread_within_cluster(params::ExogeneousParameters)

    @unpack POP_DENSITY = params
    if POP_DENSITY == :low
        return 0.1
    elseif POP_DENSITY == :moderate
        return 0.05
    elseif POP_DENSITY == :high
        return 0.01
    else
        error("Invalid value for POP_DENSITY")
    end
end

function get_epsilon_intra(params::ExogeneousParameters)

    @unpack POP_DENSITY = params
    if POP_DENSITY == :low
        return 0.2
    elseif POP_DENSITY == :moderate
        return 0.1
    elseif POP_DENSITY == :high
        return 0.05
    else
        error("Invalid value for POP_DENSITY")
    end
end

function construct_initial_graph(params::ExogeneousParameters)

    @unpack N_SEATS = params

    n_clusters = get_num_clusters(params)
    intra_cluster_spread = get_sigma_c(params)
    epsilon_intra = get_epsilon_intra(params)

    # assign nodes to clusters
    cluster_centers = rand(2, n_clusters)
    nodes_per_cluster = fill(floor(Int64, N_SEATS / n_clusters), n_clusters)
    for i in 1:(N_SEATS%n_clusters)
        nodes_per_cluster[i] += 1  # just add one seat to a few clusters for remainder seats
    end

    # Place nodes within clusters
    nodes = zeros(2, N_SEATS)
    cluster_assignments = zeros(Int64, N_SEATS) # the assignment to a cluster for each node
    idx = 1
    for cluster in 1:n_clusters
        n_c = nodes_per_cluster[cluster]
        mvn = MvNormal(cluster_centers[:, cluster], intra_cluster_spread^2 * I(2))
        for _ in 1:n_c
            node_pos = rand(mvn)
            node_pos = clamp.(node_pos, 0.0, 1.0)
            nodes[:, idx] = node_pos
            cluster_assignments[idx] = cluster
            idx += 1
        end
    end

    # construct graph
    g = SimpleGraph(N_SEATS)

    return InitialGraphAttrs(
        g, n_clusters, intra_cluster_spread, epsilon_intra, cluster_centers, nodes_per_cluster,
        nodes, cluster_assignments
    )

end

function connect_graph(initial_graph_attributes::InitialGraphAttrs)

    @unpack g, n_clusters, intra_cluster_spread, epsilon_intra = initial_graph_attributes
    @unpack cluster_centers, nodes_per_cluster nodes = initial_graph_attributes
    @unpack cluster_assignments = initial_graph_attributes

    # Intra-cluster connections
    for cluster in 1:n_clusters
        cluster_indices = findall(cluster_assignments .== cluster)
        for i in cluster_indices
            for j in cluster_indices
                if i < j && norm(nodes[:, i] - nodes[:, j]) ≤ epsilon_intra
                    add_edge!(g, i, j)
                end
            end
        end
    end

    # Intra-cluster connections using KDTree
    for cluster in 1:n_clusters
        indices_with_this_cluster = findall(cluster_assignments .== cluster)
        cluster_centers[:, cluster] = mean(nodes[:, indices_with_this_cluster], dims=2)
    end

    # Compute distances between cluster centers
    distances = zeros(n_clusters, n_clusters)
    for i in 1:n_clusters
        for j in i+1:n_clusters
            dist = norm(cluster_centers[:, i] - cluster_centers[:, j])
            distances[i, j] = dist
            distances[j, i] = dist
        end
    end

    # Create a complete graph of cluster centers with edge weights
    cluster_graph = Graph(n_clusters)
    edge_weights = Dict{Edge{Int64},Float64}()
    for i in 1:n_clusters
        for j in i+1:n_clusters
            add_edge!(cluster_graph, i, j)
            edge_weights[Edge(i, j)] = distances[i, j]
        end
    end

    # Compute the minimum spanning tree using Kruskal's algorithm
    mst = kruskal_mst(cluster_graph, edge_weights)

    # Connect clusters via the MST
    for e in edges(mst)
        c1, c2 = src(e), dst(e)
        indices_c1 = findall(cluster_assignments .== c1)
        indices_c2 = findall(cluster_assignments .== c2)
        min_dist = Inf
        closest_pair = (0, 0)
        for i in indices_c1
            for j in indices_c2
                dist = norm(nodes[:, i] - nodes[:, j])
                if dist < min_dist
                    min_dist = dist
                    closest_pair = (i, j)
                end
            end
        end
        add_edge!(g, closest_pair[1], closest_pair[2])
    end

    return g

end

# --------------------------------------- #
# Gonna wanna put the following block into its own module

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

# --------------------------------------- #

function assign_characteristics(state::ConnectedStateGraphWithDemographicDistributions,
    demographic_characteristic::DemographicCharacteristic)

    @unpack scale_type, n_categories, homogeneity, characteristic = demographic_characteristic
    @unpack g, distributions, state_params = state
    """
    Assigns state-level and charactersitic-distribution parameters and initialize 
    node-level characteristic-distribution parameters.
    """
    # Initialize dictionaries to store state-level and node-level parameters
    distributions[:state] = Dict{Symbol,Vector{Float64}}()
    distributions[:node] = Dict{Symbol,Vector{Vector{Float64}}}()

    # Generate state-level probabilities
    p_state = generate_state_level_probabilities(homogeneity, scale_type, n_categories)
    distributions[:state][characteristic] = p_state

    # Initialize node-level parameters as copies of the state-level probabilities
    distributions[:node][characteristic] = [copy(p_state) for _ in 1:state_params.N_SEATS]

end

# Function to apply spatial autocorrelation
function apply_spatial_autocorrelation(state::ConnectedStateGraphWithDemographicDistributions)

    @unpack distributions, params = state
    # Map A to weight w

    if params.SPATIAL_AUTOCOR == :none
        w = 0.0
    elseif params.SPATIAL_AUTOCOR == :low
        w = 0.25
    elseif params.SPATIAL_AUTOCOR == :moderate
        w = 0.50
    elseif params.SPATIAL_AUTOCOR == :high
        w = 0.75
    else
        throw(DomainError("invalid value for params.SPATIAL_AUTOCOR"))
    end

    # Adjust node-level parameters for all characteristics
    for characteristic in keys(distributions[:node])

        characteristic_distributions = distributions[:node][characteristic]

        for iter in 1:5

            new_characteristic_distributions = deepcopy(characteristic_distributions)

            for i in 1:params.N_SEATS

                neighbor_indices = neighbors(g, i)

                if !isempty(neighbor_indices)

                    neighbor_params = [characteristic_distributions[j] for j in neighbor_indices]
                    mean_neighbor = mean(hcat(neighbor_params...); dims=2)
                    mean_neighbor = vec(mean_neighbor)
                    # Adjust node-level prbobailities
                    begin
                        new_characteristic_distributions[i] =
                            (1 - w) * characteristic_distributions[i] + w * mean_neighbor
                    end
                    # Normalize to ensure the probabilities sum to 1
                    new_characteristic_distributions[i] /= sum(new_characteristic_distributions[i])

                end
            end
            characteristic_distributions = new_characteristic_distributions
        end
        state.theta[:node][characteristic] = characteristic_distributions
    end
end

function simulate_population(state::ConnectedStateGraphWithDemographicDistributions)

    @unpack params, distributions = state
    @unpack N_SEATS, N_POP = params

    population_per_node = floor(Int, N_POP / N_SEATS)
    remainder = N_POP % N_SEATS
    population = Vector{Dict}()

    for i in 1:N_SEATS

        n_i = population_per_node + (i <= remainder ? 1 : 0)

        person_records = [Dict{Symbol,Any}() for _ in 1:n_i]

        for characteristic in keys(distributions[:node])

            probs = state.theta[:node][characteristic][i]

            # Sample from the node-level distribution
            dist = Categorical(probs)
            samples = rand(dist, n_i)

            # Assign samples to person records
            for (idx, person) in enumerate(person_records)
                person[characteristic] = samples[idx]
            end
        end

        # Record node assignment
        for person in person_records
            person[:node] = i
        end

        append!(population, person_records)
    end
    return population
end

end # module DemographicSimulation
