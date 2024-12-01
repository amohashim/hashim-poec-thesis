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

struct ConnectStateGraphWithDemographicDistributions

    # each dict in `thetas` maps a characteristic (e.g. :age) to the parameters of their 
    # distribution, for a given node (the vector is a vector of nodes)

    g::SimpleGraph # from ConnectedStateGraph
    thetas::Vector{Dict{Symbol,Float}}
    params::ExogeneousParameters

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

# Function to generate the state
# function generate_state(S::Int, D::String, P::String)

#     # Map parameters to numerical values
#     C = get_num_clusters(P, S)
#     sigma_c = get_sigma_c(D)
#     epsilon_intra = get_epsilon_intra(D)

#     # Generate cluster centers
#     cluster_centers = rand(2, C)

#     # Assign nodes to clusters
#     nodes_per_cluster = fill(floor(Int, S / C), C)
#     for i in 1:(S%C)
#         nodes_per_cluster[i] += 1
#     end

#     # Place nodes within clusters
#     nodes = zeros(2, S)
#     cluster_assignments = zeros(Int, S)
#     idx = 1
#     for c in 1:C
#         n_c = nodes_per_cluster[c]
#         mvn = MvNormal(cluster_centers[:, c], sigma_c^2 * I(2))
#         for j in 1:n_c
#             node_pos = rand(mvn)
#             node_pos = clamp.(node_pos, 0.0, 1.0)
#             nodes[:, idx] = node_pos
#             cluster_assignments[idx] = c
#             idx += 1
#         end
#     end

#     # Construct the graph
#     g = SimpleGraph(S)

#     # Intra-cluster connections
#     for c in 1:C
#         cluster_indices = findall(cluster_assignments .== c)
#         for i in cluster_indices
#             for j in cluster_indices
#                 if i < j && norm(nodes[:, i] - nodes[:, j]) ≤ epsilon_intra
#                     add_edge!(g, i, j)
#                 end
#             end
#         end
#     end

#     # Inter-cluster connections via MST
#     # Compute distances between cluster centers
#     distances = pairwise(Euclidean(), cluster_centers)
#     cluster_graph = SimpleWeightedGraph(C)
#     for i in 1:C
#         for j in i+1:C
#             add_edge!(cluster_graph, i, j, distances[i, j])
#         end
#     end

#     # Minimum Spanning Tree
#     mst_edges = minimum_spanning_tree(cluster_graph)

#     # Connect clusters via the MST
#     for e in mst_edges
#         c1, c2 = src(e), dst(e)
#         indices_c1 = findall(cluster_assignments .== c1)
#         indices_c2 = findall(cluster_assignments .== c2)
#         min_dist = Inf
#         closest_pair = (0, 0)
#         for i in indices_c1
#             for j in indices_c2
#                 dist = norm(nodes[:, i] - nodes[:, j])
#                 if dist < min_dist
#                     min_dist = dist
#                     closest_pair = (i, j)
#                 end
#             end
#         end
#         add_edge!(g, closest_pair[1], closest_pair[2])
#     end

#     # Return the state object
#     return State(S, 0, D, P, "", "", "", nodes, g, cluster_assignments, Dict())
# end

# Function to assign characteristics
function assign_characteristics(state::State, H::String)
    # Example for Race characteristic
    K_race = 5  # Number of race categories

    # Get Dirichlet parameters based on H
    function get_alpha(H::String, K::Int)
        if H == "Perfect"
            alpha = ones(K)
            alpha[rand(1:K)] = 1000.0
        elseif H == "High"
            alpha = ones(K)
            alpha[rand(1:K)] = 100.0
        elseif H == "Moderate"
            alpha = fill(10.0, K)
        elseif H == "Low"
            alpha = ones(K)
        else
            error("Invalid value for H")
        end
        return alpha
    end

    alpha_race = get_alpha(H, K_race)
    dirichlet_race = Dirichlet(alpha_race)
    p_state_race = rand(dirichlet_race)

    # Assign state-level parameters
    state.theta[:race_state] = p_state_race

    # Initialize node-level parameters
    state.theta[:race_node] = [copy(p_state_race) for _ in 1:state.S]
end

# Function to apply spatial autocorrelation
function apply_spatial_autocorrelation(state::State, A::String)
    # Map A to weight w
    w = A == "None" ? 0.0 : A == "Low" ? 0.25 : A == "Moderate" ? 0.5 : 0.75

    # Adjust node-level parameters for Race
    theta_race = state.theta[:race_node]
    for iter in 1:5
        theta_race_new = deepcopy(theta_race)
        for i in 1:state.S
            neighbors = neighbors(state.g, i)
            if !isempty(neighbors)
                neighbor_params = [theta_race[j] for j in neighbors]
                mean_neighbor = mean(reduce(hcat, neighbor_params); dims=2)
                theta_race_new[i] = (1 - w) * theta_race[i] + w * vec(mean_neighbor)
            end
        end
        theta_race = theta_race_new
    end
    state.theta[:race_node] = theta_race
end

# Function to apply urban heterogeneity
function apply_urban_heterogeneity(state::State, U::String)
    # Map U to a numerical factor
    U_factor = U == "Low" ? 0.1 : U == "Moderate" ? 0.5 : 1.0

    # Adjust homogeneity based on node degree
    degrees = degree(state.g)
    d_min, d_max = minimum(degrees), maximum(degrees)

    for i in 1:state.S
        degree_norm = (degrees[i] - d_min) / (d_max - d_min)
        H_i = 1.0 + U_factor * degree_norm
        # Adjust node-level parameters
        # For Race, we can flatten the Dirichlet parameters towards uniform
        theta_i = state.theta[:race_node][i]
        theta_i = theta_i .^ (1 / H_i)
        theta_i = theta_i / sum(theta_i)
        state.theta[:race_node][i] = theta_i
    end
end

# Function to simulate population
function simulate_population(state::State, N::Int)
    population_per_node = floor(Int, N / state.S)
    remainder = N % state.S
    population = []

    for i in 1:state.S
        n_i = population_per_node + (i <= remainder ? 1 : 0)
        # Sample Race
        race_dist = Categorical(state.theta[:race_node][i])
        races = rand(race_dist, n_i)
        # Create individual records (for simplicity, only Race is included)
        for r in races
            push!(population, Dict(:node => i, :race => r))
        end
    end
    return population
end

end # module DemographicSimulation
