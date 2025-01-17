cd("/Users/alihashim/Desktop/Online_Academic_Submissions/poec_thesis/simulation_package")
include("simulate_spatial_characteristics.jl")
include("simulate_demographic_characteristics.jl")
include("simulating_preferences.jl")
include("party_simulation.jl")
include("candidate_simulation.jl")
include("election_simulation.jl")
include("tangian_indices.jl")
include("evaluation_metrics.jl")

using Random
using StaticArrays
using Distributions
using DataStructures
using .ExogeneousDemographicCharacteristics
using .SpatialCharacteristics
using .EndogeneousDemogprahicCharacteristics
using .SimulateIssuePreferences
using .SimulateQuestionPreferences
using .PartySimulation
using .CandidateSimulation
using .ElectionSimulation
using .TangianIndices
using .ProportionalEvaluationMetrics

# Global Constants
const RNG = MersenneTwister(2024)
rng = RNG

const N_CHARACTERISTICS = 6

# Statewide Distributions
const CHARACTERISTICS = SVector{N_CHARACTERISTICS,Symbol}(
    [:income, :age, :education, :race, :religion, :immigrant_status]
)
const CHARACTERISTIC_TYPE = SVector{N_CHARACTERISTICS,Symbol}(
    [:ordinal, :ordinal, :ordinal, :nominal, :nominal, :nominal]
)
const N_GROUPS = SVector{N_CHARACTERISTICS,Int}(
    [3, 3, 3, 5, 5, 2]
)
const HOMOGENEITY = SVector{N_CHARACTERISTICS,Symbol}(
    [:high, :low, :high, :moderate, :low, :high]
)

# Spatial Characteristics
const N_METROS = 3
const URBANIZATION = 0.9
const URBAN_SPRAWL = 0.05
const SPATIAL_DISPERSION = 0.05
const N_SEATS = 50
n_seats = N_SEATS
const A_VALS = SVector{N_CHARACTERISTICS,Float64}([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# Exogeneous Demographic Characteristics
const SALIENCE_TO_PROBS = Dict{Symbol,Vector{Float64}}(
    :none => [1.0, 0.0, 0.0, 0.0],
    :low => [0.15, 0.70, 0.10, 0.05],
    :moderate => [0.05, 0.10, 0.70, 0.15],
    :high => [0.05, 0.10, 0.15, 0.7],
)
const DEMOGRAPHIC_SALIENCE = SVector{N_CHARACTERISTICS,Symbol}(
    [:moderate, :moderate, :moderate, :moderate, :moderate, :moderate]
)
const POP_PER_NODE = 5_000
const POP_PER_SEAT = 5_000
pop_per_seat = POP_PER_SEAT

# Ideal Point Preference Parameters
const N_ISSUES = 3
n_issues = N_ISSUES
const ISSUE_DIMS = SVector{N_ISSUES,Int64}([1, 2, 2])
issue_dimensions = ISSUE_DIMS
const DEMOGRAPHIC_CLEAVAGE_SALIENCE = SMatrix{N_CHARACTERISTICS,N_ISSUES,Int}(
    [2 3 2; 2 3 0; 2 2 3; 1 2 2; 0 2 2; 2 1 2]
)

const σ_NONE = 20.0
const σ_LOW = 10.0
const σ_MODERATE = 5.0
const σ_HIGH = 1.0

# Question Position Preference parameters
const N_QUESTIONS = SVector{N_ISSUES,Int}([3, 3, 3]) # questions per issue
n_questions = N_QUESTIONS
const N_POSITIONS = SVector{N_ISSUES,Int}([5, 5, 5]) # positions per question, for each issue
n_positions = N_POSITIONS
const GAMMA = 1.0

# Structured Party Preferences
const N_PARTIES = 5
n_parties = N_PARTIES

# Exogeneous Candidates Preferences
const α = 0.01
const P_NORM = 5.0
const N_CANDIDATES = 5
n_candidates = N_CANDIDATES

# Evaluation
const N_ITERS = 100
n_iterations = N_ITERS
const SAMPLE_SIZE = 400
sample_size = 400
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

"""
Array with the following dimensions:
N x C x d_max

where N is the number of districts, C is the number of charactersitics, and d_max is the maximum
number of groups

"""
function generate_spatial_characteristics(n_metros::Int, spatial_dispersion::Float64,
    n_seats::Int, urbanization::Float64, urban_sprawl::Float64, a_vals::AbstractVector{Float64},
    statewide_distributions::Vector{Vector{Float64}},
)::Array{Float64,3}

    centers = SpatialCharacteristics.generate_cluster_centers(n_metros, spatial_dispersion, RNG)
    coords, is_urban = SpatialCharacteristics.place_nodes(n_seats, urbanization, centers,
        urban_sprawl, RNG)
    node_dists = SpatialCharacteristics.build_node_distributions(coords, statewide_distributions,
        a_vals, RNG)

    return node_dists

end

"""
N x A x C x 3 array

where N is the number of districts, A is the numnber of people per district, C is the number of 
charactersitics, and 3 corresponds to each of the following: 
1 => what group in the characteristic you're in (can be [1,2,...,d_max])
2 => how salient your identity is (can be [0,1,2,3])
3 => whether your identity is salient or not (can be [0,1])

"""
function generate_voters(demographic_salience::AbstractVector{Symbol},
    n_characteristics::Int, mapping::Dict{Symbol,Vector{Float64}},
    node_dists::Array{Float64,3}, pop_per_node::Int
)::Array{Int,4}

    salience_probs = Matrix{Float64}(undef, n_characteristics, 4)
    @inbounds for i in 1:n_characteristics
        salience_probs[i, :] = mapping[demographic_salience[i]]
    end

    salience_lookup = EndogeneousDemogprahicCharacteristics.precompute_salience_lookup_table(
        SMatrix{n_characteristics,4,Float64}(salience_probs)
    )

    voter_demographics = EndogeneousDemogprahicCharacteristics.simulate_agents_with_salience(
        node_dists, salience_lookup, pop_per_node; rng=RNG
    )

    return voter_demographics

end

"""
Returns: 
1. Vector of 3D arrays with dimensions N x A x d_I, where d_I is the dimension of the issue I
2. Vector of means of ideal points in each dimension in each issue space
3. Vector of variances of ideal points in each dimension of each issue space

"""
function generate_scaled_ideal_points(demographic_cleavage_salience::AbstractMatrix{Int},
    n_characteristics::Int, n_groups::AbstractVector{Int}, n_issues::Int,
    issue_dimensions::AbstractVector{Int}, voters::Array{Int,4}, σ_none::Float64, σ_low::Float64,
    σ_moderate::Float64, σ_high::Float64
)

    ideal_points_per_issue = Vector{Array{Float64,3}}(undef, n_issues)
    max_n_groups = maximum(n_groups)


    for issue in 1:n_issues

        θ = SimulateIssuePreferences.generate_theta(
            demographic_cleavage_salience[:, issue], n_characteristics, max_n_groups,
            issue_dimensions[issue], RNG
        )

        ideal_points_per_issue[issue] = SimulateIssuePreferences.generate_ideal_points(
            voters, θ, σ_none, σ_low, σ_moderate, σ_high, issue_dimensions[issue]; rng=RNG
        )

    end

    results = SimulateQuestionPreferences.z_scale_points_for_tangian.(ideal_points_per_issue)

    ideal_points_scaled, scaled_means, scaled_variances = map(x -> getindex.(results, x), 1:3)

    return ideal_points_scaled, scaled_means, scaled_variances

end


"""
for a given issue (a single vector of matricies)

"""
function find_active_dimensions(n_questions::Int, available_positions::AbstractVector)::Vector{Int}

    # find active dimensions
    active_positions = Vector{Int}(undef, n_questions)
    @inbounds for (q, mat) in enumerate(available_positions)
        for (i, row) in enumerate(eachrow(mat))
            # Check if any element in row i is nonzero.
            if any(x -> x != 0.0, row)
                active_positions[q] = i
            end
        end
    end
    return active_positions

end

"""
Vector of N x A x Q_i arrays, where Q_i is the number of questions in a given issue.
So, question_positions[1][1,1,:] gives us the positions of agent 1 on 

"""
function generate_question_positions(issue_dimensions::AbstractVector{Int}, n_issues::Int,
    n_questions::AbstractVector{Int64}, n_positions::AbstractVector{Int64},
    ideal_points::AbstractVector{Array{Float64,3}},
    gamma::Float64, n_seats::Int, pop_per_seat::Int)::Vector

    available_positions = Vector{AbstractVector{AbstractMatrix{Float64}}}(undef, n_issues)

    for issue in 1:n_issues

        number_of_questions = n_questions[issue]
        number_of_positions = n_positions[issue]
        issue_dimension = issue_dimensions[issue]

        q_specs = map(
            random_dim -> (random_dim, number_of_positions),
            rand(1:issue_dimension, number_of_questions)
        )

        available_positions[issue] = SimulateQuestionPreferences.build_tangian_questions_multiissue(
            issue_dimension, q_specs
        )

    end

    available_positions = SVector{n_issues,AbstractVector{AbstractMatrix{Float64}}}(
        available_positions
    )

    question_positions = SimulateQuestionPreferences.map_voters_to_positions!(ideal_points,
        available_positions, gamma, n_issues, n_questions, n_positions, n_seats, pop_per_seat,
        issue_dimensions)

    return question_positions

end

function generate_parties(ideal_point_means::AbstractVector{Vector{Float64}},
    ideal_point_variances::AbstractVector{Vector{Float64}},
    n_parties::Int, n_issues::Int, issue_dimensions::AbstractVector{Int})

    covariances = PartySimulation.build_covariances(ideal_point_variances, n_issues)
    party_ideal_points = PartySimulation.structured_noise_generate_party_ideal_points(n_parties,
        ideal_point_means, covariances, n_issues, issue_dimensions, RNG
    )

    return party_ideal_points
end

@inline function convert_party_ideal_points_to_arrs(n_issues::Int, n_parties::Int,
    issue_dims::AbstractVector{Int}, party_ideal_points::AbstractVector{Matrix{Float64}}
)

    reshaped_party_ideal_points = Vector{Array{Float64,3}}(undef, n_issues)  # Adjust type

    @inbounds for k in 1:N_ISSUES
        reshaped_party_ideal_points[k] = Array{Float64,3}(undef, 1, n_parties, issue_dims[k])
        reshaped_party_ideal_points[k][1, :, :] .= party_ideal_points[k]
    end

    return SVector{n_issues,Array{Float64,3}}(reshaped_party_ideal_points)

end

function candidate_entry(ideal_points::AbstractVector{Array{Float64,3}}, α::Float64,
    p_norm::Float64, n_candidates::Int, n_seats::Int, pop_per_seat::Int
)

    _, is_political_class = CandidateSimulation.compute_engagement(ideal_points, α, p_norm)
    candidate_indices = Vector{Vector{Int}}(undef, n_seats)

    @inbounds for seat in 1:n_seats

        local_indices = collect(1:pop_per_seat)[@view is_political_class[seat, :]]
        candidate_indices[seat] = sample(local_indices, n_candidates)

    end

    return candidate_indices
end


function run_majoritarian_election(ideal_points::AbstractVector{Array{Float64,3}},
    candidates::Vector{Vector{Int}}, n_seats::Int, n_candidates::Int, n_issues::Int,
    pop_per_seat::Int, issue_dimensions::AbstractVector)

    begin
        first_round_results, voter_utilities, first_round_candidate_choices =
            ElectionSimulation.run_single_round_election(
                ideal_points, candidates, n_seats, n_candidates, n_issues, pop_per_seat, issue_dimensions
            )
    end

    run_off_candidates = ElectionSimulation.tally_top_2(first_round_results, n_seats)

    second_round_results, _, _ = ElectionSimulation.run_single_round_election(
        ideal_points, run_off_candidates, n_seats, 2, n_issues, pop_per_seat, issue_dimensions
    )

    winning_candidates = ElectionSimulation.tally_top_1(second_round_results, n_seats)

    return winning_candidates, voter_utilities, first_round_candidate_choices

end

@inline function run_proportional_election(preferred_parties::Matrix{Int}, n_seats::Int,
    pop_per_seat::Int)

    counts = counter(preferred_parties)
    proportional_results = Dict(
        party => count / (n_seats * pop_per_seat) for (party, count) in counts
    )

    return proportional_results

end

function evaluate_party_profiles(n_parties::Int, n_issues::Int, n_seats::Int,
    pop_per_seat::Int, voter_question_positions::AbstractVector{Array{Float64,3}},
    party_question_positions::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3}, n_questions::AbstractVector{Int})

    coalition_options = ProportionalEvaluationMetrics.build_coalition_options(n_parties)
    n_coalitions = length(coalition_options)
    begin
        coalition_profiles, coalition_unanimities, coalition_qualified_unanimities =
            ProportionalEvaluationMetrics.build_party_profile_options(coalition_options,
                party_question_positions, n_issues, n_questions)

    end

    profile_utilities = ProportionalEvaluationMetrics.compute_utilities_for_party_profiles(
        voter_question_positions, coalition_profiles, voter_issue_weights, n_coalitions,
        n_seats, pop_per_seat, n_issues
    )

    begin
        return coalition_options, coalition_profiles, profile_utilities, coalition_unanimities,
        coalition_qualified_unanimities
    end

end

@inline function scale_utilities(profile_utilities::Array{Float64,3})

    min_val = minimum(profile_utilities)
    max_val = maximum(profile_utilities)
    normalized_utilities = (profile_utilities .- min_val) ./ (max_val - min_val)

    return normalized_utilities
end


function main()

    @time statewide_demographic_dists = generate_statewide_distributions(HOMOGENEITY,
        CHARACTERISTIC_TYPE, N_GROUPS
    )

    @time district_dists = generate_spatial_characteristics(N_METROS, SPATIAL_DISPERSION,
        N_SEATS, URBANIZATION, URBAN_SPRAWL, A_VALS, statewide_demographic_dists,
    )

    @time voters = generate_voters(DEMOGRAPHIC_SALIENCE, N_CHARACTERISTICS,
        SALIENCE_TO_PROBS, district_dists, POP_PER_NODE
    )

    @time ideal_points, ideal_means, ideal_variances = generate_scaled_ideal_points(
        DEMOGRAPHIC_CLEAVAGE_SALIENCE, N_CHARACTERISTICS, N_GROUPS, N_ISSUES, ISSUE_DIMS, voters,
        σ_NONE, σ_LOW, σ_MODERATE, σ_HIGH
    )

    @time voter_issue_weights = ProportionalEvaluationMetrics.find_issue_weights(ideal_points,
        N_ISSUES, N_SEATS, POP_PER_NODE, ISSUE_DIMS
    )

    @time ideal_points, ideal_means, ideal_variances = collect.(
        [ideal_points, ideal_means, ideal_variances]
    )

    ideal_points = SVector{3,Array{Float64,3}}(ideal_points)

    @time voter_question_positions = generate_question_positions(ISSUE_DIMS, N_ISSUES, N_QUESTIONS,
        N_POSITIONS, ideal_points, GAMMA, N_SEATS, POP_PER_NODE
    )

    # Non-adaptive parties or candidates

    @time party_ideal_points = generate_parties(ideal_means, ideal_variances, N_PARTIES,
        N_ISSUES, ISSUE_DIMS
    )

    @time preferred_parties, proportional_voter_utilities =
        ElectionSimulation.assign_voters_to_parties!(
            ideal_points, party_ideal_points, N_PARTIES, N_ISSUES, N_SEATS, POP_PER_NODE
        )

    @time party_ideal_points = convert_party_ideal_points_to_arrs(N_ISSUES, N_PARTIES, ISSUE_DIMS,
        party_ideal_points
    )

    @time party_issue_weights = ProportionalEvaluationMetrics.find_issue_weights(party_ideal_points,
        N_ISSUES, 1, N_PARTIES, ISSUE_DIMS
    )

    @time party_question_positions = generate_question_positions(ISSUE_DIMS, N_ISSUES, N_QUESTIONS,
        N_POSITIONS, party_ideal_points, GAMMA, 1, N_PARTIES)

    @time candidates = candidate_entry(ideal_points, α, P_NORM, N_CANDIDATES, N_SEATS, POP_PER_NODE)

    @time winning_candidates, majoritarian_voter_utilities, first_round_candidate_choices =
        run_majoritarian_election(ideal_points, candidates, N_SEATS, N_CANDIDATES, N_ISSUES,
            POP_PER_NODE, ISSUE_DIMS
        )

    @time winning_parties = run_proportional_election(
        preferred_parties, N_SEATS, POP_PER_NODE
    )

    @time begin
        coalition_options, coalition_profiles, profile_utilities, coalition_unanimities,
        coalition_qualified_unanimities =
            evaluate_party_profiles(
                N_PARTIES, N_ISSUES, N_SEATS, POP_PER_NODE, voter_question_positions,
                party_question_positions, voter_issue_weights, N_QUESTIONS
            )
    end

    profile_utilities = scale_utilities(profile_utilities)

    social_utilities = sum(profile_utilities, dims=(1, 2))[:]
    coalition_profiles = convert.(Array, coalition_profiles)

    begin
        @time party_coalition_utilities =
            ProportionalEvaluationMetrics.compute_utilities_for_party_profiles(
                coalition_profiles, coalition_profiles, party_issue_weights, length(coalition_options),
                1, N_PARTIES, N_ISSUES
            )
    end

    party_coalition_utilities = scale_utilities(party_coalition_utilities)

    @time winning_coalition = ProportionalEvaluationMetrics.find_winning_coalition(N_QUESTIONS,
        winning_parties, coalition_options, coalition_qualified_unanimities,
        party_coalition_utilities)

    # FIRST METRIC
    utility_maxer_chosen = argmax(social_utilities) == winning_coalition ? 1 : 0

    # SECOND METRIC
    prop_vse = ProportionalEvaluationMetrics.compute_vse(N_ITERS, preferred_parties, rng, N_SEATS,
        POP_PER_SEAT, N_PARTIES, SAMPLE_SIZE, N_QUESTIONS, coalition_options,
        coalition_qualified_unanimities, party_coalition_utilities, social_utilities,
        winning_coalition)

    # THIRD AND FOURTH METRICS
    tangian_indices = TangianIndices.computeTangianIndices(
        voter_question_positions, party_question_positions, winning_candidates,
        winning_parties, N_QUESTIONS, N_POSITIONS, N_ISSUES, N_SEATS, POP_PER_NODE,
        N_PARTIES
    )

end



if abspath(PROGRAM_FILE) == @__FILE__
    main()
end