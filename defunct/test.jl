cd("/Users/alihashim/Desktop/Online_Academic_Submissions/poec_thesis/simulation_package")
include("simulate_spatial_characteristics.jl")
include("simulate_demographic_characteristics.jl")
include("simulating_preferences.jl")
include("party_simulation.jl")
include("candidate_simulation.jl")
include("election_simulation.jl")
include("tangian_indices.jl")

using Random
using StaticArrays
using Distributions
using Plots
using .ExogeneousDemographicCharacteristics
using .SpatialCharacteristics
using .EndogeneousDemogprahicCharacteristics
using .SimulateIssuePreferences
using .SimulateQuestionPreferences
using .PartySimulation
using .ElectionSimulation

RNG = MersenneTwister(2024)

N_CHARACTERISTICS = 3

# Statewide Distributions
CHARACTERISTICS = SVector{N_CHARACTERISTICS,Symbol}(
    [:race, :immigrant, :income]
)
CHARACTERISTIC_TYPE = SVector{N_CHARACTERISTICS,Symbol}(
    [:nominal, :nominal, :ordinal]
)
N_GROUPS = SVector{N_CHARACTERISTICS,Int}(
    [5, 5, 3]
)
HOMOGENEITY = SVector{N_CHARACTERISTICS,Symbol}(
    [:moderate, :low, :low]
)

# Spatial Characteristics
N_METROS = 3
URBANIZATION = 0.9
URBAN_SPRAWL = 0.05
SPATIAL_DISPERSION = 0.05
N_SEATS = 50
A_VALS = SVector{N_CHARACTERISTICS,Float64}([1.0, 1.0, 1.0])

# Exogeneous Demographic Characteristics
SALIENCE_TO_PROBS = Dict{Symbol,Vector{Float64}}(
    :none => [1.0, 0.0, 0.0, 0.0],
    :low => [0.15, 0.70, 0.10, 0.05],
    :moderate => [0.05, 0.10, 0.70, 0.15],
    :high => [0.05, 0.10, 0.15, 0.7],
)
DEMOGRAPHIC_SALIENCE = SVector{N_CHARACTERISTICS,Symbol}(
    [:low, :high, :moderate]
)
POP_PER_NODE = 5_000

# Ideal Point Preference Parameters
N_ISSUES = 4
ISSUE_DIMS = SVector{N_ISSUES,Int64}([2, 1, 1, 2])
DEMOGRAPHIC_CLEAVAGE_SALIENCE = SMatrix{N_CHARACTERISTICS,N_ISSUES,Int}(
    [0 1 2 3; 0 1 2 3; 0 1 2 3]
)
σ_NONE = 20.0
σ_LOW = 10.0
σ_MODERATE = 5.0
σ_HIGH = 1.0

# Question Position Preference parameters
N_QUESTIONS = SVector{N_ISSUES,Int}([3, 3, 3, 3]) # questions per issue
N_POSITIONS = SVector{N_ISSUES,Int}([5, 5, 5, 5]) # positions per question, for each issue
GAMMA = 1.0

# Structured Party Preferences
N_PARTIES = 5

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
        σ_NONE, σ_LOW, σ_MODERATE, σ_HIGH,
    )

    @time ideal_points, ideal_means, ideal_variances = collect.(
        [ideal_points, ideal_means, ideal_variances]
    )

    ideal_points = SVector{N_ISSUES,Array{Float64,3}}(ideal_points)

    @time question_positions = generate_question_positions(ISSUE_DIMS, N_ISSUES, N_QUESTIONS,
        N_POSITIONS, ideal_points, GAMMA, N_SEATS, POP_PER_NODE)


    # Non-adaptive parties or candidates

    @time party_ideal_points = generate_parties(ideal_means, ideal_variances,
        N_PARTIES, N_ISSUES, ISSUE_DIMS
    )

    @time preferred_parties = ElectionSimulation.assign_voters_to_parties!(ideal_points,
        party_ideal_points, N_PARTIES, N_ISSUES
    )

    tangian_indices = 

end


function analyze(ideal_points::AbstractVector, issue::Int; dimension::Int=1)

    arr = ideal_points[issue][:, :, dimension]
    histogram(vec(arr))

end

function analyze(ideal_points::AbstractVector, party_ideal_points::AbstractVector, issue::Int)

    arr = ideal_points[issue]
    party_arr = party_ideal_points[issue]
    party_x, party_y = eachcol(party_arr)
    colors = distinguishable_colors(size(party_x, 1))

    @assert size(arr)[3] > 1

    scatter(vec(@view arr[:, :, 1]), vec(@view arr[:, :, 2]))
    scatter!(party_x, party_y, colors=colors)


end

function analyze(preferred_parties::Matrix{Int})

    println("Votes:")
    @show Dict(party => count / 250000 for (party, count) in counter(preferred_parties))
    histogram(vec(preferred_parties))

end