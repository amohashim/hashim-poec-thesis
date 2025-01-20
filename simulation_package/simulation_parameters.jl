module SimulationParameters

using Random
using StaticArrays

export FixedParams, DemographicCharacteristicParams, SpatialCharacteristicParams
export SalienceStructure, IssueStructure, QuestionStructure, RepresentativesParams
export BranchParams

struct FixedParams{K}

    rng::AbstractRNG

    n_characteristics::Int
    n_groups::SVector{K,Int}
    salience_to_probs::Dict{Symbol,SVector{4,Float64}}

    n_seats::Int
    pop_per_seat::Int

    σ_non