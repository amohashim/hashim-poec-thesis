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

    σ_none::Float64
    σ_low::Float64
    σ_moderate::Float64
    σ_high::Float64

    gamma::Float64

    n_iterations::Int
    sample_size::Int
    mantel_permutations::Int

end

struct DemographicCharacteristicParams{K}

    characteristic_type::SVector{K,Symbol}
    homogeneity::SVector{K,Symbol}

end

struct SpatialCharacteristicParams{K}

    n_metros::Int
    urbanization::Float64
    urban_sprawl::Float64
    spatial_dispersion::Float64
    a_vals::SVector{K,Float64}

end

struct IssueStructure{J}

    n_issues::Int
    issue_dimensions::SVector{J,Int}

end

struct SalienceStructure{K,J}

    demographic_salience::SVector{K,Symbol}
    demographic_cleavage_salience::SMatrix{K,J,Int}

end

struct QuestionStructure{J}

    n_questions::SVector{J,Int}
    n_positions::SVector{J,Int}

end

struct RepresentativesParams

    α_political_class::Float64
    p_norm::Float64
    α_candidate_entry::Float64
    party_threshold::Float64

end

struct BranchParams{K}

    n_parties::Int
    n_candidates::Int

    turnout_level::Float64
    strategic_level::Float64
    directional_utility::Bool

end


end