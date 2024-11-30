module TangianIndices

using LinearAlgebra
using Parameters
using Random
using StaticArrays
using DataStructures

export calculate_representativeness_index, calculate_popularity_index, calculate_universality_index
export calc_rep_indices_for_individual

function calculate_representativeness_index(candidate_position::Symbol, n_protagonists::Int64,
    n_antagonists::Int64)

    if !(candidate_position ∈ [:protagonist, :antaognist])
        throw(DomainError("`candidate_position` must be in [:protagonist, :antaognist]"))
    end

    total_voting = n_protagonists + n_antagonists
    r = candidate_position == :protagonist ? n_protagonists / total_voting : n_antagonists / total_voting

    return r

end

function calculate_popularity_index(n_questions::Int64, rep_indices::AbstractVector{Float64})

    return sum(rep_indices) / n_questions

end

function calculate_universality_index(n_questions::Int64, rep_indices::AbstractVector{Float64})

    return sum(round.(rep_indices)) / n_questions

end

"""
use row vector for candidate_prefs; computes index on all questions for a given candidate

assumes population_prefs matrix is NxQ

"""
function calc_rep_indices_for_individual(candidate_prefs::AbstractVector{Symbol},
    population_prefs::Matrix{Symbol})

    public_opinion = vec(mapslices(counter, population_prefs; dims=1))

    rep_indices = calculate_representativeness_index.(
        candidate_prefs,
        getindex.(public_opinion, :protagonist),
        getindex.(public_opinion, :antaognist)
    )

    return rep_indices

end

end