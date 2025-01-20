module HelpfulFunctions

using DataStructures
using Random
using Statistics
using StaticArrays
using StatsBase

export z_scale_points, scale_utilities, convert_party_ideal_points_to_arrs
export generate_question_positions, find_issue_weights, filter_parties_below_threshold

"""
    z_scale_points_for_tangian(points::Array{Float64,3})
        -> (scaled::Array{Float64,3}, means::Vector{Float64}, stds::Vector{Float64})

Z-scale the 3D array (N, A, d) across the d dimensions, 
flattening N*A as needed. Return (scaled, means, stds).

** For a given issue **
"""
function z_scale_points(points::Array{Float64,3})

    N, A, d = size(points)
    bigN = N * A

    # Flatten to (bigN, d)
    mat_2d = reshape(points, bigN, d)
    scaled_2d = copy(mat_2d)
    means = Vector{Float64}(undef, d)
    stds = Vector{Float64}(undef, d)

    @inbounds for j in 1:d
        col_j = @view scaled_2d[:, j]
        m_j = mean(col_j)
        s_j = std(col_j)
        means[j] = m_j
        stds[j] = (s_j < 1e-12) ? 1.0 : s_j
        @inbounds for i in 1:bigN
            col_j[i] = (col_j[i] - m_j) / stds[j]
        end
    end
    scaled_3d = reshape(scaled_2d, N, A, d)

    means = [mean(scaled_3d[:, :, dim]) for dim in 1:d]
    stds = [std(scaled_3d[:, :, dim]) for dim in 1:d]

    return (scaled_3d, means, stds)
end


"""
min-max scaling

"""
function scale_utilities(profile_utilities::AbstractArray{Float64})

    min_val = minimum(profile_utilities)
    max_val = maximum(profile_utilities)
    normalized_utilities = (profile_utilities .- min_val) ./ (max_val - min_val)

    return normalized_utilities
end

function convert_party_ideal_points_to_arrs(party_ideal_points::AbstractVector{Matrix{Float64}},
    n_issues::Int, n_parties::Int, issue_dims::AbstractVector{Int},
)

    reshaped_party_ideal_points = Vector{Array{Float64,3}}(undef, n_issues)  # Adjust type

    @inbounds for k in 1:n_issues
        reshaped_party_ideal_points[k] = Array{Float64,3}(undef, 1, n_parties, issue_dims[k])
        reshaped_party_ideal_points[k][1, :, :] .= party_ideal_points[k]
    end

    return SVector{n_issues,Array{Float64,3}}(reshaped_party_ideal_points)

end


function find_issue_weights(ideal_points::AbstractVector{Array{Float64,3}}, n_issues::Int,
    n_seats::Int, pop_per_seat::Int, issue_dimensions::AbstractVector{Int})

    voter_magnitudes = Array{Float64,3}(undef, n_issues, n_seats, pop_per_seat)

    @inbounds for issue in 1:n_issues

        issue_dimension = issue_dimensions[issue]
        points_for_issue = ideal_points[issue]
        norms = similar(@view points_for_issue[:, :, 1])

        mats = [@view points_for_issue[:, :, d] for d in 1:issue_dimension]
        @inbounds for i in eachindex(norms)
            sum_of_squares = 0.0
            @inbounds for M in mats
                sum_of_squares += M[i]^2
            end
            norms[i] = sqrt(sum_of_squares)
        end

        voter_magnitudes[issue, :, :] = norms ./ Ref(sqrt(issue_dimension))

    end

    return voter_magnitudes

end

@inline function filter_dict(dict::AbstractDict, t::Float64)
    for k in keys(dict)
        if dict[k] <= t
            delete!(dict, k)
        end
    end
    return dict
end

function filter_parties_below_threshold(raw_vote_counts::Dict{Int,Int},
    party_vote_count_threshold::Float64, n_parties::Int,
    proportions::Bool)

    qualified_parties = filter_dict(raw_vote_counts, party_vote_count_threshold)

    if proportions
        counting_votes = sum(values(qualified_parties))
        proportional_results = Dict(
            party => count / counting_votes for (party, count) in raw_vote_counts
        )
    else
        proportional_results = qualified_parties
    end

    @inbounds for party in 1:n_parties
        if !haskey(proportional_results, party)
            proportional_results[party] = proportions ? 0.0 : 0
        end
    end

    return proportional_results

end

end