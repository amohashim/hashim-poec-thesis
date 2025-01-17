"""

"""
function find_candidate_profiles_by_district(
    voter_question_positions::AbstractVector{Array{Float64,3}},
    candidates::AbstractVector{Vector{Int}}, n_issues::Int, n_questions::AbstractVector{Int},
    n_seats::Int, n_candidates::Int)

    candidate_profiles = Vector{Array{Float64,3}}(undef, n_issues)

    for issue in 1:n_issues

        n_q = n_questions[issue]
        issue_profiles = Array{Float64,3}(undef, n_q, n_seats, n_candidates)
        issue_question_positions = voter_question_positions[issue]

        for seat in 1:n_seats
            district_candidates = candidates[seat]
            begin
                issue_profiles[:, seat, 1:n_candidates] =
                    issue_question_positions[:, seat, district_candidates]
            end
        end

        candidate_profiles[issue] = issue_profiles

    end

    return candidate_profiles

end

function compute_utilities_for_candidate_profiles(
    voter_question_positions::AbstractVector{Array{Float64,3}},
    candidate_profiles::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3},
    n_candidates_per_district::Int, n_seats::Int, pop_per_seat::Int, n_issues::Int)

    voter_utilities = Array{Float64,3}(undef, n_seats, pop_per_seat, n_candidates_per_district)

    @inbounds for issue in 1:n_issues

        @inbounds for seat in 1:n_seats

            profile = @view candidate_profiles[issue][:, seat, :]

            district_positions = @view voter_question_positions[issue][:, seat, :]
            seat_issue_weights = @view voter_issue_weights[issue, seat, :]
            utilities = [
                sum(abs.(district_positions[:, i] .- profile[:, j]))
                for i in 1:pop_per_seat, j in 1:n_candidates_per_district
            ]

            voter_utilities[seat, :, :] = -1 .* utilities .* @view seat_issue_weights[:, :]

        end
    end

    return voter_utilities

end

function evaluate_candidate_profiles(voter_question_positions::AbstractVector{Array{Float64,3}},
    candidates::AbstractVector{Vector{Int}}, voter_issue_weights::Array{Float64,3},
    n_issues::Int, n_questions::AbstractVector{Int}, pop_per_seat::Int,
    n_seats::Int, n_candidates::Int)

    candidate_profiles = find_candidate_profiles_by_district(voter_question_positions, candidates,
        n_issues, n_questions, n_seats, n_candidates)

    voter_utilities_from_candidate_profiles = compute_utilities_for_candidate_profiles(
        voter_question_positions, candidate_profiles, voter_issue_weights, n_candidates,
        n_seats, pop_per_seat, n_issues)

    voter_utilities_from_candidate_profiles = scale_utilities(
        voter_utilities_from_candidate_profiles
    )

    return candidate_profiles, voter_utilities_from_candidate_profiles

end

@inline function sample_from_dirichlet(prefered_candidates::Vector{Int}, sample_size::Int,
    rng::AbstractRNG, pop_per_seat::Int, n_candidates::Int)

    sampled_voters = sample(RNG, 1:pop_per_seat, sample_size, replace=false)
    candidate_counts = Dict(counter(prefered_candidates[sampled_voters]))

    for candidate in 1:n_candidates

        if !haskey(candidate_counts, candidate)
            candidate_counts[candidate] = 0

        end

    end

    candidate_counts = OrderedDict(
        k => candidate_counts[k] for k in sort(collect(keys(candidate_counts)))
    )

    alpha = [1 + candidate_counts[party] for party in 1:n_candidates]
    dirichlet_sample = rand(rng, Dirichlet(alpha))

    return Dict{Int,Float64}(
        c => dirichlet_sample[c] for c in 1:n_candidates
    )

end

function compute_vse(voter_utilities_from_candidate_profiles::Array{Float64,3}, n_seats::Int,
    candidates::Vector{Vector{Int}}, winning_candidates::Vector{Int},
    first_round_candidate_choices::Matrix{Int}, sample_size::Int, rng::AbstractRNG,
    pop_per_seat::Int, n_candidates::Int, n_iterations::Int,
    ideal_points::AbstractVector{Array{Float64,3}}, issue_dimensions::AbstractVector{Int},
    n_issues::Int
)

    n_maximizer_as_winner = 0
    vse_by_district = Vector{Float64}(undef, n_seats)

    @inbounds for seat in 1:n_seats

        district_candidates = candidates[seat]
        winning_candidate = winning_candidates[seat]
        winner_local_indx = findfirst(==(winning_candidate), district_candidates)

        utilities_by_candidate = @view voter_utilities_from_candidate_profiles[seat, :, :]
        social_utilities = sum(utilities_by_candidate, dims=1)[:]

        utility_maximizer = argmax(social_utilities)

        if district_candidates[utility_maximizer] == winning_candidate
            n_maximizer_as_winner += 1
        end

        prefered_candidates = first_round_candidate_choices[seat, :]

        iter_winners = Vector{Int}(undef, n_iterations)

        @inbounds for i in 1:n_iterations

            first_round_results = sample_from_dirichlet(prefered_candidates, sample_size, rng,
                pop_per_seat, n_candidates)

            run_off_candidates = ElectionSimulation.tally_top_2([first_round_results], 1)[1]
            run_off_candidates = district_candidates[run_off_candidates]

            run_off_candidate_points, row_to_candidate_map =
                ElectionSimulation.find_candidate_ideal_points(ideal_points, run_off_candidates,
                    n_candidates, seat, n_issues
                )

            results, _ = ElectionSimulation.assign_voters_to_candidates!(ideal_points,
                run_off_candidate_points, 2, n_issues, pop_per_seat, issue_dimensions,
                seat)

            proportions = Dict(idx => ct / pop_per_seat for (idx, ct) in counter(results))

            election_proportions = ElectionSimulation.map_global_indx_to_props(
                row_to_candidate_map, proportions
            )

            iter_winners[i] = ElectionSimulation.tally_top_1([election_proportions], 1)[1]

        end

        probs = Dict{Int,Float64}(
            candidate => votes / n_iterations for (candidate, votes) in counter(iter_winners)
        )

        for candidate in district_candidates

            if !haskey(probs, candidate)
                probs[candidate] = 0.0
            end
        end

        expected_utilities = [
            probs[candidate] * social_utilities[i] for (i, candidate) in pairs(district_candidates)
        ]

        e_winner = expected_utilities[winner_local_indx]
        e_maximizer = expected_utilities[utility_maximizer]
        e_average = mean(social_utilities)

        vse_by_district[seat] = (e_winner - e_average) / (e_maximizer - e_average)

    end

    return vse_by_district

end
