module CondorcetSmithFunctions

export find_condorcet_and_smith_sets_for_district, find_condorcet_and_smith_sets_for_state
using Combinatorics

"""
    process_district(rankings::AbstractMatrix{Int})

Process one district’s ballots (an A×C matrix) to compute:
  - the Condorcet winner (if one exists),
  - the Smith set (the minimal set that dominates the other candidates),
  - the Smith winner (using a simple tie-break rule on pairwise wins).
  
Returns a triple: (condorcet, smith_set, smith_winner)
"""
function find_condorcet_and_smith_sets_for_district(rankings::AbstractMatrix{Int})

    A, C = size(rankings)
    total_votes = A
    # Preallocate a C×C wins matrix where wins[i,j] is the number of voters
    # who prefer candidate i over candidate j.
    wins = zeros(Int, C, C)
    @inbounds for i in 1:C
        for j in 1:C
            if i == j
                continue
            end
            s = 0
            @simd for a in 1:A
                s += ifelse(rankings[a, i] < rankings[a, j], 1, 0)
            end
            wins[i, j] = s
        end
    end

    # --------------------------
    # 1. Compute the Condorcet winner.
    # A candidate i wins if for every other candidate j, wins[i,j] is more than half the votes.
    # --------------------------
    condorcet = nothing
    for i in 1:C
        is_winner = true
        for j in 1:C
            if i == j
                continue
            end
            if wins[i, j] <= (total_votes ÷ 2)
                is_winner = false
                break
            end
        end
        if is_winner
            condorcet = i
            break
        end
    end

    # --------------------------
    # 2. Compute the Smith set.
    #
    # The Smith set is the smallest nonempty set S ⊆ {1, …, C} such that for every candidate i ∈ S
    # and every candidate j not in S, wins[i,j] > wins[j,i].
    # Since the number of candidates is small, we enumerate subsets by increasing size.
    # --------------------------
    smith_set = nothing
    all_candidates = collect(1:C)
    found = false
    for k in 1:C
        for subset in combinations(all_candidates, k)
            # Using a Set for membership testing.
            subset_set = Set(subset)
            valid = true
            for i in subset
                for j in setdiff(all_candidates, subset_set)
                    if wins[i, j] <= wins[j, i]
                        valid = false
                        break
                    end
                end
                if !valid
                    break
                end
            end
            if valid
                # For deterministic tie–breaking later, we sort the Smith set.
                smith_set = sort(collect(subset_set))
                found = true
                break
            end
        end
        if found
            break
        end
    end

    # --------------------------
    # 3. Choose a Smith winner.
    #
    # If a Condorcet winner exists, choose it.
    # Otherwise, select from the Smith set the candidate with the highest number of pairwise wins.
    # (Note: If there is a tie, the one encountered first is chosen.)
    # --------------------------
    smith_winner = nothing
    if condorcet !== nothing
        smith_winner = condorcet
    elseif smith_set !== nothing && !isempty(smith_set)
        best_score = -1
        for i in smith_set
            score = 0
            for j in all_candidates
                if i == j
                    continue
                end
                score += ifelse(wins[i, j] > wins[j, i], 1, 0)
            end
            if score > best_score
                best_score = score
                smith_winner = i
            end
        end
    end

    return condorcet, smith_set, smith_winner
end


"""
    process_all_districts(rankings::AbstractArray{Int,3})

Process a 3D rankings array of dimensions N×A×C (districts × voters × candidates)
and returns a triple of vectors: (condorcet_winners, smith_sets, smith_winners)
for the districts.
"""
function find_condorcet_and_smith_sets_for_state(rankings::AbstractArray{Int,3})
    N, A, C = size(rankings)
    condorcet_winners = Vector{Union{Int,Nothing}}(undef, N)
    smith_sets = Vector{Vector{Int}}(undef, N)
    smith_winners = Vector{Union{Int,Nothing}}(undef, N)
    @inbounds for n in 1:N
        cd, ss, sw = find_condorcet_and_smith_sets_for_district(@view rankings[n, :, :])
        condorcet_winners[n] = cd
        smith_sets[n] = ss === nothing ? Int[] : ss
        smith_winners[n] = sw
    end
    return condorcet_winners, smith_sets, smith_winners
end

"""
    process_all_districts(rankings::AbstractArray{Int,3})

Process a 3D rankings array of dimensions N×A×C (districts × voters × candidates)
and returns a triple of vectors: (condorcet_winners, smith_sets, smith_winners)
for the districts.
"""
function find_condorcet_and_smith_sets_for_state(rankings::Vector{Matrix{Int}})

    N = size(rankings)[1]
    condorcet_winners = Vector{Union{Int,Nothing}}(undef, N)
    smith_sets = Vector{Vector{Int}}(undef, N)
    smith_winners = Vector{Union{Int,Nothing}}(undef, N)
    @inbounds for n in 1:N

        cd, ss, sw = find_condorcet_and_smith_sets_for_district(@view rankings[n, :, :])
        condorcet_winners[n] = cd
        smith_sets[n] = ss === nothing ? Int[] : ss
        smith_winners[n] = sw
    end
    return condorcet_winners, smith_sets, smith_winners
end


end
