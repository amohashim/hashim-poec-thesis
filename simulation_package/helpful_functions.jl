module HelpfulFunctions

using DataStructures
using LinearAlgebra
using Random
using Statistics
using StaticArrays
using StatsBase

using ..SimulationParameters

export z_scale_points, scale_utilities, convert_party_ideal_points_to_arrs
export generate_question_positions, find_issue_weights, filter_parties_below_threshold
export flatten_into_dict, find_geometric_median, get_bitmatrix_from_matrix_quartiles

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
min-max scaling to [0,1]

"""
function scale_utilities(profile_utilities::AbstractArray{Float64})

    min_val = minimum(profile_utilities)
    max_val = maximum(profile_utilities)
    normalized_utilities = (profile_utilities .- min_val) ./ (max_val - min_val)

    return normalized_utilities
end

"""
min-max to [a,b]

"""
function scale_utilities(profile_utilities::AbstractArray{Float64}, a::Float64, b::Float64)

    min_val = minimum(profile_utilities)
    max_val = maximum(profile_utilities)
    normalized_utilities = (profile_utilities .- min_val) ./ (max_val - min_val)
    scaled = normalized_utilities .* ((b - a) + a)
    return scaled
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

    return scale_utilities(voter_magnitudes)

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

_is_leaf(x) = (isa(x, Bool) || isa(x, Number))

# Check if something is a composite structure (non-leaf, non-vector, and has fields)
@inline function _is_composite(x)
    return !(_is_leaf(x)) && !isa(x, AbstractVector) && !isempty(fieldnames(typeof(x)))
end

# Updated flatten_result that first checks for a vector regardless of element type.
function flatten_result(x; prefix="")
    result = OrderedDict{Symbol,Any}()
    for field in fieldnames(typeof(x))
        fieldname = Symbol(field)  # Use Symbol instead of String for keys

        # Skip fields containing "sd" or "p_val"
        if occursin("sd", String(fieldname)) || occursin("p_val", String(fieldname))
            continue
        end

        # Build the prefixed column name as a Symbol
        new_prefix = isempty(prefix) ? fieldname : Symbol(string(prefix, "_", fieldname))
        val = getfield(x, field)

        if isa(val, AbstractVector)
            for (i, element) in enumerate(val)
                result[Symbol(string(new_prefix, "_", i))] = element
            end
        elseif _is_leaf(val)
            result[new_prefix] = val
        elseif _is_composite(val)
            subdict = flatten_result(val; prefix=string(new_prefix))
            merge!(result, subdict)
        else
            # Skip other types
        end
    end
    return result
end
"""
    is_structlike(x)

Returns true if `x` looks like a struct or a NamedTuple that we can iterate via `fieldnames`.
"""
function is_structlike(x)
    # NamedTuple is easy to check
    if x isa NamedTuple
        return true
    end
    # For most concrete structs you define, fieldnames(...) is non-empty
    # or at least recognized. (Be sure x is not nothing or missing before calling!)
    T = typeof(x)
    # skip "primitive" (like Int, Float64, etc.) or abstract types
    if isprimitivetype(T) || isabstracttype(T)
        return false
    end
    # Heuristic: struct if it has at least one field name
    return !isempty(fieldnames(T))
end

"""
    flatten_struct!(acc, obj; prefix = "")

Recursively flattens `obj` into the dictionary `acc`, with column names formed
from `prefix` plus the field name. Skips any field whose name contains "sd" or "p_val".
Expands numeric vectors. Stores bool/int/float fields as single values.
Handles `missing` and `nothing` by storing them as single columns (unless prefix is empty).
"""
function flatten_struct!(acc::AbstractDict{String,Any}, obj; prefix::String="")
    # 1) If the object is `missing`, store it if we have a prefix, then return
    if obj === missing
        if !isempty(prefix)
            acc[prefix] = missing
        end
        return
    end

    # 2) If the object is `nothing`, we can either store it or skip:
    if obj === nothing
        if !isempty(prefix)
            acc[prefix] = nothing
        end
        return
    end

    # 3) If the object is a Bool, Int, Float, etc. store under prefix (if prefix != "")
    if obj isa Number || obj isa Bool
        if !isempty(prefix)
            acc[prefix] = obj
        end
        return
    end

    # 4) If the object is a numeric Vector, expand each element into prefix_1, prefix_2, ...
    if obj isa AbstractVector{<:Number}
        if !isempty(prefix)
            for (i, val) in enumerate(obj)
                acc["$(prefix)_$(i)"] = val
            end
        end
        return
    end

    # 5) If it's some other vector (non-numeric), decide how to handle. Example: skip or flatten recursively.
    #    For demonstration, let's skip them or store as-is. 
    if obj isa AbstractVector
        # If you want to store them as is:
        if !isempty(prefix)
            acc[prefix] = obj
        end
        return
    end

    # 6) If it's struct-like, we attempt to flatten its fields
    if is_structlike(obj)
        # flatten each field
        for fn in fieldnames(typeof(obj))
            fn_str = String(fn)
            # skip if the name has "sd" or "p_val"
            if occursin("sd", fn_str) || occursin("p_val", fn_str)
                continue
            end
            fieldval = getfield(obj, fn)
            # build next prefix
            new_prefix = isempty(prefix) ? lowercase(fn_str) : string(prefix, "_", lowercase(fn_str))
            flatten_struct!(acc, fieldval; prefix=new_prefix)
        end
        return
    end

    # 7) If none of these matched, we can store it as-is if we have a prefix
    if !isempty(prefix)
        acc[prefix] = obj
    end
end


"""
    flatten_into_dict(result_structs...; remove_substring = "")

Flattens one or more result objects into a single dictionary.
If `remove_substring` is provided, it removes that substring from keys
before converting them to Symbols.
"""
function flatten_into_dict(result_structs...; remove_substring="")
    acc = Dict{String,Any}()
    for rs in result_structs
        # Use the type's name as a top-level prefix, e.g., "spatialautocorrelationmeasurement"
        type_prefix = lowercase(string(typeof(rs).name.wrapper))

        flatten_struct!(acc, rs; prefix=type_prefix)
    end

    # Step 1: Remove the specified substring from all keys
    if !isempty(remove_substring)
        acc = Dict(replace(key, remove_substring => "") => value for (key, value) in acc)
    end

    return acc
end

"""
For moderate 1 issue:
L = 1, U = 2, W = 10

For moderate 3 issue:
L = 3, U = 6, W = 30

For moderate 5 issue:
L = 5, U = 10, W = 50

L is lower bound on sum of rows, U is upper bound on sum of rows, W is summer bound no sum of 
    all entries
"""
function generate_random_matrix(D, L, U, W; max_attempts=10000)
    attempts = 0

    while attempts < max_attempts
        attempts += 1

        # Generate a random 5x3 matrix with entries in {0, 1, 2, 3}
        mat = rand(0:3, 5, D)

        # Check row-sum constraints
        row_sums = sum(mat, dims=2)  # Sum along rows
        if all(L .<= row_sums .<= U)  # Ensure row sums are within bounds
            # Check total matrix sum constraint
            if sum(mat) >= W
                println("Solution found after $attempts attempts.")
                return mat  # Return valid matrix
            end
        end
    end

    # If no solution is found within the max_attempts, return nothing or raise an error
    println("No solution found after $max_attempts attempts.")
    return nothing
end

function generate_matrix_with_row_mean(mean, rows, cols; max_attempts=10_000)
    total_sum = mean * cols  # Total sum per row
    attempts = 0

    while attempts < max_attempts
        attempts += 1

        # Generate a random 5x3 matrix with entries in {0, 1, 2, 3}
        mat = rand(0:3, rows, cols)

        # Check if each row sums to total_sum
        row_sums = sum(mat, dims=2)  # Sum along rows
        if all(row_sums .== total_sum)
            println("Solution found after $attempts attempts.")
            return mat  # Return valid matrix
        end
    end

    # If no solution is found within the max_attempts, return nothing
    println("No solution found after $max_attempts attempts.")
    return nothing
end

function find_geometric_median(points::AbstractArray; tol=1e-6, max_iter=1000)
    """
    Computes the geometric median of a set of points using Weiszfeld's algorithm.

    Arguments:
    - points: A matrix where each column represents a point in space (e.g., 2D or 3D).
    - tol: Convergence tolerance for stopping criterion (default 1e-6).
    - max_iter: Maximum number of iterations (default 1000).

    Returns:
    - The estimated geometric median as a vector.
    """
    n, m = size(points)  # `n`: dimensions, `m`: number of points
    x = mean(points, dims=2)  # Start with the centroid as the initial guess

    for iter in 1:max_iter
        # Compute weights for each point
        distances = sqrt.(sum((points .- x) .^ 2, dims=1))
        if any(distances .== 0)
            # If the current estimate coincides with any point, return it
            return x
        end
        weights = 1.0 ./ distances

        # Update estimate using weighted average
        new_x = sum(points .* weights, dims=2) / sum(weights)

        # Check for convergence
        if norm(new_x - x) < tol
            return new_x
        end
        x = new_x
    end

    error("Failed to converge within $max_iter iterations")
end


"""
    get_bitmatrix_from_matrix_quartiles(E, p)

Returns a BitMatrix `S` of the same size as `E`, where:
`S[i,j] = true` if `E[i,j] > threshold`
and
`S[i,j] = false` otherwise.

The `threshold` is the `p`-th percentile of all values in `E`.
For example, `p = 0.75` yields the 75th percentile.
"""
function get_bitmatrix_from_matrix_quartiles(E::AbstractMatrix{Float64}, p::Float64,
    above_threshold::Bool)
    # Flatten E into a single vector
    vals = vec(E)

    # Compute the p-th percentile
    threshold = quantile(vals, p)

    # Compare each element of E to threshold, producing a BitMatrix

    if above_threshold
        return BitArray(E .> threshold)
    else
        return BitArray(E .< threshold)
    end
end

end