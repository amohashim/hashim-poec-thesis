module PreferenceSimulation

export QuestionAndIssueStructure, SimulationTools

module QuestionAndIssueStructure

using StaticArrays
using OrderedCollections

export AbstractQuestion
export NonDichotomousQuestion, DichotomousQuestion
export DecomposedDichotomousQuestion, MappedDecomposedDichotomousQuestion

export AbstractIssue
export ArbitraryIssue, MappedDecomposedIssue

abstract type AbstractQuestion{T} end
abstract type AbstractIssue{D} end

struct NonDichotomousQuestion{T} <: AbstractQuestion{T}

    question::T

end

struct DichotomousQuestion{T} <: AbstractQuestion{T}

    question::T

end

struct DecomposedDichotomousQuestion{T} <: AbstractQuestion{T}

    # Set of mutually exclusive DichotomousQuestion objects

    n_component_questions::Int64
    component_questions::Vector{DichotomousQuestion{T}}

end

struct MappedDecomposedDichotomousQuestion{D,T} <: AbstractQuestion{T}

    n_component_questions::Int64
    component_questions::Vector{DichotomousQuestion{T}}
    question_positions::Vector{SVector{D,Float64}}
    question_to_position::OrderedDict{DichotomousQuestion{T},SVector{D,Float64}}

    function MappedDecomposedDichotomousQuestion(
        n_component_questions::Int64, component_questions::Vector{DichotomousQuestion{T}},
        question_positions::Vector{SVector{D,Float64}}
    ) where {D,T}

        question_to_position = OrderedDict{DichotomousQuestion{T},SVector{D,Float64}}(
            q => pos for (q, pos) in zip(component_questions, question_positions)
        )

        new{D,T}(
            n_component_questions, component_questions, question_positions, question_to_position
        )

    end

end

struct ArbitraryIssue{D,S,T} <: AbstractIssue{D}

    dimensionality::Int64
    dimensions::SVector{D,S}
    questions::Vector{AbstractQuestion{T}}

    function ArbitraryIssue(
        dimensions::AbstractVector, questions::AbstractVector
    ) where {D,S,T}
        new{D,S,T}(D, dimensions, questions)
    end

end

struct MappedDecomposedIssue{D,S,T} <: AbstractIssue{D}

    dimensionality::Int64
    dimensions::SVector{D,S}
    questions_by_decomposition::Vector{MappedDecomposedDichotomousQuestion{D,T}}
    questions::Vector{DichotomousQuestion{T}}
    question_positions::Vector{SVector{D,Float64}}
    question_to_position::OrderedDict{DichotomousQuestion{T},SVector{D,Float64}}

    function MappedDecomposedIssue(
        dimensions::SVector{D,S},
        questions_by_decomposition::Vector{MappedDecomposedDichotomousQuestion{D,T}}
    ) where {D,S,T}

        questions = vcat(getfield.(questions_by_decomposition, :component_questions)...)
        question_positions = vcat(getfield.(questions_by_decomposition, :question_positions)...)
        question_to_positions = reduce(
            merge, getfield.(questions_by_decomposition, :question_to_position)
        )

        new{D,S,T}(
            D, dimensions, questions_by_decomposition, questions, question_positions,
            question_to_positions
        )

    end

end

end

module SimulationEssentials

using Distributions
using LinearAlgebra
using Random
using StatsFuns
using StaticArrays

using .QuestionAndIssueStructure

export compute_prob_of_positive_along_dimension_d, determine_typology
export simulate_from_beta_glm, sample_points, find_probabilities_of_positions
export sample_positions_on_issue

"""
computes probabilities for a given typology (parameterized by θ)

"""
function compute_prob_of_positive_along_dimension_d(factors::BitVector, θ::AbstractVector{Float64})

    dot_product = dot(θ, factors)

    return (exp(dot_product) / (1 + exp(dot_product)))

end

function compute_prob_of_positive_along_dimension_d(factors::BitMatrix, θ::AbstractVector{Float64})

    dot_products = factors * θ

    return (exp.(dot_products) ./ (1 .+ exp.(dot_products)))

end

"""
θ is CxD

"""
function determine_typology(factors::BitVector, θ::AbstractMatrix{Float64})

    probabilities = hcat(compute_prob_of_positive_along_dimension_d.(Ref(factors), eachcol(θ))...)

    typology = hcat(map(t -> rand.(Bernoulli.(t)), eachcol(probabilities))...)

    return BitArray(typology)

end

"""
X: matrix of characteristics (N x C)
β: vector of coefficients (Cx1)
offset: constant with which to shift η when X = 0
ϕ: dispersion parameter
simulate: whether to simulate or predict mean
ϵ: lower bound on Beta parameters

returns: N-vector with elements between 0 and 1

"""
# Function to compute predicted values or simulate data from Beta GLM
function simulate_from_beta_glm(X::BitMatrix, β::AbstractVector{Float64}; offset::Float64=0.0,
    ϕ::Float64=10.0, simulate::Bool=true, ϵ::Float64=1e-6)
    # Ensure that X and β have compatible dimensions
    if size(X, 2) != length(β)
        throw(DimensionMismatch("Number of columns in X must match the length of β"))
    end

    # Step 1: Compute linear predictor
    # Assumes that when X = 0, η = 0 + offset
    η = X * β .+ offset # Linear predictor (n x 1 vector)

    # Step 2: Apply inverse link function (logistic function for logit link)
    μ = @. 1 / (1 + exp(-η))  # Mean of Beta distribution

    if simulate
        # Step 3: Compute α and β parameters of Beta distribution
        α = @. max(μ * ϕ, ϵ)
        β_params = @. max((1 - μ) * ϕ, ϵ)  # Avoid naming conflict with coefficients β

        # Step 4: Generate response variable Y ~ Beta(α, β_params)
        Y = rand.(Beta.(α, β_params))
        return Y
    else
        # Return the predicted mean values
        return μ
    end
end

"""
for an observation

"""
function simulate_from_beta_glm(X::BitVector, β::AbstractVector{Float64}; offset::Float64=0.0,
    ϕ::Float64=10.0, simulate::Bool=true, ϵ::Float64=1e-6)
    # Ensure that X and β have compatible dimensions
    if length(X) != length(β)
        throw(DimensionMismatch("Number of columns in X must match the length of β"))
    end

    # Step 1: Compute linear predictor
    η = dot(X, β) + offset # Linear predictor (n x 1 vector)

    # Step 2: Apply inverse link function (logistic function for logit link)
    μ = 1 / (1 + exp(-η))  # Mean of Beta distribution

    if simulate
        # Step 3: Compute α and β parameters of Beta distribution
        α = max(μ * ϕ, ϵ)
        β_params = max((1 - μ) * ϕ, ϵ)  # Avoid naming conflict with coefficients β

        # Step 4: Generate response variable Y ~ Beta(α, β_params)
        Y = rand(Beta(α, β_params))
        return Y
    else
        # Return the predicted mean values
        return μ
    end
end

"""
samples a point in a unit hypercube (in a given typology) for an individual/individuals with given 
    characteristics and characteristic-coefficients

factors: vector of characteristics
coefficient_matrix: CxD matrix of coefficients

"""

function sample_points(dimensionality::Int64, typology::SVector{D,Int64},
    factors_matrix::BitArray, factor_coefficient_matrix::AbstractMatrix{Float64},
    ϕ_vector::SVector{D,Float64}, offset_vector::SVector{D,Float64}) where {D}

    F, Ψ = factors_matrix, factor_coefficient_matrix

    point = map(
        t -> simulate_from_beta_glm(F, t[1], offset=t[2], ϕ=t[3]),
        zip(eachcol(Ψ), ϕ_vector, offset_vector)
    ) # should work for a BitMatrix or BitArray because simulate_from_beta_glm has two methods

    if typeof(F) == BitVector
        return SVector{dimensionality,Float64}(point .* typology)
    else
        return hcat(points...)
    end

end

function find_nearest_position(point::SVector{D,Float64},
    question::MappedDecomposedDichotomousQuestion{D,T}) where {D,T}

    question_to_position = getfield(question, :question_to_position)

    min_distance = Inf
    best_key = nothing
    ties = 0

    @inbounds for (key, question_position) in question_to_position

        sum_of_squares = dot(point - question_position, point - question_position)

        if sum_of_squares < min_distance
            min_distance = sum_of_squares
            best_key = key
            ties = 1
        elseif sum_of_squares == min_distance
            # With a tie, replace the current best key randomly with probability 1/(ties+1)
            ties += 1
            if rand() < 1 / ties
                best_key = key
            end
        end
    end

    return best_key

end

"""
Softmax function; converts distances to probabilities as follows:

    `P_iq = exp(gamma*D_iq) / (sum_{q ∈ Q} exp(gamma*D_iq))`

Probabilities add up to 1.

"""
function map_distances_to_probabilities(n_component_questions::Int64,
    scores::AbstractVector{Float64}, negative_gamma::Float64)

    exp_terms = exp.(negative_gamma .* scores)
    denominator = sum(exp_terms)

    probs = (exp_term -> exp_term / denominator).(exp_terms)

    return SVector{n_component_questions,Float64}(probs)

end

function find_probabilities_of_positions(point::SVector{D,Float64},
    question::MappedDecomposedDichotomousQuestion{D,T}, negative_gamma::Float64) where {D,T}

    positions = getfield(question, :question_positions)
    distances = norm.(Ref(point) .- positions)
    probabilities = map_distances_to_probabilities(
        question.n_component_questions, distances, negative_gamma
    )

    question_to_probs = OrderedDict{DichotomousQuestion{T},Float64}(
        q => probs for (q, probs) in zip(question.component_questions, probabilities)
    )

    return question_to_probs

end

function sample_from_ordered_dict(dictionary::OrderedDict{Any,Float64})

    categories, probabilities = keys(dictionary), collect(values(dictionary))
    categorical_dist = Categorical(probabilities)

    sampled_index = rand(categorical_dist)

    return categories[sampled_index]

end

function sample_positions_on_issue(point::SVector{D,Float64}, issue::MappedDecomposedIssue{D,S,T},
    negative_gamma::Float64) where {D,S,T}

    positions = OrderedDict{DichotomousQuestion{T},Int64}(
        question => 0 for question in issue.questions
    )

    @inbounds for decomposed_question in issue.questions_by_decomposition

        distribution = find_probabilities_of_positions(point, decomposed_question, negative_gamma)
        yes_question = sample_from_ordered_dict(distribution)

        @inbounds for component_question in decomposed_question.component_questions

            positions[component_question] = component_question == yes_question ? 1 : -1

        end

    end

    return positions

end

end

module SimulationUtilities

using .QuestionAndIssueStructure

function generate_labels(numbers::Vector{Int})
    labels = String[]
    alphabet = 'a':'z'

    for n in numbers
        # Decrement n to align with 0-based indexing (makes handling carryover easier)
        n -= 1
        label = ""
        while n >= 0
            letter = alphabet[mod(n, 26)+1]
            label = string(letter, label)
            n = div(n, 26) - 1
        end
        push!(labels, label)
    end

    return labels
end

# assumes the question only matters wrt a single dimension
function randomly_generate_a_MappedDecomposedDichotomousQuestion(
    n_component_questions::Int64, question_label::Int64, issue_name::String,
    issue_dimensionality::Int64, question_dimension::Int64
)

    question_labels = generate_labels(collect(1:n_component_questions))
    question_names = Symbol.(issue_name * "_question_$question_label" .* question_labels)
    positions = LinRange(-1, 1, n_component_questions)
    positions = (t -> t == 0.0 ? 0.0001 : t).(positions)

    component_questions = DichotomousQuestion{Symbol}.(question_names)
    question_positions = [Tuple(zeros(issue_dimensionality)) for _ in 1:n_component_questions]
    question_positions = SVector{issue_dimensionality,Float64}.(
        setindex.(question_positions, positions, question_dimension)
    )

    dimension

    ob = MappedDecomposedDichotomousQuestion(
        n_component_questions, component_questions, question_positions
    )

    return ob

end

function randomly_generate_a_MappedDecomposedIssue(dimensionality::Int64, n_questions::Int64,
    issue_name::String
    
    for i in
    )

end

end


end