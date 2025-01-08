# Function to generate state-level probabilities for Low Homogeneity
function get_base_probabilities_low(n_categories::Int64)
    probs = ones(n_categories) / n_categories
    return probs
end

# Function to generate state-level probabilities for Medium Homogeneity (Nominal)
function get_base_probabilities_medium_nominal(n_categories::Int64)

    n_prominent = min(3, n_categories)  # Choose 2 or 3 categories
    prominent_categories = randperm(n_categories)[1:n_prominent]
    n_non_prominent = n_categories - n_prominent
    # Assign weights to prominent categories
    α_prominent = fill(2.0, n_prominent)
    ω_prominent = rand(Dirichlet(α_prominent))
    # Assign weights to non-prominent categories
    if n_non_prominent > 0
        α_non_prominent = ones(n_non_prominent)
        ω_non_prominent = rand(Dirichlet(α_non_prominent))
    else
        ω_non_prominent = Float64[]
    end
    # Combine and normalize

    probs = zeros(n_categories)
    idx = 1
    for cat in 1:n_categories
        if cat in prominent_categories
            probs[cat] = ω_prominent[findfirst(==(cat), prominent_categories)]
        else
            if n_non_prominent > 0
                probs[cat] = ω_non_prominent[idx]
                idx += 1
            end
        end
    end
    probs /= sum(probs)
    return probs

end

# Function to generate state-level probabilities for Medium Homogeneity (Ordinal)
function get_base_probabilities_medium_ordinal(n_categories::Int64)

    peak_category = rand(1:n_categories)
    distances = abs.(collect(1:n_categories) .- peak_category)
    base_ω = exp.(-distances / 2.0)
    noise = rand(n_categories) .* 0.4 .+ 0.8  # Uniform(0.8, 1.2)
    ω = base_ω .* noise
    probabilities = ω / sum(ω)

    return probabilities
end

# Function to generate state-level probabilities for High Homogeneity (Ordinal)
function get_base_probabilities_high(n_categories::Int64)

    peak_category = rand(1:n_categories)
    peak_prob = rand(Uniform(0.70, 0.85))
    non_peak_categories = setdiff(1:n_categories, [peak_category])
    remaining_prob = 1.0 - peak_prob
    n_non_peak = n_categories - 1

    if n_non_peak > 0
        α_non_peak = ones(n_non_peak)
        ω_non_peak = rand(Dirichlet(α_non_peak))
        non_peak_probs = ω_non_peak .* remaining_prob
    else
        non_peak_probs = Float64[]
    end
    probs = zeros(n_categories)
    probs[peak_category] = peak_prob
    idx = 1
    for cat in non_peak_categories
        probs[cat] = non_peak_probs[idx]
        idx += 1
    end
    return probs
end

# Function to generate state-level probabilities for Perfect Homogeneity
function get_base_probabilities_perfect(n_categories::Int64)
    peak_category = rand(1:n_categories)
    probs = zeros(n_categories)
    probs[peak_category] = 1.0
    return probs
end

# Main function to generate state-level probabilities based on homogeneity and characteristic type
function generate_state_level_probabilities(homogeneity::Symbol, characteristic_type::Symbol,
    n_categories::Int64)
    if homogeneity == :low
        return get_base_probabilities_low(n_categories)
    elseif homogeneity == :moderate
        if characteristic_type == :nominal
            return get_base_probabilities_medium_nominal(n_categories)
        elseif characteristic_type == :ordinal
            return get_base_probabilities_medium_ordinal(n_categories)
        else
            error("Invalid characteristic type: $characteristic_type")
        end
    elseif homogeneity == :high
        return get_base_probabilities_high(n_categories)
    elseif homogeneity == :perfect
        return get_base_probabilities_perfect(n_categories)
    else
        error("Invalid homogeneity level: $homogeneity")
    end
end