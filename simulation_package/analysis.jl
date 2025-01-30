using StatsModels, Combinatorics
using GLM, RegressionTables, DataFrames

function generate_specific_interaction_formula(response::Symbol, predictors::Vector{Symbol}, interaction_var::Symbol, order::Int)
    terms = [string(v) for v in predictors if v != interaction_var]  # Exclude interaction_var from main effects
    interaction_terms = []

    for k in 1:order-1  # Only interact with other variables (not itself)
        append!(interaction_terms, [string(interaction_var, " & ", join(comb, " & ")) for comb in combinations(terms, k)])
    end

    formula_str = "$(response) ~ " * join(predictors, " + ")  # Include all main effects
    if !isempty(interaction_terms)
        formula_str *= " + " * join(interaction_terms, " + ")  # Add interactions if any
    end

    return eval(Meta.parse("@formula($formula_str)"))
end


using StatsModels, Combinatorics

function generate_interaction_formula(response::Symbol, predictors::Vector{Symbol}, order::Int)
    terms = [string(v) for v in predictors]

    interaction_terms = []
    for k in 1:order
        append!(interaction_terms, [join(comb, " & ") for comb in combinations(terms, k)])
    end

    formula_str = "$(response) ~ " * join(terms, " + ") * " + " * join(interaction_terms, " + ")

    return eval(Meta.parse("@formula($formula_str)"))
end

using GLM, RegressionTables, DataFrames

function glm_to_table(model::StatsModels.TableRegressionModel; file_path::Union{String,Nothing}=nothing)
    table = regtable(model, render=:ascii)  # Generates a readable table

    if file_path !== nothing
        open(file_path, "w") do f
            write(f, regtable(model, render=:text))  # Save as text file
        end
        println("Table saved to: $file_path")
    else
        println(table)  # Print the table in readable format
    end
end

function glm_to_table(model::RegressionModel; file_path::Union{String,Nothing}=nothing)
    table_output = regtable((model,); render=:ascii)  # Ensure model is treated as a tuple

    if file_path !== nothing
        open(file_path, "w") do f
            write(f, regtable((model,); render=:text))  # Save to file
        end
        println("Table saved to: $file_path")
    else
        println(table_output)  # Print table in the terminal
    end
end

FILE_PATH = "/Users/alihashim/Desktop/cleaned_data.csv"
function main(file_path::String)

    df = CSV.read(file_path, DataFrame)
    df = coalesce(df)
    response = BitVector(df.tangian_adj_pop_parties .> 0.5)

    df[!, :tangian_pop_party_positive] = Int.(response)

    df_issue_1 = df[df.E.=="L1", :]

    df_issue_1 =
        formula = generate_specific_interaction_formula(:response, cat_cols, :F, 2)
    model = glm(formula, data, Binomial(), LogitLink())
    regtable(
        model; significance_levels=[0.01, 0.05, 0.1], digits=2,
        extralines=["Note: * p<0.1, ** p<0.05, *** p<0.01"]
    )


end

