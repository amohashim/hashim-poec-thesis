module ExperimentDesignInterfaceTools

export read_in_subdesign, initialize_output_dataframe, intitialize_simulation_run
export create_results_frame, get_parameter_columns_for_unpacking, define_varing_parameters
export generate_DemographicCharacteristicsParams, generate_SpatialCharacteristicParams
export generate_IssueStructure, generate_SalienceStructure, generate_QuestionStructure
export generate_RepresentativesParams

using CSV
using DataFrames
using DataStructures
using StaticArrays
using Statistics

using ..ExperimentParameters
using ..SimulationParameters

function generate_DemographicCharacteristicsParams(factor_A_level::String)

    characteristic_type = FACTOR_A_LOOKUP_TABLE_T[factor_A_level]
    homogeneity = FACTOR_A_LOOKUP_TABLE_H[factor_A_level]

    return DemographicCharacteristicParams(characteristic_type, homogeneity)

end

function generate_SpatialCharacteristicParams(factor_B_level::String)

    n_metros, urbanization, urban_sprawl, spatial_dispersion = values(FACTOR_B_FIXED_PARAMS)
    a_vals = FACTOR_B_LOOKUP_TABLE[factor_B_level]

    return SpatialCharacteristicParams(
        n_metros, urbanization, urban_sprawl, spatial_dispersion, a_vals
    )
end

function generate_IssueStructure(factor_E_level::String)

    n_issues = FACTOR_E_LOOK_UP_TABLE_N[factor_E_level]
    issue_dimensions = FACTOR_E_LOOK_UP_TABLE_D[factor_E_level]

    return IssueStructure(n_issues, issue_dimensions)

end

function generate_SalienceStructure(factor_C_level::String, factor_D_level::String)

    demographic_salience = FACTOR_C_LOOKUP_TABLE[factor_C_level]
    demographic_cleavage_salience = FACTOR_D_LOOKUP_TABLE[factor_D_level]

    return SalienceStructure(demographic_salience, demographic_cleavage_salience)
end

function generate_QuestionStructure(factor_F_level::String)

    n_questions = FACTOR_F_LOOK_UP_TABLE_Q[factor_F_level]
    n_positions = FACTOR_F_LOOK_UP_TABLE_P[factor_F_level]

    return QuestionStructure(n_questions, n_positions)

end

function generate_RepresentativesParams(factor_G_level::String, factor_H_level::String,
    factor_I_level::String)

    α_political_class = FACTOR_G_LOOKUP_TABLE_ALPHA[factor_G_level]
    p_norm = FACTOR_G_LOOKUP_TABLE_P[factor_G_level]
    party_threshold = FACTOR_H_LOOKUP_TABLE[factor_H_level]
    α_candidate_entry = FACTOR_I_LOOKUP_TABLE[factor_I_level]

    return RepresentativesParams(α_political_class, p_norm, party_threshold, α_candidate_entry)

end

function generate_BranchParams(factor_J_level::String, factor_K_level::String,
    factor_L_level::String, factor_M_level::String, factor_N_level::String)

    n_parties = FACTOR_J_LOOKUP_TABLE[factor_J_level]
    n_candidates = FACTOR_K_LOOKUP_TABLE[factor_K_level]
    turnout_level = FACTOR_L_LOOKUP_TABLE[factor_L_level]
    strategic_level = FACTOR_M_LOOKUP_TABLE[factor_M_level]
    demographic_attitudes = FACTOR_N_LOOKUP_TABLE[factor_N_level]

    return BranchParams(
        n_parties, n_candidates, turnout_level, strategic_level, demographic_attitudes
    )

end

function define_varing_parameters(factor_E_level::String, factor_A_level::String,
    factor_B_level::String, factor_C_level::String, factor_D_level::String, factor_F_level::String,
    factor_G_level::String, factor_H_level::String, factor_I_level::String, factor_J_level::String,
    factor_K_level::String, factor_L_level::String, factor_M_level::String, factor_N_level::String)

    dem_char_params = generate_DemographicCharacteristicsParams(factor_A_level)
    spatial_params = generate_SpatialCharacteristicParams(factor_B_level)
    issue_structure = generate_IssueStructure(factor_E_level)
    salience_structure = generate_SalienceStructure(factor_C_level, factor_D_level)
    question_structure = generate_QuestionStructure(factor_F_level)
    representative_params = generate_RepresentativesParams(
        factor_G_level, factor_H_level, factor_I_level)
    branch_params = generate_BranchParams(
        factor_J_level, factor_K_level, factor_L_level, factor_M_level, factor_N_level
    )

    begin
        return dem_char_params, spatial_params, issue_structure, salience_structure, question_structure,
        representative_params, branch_params
    end

end

function read_in_subdesign(file_name::String)

    design_mat = CSV.read(file_name, DataFrame)
    design_mat = select!(design_mat, Not(1))
    # design_mat_clean = design_mat[
    #     :, map(x -> !all(ismissing, design_mat[!, x]), propertynames(design_mat))
    # ]
    design_mat = coalesce.(design_mat, "L0")

    return design_mat

end

@inline function initialize_output_dataframe(output_path::String)

    factors_frame = DataFrame([name => Union{Missing,Any}[] for name in FACTOR_ORDER]...)
    responses_frame = DataFrame(
        [response => Union{Missing,Any}[] for response in values(RAW_NAME_TO_NICE_NAME)]
    )
    output_frame = hcat(factors_frame, responses_frame)
    CSV.write(output_path, output_frame)
    return output_frame

end

@inline function initialize_output_dataframe()

    factors_frame = DataFrame([name => Union{Missing,Any}[] for name in FACTOR_ORDER]...)
    responses_frame = DataFrame(
        [response => Union{Missing,Any}[] for response in values(RAW_NAME_TO_NICE_NAME)]
    )
    output_frame = hcat(factors_frame, responses_frame)
    return output_frame

end

"""
to unpack the subdesign parameters into define_varying_parameters(), assuming the proper
order of FACTOR_ORDER

"""
@inline function get_parameter_columns_for_unpacking(design_matrix_row::DataFrameRow)

    df = DataFrame(design_matrix_row)

    factors = [factor for factor in FACTOR_ORDER if factor ∉ [:X, :Y, :Z]]
    select!(df, factors...)

    return df[1, :]

end

@inline function convert_raw_name_to_nice_name(raw_name_to_nice_name::OrderedDict{String,Symbol},
    measurements::AbstractDict)

    return OrderedDict(
        raw_name_to_nice_name[key] => measurements[key] for key in keys(raw_name_to_nice_name
        )
    )

end
function intitialize_simulation_run(run_number::Int, design_matrix::DataFrame)

    design_matrix_row = design_matrix[run_number, :]
    run_parameters = get_parameter_columns_for_unpacking(design_matrix_row)
    run_parameters_vec = convert.(String, collect(run_parameters))

    begin
        dem_char_params, spatial_params, issue_structure, salience_structure,
        question_structure, representative_params, branch_params =
            define_varing_parameters(run_parameters_vec...)
    end
    begin
        return dem_char_params, spatial_params, issue_structure, salience_structure,
        question_structure, representative_params, branch_params, design_matrix_row
    end

end

"""
`design` assumed to have the proper columns (i.e. `design_mat[row,:]` from `read_in_subdesign()`)

returns dataframe of [factors..., results...]
"""
function create_results_frame(results_dictionary::OrderedDict, design::DataFrameRow,
    factor_order::SVector{17,Symbol})

    des = DataFrame(design)
    select!(des, factor_order...)
    # des = coalesce.(des, "L0")

    return hcat(des, DataFrame(results_dictionary))

end

@inline function append_to_results(existing_results_frame::DataFrame,
    new_results::DataFrame)

    append!(existing_results_frame, new_results)

end


end