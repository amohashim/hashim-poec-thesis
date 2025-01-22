module ExperimentalDesign

export FACTOR_ORDER, FACTOR_NAMES, SPATIAL_MEASUREMENT_ORDER

using StaticArrays

const FACTOR_ORDER = SVector{17,Symbol}(
    [:E, :X, :Y, :Z, :A, :B, :C, :D, :F, :G, :H, :I, :J, :K, :L, :M, :N]
)

const RAW_NAME_TO_NICE_NAME = OrderedDict{String,Symbol}(
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_utility_efficiency" => :prop_strict_util_eff,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_utility_efficiency_median_" => :major_util_eff_med,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_utility_from_maximizer" => :prop_qual_util_from_max,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_single_party_win" => :prop_strict_single_party,
    "spatialcharacteristics.spatialautocorrelationmeasurement_characteristic_level_corr_1" => :spat_char_corr1,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_election_failed" => :prop_strict_elec_fail,
    "spatialcharacteristics.spatialautocorrelationmeasurement_median_entropies_3" => :spat_med_entropy3,
    "spatialcharacteristics.spatialautocorrelationmeasurement_average_entropies_1" => :spat_avg_entropy1,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_winner_in_smith_set" => :prop_qual_winner_smith,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_util_maxer_elected" => :prop_strict_maxer_elected,
    "spatialcharacteristics.spatialautocorrelationmeasurement_average_entropies_5" => :spat_avg_entropy5,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_condorcet_paradox" => :prop_strict_condorcet_paradox,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_single_party_win" => :prop_qual_single_party,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_util_maximizer_election_rate" => :major_maximizer_rate,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_utility_from_winner_avg" => :major_util_from_winner_avg,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_unanimity" => :prop_strict_unanimity,
    "tangianindicesresults_raw_uni_parties" => :tangian_raw_uni_parties,
    "tangianindicesresults_adj_uni_body" => :tangian_adj_uni_body,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_unanimity_rate" => :prop_strict_unanimity_rate,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_condorcet_winner_elected" => :prop_strict_condorcet_winner,
    "spatialcharacteristics.spatialautocorrelationmeasurement_median_entropies_5" => :spat_med_entropy5,
    "spatialcharacteristics.spatialautocorrelationmeasurement_characteristic_level_corr_3" => :spat_char_corr3,
    "spatialcharacteristics.spatialautocorrelationmeasurement_characteristic_level_corr_5" => :spat_char_corr5,
    "majoritarianevaluationmetrics.majoritarianevaluation_voting_efficiency_stats_n_condorcet_paradox" => :major_vot_eff_n_cond_paradox,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_minority_govt_indicators_size" => :prop_qual_min_govt_size,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_winner_in_smith_set" => :prop_strict_winner_smith,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_median_position" => :prop_qual_med_pos,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_utility_from_winner" => :prop_qual_util_from_winner,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_minority_govt_indicators_unstable" => :prop_qual_min_govt_unstable,
    "spatialcharacteristics.spatialautocorrelationmeasurement_average_entropies_3" => :spat_avg_entropy3,
    "tangianindicesresults_raw_uni_body" => :tangian_raw_uni_body,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_utility_from_maximizer_avg" => :major_util_from_max_avg,
    "spatialcharacteristics.spatialautocorrelationmeasurement_median_entropies_2" => :spat_med_entropy2,
    "spatialcharacteristics.spatialautocorrelationmeasurement_average_entropies_2" => :spat_avg_entropy2,
    "spatialcharacteristics.spatialautocorrelationmeasurement_median_entropies_1" => :spat_med_entropy1,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_utility_from_maximizer" => :prop_strict_util_from_max,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_non_zero_positions" => :prop_qual_nonzero_pos,
    "majoritarianevaluationmetrics.majoritarianevaluation_qualified_decisive_body_metrics_non_zero_positions" => :major_qual_nonzero_pos,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_minority_govt_indicators_elected" => :prop_strict_min_govt_elected,
    "majoritarianevaluationmetrics.majoritarianevaluation_voting_efficiency_stats_condorcet_efficiency" => :major_vot_eff_cond_eff,
    "majoritarianevaluationmetrics.majoritarianevaluation_voting_efficiency_stats_smith_efficiency" => :major_vot_eff_smith_eff,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_condorcet_paradox" => :prop_qual_condorcet_paradox,
    "tangianindicesresults_raw_pop_parties" => :tangian_raw_pop_parties,
    "tangianindicesresults_adj_pop_parties" => :tangian_adj_pop_parties,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_utility_from_maximizer_median_" => :major_util_from_max_med,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_utility_from_winner_median_" => :major_util_from_win_med,
    "majoritarianevaluationmetrics.majoritarianevaluation_strict_decisive_body_metrics_median_position" => :major_strict_med_pos,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_utility_efficiency_avg" => :major_util_eff_avg,
    "tangianindicesresults_adj_uni_parties" => :tangian_adj_uni_parties,
    "majoritarianevaluationmetrics.majoritarianevaluation_maj_gallagher_index" => :major_gallagher,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_election_failed" => :prop_qual_elec_fail,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_util_maxer_elected" => :prop_qual_maxer_elected,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_utility_efficiency" => :prop_qual_util_eff,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_unanimity_rate" => :prop_qual_unanimity_rate,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_minority_govt_indicators_size" => :prop_strict_min_govt_size,
    "majoritarianevaluationmetrics.majoritarianevaluation_strict_decisive_body_metrics_non_zero_positions" => :major_strict_nonzero_pos,
    "proportionalevaluationmetrics.proportionalevaluation_prop_gallagher_index" => :prop_gallagher,
    "majoritarianevaluationmetrics.majoritarianevaluation_strict_decisive_body_metrics_social_utility_from_body" => :major_strict_social_util,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_non_zero_positions" => :prop_strict_nonzero_pos,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_median_position" => :prop_strict_med_pos,
    "majoritarianevaluationmetrics.majoritarianevaluation_qualified_decisive_body_metrics_social_utility_from_body" => :major_qual_social_util,
    "spatialcharacteristics.spatialautocorrelationmeasurement_aggregated_distance_corr" => :spat_agg_dist_corr,
    "majoritarianevaluationmetrics.majoritarianevaluation_qualified_decisive_body_metrics_median_position" => :major_qual_med_pos,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_condorcet_winner_elected" => :prop_qual_condorcet_winner,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_minority_govt_indicators_unstable" => :prop_strict_min_govt_unstable,
    "spatialcharacteristics.spatialautocorrelationmeasurement_characteristic_level_corr_2" => :spat_char_corr2,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_utility_from_winner" => :prop_strict_util_from_winner,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_minority_govt_indicators_elected" => :prop_qual_min_govt_elected,
    "tangianindicesresults_adj_pop_body" => :tangian_adj_pop_body,
    "spatialcharacteristics.spatialautocorrelationmeasurement_average_entropies_4" => :spat_avg_entropy4,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_unanimity" => :prop_qual_unanimity,
    "spatialcharacteristics.spatialautocorrelationmeasurement_median_entropies_4" => :spat_med_entropy4,
    "spatialcharacteristics.spatialautocorrelationmeasurement_characteristic_level_corr_4" => :spat_char_corr4,
    "tangianindicesresults_raw_pop_body" => :tangian_raw_pop_body
)


const FACTOR_NAMES = Dict{Symbol,Symbol}(
    :E => :issue_structure,
    :X => :static_candidates, # baseline (L1 or 0) is dynamic candidates
    :Y => :exogeneous_parties, # baseline (L1 or 0) is endogeneous parties
    :Z => :perfect_voters, # baseline (L1 or 0) is imperfect voters
    :A => :demographic_chars, :B => :spatial_chars,
    :C => :demographic_salience, :D => :demographic_cleavage_salience,
    :F => :question_structure, :G => :engagement_structure,
    :H => :party_threshold, :I => :α_candidate_entry,
    :J => :n_parties, :K => :n_candidates,
    :L => :turnout_level, :M => :strategic_level,
    :N => :demographic_attitudes
)

@inline function convert_raw_name_to_nice_name(raw_name_to_nice_name::OrderedDict{String,Symbol},
    measurements::AbstractDict)

    return OrderedDict(
        raw_name_to_nice_name[key] => measurements[key] for key in keys(raw_name_to_nice_name
        )
    )

end

function read_in_subdesign(file_name::String)

    design_mat = CSV.read(file_name, DataFrame)
    design_mat = select!(design_mat, Not(1))
    design_mat_clean = design_mat[
        :, map(x -> !all(ismissing, design_mat[!, x]), propertynames(design_mat))
    ]

    return design_mat, design_mat_clean

end

@inline function initialize_names_dataframe(factor_order::SVector{17,Symbol})

    return DataFrame([name => Union{Missing,Any}[] for name in factor_order]...)
end

"""
`design` assumed to have the proper columns (i.e. `design_mat[row,:]` from `read_in_subdesign()`)

returns dataframe of [factors..., results...]
"""
function create_results_frame(results_dictionary::OrderdDict, design::DataFrameRow,
    factor_order::SVector{17,Symbol})

    des = DataFrame(design)
    select!(des, factor_order...)
    des = coalesce.(des, "L0")

    return hcat(des, DataFrame(results_dictionary))

end




end