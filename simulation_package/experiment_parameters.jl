module ExperimentParameters

begin
    export FACTOR_ORDER, FACTOR_NAMES, RAW_NAME_TO_NICE_NAME, FACTOR_A_LOOKUP_TABLE_H,
        FACTOR_A_LOOKUP_TABLE_T, FACTOR_B_LOOKUP_TABLE, FACTOR_B_FIXED_PARAMS,
        FACTOR_C_LOOKUP_TABLE, FACTOR_D_LOOKUP_TABLE, FACTOR_E_LOOK_UP_TABLE_N,
        FACTOR_E_LOOK_UP_TABLE_D, FACTOR_F_LOOK_UP_TABLE_Q, FACTOR_F_LOOK_UP_TABLE_P,
        FACTOR_G_LOOKUP_TABLE_ALPHA, FACTOR_G_LOOKUP_TABLE_P, FACTOR_H_LOOKUP_TABLE,
        FACTOR_I_LOOKUP_TABLE, FACTOR_J_LOOKUP_TABLE, FACTOR_K_LOOKUP_TABLE, FACTOR_L_LOOKUP_TABLE,
        FACTOR_M_LOOKUP_TABLE, FACTOR_N_LOOKUP_TABLE
end

using StaticArrays
using DataStructures

const FACTOR_ORDER = SVector{17,Symbol}(
    [:X, :Y, :Z, :E, :A, :B, :C, :D, :F, :G, :H, :I, :J, :K, :L, :M, :N]
)

const RAW_NAME_TO_NICE_NAME = OrderedDict{String,Symbol}(
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_median_position" => :prop_strict_med_pos,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_n_winning_parties" => :prop_qual_n_winning_parties,
    "majoritarianevaluationmetrics.pluratarianevaluation_utility_metrics_utility_from_winner_avg" => :plur_util_winner_avg,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_minority_govt_indicators_unstable" => :prop_qual_min_govt_unstable,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_condorcet_winner_elected" => :prop_qual_condorcet_winner,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_minority_govt_indicators_size" => :prop_strict_min_govt_size,
    "spatialcharacteristics.spatialautocorrelationmeasurement_characteristic_level_corr_4" => :spat_char_corr4,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_utility_from_winner" => :prop_strict_util_winner,
    "majoritarianevaluationmetrics.majoritarianevaluation_qualified_decisive_body_metrics_median_position" => :major_qual_med_pos,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_unanimity" => :prop_strict_unanimity,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_size_of_coalition" => :prop_qual_coalition_size,
    "majoritarianevaluationmetrics.pluratarianevaluation_utility_metrics_utility_from_maximizer_median_" => :plur_util_max_med,
    "proportionalevaluationmetrics.proportionalevaluation_prop_gallagher_index" => :prop_gallagher_index,
    "tangianindicesresults_adj_pop_parties" => :tangian_adj_pop_parties,
    "spatialcharacteristics.spatialautocorrelationmeasurement_characteristic_level_corr_5" => :spat_char_corr5,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_utility_efficiency" => :prop_qual_util_eff,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_utility_from_maximizer_avg" => :major_util_max_avg,
    "majoritarianevaluationmetrics.majoritarianevaluation_voting_efficiency_stats_condorcet_efficiency" => :major_cond_eff,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_non_zero_positions" => :prop_strict_nonzero_pos,
    "majoritarianevaluationmetrics.majoritarianevaluation_qualified_decisive_body_metrics_non_zero_positions" => :major_qual_nonzero_pos,
    "majoritarianevaluationmetrics.majoritarianevaluation_vse" => :major_vse,
    "majoritarianevaluationmetrics.pluratarianevaluation_vse" => :plur_vse,
    "spatialcharacteristics.spatialautocorrelationmeasurement_median_entropies_1" => :spat_med_entropy1,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_utility_from_maximizer" => :prop_strict_util_max,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_util_maximizer_election_rate" => :major_util_max_elec_rate,
    "majoritarianevaluationmetrics.majoritarianevaluation_voting_efficiency_stats_n_condorcet_paradox" => :major_cond_paradox,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_election_failed" => :prop_qual_elec_failed,
    "majoritarianevaluationmetrics.majoritarianevaluation_strict_decisive_body_metrics_non_zero_positions" => :major_strict_nonzero_pos,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_utility_from_maximizer_median_" => :major_util_max_med,
    "tangianindicesresults_raw_pop_body" => :tangian_raw_pop_body,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_utility_from_winner" => :prop_qual_util_winner,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_non_zero_positions" => :prop_qual_nonzero_pos,
    "spatialcharacteristics.spatialautocorrelationmeasurement_median_entropies_3" => :spat_med_entropy3,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_condorcet_paradox" => :prop_strict_cond_paradox,
    "majoritarianevaluationmetrics.pluratarianevaluation_voting_efficiency_stats_smith_efficiency" => :plur_smith_eff,
    "majoritarianevaluationmetrics.majoritarianevaluation_strict_decisive_body_metrics_median_position" => :major_strict_med_pos,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_util_maxer_elected" => :prop_strict_util_maxer_elected,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_minority_govt_indicators_size" => :prop_qual_min_govt_size,
    "majoritarianevaluationmetrics.pluratarianevaluation_utility_metrics_utility_efficiency_median_" => :plur_util_eff_med,
    "majoritarianevaluationmetrics.pluratarianevaluation_voting_efficiency_stats_condorcet_efficiency" => :plur_cond_eff,
    "spatialcharacteristics.spatialautocorrelationmeasurement_median_entropies_2" => :spat_med_entropy2,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_utility_efficiency_avg" => :major_util_eff_avg,
    "majoritarianevaluationmetrics.pluratarianevaluation_voting_efficiency_stats_n_condorcet_paradox" => :plur_cond_paradox,
    "spatialcharacteristics.spatialautocorrelationmeasurement_average_entropies_2" => :spat_avg_entropy2,
    "spatialcharacteristics.spatialautocorrelationmeasurement_median_entropies_5" => :spat_med_entropy5,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_minority_govt_indicators_unstable" => :prop_strict_min_govt_unstable,
    "spatialcharacteristics.spatialautocorrelationmeasurement_aggregated_distance_corr" => :spat_agg_dist_corr,
    "majoritarianevaluationmetrics.pluratarianevaluation_maj_gallagher_index" => :plur_gallagher_index,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_n_parties_effective" => :prop_qual_n_parties_eff,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_util_maxer_elected" => :prop_qual_util_maxer_elected,
    "tangianindicesresults_raw_uni_parties" => :tangian_raw_uni_parties,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_single_party_win" => :prop_qual_single_party_win,
    "majoritarianevaluationmetrics.pluratarianevaluation_utility_metrics_utility_from_maximizer_avg" => :plur_util_max_avg,
    "tangianindicesresults_adj_uni_parties" => :tangian_adj_uni_parties,
    "spatialcharacteristics.spatialautocorrelationmeasurement_characteristic_level_corr_2" => :spat_char_corr2,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_utility_from_winner_avg" => :major_util_winner_avg,
    "tangianindicesresults_adj_uni_body" => :tangian_adj_uni_body,
    "majoritarianevaluationmetrics.pluratarianevaluation_qualified_decisive_body_metrics_social_utility_from_body" => :plur_qual_social_util,
    "spatialcharacteristics.spatialautocorrelationmeasurement_median_entropies_4" => :spat_med_entropy4,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_median_position" => :prop_qual_med_pos,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_n_parties_effective" => :prop_strict_n_parties_eff,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_minority_govt_indicators_elected" => :prop_strict_min_govt_elected,
    "majoritarianevaluationmetrics.majoritarianevaluation_n_parties_effective" => :major_n_parties_eff,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_unanimity_rate" => :prop_qual_unanimity_rate,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_winner_in_smith_set" => :prop_qual_winner_smith,
    "majoritarianevaluationmetrics.pluratarianevaluation_n_parties_effective" => :plur_n_parties_eff,
    "majoritarianevaluationmetrics.pluratarianevaluation_qualified_decisive_body_metrics_non_zero_positions" => :plur_qual_nonzero_pos,
    "majoritarianevaluationmetrics.pluratarianevaluation_utility_metrics_utility_efficiency_avg" => :plur_util_eff_avg,
    "majoritarianevaluationmetrics.majoritarianevaluation_voting_efficiency_stats_smith_efficiency" => :major_smith_eff,
    "spatialcharacteristics.spatialautocorrelationmeasurement_average_entropies_5" => :spat_avg_entropy5,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_election_failed" => :prop_strict_elec_failed,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_winner_in_smith_set" => :prop_strict_winner_smith,
    "majoritarianevaluationmetrics.pluratarianevaluation_strict_decisive_body_metrics_non_zero_positions" => :plur_strict_nonzero_pos,
    "majoritarianevaluationmetrics.majoritarianevaluation_strict_decisive_body_metrics_social_utility_from_body" => :major_strict_social_util,
    "majoritarianevaluationmetrics.majoritarianevaluation_n_winning_parties" => :major_n_winning_parties,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_utility_from_maximizer" => :prop_qual_util_max,
    "spatialcharacteristics.spatialautocorrelationmeasurement_characteristic_level_corr_3" => :spat_char_corr3,
    "majoritarianevaluationmetrics.pluratarianevaluation_qualified_decisive_body_metrics_median_position" => :plur_qual_med_pos,
    "majoritarianevaluationmetrics.pluratarianevaluation_strict_decisive_body_metrics_median_position" => :plur_strict_med_pos,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_condorcet_paradox" => :prop_qual_cond_paradox,
    "spatialcharacteristics.spatialautocorrelationmeasurement_average_entropies_4" => :spat_avg_entropy4,
    "majoritarianevaluationmetrics.majoritarianevaluation_qualified_decisive_body_metrics_social_utility_from_body" => :major_qual_social_util,
    "majoritarianevaluationmetrics.pluratarianevaluation_utility_metrics_utility_from_winner_median_" => :plur_util_winner_med,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_size_of_coalition" => :prop_strict_coalition_size,
    "spatialcharacteristics.spatialautocorrelationmeasurement_average_entropies_1" => :spat_avg_entropy1,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_condorcet_winner_elected" => :prop_strict_condorcet_winner,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_indicators_minority_govt_indicators_elected" => :prop_qual_min_govt_elected,
    "tangianindicesresults_raw_pop_parties" => :tangian_raw_pop_parties,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_unanimity_rate" => :prop_strict_unanimity_rate,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_n_parties" => :prop_strict_n_parties,
    "proportionalevaluationmetrics.proportionalevaluation_strict_indicators_single_party_win" => :prop_strict_single_party_win,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_utility_efficiency" => :prop_strict_util_eff,
    "tangianindicesresults_raw_uni_body" => :tangian_raw_uni_body,
    "majoritarianevaluationmetrics.pluratarianevaluation_strict_decisive_body_metrics_social_utility_from_body" => :plur_strict_social_util,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_unanimity" => :prop_qual_unanimity,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_utility_efficiency_median_" => :major_util_eff_med,
    "majoritarianevaluationmetrics.majoritarianevaluation_utility_metrics_utility_from_winner_median_" => :major_util_winner_med,
    "spatialcharacteristics.spatialautocorrelationmeasurement_characteristic_level_corr_1" => :spat_char_corr1,
    "majoritarianevaluationmetrics.pluratarianevaluation_utility_metrics_util_maximizer_election_rate" => :plur_util_max_elec_rate,
    "tangianindicesresults_adj_pop_body" => :tangian_adj_pop_body,
    "proportionalevaluationmetrics.proportionalevaluation_strict_measures_n_winning_parties" => :prop_strict_n_winning_parties,
    "spatialcharacteristics.spatialautocorrelationmeasurement_average_entropies_3" => :spat_avg_entropy3,
    "majoritarianevaluationmetrics.majoritarianevaluation_maj_gallagher_index" => :major_gallagher_index,
    "majoritarianevaluationmetrics.pluratarianevaluation_n_winning_parties" => :plur_n_winning_parties,
    "proportionalevaluationmetrics.proportionalevaluation_qualified_measures_n_parties" => :prop_qual_n_parties,
    "proportionalevaluationmetrics.proportionalevaluation_prop_true_gallagher_index" => :prop_true_gallagher_index,
    "majoritarianevaluationmetrics.majoritarianevaluation_maj_true_gallagher_index" => :maj_true_gallagher_index,
    "majoritarianevaluationmetrics.pluratarianevaluation_plur_true_gallagher_index" => :plur_true_gallagher_index
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

# Demographic Characteristics homogeneity
const FACTOR_A_LOOKUP_TABLE_H = Dict{String,SVector{5,Symbol}}(
    "L1" => SVector{5,Symbol}([:perfect, :perfect, :perfect, :perfect, :perfect]),
    "L2" => SVector{5,Symbol}([:perfect, :high, :moderate, :moderate, :low]),
    "L3" => SVector{5,Symbol}([:low, :low, :low, :low, :low])
)

# Demographic Characteristics type
const FACTOR_A_LOOKUP_TABLE_T = Dict{String,SVector{5,Symbol}}(
    "L1" => SVector{5,Symbol}([:ordinal, :ordinal, :nominal, :nominal, :nominal]),
    "L2" => SVector{5,Symbol}([:ordinal, :ordinal, :nominal, :nominal, :nominal]),
    "L3" => SVector{5,Symbol}([:ordinal, :ordinal, :nominal, :nominal, :nominal]),
)

# Spatial Characteristics
const FACTOR_B_LOOKUP_TABLE = Dict(
    "L1" => SVector{5,Float64}([0.0, 0.0, 0.0, 0.0, 0.0]), # none to low
    "L2" => SVector{5,Float64}([10.0, 10.0, 10.0, 10.0, 10.0]), # moderate (overload latent field)
    "L3" => SVector{5,Float64}([0.25, 0.25, 0.25, 0.25, 0.25]), # high
)

const FACTOR_B_FIXED_PARAMS = OrderedDict{Symbol,Union{Float64,Int}}(:n_metros => 3, :urbanization => 0.8, :urban_sprawl => 0.05,
    :SPATIAL_DISPERSION => 0.05
)

# Demographic Salience
const FACTOR_C_LOOKUP_TABLE = Dict{String,SVector{5,Symbol}}(
    "C1_1" => SVector{5,Symbol}([:none, :none, :none, :none, :none]),
    "C1_2" => SVector{5,Symbol}([:none, :low, :moderate, :high, :moderate]),
    "C1_3" => SVector{5,Symbol}([:high, :high, :high, :high, :high]),
    "C2_1" => SVector{5,Symbol}([:none, :none, :none, :none, :none]),
    "C2_2" => SVector{5,Symbol}([:none, :low, :moderate, :high, :moderate]),
    "C2_3" => SVector{5,Symbol}([:high, :high, :high, :high, :high]),
    "C3_1" => SVector{5,Symbol}([:none, :none, :none, :none, :none]),
    "C3_2" => SVector{5,Symbol}([:none, :low, :moderate, :high, :moderate]),
    "C3_3" => SVector{5,Symbol}([:high, :high, :high, :high, :high])
)

# Demographic Cleavage Saliece
const FACTOR_D_LOOKUP_TABLE = Dict{String,SArray}(
    "D1_1" => SMatrix{5,1,Int}([0; 0; 0; 0; 0]),
    "D1_2" => SMatrix{5,1,Int}([2; 2; 2; 2; 2]),
    "D1_3" => SMatrix{5,1,Int}([3; 3; 3; 3; 3]),
    "D2_1" => SMatrix{5,3,Int}([0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0]),
    "D2_2" => SMatrix{5,3,Int}([3 2 1; 3 1 2; 3 0 3; 3 3 0; 3 2 1]),
    "D2_3" => SMatrix{5,3,Int}([3 3 3; 3 3 3; 3 3 3; 3 3 3; 3 3 3]),
    "D3_1" => SMatrix{5,5,Int}(
        [0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0]
    ),
    "D3_2" => SMatrix{5,5,Int}(
        [2 2 1 2 3; 3 2 0 2 3; 3 3 1 0 3; 3 3 1 1 2; 3 1 2 2 2]
    ),
    "D3_3" => SMatrix{5,5,Int}(
        [3 3 3 3 3; 3 3 3 3 3; 3 3 3 3 3; 3 3 3 3 3; 3 3 3 3 3]
    )
)

# Issue Structure numnber of issues
const FACTOR_E_LOOK_UP_TABLE_N = Dict{String,Int}(
    "L1" => 1, "L2" => 3, "L3" => 5
)

# Issue Structure dimensions
const FACTOR_E_LOOK_UP_TABLE_D = Dict{String,SArray}(
    "L1" => SVector{1,Int}([1]),
    "L2" => SVector{3,Int}([2, 2, 2]),
    "L3" => SVector{5,Int}([1, 2, 2, 3, 3])
)

# n_questions
const FACTOR_F_LOOK_UP_TABLE_Q = Dict{String,SArray}(
    "F1_1" => SVector{1,Int}([20]),
    "F1_2" => SVector{1,Int}([20]),
    "F1_3" => SVector{1,Int}([20]),
    "F2_1" => SVector{3,Int}([10, 5, 5]),
    "F2_2" => SVector{3,Int}([10, 5, 5]),
    "F2_3" => SVector{3,Int}([10, 5, 5]),
    "F3_1" => SVector{5,Int}([2, 3, 5, 5, 5]),
    "F3_2" => SVector{5,Int}([2, 3, 5, 5, 5]),
    "F3_3" => SVector{5,Int}([2, 3, 5, 5, 5])
)

# n_positions
const FACTOR_F_LOOK_UP_TABLE_P = Dict{String,SArray}(
    "F1_1" => SVector{1,Int}([2]),
    "F1_2" => SVector{1,Int}([3]),
    "F1_3" => SVector{1,Int}([5]),
    "F2_1" => SVector{3,Int}([2, 2, 2]),
    "F2_2" => SVector{3,Int}([3, 3, 3]),
    "F2_3" => SVector{3,Int}([5, 5, 5]),
    "F3_1" => SVector{5,Int}([2, 2, 2, 2, 2]),
    "F3_2" => SVector{5,Int}([3, 3, 3, 3, 3]),
    "F3_3" => SVector{5,Int}([5, 5, 5, 5, 5])
)

# engagment alpha
const FACTOR_G_LOOKUP_TABLE_ALPHA = Dict{String,Float64}(
    "L1" => 0.99, "L2" => 0.5, "L3" => 0.01
)

# engagement p-norm
const FACTOR_G_LOOKUP_TABLE_P = Dict{String,Float64}(
    "L1" => 3.0, "L2" => 3.0, "L3" => 3.0
)

#party threshold
const FACTOR_H_LOOKUP_TABLE = Dict{String,Float64}(
    "L1" => 0.02, "L2" => 0.05, "L3" => 0.1
)

# alpha candidate entry
const FACTOR_I_LOOKUP_TABLE = Dict{String,Float64}(
    "L1" => 0.99, "L2" => 0.05, "L3" => 0.01
)

# n parties
const FACTOR_J_LOOKUP_TABLE = Dict{String,Int}(
    "L0" => 404, "L1" => 3, "L2" => 5, "L3" => 7
)

# n candidates
const FACTOR_K_LOOKUP_TABLE = Dict{String,Int}(
    "L0" => 404, "L1" => 3, "L2" => 5, "L3" => 7
)

# turnout
const FACTOR_L_LOOKUP_TABLE = Dict{String,Float64}(
    "L0" => 404.0, "L_1" => 0.99, "L_2" => 0.75, "L_3" => 0.5
)

# strategic voting
const FACTOR_M_LOOKUP_TABLE = Dict{String,Float64}(
    "L0" => 404.0, "M_1" => 0.01, "M_2" => 0.4
)

# demographic attitudes
const FACTOR_N_LOOKUP_TABLE = Dict{String,Bool}(
    "L0" => false,
    "N1_1" => false,
    "N1_2" => true,
    "N2_1" => false,
    "N2_2" => true,
    "N3_1" => false,
    "N3_2" => true
)

end