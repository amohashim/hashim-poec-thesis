module HashimPoecThesisSimulationPackage

export SimulationParameters, HelpfulFunctions, ExperimentParameters, ExperimentDesignInterfaceTools
export ExogeneousDemographicCharacteristics, EndogeneousDemogprahicCharacteristics
export SpatialCharacteristics, SimulateIssuePreferences, SimulateQuestionPreferences
export PartySimulation, CandidateSimulation, ElectionSimulation, StrategicExit
export CondorcetSmithFunctions, ProportionalEvaluationMetrics, MajoritarianEvaluationMetrics
export TangianIndices
export BranchAgnosticSequences, Branch1Sequences, Branch2Sequences, Branch3Sequences
export Branch4Sequences, Branch5Sequences

include("simulation_parameters.jl")
include("helpful_functions.jl")
include("experiment_parameters.jl")
include("experimental_design_tools.jl")
include("simulate_demographic_characteristics.jl")
include("simulate_spatial_characteristics.jl")
include("simulating_preferences.jl")
include("party_simulation.jl")
include("candidate_simulation.jl")
include("election_simulation.jl")
include("condorcet_smith_functions.jl")
include("proportional_evaluation_metrics.jl")
include("majoritarian_evaluation_metrics.jl")
include("tangian_indices.jl")
include("branch_agnostic_sequences.jl")
include("branch_1_sequences.jl")
include("branch_2_sequences.jl")
include("branch_3_sequences.jl")
include("branch_4_sequences.jl")
include("branch_5_sequences.jl")

end