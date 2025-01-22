module HashimPoecThesisSimulationPackage

export SimulationParameters, HelpfulFunctions
export ExogeneousDemographicCharacteristics, EndogeneousDemogprahicCharacteristics
export SpatialCharacteristics, SimulateIssuePreferences, SimulateQuestionPreferences
export PartySimulation, CandidateSimulation, ElectionSimulation
export CondorcetSmithFunctions, ProportionalEvaluationMetrics, MajoritarianEvaluationMetrics
export TangianIndices
export BranchAgnosticSequences, Branch1Sequences

include("simulation_parameters.jl")
include("helpful_functions.jl")
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


end