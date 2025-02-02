cd("/Users/alihashim/Desktop/Online_Academic_Submissions/poec_thesis/simulation_package")
include("SimulationPackage.jl") # Li_L1_L1_L2

# Endogeneous Parties (Factor X) => YES (L1)
# Dynamic Candidates (Factor Y) => YES (L1)
# Imperfect Voters (Factor Z) => YES (L2)

using Base.Threads
using Random
using StaticArrays
using Parameters
using CSV
using DataFrames
using Logging, LoggingExtras

using .HashimPoecThesisSimulationPackage
using ..SimulationParameters
using ..ExperimentDesignInterfaceTools: read_in_subdesign, initialize_output_dataframe
using ..ExperimentDesignInterfaceTools: intitialize_simulation_run
using ..ExperimentParameters
using ..BranchAgnosticSequences
using ..Branch5Sequences
using ..TangianIndices

function define_parameters()

    rng = MersenneTwister(rand(1:10_000_000))

    n_characteristics = 5
    n_groups = SVector{5,Int}([3, 3, 5, 5, 2])
    salience_to_probs = Dict{Symbol,SVector{4,Float64}}(
        :none => SVector{4,Float64}([1.0, 0.0, 0.0, 0.0]),
        :low => SVector{4,Float64}([0.15, 0.70, 0.10, 0.05]),
        :moderate => SVector{4,Float64}([0.05, 0.10, 0.70, 0.15]),
        :high => SVector{4,Float64}([0.05, 0.10, 0.15, 0.7]),
    )

    n_seats = 50
    pop_per_seat = 5000

    σ_none = 20.0
    σ_low = 10.0
    σ_moderate = 5.0
    σ_high = 1.0

    gamma = 1.0

    n_iterations = 100
    sample_size = 400
    mantel_permutations = 1000

    fixed_params = FixedParams(rng, n_characteristics, n_groups, salience_to_probs, n_seats,
        pop_per_seat, σ_none, σ_low, σ_moderate, σ_high, gamma, n_iterations, sample_size,
        mantel_permutations
    )

    return fixed_params

end

function io_task(io_channel::Channel{DataFrame}, output_path::String, io_chunk_size::Int)
    temp_output = DataFrame()  # Local buffer for I/O task

    for chunk in io_channel
        append!(temp_output, chunk)

        if size(temp_output, 1) >= io_chunk_size
            # Write the buffered rows to the file
            CSV.write(output_path, temp_output; append=true)
            println("wrote to cscv")
            empty!(temp_output)  # Clear the buffer
        end
    end

    # Write any remaining data after the channel closes
    if !isempty(temp_output)
        CSV.write(output_path, temp_output; append=true)
    end
end

const SUBDESIGN_FILE_NAME::String = "/Users/alihashim/Desktop/Online_Academic_Submissions/poec_thesis/design_matrix/issue_L1_party_L1_cand_L1_voter_L2.csv"
const OUTPUT_PATH::String = "L1_L1_L1_L2_30001_to_31104_replications.csv"
const RUN_RANGE::UnitRange = 30_001:31_104
const IO_CHUNK_SIZE::Int = 1
const N_THREADS::Int = 4

function main()
    # Open a log file
    log_file = open("error_log.txt", "w")
    file_logger = FileLogger(log_file)
    global_logger(file_logger)  # Set the file logger as the global logger

    design_matrix = read_in_subdesign(SUBDESIGN_FILE_NAME)
    output = initialize_output_dataframe(OUTPUT_PATH)

    # # io_run_iteration = 1
    # temp_output = initialize_output_dataframe()

    io_channel = Channel{DataFrame}(N_THREADS * 2)
    @spawn io_task(io_channel, OUTPUT_PATH, IO_CHUNK_SIZE)

    Threads.@threads for simulation_run in RUN_RANGE

        for _ in 1:1

            println(simulation_run)

            fixed_params = define_parameters() # need this in the loop for thread-safe RNG

            try
                begin
                    dem_char_params, spatial_params, issue_structure, salience_structure,
                    question_structure, representative_params, branch_params, params_row =
                        intitialize_simulation_run(
                            simulation_run, design_matrix)
                end

                statewide_demographic_dists, district_dists, coords = run_spatial_dist_sequence(
                    fixed_params, dem_char_params, spatial_params
                )

                spatial_corr_measurements = run_endogeneous_param_measurement_sequence(
                    fixed_params, district_dists, coords
                )

                begin
                    voters, voter_ideal_points = run_voter_information_sequence(
                        fixed_params, salience_structure, issue_structure, question_structure,
                        district_dists; return_only_voters_and_ideal_points=true
                    )
                end

                begin
                    prop_eval_metrics, tangian_inputs, preferred_parties,
                    voter_ideal_points, voter_question_positions, voter_issue_weights =
                        run_proportional_election_sequence(
                            fixed_params, issue_structure, representative_params, question_structure,
                            branch_params, voter_ideal_points
                        )
                end

                begin
                    majoritarian_eval_metrics, pluralility_evaluation_metrics, winning_candidates =
                        run_majoritarian_sequence(
                            fixed_params, issue_structure, branch_params, question_structure,
                            representative_params, voter_ideal_points, voter_question_positions,
                            voter_issue_weights, preferred_parties, 0.5
                        )
                end
                party_question_positions, winning_parties, n_parties = tangian_inputs

                tangian_indices = computeTangianIndices(fixed_params, issue_structure,
                    question_structure, voter_question_positions, party_question_positions,
                    winning_candidates, winning_parties, n_parties; single_collapsed_space=true
                )

                results_row = run_compile_results_sequence(spatial_corr_measurements,
                    prop_eval_metrics, majoritarian_eval_metrics, pluralility_evaluation_metrics,
                    tangian_indices, params_row)

                put!(io_channel, DataFrame([results_row]))

            catch e

                # Log error with simulation_run and stacktrace
                @error "Error in simulation_run $simulation_run: $e"
                @error "Simulation run failed on iteration $simulation_run"
                @error "Stacktrace: $(stacktrace(e))"
                continue

            end
        end

    end

    close(io_channel)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end