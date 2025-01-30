cd("/Users/alihashim/Desktop/Online_Academic_Submissions/poec_thesis/simulation_package")
include("SimulationPackage.jl")

module SimulationPlotting

using Plots
using Measures

using .HashimPoecThesisSimulationPackage
using ..SimulationParameters
using ..ExperimentDesignInterfaceTools: read_in_subdesign, initialize_output_dataframe
using ..ExperimentDesignInterfaceTools: intitialize_simulation_run
using ..ExperimentParameters
using ..BranchAgnosticSequences

function define_parameters()

    rng = MersenneTwister(2024)

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

function plot_ideal_points_under_assumptions(voter_ideal_points::AbstractVector{Array{Float64,3}})

    return histogram(vec(voter_ideal_points[1]),
        xlabel="Ideal Points",
        ylabel="Frequency",
        color=:gray,
        linecolor=:black, linewidth=0.5,

        # Remove grid and tick lines
        grid=false,  # Removes tick marks on X-axis
        yticks=:none,  # Removes tick marks on Y-axis
        framestyle=:box,  # Keeps a clean box around the plot

        # Use sans-serif fonts
        titlefont=font("Arial", 12),
        guidefont=font("Arial", 11),
        tickfont=font("Arial", 10),
        fontfamily="Arial",

        # Figure size
        size=(600, 400),
        margins=5mm,
        legend=false
    )

end

macro Name(arg)
    string(arg)
end

if abspath(PROGRAM_FILE) == @__FILE__

    SUBDESIGN_FILE_NAME = "/Users/alihashim/Desktop/Online_Academic_Submissions/poec_thesis/design_matrix/issue_L1_party_L2_cand_L2_voter_L2.csv"
    simulation_run = 1
    design_matrix = read_in_subdesign(SUBDESIGN_FILE_NAME)
    fixed_params = define_parameters() # need this in the loop for thread-safe RNG

    begin
        dem_char_params, spatial_params, issue_structure, salience_structure,
        question_structure, representative_params, branch_params, params_row =
            intitialize_simulation_run(
                simulation_run, design_matrix)
    end

    perf_homo_no_salience_no_cleavage = (
        DemographicCharacteristicParams{5}(
            [:ordinal, :ordinal, :nominal, :nominal, :nominal],
            [:perfect, :perfect, :perfect, :perfect, :perfect]
        ),
        SalienceStructure{5,1}([:none, :none, :none, :none, :none], [0; 0; 0; 0; 0;;])
    )

    perf_homo_high_salience_no_cleavage = (
        DemographicCharacteristicParams{5}(
            [:ordinal, :ordinal, :nominal, :nominal, :nominal],
            [:perfect, :perfect, :perfect, :perfect, :perfect]
        ),
        SalienceStructure{5,1}([:high, :high, :high, :high, :high], [0; 0; 0; 0; 0;;])
    )

    low_homo_high_salience_no_cleavage = (
        DemographicCharacteristicParams{5}(
            [:ordinal, :ordinal, :nominal, :nominal, :nominal],
            [:low, :low, :low, :low, :low]
        ),
        SalienceStructure{5,1}([:high, :high, :high, :high, :high], [1; 1; 1; 1; 1;;])
    )

    low_homo_high_salience_high_cleavage = (
        DemographicCharacteristicParams{5}(
            [:ordinal, :ordinal, :nominal, :nominal, :nominal],
            [:low, :low, :low, :low, :low]
        ),
        SalienceStructure{5,1}([:high, :high, :high, :high, :high], [3; 3; 3; 3; 3;;])
    )

    var_names = [
        "perf_homo_no_salience_no_cleavage", "perf_homo_high_salience_no_cleavage",
        "low_homo_high_salience_low_cleavage", "low_homo_high_salience_high_cleavage"
    ]

    parameters = [
        perf_homo_no_salience_no_cleavage, perf_homo_high_salience_no_cleavage,
        low_homo_high_salience_no_cleavage, low_homo_high_salience_high_cleavage
    ]

    for (name, params) in zip(var_names, parameters)

        dem_char_params, salience_structure = params

        statewide_demographic_dists, district_dists, coords = run_spatial_dist_sequence(
            fixed_params, dem_char_params, spatial_params
        )
        begin
            voters, voter_ideal_points = run_voter_information_sequence(
                fixed_params, salience_structure, issue_structure, question_structure,
                district_dists; return_only_voters_and_ideal_points=true
            )
        end

        plot_ideal_points_under_assumptions(voter_ideal_points)
        savefig(name * "_ideal_points_plot.png")

    end

end

end
