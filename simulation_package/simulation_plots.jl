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

# translated from python using chatgpt model o3-mini-high on February 1, 2025
function plot_structured_parties()

    # Set random seed for reproducibility.
    Random.seed!(1)

    # -------------------------------
    # Figure 1: Gaussian Population
    # -------------------------------
    n_points = 1000  # number of individuals in the population

    # Generate a homogeneous (bivariate normal) population.
    pop_mean_true = [0.0, 0.0]
    pop_cov = [1.0 0.0; 0.0 1.0]
    population = rand(MvNormal(pop_mean_true, pop_cov), n_points)'  # n_points × 2

    # Compute the sample population mean.
    pop_mean = vec(mean(population, dims=1))

    # Structured Noise Model:
    num_parties = 5
    noise_sigma = 1.0
    party_points = Array{Float64}(undef, num_parties, 2)
    for i in 1:num_parties
        error = rand(MvNormal([0.0, 0.0], [noise_sigma^2 0.0; 0.0 noise_sigma^2]))
        party_points[i, :] = pop_mean + error
    end

    # Create the first plot with no gridlines, smaller points, and no legend.
    p1 = scatter(population[:, 1], population[:, 2],
        color="gray", alpha=0.5, markersize=2,
        grid=false, label="")  # population points (small)

    # Define a colorblind-friendly palette.
    colorblind_colors = ["#117733", "#88CCEE", "#DDCC77", "#CC6677", "#882255"]

    # Plot each party’s point with smaller markers.
    for i in 1:num_parties
        scatter!(p1, [party_points[i, 1]], [party_points[i, 2]],
            color=colorblind_colors[i], markerstrokecolor="black",
            markersize=6, label="")
    end

    xlabel!("Dimension 1")
    ylabel!("Dimension 2")
    plot!(p1, legend=false)

    # -------------------------------
    # Figure 2: Gaussian Mixture Population
    # -------------------------------
    n_components = 4
    centers = [[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]]
    mix_cov = [0.5 0.0; 0.0 0.5]
    n_each = div(n_points, n_components)

    # Generate the mixture population.
    population_mixture = Array{Float64}(undef, 0, 2)
    for center in centers
        component = rand(MvNormal(center, mix_cov), n_each)'  # each row is a point
        population_mixture = vcat(population_mixture, component)
    end

    # Compute the overall population mean.
    pop_mixture_mean = vec(mean(population_mixture, dims=1))

    # Generate structured noise for the mixture.
    party_points_mixture = Array{Float64}(undef, num_parties, 2)
    for i in 1:num_parties
        error = rand(MvNormal([0.0, 0.0], [noise_sigma^2 0.0; 0.0 noise_sigma^2]))
        party_points_mixture[i, :] = pop_mixture_mean + error
    end

    # Create the second plot with the same modifications.
    p2 = scatter(population_mixture[:, 1], population_mixture[:, 2],
        color="gray", alpha=0.5, markersize=2,
        grid=false, label="")

    for i in 1:num_parties
        scatter!(p2, [party_points_mixture[i, 1]], [party_points_mixture[i, 2]],
            color=colorblind_colors[i], markerstrokecolor="black",
            markersize=6, label="")
    end

    xlabel!("Dimension 1")
    ylabel!("Dimension 2")
    plot!(p2, legend=false)

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
