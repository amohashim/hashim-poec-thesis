
# Issue: Educaction
# Dimensionality: 2
# Dimensions: (Public Spending, Private Spending)

V, eps = SVector{2,Float64}, 0.0001
q1a, q1b, q1c = "Should we raise public education spending?", "Should we maintain public education spending?", "Should we reduce public education spending?"
q1a_pos, q1b_pos, q1c_pos = V([1.0, 0.0]), V([eps, 0.0]), V([-1.0, 0.0])
q2a, q2b, q2c = "Should we raise private education spending?", "Should we maintain private education spending?", "Should we reduce private education spending?"
q2a_pos, q2b_pos, q2c_pos = V([0.0, 1.01]), V([0.0, eps]), V([0.0, -1.0])

# Issue: abortion
# Dimensionality: 2
# Dimensions: (Weeks, Moral Exceptions)

V, eps = SVector{2,Float64}, 0.0001
q3a, q3b, q3c, q3d = "Should we ban abortion after 0 weeks?", "Should we ban abortion after 6 weeks?", "Should we ban abortion after 12 weeks?", "Should we ban abortion after 16 weeks?"
q3a_pos, q3b_pos, q3c_pos, q3d_pos = V([-1.0, 0]), V([-2 / 3, 0]), V([-1 / 3, 0]), V([1 / 3, 0])
q3e, q3f = "Should we ban abortion after 18 weeks?", "Should we ban abortion after 24 weeks?"
q3e_pos, q3f_pos = V([2 / 3, 0]), V([1, 0])
q4a, q4b = "Given that we ban abortion after 6 weeks, should we allow exceptions for rape and incest?", "Given that we ban abortion after 6 weeks, should we NOT allow exceptions for rape and incest?"
q4a_pos, q4b_pos = V([0, 1]), V([0, -1])

q1_components = DichotomousQuestion{String}.([q1a, q1b, q1c])
q2_components = DichotomousQuestion{String}.([q2a, q2b, q2c])
q3_components = DichotomousQuestion{String}.([q3a, q3b, q3c, q3d, q3e, q3f])
q4_components = DichotomousQuestion{String}.([q4a, q4b])

q1 = DecomposedDichotomousQuestion{String}(3, q1_components)
q2 = DecomposedDichotomousQuestion{String}(3, q2_components)
q3 = DecomposedDichotomousQuestion{String}(5, q3_components)
q4 = DecomposedDichotomousQuestion{String}(2, q4_components)

q1_positions = Vector{SVector{2,Float64}}([q1a_pos, q1b_pos, q1c_pos])
q2_positions = Vector{SVector{2,Float64}}([q2a_pos, q2b_pos, q2c_pos])
q3_positions = Vector{SVector{2,Float64}}([q3a_pos, q3b_pos, q3c_pos, q3d_pos, q3e_pos, q3f_pos])
q4_positions = Vector{SVector{2,Float64}}([q4a_pos, q4b_pos])

q1_mapped = MappedDecomposedDichotomousQuestion(3, q1_components, q1_positions)
q2_mapped = MappedDecomposedDichotomousQuestion(3, q2_components, q2_positions)
q3_mapped = MappedDecomposedDichotomousQuestion(6, q3_components, q3_positions)
q4_mapped = MappedDecomposedDichotomousQuestion(2, q4_components, q4_positions)

issue_education = MappedDecomposedIssue(
    SVector{2,String}(["Public Spending", "Private Spending"]),
    Vector{MappedDecomposedDichotomousQuestion{2,String}}([q1_mapped, q2_mapped])
)

issue_abortion = MappedDecomposedIssue(
    SVector{2,String}(["Weeks", "Moral Exceptions"]),
    Vector{MappedDecomposedDichotomousQuestion{2,String}}([q3_mapped, q4_mapped])
)

# Number of individuals (N), characteristics (C), and dimensions (d)
N = 500_000
C = 100
d = 3  # For simplicity, expand to multiple dimensions if needed

# Generate random binary characteristics matrix (N x C)
X = bitrand(N, C)

# Generate random weights vector (C x 1)
w = randn(C, d)  # Weights can be positive or negative

# Bias term
b = 0.0  # Adjust as needed

# Generate samples
samples = generate_samples_weighted(
    X, w, b,
    variance_setting=:fixed,
    σ²_fraction=0.9
)

# Inspect samples
println("Mean of samples: ", mean(samples))
println("Variance of samples: ", var(samples))
