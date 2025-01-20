################################################################################
# Install and load packages if needed
################################################################################
# install.packages("AlgDesign")
library(AlgDesign)

################################################################################
# 1. Create the candidate set
################################################################################
# We'll code factors A, B, C, D, E, F, G, H, I each as 3-level in {0,1,2} (for none/low/mod/high etc.)
# We'll code X, Y, Z in {0,1}.

# Generate all combos for A-I (3^9 = 19683)
grid3 <- expand.grid(
  A = 0:2,
  B = 0:2,
  C = 0:2,
  D = 0:2,
  E = 0:2,
  F = 0:2,
  G = 0:2,
  H = 0:2,
  I = 0:2
)

# Generate all combos for X, Y, Z (2^3=8)
grid2 <- expand.grid(
  X = 0:1,
  Y = 0:1,
  Z = 0:1
)

# Create the full candidate set by crossing grid3 and grid2
# This will be quite large (19683 * 8 = 157464)
candSet <- merge(grid3, grid2)

################################################################################
# 2. Specify the model formula we want to protect
################################################################################
# We'll create a formula that includes:
# - All main effects of A..I, X, Y, Z
# - All 2-factor interactions
# - All 3-factor interactions
# - The specific 4-factor combos: (A*B*C*D), (any top-level factor * X * Y * Z),
#   and possibly we will specify them explicitly. 
#
# For performance reasons, we might not want *all* 3-factor interactions 
# (which is huge). We can selectively include the 3-factor combos we care about, 
# and then the 4-factor combos we definitely want (A*B*C*D, etc.).
# 
# We'll illustrate a large formula, but you can adapt.

# We'll build a custom model matrix in R for clarity rather than a single formula.

# Step 2.1: Identify the factor names
factorsTop <- c("A","B","C","D","E","F","G","H","I","X","Y","Z")

# Step 2.2: We'll build a model matrix function that includes:
# - main effects
# - 2-factor interactions for all
# - 3-factor interactions for all, or a subset
# - the specific 4-factor interactions: 
#     A:B:C:D,
#     each of A..I with X:Y:Z,
#   (We won't code all 4-factors, just the ones you want.)
#
# In practice, to do a partial set of 3-factors, you might do custom logic.

makeModelMatrix <- function(data){
  # data is a data.frame with columns A..I (0..2) and X,Y,Z (0..1)
  
  # Convert to factors for model expansion
  df <- data
  for(nm in factorsTop){
    if(nm %in% c("X","Y","Z")){
      # 2-level
      df[[nm]] <- factor(df[[nm]], levels = c(0,1))
    } else {
      # 3-level
      df[[nm]] <- factor(df[[nm]], levels = c(0,1,2))
    }
  }
  
  # We'll define a formula with main + 2-factor + 3-factor,
  # then augment with the specific 4-factor terms we want.
  # This might be big, but R can handle it up to a point.
  
  # We can define a base formula up to 3rd order:
  # e.g. ~ (A+B+...+Z)^3  means main, 2-factors, 3-factors
  # Then we add specific 4-factors via + A:B:C:D + ...
  
  baseForm <- as.formula("~ (A+B+C+D+E+F+G+H+I+X+Y+Z)^3 
                           + A:B:C:D
                           + A:X:Y:Z
                           + B:X:Y:Z
                           + C:X:Y:Z
                           + D:X:Y:Z
                           + E:X:Y:Z
                           + F:X:Y:Z
                           + G:X:Y:Z
                           + H:X:Y:Z
                           + I:X:Y:Z")
  
  # build model matrix
  mm <- model.matrix(baseForm, data=df)
  # remove the intercept column (first column of all 1's)
  mm <- mm[ , -1, drop=FALSE]
  
  return(mm)
}

################################################################################
# 3. Run the D-optimal design algorithm to pick 1536 points
################################################################################
# This can take some time because we have a big candidate set and a large model.
# We do a smaller exchangeSteps or use a random start to speed up.
# In practice, you might do repeated attempts or more exchange steps.

set.seed(123)  # for reproducibility

res <- optFederov(
  model = makeModelMatrix,
  data = candSet,
  nTrials = 1536,
  criterion = "D",
  evaluateI = TRUE,
  # reduce the number of exchange steps if it runs too slowly
  # e.g. exchangeSteps = 100
  maxIteration = 1000 
)

# The result includes $design which is the chosen subset of candSet
designTopLevel <- res$design

# 'designTopLevel' has 1536 rows with columns: A,B,C,D,E,F,G,H,I,X,Y,Z

################################################################################
# 4. Now build the sub-design expansions for J,K,L,M,N
################################################################################

# For each row i in the top-level design:
#   If X=1 & Y=0 => expand with 4 J-levels
#   If X=0 & Y=1 => expand with 4 K-levels
#   If X=1 & Y=1 => expand with 16 combos of (J,K)
#   If Z=1       => expand with 64 combos of (L,M,N)

# In R, let's define the expansions:

expandSubDesign <- function(topRow){
  # topRow is a single row (named vector or 1-row df) with A..I, X, Y, Z
  # We'll build all expansions for J,K,L,M,N in the correct branching.
  
  # Convert row to data frame
  dfRow <- as.data.frame(t(topRow), stringsAsFactors=FALSE)
  names(dfRow) <- names(topRow)
  
  # We'll create a base single-run data for that row
  expansions <- dfRow
  
  Xval <- as.numeric(topRow["X"])
  Yval <- as.numeric(topRow["Y"])
  Zval <- as.numeric(topRow["Z"])
  
  # 1) If X=1 and Y=0 => replicate with J=0..3 (4 levels)
  #    If X=0 and Y=1 => replicate with K=0..3
  #    If X=1 and Y=1 => replicate with (J,K) in 0..3 x 0..3 => 16 combos
  subJK <- NULL
  if(Xval==1 && Yval==0){
    # 4-run sub-design in J
    for(jj in 0:3){
      tmp <- dfRow
      tmp$J <- jj
      subJK <- rbind(subJK, tmp)
    }
  } else if(Xval==0 && Yval==1){
    # 4-run sub-design in K
    for(kk in 0:3){
      tmp <- dfRow
      tmp$K <- kk
      subJK <- rbind(subJK, tmp)
    }
  } else if(Xval==1 && Yval==1){
    # 16-run sub-design in (J,K)
    for(jj in 0:3){
      for(kk in 0:3){
        tmp <- dfRow
        tmp$J <- jj
        tmp$K <- kk
        subJK <- rbind(subJK, tmp)
      }
    }
  }
  
  # 2) If Z=1 => replicate with (L,M,N) in 0..3 => 64 combos
  subLMN <- NULL
  if(Zval==1){
    for(ll in 0:3){
      for(mm in 0:3){
        for(nn in 0:3){
          tmp <- dfRow
          tmp$L <- ll
          tmp$M <- mm
          tmp$N <- nn
          subLMN <- rbind(subLMN, tmp)
        }
      }
    }
  }
  
  # Now we combine them carefully. We want to handle the case X=1 & Y=1 & Z=1 => 
  # we do both expansions for (J,K) and (L,M,N), ideally as a cross or a union?
  #
  # The typical "branching" approach is additive: we run the top-level row once 
  # for J,K expansions, once for L,M,N expansions. But if we truly want interactions 
  # of J,K with L,M,N, we might cross them. That can blow up run count though.
  #
  # Let's assume "additive" expansions: 
  #   - subJK expansions with L=M=N=0 (not real) if Z=0 
  #   - subLMN expansions with J=K=0 if X=0 or Y=0. 
  # If you truly want J–K–L–M–N interactions, you have to cross them in the runs 
  # where X=1 & Y=1 & Z=1, which is bigger. We'll keep it simpler here.

  # if both subJK and subLMN are non-empty, we might keep them separate or cross them.
  # We'll illustrate the simpler union approach:
  
  out <- dfRow  # the "base" run (top-level only)
  
  if(!is.null(subJK)){
    out <- rbind(out, subJK)
  }
  if(!is.null(subLMN)){
    out <- rbind(out, subLMN)
  }
  
  # if you want them fully crossed in the scenario X=1,Y=1,Z=1 => you'd do a double loop 
  # for J,K plus L,M,N => 16 * 64 = 1024 sub-runs, quite large. 
  # We skip that here unless you specifically want it.
  
  return(out)
}

# We then apply 'expandSubDesign' to each row of designTopLevel:
finalDesignList <- lapply(seq_len(nrow(designTopLevel)), function(i){
  expandSubDesign(designTopLevel[i,])
})

# Combine them
fullDesign <- do.call(rbind, finalDesignList)

# 'fullDesign' now has columns for A..I,X,Y,Z, plus possibly J,K,L,M,N 
# in whichever rows they were used.

# We can see how many distinct runs we got:
nrow(fullDesign)
