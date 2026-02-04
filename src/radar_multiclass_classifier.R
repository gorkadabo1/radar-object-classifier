# ==============================================================================
# Radar-Based Multiclass Object Classifier
# Author: Gorka Dabó
# ==============================================================================

# 0) Libraries ####
library(ggplot2)
library(xgboost)
library(glmnet)   
library(dplyr)
library(effsize)
library(lubridate)
library(tidyr)
library(stringr)
library(ROCR)   

# 1) Data Loading ####
load("data/data.RData")

# 1.5) Robust Outlier Cleaning ####
# Outlier removal improves XGBoost validation mAP substantially (from ~0.43 to ~0.51)
# Define numeric columns (all columns in X)
num_cols <- colnames(X)

# Robust z-score threshold (median/MAD)
# If any feature in a row is more than 6 MADs from the median,
# that row is considered a global outlier and removed
rz_thr <- 6

# Function for robust z-score calculation
robust_z <- function(v) {
  m  <- median(v, na.rm = TRUE)
  md <- mad(v, constant = 1.4826, na.rm = TRUE) # MAD scaled to ~sigma
  if (is.na(md) || md == 0) return(rep(0, length(v)))
  (v - m) / md
}

# Calculate if a row is an outlier in any column
rz_mat <- sapply(num_cols, function(cn) robust_z(X[, cn]))
if (!is.matrix(rz_mat)) rz_mat <- matrix(rz_mat, ncol = 1)
row_is_outlier <- apply(abs(rz_mat) > rz_thr, 1, any)

cat("\n[OUTLIERS] Rows detected as outliers (>|z|_robust >", rz_thr, "): ",
    sum(row_is_outlier), " of ", nrow(X), "\n", sep = "")

# Filter X and y together
X_clean <- X[!row_is_outlier, , drop = FALSE]
y_clean <- y[!row_is_outlier]

cat("[OUTLIERS] Rows after cleaning: ", nrow(X_clean), "\n", sep = "")

# Use cleaned X/y from here on
X <- X_clean
y <- y_clean

# 2) Label Reassignment ####
# Relabel truck (2) and bus (3) as large vehicle (1)
# Relabel group of people (8) as person (7)

# Show initial label distribution
table(y)

# Reassign labels according to assignment instructions
y[y == 2] <- 1   # Truck → Large vehicle
y[y == 3] <- 1   # Bus → Large vehicle
y[y == 8] <- 7   # Group of people → Person

# Verify reassignment
table(y)

# 3) Train/Validation Split ####
# Attempt to reproduce 200k/200k split; adapt safely if cleaning reduced total
n_total <- nrow(X)
target_train <- 200000
target_val   <- 200000

if (n_total >= (target_train + target_val)) {
  n_train <- target_train
  n_val   <- target_val
} else {
  # 50/50 split if we don't have 400k rows
  n_train <- floor(0.5 * n_total)
  n_val   <- n_total - n_train
  cat("[SPLIT] Not enough rows for 400k after cleaning. Using 50/50 split: ",
      n_train, " train / ", n_val, " val\n", sep = "")
}

X_train <- X[1:n_train, , drop = FALSE]
y_train <- y[1:n_train]
X_val   <- X[(n_train + 1):(n_train + n_val), , drop = FALSE]
y_val   <- y[(n_train + 1):(n_train + n_val)]

# Verification
dim(X_train); dim(X_val)
length(y_train); length(y_val)


# 4) RCS Median ± MAD vs Distance (Train) ####

# Correction parameters for radar physics
k   <- 4       # R^4 compensation exponent typical in radar
R0  <- 1.0     # Reference distance (1 m)
eps <- 1e-3    # Avoids log10(0) at very small distances

# Create training dataframe with relevant variables
df_train <- data.frame(
  range_sc = X_train[, "range_sc"],            # Distance to object
  rcs      = X_train[, "radar_cross_section"], # Measured RCS (with compensation)
  clase    = y_train                           # Target class
) %>%
  # Filter only classes of interest: 1 (large vehicle) and 7 (person)
  filter(clase %in% c(1, 7)) %>%
  # Create factors with readable labels and undo R^4 compensation
  mutate(
    clase = factor(clase, levels = c(1,7),
                   labels = c("Large Vehicle (1)", "Person (7)")),
    R_m   = pmax(range_sc, eps),                       # Force R>0
    rcs_uncomp = rcs - 10 * k * log10(R_m / R0)        # Uncompensat RCS (remove distance effect)
  )

# Define 5-meter distance bins
bin_width   <- 5
rmin        <- floor(min(df_train$range_sc, na.rm = TRUE))
rmax        <- ceiling(max(df_train$range_sc, na.rm = TRUE))
breaks_range <- seq(rmin, rmax, by = bin_width)
if (length(breaks_range) < 2) breaks_range <- c(rmin, rmax)

# Assign each detection to its distance bin
df_train_binned <- df_train %>%
  mutate(range_bin = cut(range_sc, breaks = breaks_range,
                         include.lowest = TRUE, right = FALSE))

# Calculate statistics per bin and class
summary_df <- df_train_binned %>%
  group_by(clase, range_bin) %>%
  summarise(
    median_rcs = median(rcs_uncomp, na.rm = TRUE),  # RCS median
    mad_rcs    = mad(rcs_uncomp, na.rm = TRUE),     # Median absolute deviation
    n          = sum(!is.na(rcs_uncomp)),           # Observations per bin
    .groups    = "drop"
  ) %>%
  # Calculate bin midpoint for X axis
  mutate(range_bin_chr = as.character(range_bin),
         range_bin_clean = stringr::str_replace_all(range_bin_chr, "\\[|\\(|\\]|\\)", "")) %>%
  tidyr::separate(range_bin_clean, into = c("lo","hi"), sep = ",", convert = TRUE, fill = "right") %>%
  mutate(range_mid = (as.numeric(lo) + as.numeric(hi)) / 2) %>%
  # Keep only bins with sufficient data
  filter(n >= 50) %>%
  select(clase, range_bin, range_mid, median_rcs, mad_rcs, n)

# Plot median ± MAD by distance
ggplot(summary_df, aes(x = range_mid, y = median_rcs, group = clase)) +
  geom_ribbon(aes(ymin = median_rcs - mad_rcs,
                  ymax = median_rcs + mad_rcs,
                  fill = clase), alpha = 0.25) +  # Dispersion band (MAD)
  geom_line(aes(color = clase), linewidth = 0.9) +  # Median curve
  labs(
    title    = "RCS (Uncompensated) Median ± MAD vs Distance (Train)",
    subtitle = paste0("Distance bin: ", bin_width, " m"),
    x        = "Distance (range_sc) [m]",
    y        = "Radar Cross Section (RCS) [dBsm]",
    color    = "Class",
    fill     = "Class"
  ) +
  theme_minimal(base_size = 12)

# OBSERVATIONS:
# In the plot (uncompensated RCS), both classes show expected physical behavior:
# RCS decreases with distance due to attenuation (after removing ~R^4 compensation).
#
# The "Large Vehicle (1)" class maintains systematically higher RCS values
# than "Person (7)" across almost the entire distance range.
#
# The separation between medians is clear and the variability bands (±MAD) 
# don't overlap much. This suggests RCS is a useful discriminative feature 
# between both classes in that range.
#
# CONCLUSION for section 4: YES, there are consistent differences in RCS
# between "Large Vehicle (1)" and "Person (7)", supporting its use as a 
# discriminative variable for the classifier.


# Statistical confirmation: RCS comparison between classes
rcs_persona  <- X_train[y_train == 7, "radar_cross_section"]
rcs_vehiculo <- X_train[y_train == 1, "radar_cross_section"]

# Normality tests
shapiro.test(sample(rcs_persona, 5000))
shapiro.test(sample(rcs_vehiculo))
# Both results are highly significant, suggesting we should reject normality.
# However, with such large sample sizes, even minimal differences are detected.
# The high W values suggest distributions fit reasonably well to normal.

# Create dataframe to plot distributions
df_rcs <- data.frame(
  rcs = c(rcs_persona, rcs_vehiculo),
  clase = factor(c(rep("Person (7)", length(rcs_persona)),
                   rep("Large Vehicle (1)", length(rcs_vehiculo))))
)

# Comparative histogram
ggplot(df_rcs, aes(x = rcs, fill = clase)) +
  geom_histogram(alpha = 0.5, bins = 60, position = "identity") +
  labs(title = "RCS Distribution by Class",
       x = "Radar Cross Section (dBsm)",
       y = "Frequency") +
  theme_minimal(base_size = 12)
# Person RCS appears to follow normal distribution;
# Large vehicle is less clear. We use non-parametric Wilcoxon test.

# Non-parametric Wilcoxon test
wilcox.test(rcs_persona, rcs_vehiculo,
            alternative = "two.sided",
            conf.level = 0.95)
# Highly significant result indicates a significant RCS difference between classes.

# Calculate effect size
cd <- cliff.delta(rcs_persona, rcs_vehiculo)
cd
# Result (-0.548, negative because rcs_persona < rcs_vehiculo) demonstrates
# a very large difference between both distributions.

# ROC curve to quantify RCS discriminatory capacity
df_roc <- data.frame(
  rcs = c(rcs_persona, rcs_vehiculo),
  y   = c(rep(0, length(rcs_persona)), rep(1, length(rcs_vehiculo)))
)

# Helper function for ROC points and AUC calculation
roc_df <- function(scores, labels) {
  pred <- ROCR::prediction(scores, labels)
  perf <- ROCR::performance(pred, measure = "tpr", x.measure = "fpr")
  auc  <- ROCR::performance(pred, "auc")@y.values[[1]]
  
  fpr  <- perf@x.values[[1]]
  tpr  <- perf@y.values[[1]]
  
  data.frame(fpr = fpr, tpr = tpr, auc = auc)
}

# Calculate ROC curve with RCS as score
roc_rcs <- roc_df(df_roc$rcs, df_roc$y)

# Plot ROC curve
ggplot(roc_rcs, aes(x = fpr, y = tpr)) +
  geom_line(linewidth = 1, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = "ROC Curve: Large Vehicle (1) vs Person (7)",
    subtitle = paste0("AUC = ", round(unique(roc_rcs$auc), 3)),
    x = "FPR (1 - Specificity)",
    y = "TPR (Sensitivity)"
  ) +
  theme_minimal(base_size = 12)

# AUC = 0.774 indicates very good discriminatory capacity of RCS 
# to distinguish between "Large Vehicle (1)" and "Person (7)" classes.
# In practical terms, there is a 77.4% probability that a randomly selected
# large vehicle detection has higher RCS than a random person detection.


# 5) Multiclass Model + AP/mAP + ROC/AP per class using ROCR ####
set.seed(42)

# 5a) GLMNet Multinomial Model ####
# CONFIGURATION
# - Multinomial logistic regression with elastic net regularization (alpha=0.5)
# - Sample 65% of train due to computational limitations
sample_frac    <- 0.65
alpha_elastic  <- 0.5    # 0=L2, 1=L1 (elastic mixture in glmnet)

# INPUT: X_train, y_train, X_val, y_val
train_df <- as.data.frame(X_train)
val_df   <- as.data.frame(X_val)
train_df$y <- factor(y_train)
val_df$y   <- factor(y_val, levels = levels(train_df$y))

# Training sampling
if (sample_frac < 1) {
  n_s <- floor(nrow(train_df) * sample_frac)
  idx <- sample(seq_len(nrow(train_df)), n_s)
  train_df_s <- train_df[idx, , drop = FALSE]
} else {
  train_df_s <- train_df
}

# Design matrix with interactions (linear model)
# Design: main effects + key interactions RCS×distance and RCS×azimuth
# Justification: Radar Cross Section depends on distance and observation angle
rcs_col <- "radar_cross_section"

form_lin <- as.formula(
  paste0("~ ",
         # main effects
         "range_sc + azimuth_sc + radial_velocity + vr_compensated + ",
         "x_cc + y_cc + x_seq + y_seq + ",
         rcs_col, " + ",
         # key interactions with RCS
         rcs_col, ":range_sc + ", rcs_col, ":azimuth_sc")
)

X_tr <- model.matrix(form_lin, data = train_df_s)[, -1, drop = FALSE]
X_va <- model.matrix(form_lin, data = val_df)[, -1, drop = FALSE]
y_tr <- train_df_s$y
y_va <- val_df$y

# Regularized multinomial model (glmnet)
cvfit <- cv.glmnet(
  x = X_tr, y = y_tr,
  family = "multinomial",
  type.multinomial = "ungrouped",  # prob per class
  alpha = alpha_elastic,
  nfolds = 5
)

# Class probabilities
p_tr <- predict(cvfit, newx = X_tr, s = "lambda.min", type = "response")[,,1]  # n_tr x K
p_va <- predict(cvfit, newx = X_va, s = "lambda.min", type = "response")[,,1]  # n_va x K

clases <- colnames(p_tr)  # levels

# ROCR helper functions: ROC/AUC and PR/AP
roc_rocr_df <- function(scores, labels_binary) {
  pred <- ROCR::prediction(scores, labels_binary)
  perf <- ROCR::performance(pred, measure = "tpr", x.measure = "fpr")
  auc  <- ROCR::performance(pred, "auc")@y.values[[1]]
  data.frame(FPR = perf@x.values[[1]], TPR = perf@y.values[[1]], AUC = as.numeric(auc))
}

# PR curve with ROCR: precision vs recall
pr_rocr_df <- function(scores, labels_binary) {
  pred <- ROCR::prediction(scores, labels_binary)
  prec <- ROCR::performance(pred, measure = "prec")
  rec  <- ROCR::performance(pred, measure = "rec")
  data.frame(Recall = rec@y.values[[1]], Precision = prec@y.values[[1]])
}

# AP (Average Precision) via trapezoidal integration of P-R curve
ap_from_pr <- function(pr_df) {
  df <- pr_df %>% 
    filter(!is.na(Precision), !is.na(Recall)) %>% 
    arrange(Recall, desc(Precision))
  
  if (nrow(df) < 2) return(NA_real_)
  
  # Ensure extreme points
  if (df$Recall[1] > 0) df <- rbind(data.frame(Recall = 0, Precision = df$Precision[1]), df)
  if (tail(df$Recall, 1) < 1) df <- rbind(df, data.frame(Recall = 1, Precision = tail(df$Precision, 1)))
  
  # Trapezoidal rule for area under curve
  r <- df$Recall; p <- df$Precision
  sum( (r[-1] - r[-length(r)]) * (p[-1] + p[-length(p)]) / 2 )
}

ap_one_vs_rest_rocr <- function(scores, labels_binary) {
  pr_df <- pr_rocr_df(scores, labels_binary)
  ap_from_pr(pr_df)
}

auc_one_vs_rest_rocr <- function(scores, labels_binary) {
  roc_rocr_df(scores, labels_binary)$AUC[1]
}

# Per-class metrics: AP and AUC in train/val + mAP
ap_train <- ap_val <- setNames(numeric(length(clases)), clases)
auc_train <- auc_val <- setNames(numeric(length(clases)), clases)

for (k in clases) {
  ytr_bin <- as.integer(y_tr == k)
  yva_bin <- as.integer(y_va == k)
  ap_train[k]  <- ap_one_vs_rest_rocr(p_tr[, k], ytr_bin)
  ap_val[k]    <- ap_one_vs_rest_rocr(p_va[, k], yva_bin)
  auc_train[k] <- auc_one_vs_rest_rocr(p_tr[, k], ytr_bin)
  auc_val[k]   <- auc_one_vs_rest_rocr(p_va[, k], yva_bin)
}

mAP_train <- mean(ap_train, na.rm = TRUE)
mAP_val   <- mean(ap_val,   na.rm = TRUE)

cat("AP (train) per class:\n"); print(round(ap_train, 3))
cat("AP (val)   per class:\n"); print(round(ap_val, 3))
cat("\nmAP_train =", round(mAP_train, 3), " | mAP_val =", round(mAP_val, 3), "\n")
cat("\nAUC ROC (train):\n"); print(round(auc_train, 3))
cat("AUC ROC (val):\n"); print(round(auc_val, 3))

# OBSERVATIONS:
# mAP is low in validation (and moderate in train), indicating that
# on average, precision across recall is insufficient.
# Clear degradation from train to validation per class suggests overfitting.

# ROC plot function for all classes
plot_roc_all_classes <- function(p_matrix, y_true, dataset_label = "Validation") {
  clases <- colnames(p_matrix)
  roc_list <- list()
  
  for (k in clases) {
    y_bin <- as.integer(y_true == k)
    pred  <- ROCR::prediction(p_matrix[, k], y_bin)
    perf  <- ROCR::performance(pred, "tpr", "fpr")
    auc   <- ROCR::performance(pred, "auc")@y.values[[1]]
    
    roc_df <- data.frame(
      FPR = perf@x.values[[1]],
      TPR = perf@y.values[[1]],
      Clase = k,
      AUC = round(auc, 3)
    )
    roc_list[[k]] <- roc_df
  }
  
  roc_all <- do.call(rbind, roc_list)
  
  auc_vec <- sapply(roc_list, function(x) unique(x$AUC))
  lab_map <- setNames(paste0(names(auc_vec), " (AUC=", round(auc_vec, 3), ")"),
                      names(auc_vec))
  
  ggplot(roc_all, aes(x = FPR, y = TPR, color = Clase)) +
    geom_line(linewidth = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
    coord_equal() +
    labs(
      title = paste("ROC Curves -", dataset_label),
      subtitle = "All classes (One-vs-Rest)",
      x = "FPR (1 - Specificity)",
      y = "TPR (Sensitivity)",
      color = "Class / AUC"
    ) +
    theme_minimal(base_size = 12) +
    scale_color_discrete(
      breaks = names(lab_map),
      labels = unname(lab_map)
    )
}

plot_roc_all_classes(p_va, y_va, "Validation - GLMNet")
# Best classes: Person (7) and Other Dynamic (10) — AUC ≥ 0.8
# Intermediate: Static Background (11) and Bicycle (5) — AUC ≈ 0.7
# Worst: Car (0) and Large Vehicle (1) — AUC ≤ 0.4


# 5b) XGBoost Multiclass Model ####
set.seed(42)
sample_frac_xgb <- sample_frac  # Same sampling as glmnet for comparison

# Prepare data
X_tr_xgb <- as.matrix(X_train)
X_va_xgb <- as.matrix(X_val)

# Training sampling
if (sample_frac_xgb < 1) {
  n_s_xgb <- floor(nrow(X_tr_xgb) * sample_frac_xgb)
  idx_xgb <- sample(seq_len(nrow(X_tr_xgb)), n_s_xgb)
  X_tr_xgb <- X_tr_xgb[idx_xgb, , drop = FALSE]
  y_tr_xgb <- y_train[idx_xgb]
} else {
  y_tr_xgb <- y_train
}

# Convert labels to 0-indexed format for XGBoost
y_levels <- sort(unique(y_train))
y_tr_xgb_idx <- match(y_tr_xgb, y_levels) - 1
y_va_xgb_idx <- match(y_val, y_levels) - 1

# Create DMatrix objects for XGBoost
dtrain_xgb <- xgb.DMatrix(data = X_tr_xgb, label = y_tr_xgb_idx)
dval_xgb <- xgb.DMatrix(data = X_va_xgb, label = y_va_xgb_idx)

# Model parameters
params_xgb <- list(
  objective = "multi:softprob",        # Probabilities per class
  eval_metric = "mlogloss",            # Evaluation metric
  num_class = length(y_levels),        # Number of classes
  eta = 0.1,                           # Learning rate
  max_depth = 6,                       # Maximum tree depth
  subsample = 0.8,                     # Sample fraction per tree
  colsample_bytree = 0.8,              # Feature fraction per tree
  min_child_weight = 1,
  gamma = 0
)

# Train with early stopping
cat("\nTraining XGBoost model...\n")
xgb_model <- xgb.train(
  params = params_xgb,
  data = dtrain_xgb,
  nrounds = 200,                       # Maximum iterations
  watchlist = list(train = dtrain_xgb, val = dval_xgb),
  early_stopping_rounds = 20,          # Early stopping to prevent overfitting
  verbose = 1,
  print_every_n = 10
)

cat("\nBest iteration:", xgb_model$best_iteration, "\n")

# Predictions: probability matrix (n x num_class)
p_tr_xgb <- predict(xgb_model, X_tr_xgb, reshape = TRUE)
p_va_xgb <- predict(xgb_model, X_va_xgb, reshape = TRUE)

# Assign class names to columns
colnames(p_tr_xgb) <- as.character(y_levels)
colnames(p_va_xgb) <- as.character(y_levels)

# Calculate AP and AUC per class (One-vs-Rest)
ap_train_xgb <- ap_val_xgb <- setNames(numeric(length(y_levels)), as.character(y_levels))
auc_train_xgb <- auc_val_xgb <- setNames(numeric(length(y_levels)), as.character(y_levels))

for (k in as.character(y_levels)) {
  ytr_bin_xgb <- as.integer(y_tr_xgb == as.numeric(k))
  yva_bin_xgb <- as.integer(y_val == as.numeric(k))
  
  ap_train_xgb[k]  <- ap_one_vs_rest_rocr(p_tr_xgb[, k], ytr_bin_xgb)
  ap_val_xgb[k]    <- ap_one_vs_rest_rocr(p_va_xgb[, k], yva_bin_xgb)
  auc_train_xgb[k] <- auc_one_vs_rest_rocr(p_tr_xgb[, k], ytr_bin_xgb)
  auc_val_xgb[k]   <- auc_one_vs_rest_rocr(p_va_xgb[, k], yva_bin_xgb)
}

mAP_train_xgb <- mean(ap_train_xgb, na.rm = TRUE)
mAP_val_xgb   <- mean(ap_val_xgb,   na.rm = TRUE)

# Display results
cat("\n========== XGBOOST RESULTS ==========\n")
cat("AP (train) per class:\n"); print(round(ap_train_xgb, 3))
cat("AP (val)   per class:\n"); print(round(ap_val_xgb, 3))
cat("\nmAP_train =", round(mAP_train_xgb, 3), " | mAP_val =", round(mAP_val_xgb, 3), "\n")
cat("\nAUC ROC (train):\n"); print(round(auc_train_xgb, 3))
cat("AUC ROC (val):\n"); print(round(auc_val_xgb, 3))

# OBSERVATIONS:
# XGBoost shows significantly superior performance vs GLMNet in mAP and AUC.
# mAP_train ≈ 0.933 | mAP_val ≈ 0.522 → high training fit but notable
# validation drop indicates clear overfitting.
# Per-class performance is very uneven, reflecting classification difficulty.

# GLMNet vs XGBoost comparison
cat("\n========== GLMNET vs XGBOOST COMPARISON ==========\n")
cat("mAP Train: glmnet =", round(mAP_train, 3), " | XGBoost =", round(mAP_train_xgb, 3), "\n")
cat("mAP Val:   glmnet =", round(mAP_val, 3),   " | XGBoost =", round(mAP_val_xgb, 3), "\n")
# XGBoost notably improves validation performance (+0.253 in mAP_val),
# demonstrating its ability to capture non-linear relationships and complex interactions.

# Plot ROC curves for XGBoost
plot_roc_all_classes(p_va_xgb, y_val, "Validation - XGBoost")
# Classes 7 (Person), 0 (Car) and 11 (Background) achieve AUC > 0.9 → excellent discrimination
# Classes 10 (Dynamic objects) and 1 (Large vehicle) achieve AUC > 0.75 → quite good
# Class 5 (Bicycle) still shows low AUC ≈ 0.6, highlighting the difficulty
# of building a multiclass classifier for 2D radar.

# Feature importance
importance_matrix <- xgb.importance(model = xgb_model)
cat("\nTop 10 most important features:\n")
print(head(importance_matrix, 10))
xgb.plot.importance(importance_matrix[1:min(15, nrow(importance_matrix)), ])
# Most important variable is vr_compensated, which makes sense
# as it represents how fast the object approaches or recedes from
# the radar in the world reference frame.


# 6) Does AP improve at closer distances and smaller |azimuth|? - XGBOOST ####
# Stratification in VALIDATION and metrics per bin

safe_ap_ovr <- function(scores, labels_binary, min_pos = 5, min_neg = 5) {
  n_pos <- sum(labels_binary == 1, na.rm = TRUE)
  n_neg <- sum(labels_binary == 0, na.rm = TRUE)
  if (n_pos < min_pos || n_neg < min_neg) return(NA_real_)
  ap_one_vs_rest_rocr(scores, labels_binary)
}

# Validation data with class probabilities (XGBoost)
val_probs_xgb <- as.data.frame(p_va_xgb)
clases_xgb <- colnames(p_va_xgb)

# 1) AP vs DISTANCE THRESHOLD (range_sc < t)
rmax_val <- max(val_df$range_sc, na.rm = TRUE)
dist_thresholds <- seq(5, max(5, floor(rmax_val)), by = 5)  # 5m, 10m, 15m, ...

# Calculate AP(t) table for each class
ap_vs_dist_list_xgb <- lapply(clases_xgb, function(k) {
  s <- val_probs_xgb[[k]]
  y_bin_all <- as.integer(y_val == as.numeric(k))
  data.frame(
    clase = k,
    threshold_m = dist_thresholds,
    AP = sapply(dist_thresholds, function(th) {
      idx <- which(val_df$range_sc < th)
      if (length(idx) < 20) return(NA_real_)
      safe_ap_ovr(s[idx], y_bin_all[idx])
    })
  )
})
ap_vs_dist_xgb <- do.call(rbind, ap_vs_dist_list_xgb)

# Plot function: AP vs threshold (distance) per class
plot_ap_vs_dist_threshold_one_class <- function(df, clase_target, model_name = "") {
  dfc <- df %>% filter(clase == clase_target)
  ggplot(dfc, aes(x = threshold_m, y = AP)) +
    geom_point() + geom_line() +
    labs(
      title = paste("AP vs Distance Threshold — Class:", clase_target, model_name),
      x = "Distance threshold (m), condition: range_sc < t",
      y = "Average Precision (AP)"
    ) +
    theme_minimal(base_size = 12)
}

# Print AP vs distance plots for each class
cat("\n========== AP vs DISTANCE PLOTS (XGBoost) ==========\n")

plot_ap_vs_dist_threshold_one_class(ap_vs_dist_xgb, "0", "(XGBoost)") # Car
# AP is very high at short distances (~10 m), then drops sharply to ~15 m,
# with small "bounce" near 20-22 m, stabilizing around 0.52-0.54.
# CONCLUSION: Nearby cars are easy to identify; AP drops as range increases.

plot_ap_vs_dist_threshold_one_class(ap_vs_dist_xgb, "1", "(XGBoost)") # Large Vehicle
# Sustained AP increase with distance threshold, peaking (~0.42) around 30-35 m,
# then gradually decreasing and stabilizing around 0.35-0.36.
# Large vehicles are detected better at medium distances. NOT expected behavior.

plot_ap_vs_dist_threshold_one_class(ap_vs_dist_xgb, "5", "(XGBoost)") # Bicycle
# Near-monotonic AP decrease as distance increases, from ~0.55 at short range
# to ~0.27 at long range. Expected behavior.

plot_ap_vs_dist_threshold_one_class(ap_vs_dist_xgb, "7", "(XGBoost)") # Person
# AP grows almost constantly with distance threshold, reaching ~0.85 at 50-55 m.
# This is counterintuitive (greater distance usually means lower SNR).
# Expected behavior was better detection at closer range.

plot_ap_vs_dist_threshold_one_class(ap_vs_dist_xgb, "10", "(XGBoost)") # Dynamic Object
# Coherent behavior: AP is maximum at short distances (~0.8 below 10-15 m),
# then decreases sharply, stabilizing at ~0.3 beyond 40-50 m. Expected.

plot_ap_vs_dist_threshold_one_class(ap_vs_dist_xgb, "11", "(XGBoost)") # Static Background
# Continuous AP increase with distance, from ~0.62 to ~0.77.
# Unexpected — we expected the opposite.


# 2) AP vs |AZIMUTH| THRESHOLD (abs(azimuth_sc) < t)
az_max <- suppressWarnings(max(abs(val_df$azimuth_sc), na.rm = TRUE))
az_thresholds <- seq(0.01, max(0.01, round(az_max, 2)), by = 0.01)

# Calculate AP(t) table for each class
ap_vs_az_list_xgb <- lapply(clases_xgb, function(k) {
  s <- val_probs_xgb[[k]]
  y_bin_all <- as.integer(y_val == as.numeric(k))
  data.frame(
    clase = k,
    threshold_rad = az_thresholds,
    AP = sapply(az_thresholds, function(th) {
      idx <- which(abs(val_df$azimuth_sc) < th)
      if (length(idx) < 20) return(NA_real_)
      safe_ap_ovr(s[idx], y_bin_all[idx])
    })
  )
})
ap_vs_az_xgb <- do.call(rbind, ap_vs_az_list_xgb)

# Plot function: AP vs threshold (|azimuth|) per class
plot_ap_vs_az_threshold_one_class <- function(df, clase_target, model_name = "") {
  dfc <- df %>% filter(clase == clase_target)
  ggplot(dfc, aes(x = threshold_rad, y = AP)) +
    geom_point() + geom_line() +
    labs(
      title = paste("AP vs |Azimuth| Threshold — Class:", clase_target, model_name),
      x = "|azimuth| threshold (rad), condition: |azimuth_sc| < t",
      y = "Average Precision (AP)"
    ) +
    theme_minimal(base_size = 12)
}

cat("\n========== AP vs AZIMUTH PLOTS (XGBoost) ==========\n")
plot_ap_vs_az_threshold_one_class(ap_vs_az_xgb, "0", "(XGBoost)") # Car
plot_ap_vs_az_threshold_one_class(ap_vs_az_xgb, "1", "(XGBoost)") # Large Vehicle
plot_ap_vs_az_threshold_one_class(ap_vs_az_xgb, "5", "(XGBoost)") # Bicycle
plot_ap_vs_az_threshold_one_class(ap_vs_az_xgb, "7", "(XGBoost)") # Person
plot_ap_vs_az_threshold_one_class(ap_vs_az_xgb, "10", "(XGBoost)") # Dynamic Object
plot_ap_vs_az_threshold_one_class(ap_vs_az_xgb, "11", "(XGBoost)") # Static Background

# OVERALL CONCLUSION for Section 6:
# The hypothesis is PARTIALLY confirmed.
# For some classes (Car, Bicycle, Dynamic Object), XGBoost shows expected behavior:
# AP is higher at short distances and angles close to radar axis.
#
# For other classes (Large Vehicle, Person, Static Background), opposite or
# more complex trends are observed.
#
# Regarding azimuth, the general trend is as expected: performance tends to be
# higher in frontal regions (small |azimuth|) where reflected power and angular
# resolution are maximum.


# 7) PR Threshold Selection for Person (7) Class Prioritizing Precision ####

# Using XGBoost model (better performance)
clase_persona <- "7"

# --- 7.1) PR Curve in TRAIN for Person (7) ---
y_tr_persona_bin <- as.integer(y_tr_xgb == 7)  # Binary labels: 1=Person, 0=Rest
scores_tr_persona <- p_tr_xgb[, clase_persona]  # Predicted probabilities for Person

# Generate complete PR curve with ROCR
pred_tr_persona <- ROCR::prediction(scores_tr_persona, y_tr_persona_bin)
perf_pr_tr <- ROCR::performance(pred_tr_persona, measure = "prec", x.measure = "rec")

# Extract Precision, Recall and Thresholds
df_pr_train <- data.frame(
  Recall = perf_pr_tr@x.values[[1]],
  Precision = perf_pr_tr@y.values[[1]],
  Threshold = pred_tr_persona@cutoffs[[1]]
) %>%
  filter(!is.na(Precision), !is.na(Recall)) %>%
  arrange(desc(Threshold))

# Visualize PR curve in TRAIN
ggplot(df_pr_train, aes(x = Recall, y = Precision)) +
  geom_line(linewidth = 1.2, color = "steelblue") +
  geom_point(alpha = 0.3, size = 0.8) +
  labs(
    title = "Precision-Recall Curve: Person (7) vs Rest (Train)",
    subtitle = paste("AP =", round(ap_train_xgb[clase_persona], 3)),
    x = "Recall (Sensitivity)",
    y = "Precision"
  ) +
  theme_minimal(base_size = 12) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))

# --- 7.2) Threshold Selection: MAX Recall with Precision >= precision_min ---
# Strategy: select threshold that MAXIMIZES RECALL subject to
# minimum precision constraint in TRAIN (Precision >= 0.95).
precision_min <- 0.95
tol_precision <- 1e-6  

# Candidates meeting precision constraint
df_pr_candidates <- df_pr_train %>%
  dplyr::filter(!is.na(Precision), !is.na(Recall)) %>%
  dplyr::filter(Precision + tol_precision >= precision_min)

# Objective: maximize RECALL under precision constraint.
# Tiebreakers: higher Precision; if still tied, lower threshold (less conservative).
punto_optimo <- df_pr_candidates %>%
  dplyr::arrange(dplyr::desc(Recall), dplyr::desc(Precision), Threshold) %>%
  dplyr::slice(1)

umbral_elegido  <- punto_optimo$Threshold
precision_train <- punto_optimo$Precision
recall_train    <- punto_optimo$Recall

cat("\n========== SELECTED OPERATING POINT (TRAIN) ==========\n")
cat("Probability threshold:", round(umbral_elegido, 4), "\n")
cat("Precision (Train):",     round(precision_train, 4), "\n")
cat("Recall (Train):",        round(recall_train, 4), "\n")

# --- 7.3) Visualize selected point on PR curve (Train) ---
ggplot(df_pr_train, aes(x = Recall, y = Precision)) +
  geom_line(linewidth = 1.2, color = "steelblue") +
  geom_point(data = punto_optimo, aes(x = Recall, y = Precision), 
             color = "red", size = 4, shape = 21, fill = "yellow", stroke = 2) +
  geom_text(data = punto_optimo, 
            aes(x = Recall, y = Precision, 
                label = paste0("Threshold=", round(Threshold, 3), 
                               "\nP=", round(Precision, 3), 
                               " R=", round(Recall, 3))),
            vjust = -1.5, hjust = 0.5, size = 3.5, fontface = "bold") +
  labs(
    title = "PR Curve with Selected Operating Point (Train)",
    subtitle = "Class: Person (7) vs Rest — Prioritizing Precision",
    x = "Recall (Sensitivity)",
    y = "Precision"
  ) +
  theme_minimal(base_size = 12) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))

# --- 7.4) Evaluate selected threshold in VALIDATION ---
y_va_persona_bin <- as.integer(y_val == 7)
scores_va_persona <- p_va_xgb[, clase_persona]

# Apply threshold defined in train
predicciones_val <- as.integer(scores_va_persona >= umbral_elegido)

# Calculate metrics in validation
tp_val <- sum(predicciones_val == 1 & y_va_persona_bin == 1)
fp_val <- sum(predicciones_val == 1 & y_va_persona_bin == 0)
fn_val <- sum(predicciones_val == 0 & y_va_persona_bin == 1)
tn_val <- sum(predicciones_val == 0 & y_va_persona_bin == 0)

precision_val <- tp_val / (tp_val + fp_val)
recall_val <- tp_val / (tp_val + fn_val)

cat("\n========== VALIDATION EVALUATION ==========\n")
cat("Applied threshold:", round(umbral_elegido, 4), "\n")
cat("Precision (Val):", round(precision_val, 4), "\n")
cat("Recall (Val):", round(recall_val, 4), "\n")
cat("\nConfusion Matrix (Validation):\n")
cat("                Pred=0  Pred=1\n")
cat(sprintf("Real=0 (Rest)   %6d  %6d\n", tn_val, fp_val))
cat(sprintf("Real=1 (Person) %6d  %6d\n", fn_val, tp_val))

# RESULT in TRAIN (threshold ≈ 0.293): Precision ≈ 0.951 | Recall ≈ 0.999
# When transferring that threshold to VALIDATION, the model maintains high
# sensitivity (Recall ≈ 0.973) and, as expected, Precision drops moderately
# (≈ 0.827) due to more false positives when relaxing the threshold.

# --- 7.5) Train vs Validation Comparison ---
comparacion <- data.frame(
  Metric = c("Precision", "Recall"),
  Train = round(c(precision_train, recall_train), 4),
  Validation = round(c(precision_val, recall_val), 4)
)
comparacion$Difference <- abs(comparacion$Train - comparacion$Validation)

cat("\n========== TRAIN vs VALIDATION COMPARISON ==========\n")
print(comparacion)

# Are results similar between training and validation?
# Precision: 0.951 → 0.827  (−0.124)  |  Recall: 0.999 → 0.973 (−0.026)
# Precision drop is moderate and Recall remains very high.
# The precision obtained in training doesn't transfer as well to validation
# because the threshold's precision at that recall level is much lower in
# the validation PR curve.

# --- 7.6) Comparative Visualization ---
df_comparacion <- data.frame(
  Dataset = rep(c("Train", "Validation"), each = 2),
  Metric = rep(c("Precision", "Recall"), 2),
  Value = c(precision_train, recall_train,
            precision_val, recall_val)
)

ggplot(df_comparacion, aes(x = Metric, y = Value, fill = Dataset)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = round(Value, 3)),
            position = position_dodge(width = 0.7),
            vjust = -0.5, size = 3.5) +
  labs(
    title = "Metrics Comparison: Train vs Validation",
    subtitle = paste("Class: Person (7) | Threshold =", round(umbral_elegido, 4)),
    x = "Metric",
    y = "Value",
    fill = "Dataset"
  ) +
  theme_minimal(base_size = 12) +
  coord_cartesian(ylim = c(0, 1))

# --- 7.7) Additional Analysis: Complete PR Curve in Validation ---
pred_va_persona <- ROCR::prediction(scores_va_persona, y_va_persona_bin)
perf_pr_va <- ROCR::performance(pred_va_persona, measure = "prec", x.measure = "rec")

df_pr_val <- data.frame(
  Recall = perf_pr_va@x.values[[1]],
  Precision = perf_pr_va@y.values[[1]],
  Threshold = pred_va_persona@cutoffs[[1]]
) %>%
  filter(!is.na(Precision), !is.na(Recall))

# Find validation point corresponding to selected threshold
punto_val <- df_pr_val %>%
  mutate(diff = abs(Threshold - umbral_elegido)) %>%
  filter(diff == min(diff)) %>%
  slice(1)

# Comparative PR curves plot
ggplot() +
  geom_line(data = df_pr_train, aes(x = Recall, y = Precision, color = "Train"),
            linewidth = 1) +
  geom_line(data = df_pr_val, aes(x = Recall, y = Precision, color = "Validation"),
            linewidth = 1) +
  geom_point(data = punto_optimo, aes(x = Recall, y = Precision),
             color = "red", size = 4, shape = 21, fill = "yellow", stroke = 2) +
  geom_point(data = punto_val, aes(x = Recall, y = Precision),
             color = "darkgreen", size = 4, shape = 21, fill = "lightgreen", stroke = 2) +
  labs(
    title = "Precision-Recall Curves: Train vs Validation",
    subtitle = paste("Class: Person (7) | Threshold =", round(umbral_elegido, 4)),
    x = "Recall (Sensitivity)",
    y = "Precision",
    color = "Dataset"
  ) +
  theme_minimal(base_size = 12) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  scale_color_manual(values = c("Train" = "steelblue", "Validation" = "darkorange"))

# FINAL OBSERVATIONS:
# The training PR curve is near-perfect (Precision ~1.0 across wide Recall range),
# indicating the model fits training data very well.
# The validation PR curve is more irregular and jagged: precision drops sharply
# at low recall, then stabilizes between 0.8-0.9 with notable oscillations.
#
# This pattern is due to fewer positive examples (persons) in validation,
# causing small prediction variations to produce large jumps in cumulative metrics.
# The model is conservative: it only classifies as "person" with very high probability,
# generating few true positives initially but almost no false positives.
# As threshold relaxes, more detections are included, increasing recall but
# introducing false positives, explaining the precision ups and downs.
#
# Overall, the irregular shape reflects greater uncertainty and lower stability
# in model generalization when evaluated on unseen data.
