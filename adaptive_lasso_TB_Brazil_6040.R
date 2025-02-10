set.seed(1005)
require(glmnet, "/projects/sunlab/R.lib")
require(caret,"/projects/sunlab/R.lib/")

######################################################
models <- list()
pred_data_list <- list()
pred_prob_list <- list()
impvar_list <- list()

#######################################################
dat_cv <- readRDS("/projects/sunlab/Students_Work/Pradeep_work/TB_modelling/Brazil/data_TB_Brazil_methylation.RDS")

# Load Ridge regression coefficients
cv_ridge <- readRDS("/projects/sunlab/Students_Work/Pradeep_work/TB_modelling/Brazil/adaptive_lasso/cv_ridge.RDS")

ridge_coefs <- as.vector(coef(cv_ridge, s = "lambda.min"))[-1]  ## remove intercept
weights <- 1 / abs(ridge_coefs + 1e-5)  # Add small value to avoid division by zero

# **Stratified 60:40 Train-Test Split**
train_index <- createDataPartition(dat_cv[, 1], p = 0.6, list = FALSE)  # 60% train, balanced by class

dat_train <- dat_cv[train_index, ]
dat_test <- dat_cv[-train_index, ]

# Save train-test split indices
saveRDS(train_index, "train_index.RDS")

# Fit Adaptive Lasso model
model.cv <- cv.glmnet(as.matrix(dat_train[,-1]), dat_train[,1], 
                       family = "binomial", type.measure = "class", 
                       alpha = 1, standardize = TRUE, penalty.factor = weights)

# Predict class labels
pred <- predict(model.cv, newx = as.matrix(dat_test[,-1]), s = 'lambda.min', type = "class")
colnames(pred) <- "Predicted_Class"
pred_data <- merge(pred, dat_test[, 1, drop = FALSE], by = 0)
pred_data_list <- pred_data

# Predict probability scores
pred_prob <- predict(model.cv, newx = as.matrix(dat_test[,-1]), s = 'lambda.min', type = "response")
colnames(pred_prob) <- "Predicted_Probability"
pred_score <- merge(pred_prob, dat_test[, 1, drop = FALSE], by = 0)
pred_prob_list <- pred_score

# Save trained model
models <- model.cv

# Extract important variables
impvar <- coef(model.cv, s = model.cv$lambda.min)
impvar <- data.frame(as.matrix(impvar[impvar[, 1] != 0, ]))
colnames(impvar) <- "Coefficient"
impvar$Var <- rownames(impvar)
impvar_list <- impvar

# Save results
saveRDS(models, "Models_ResultCVenet_BE_Methyl450Epic_combinedMale.RDS")
saveRDS(pred_data_list, "predicted_data_list.RDS")
saveRDS(pred_prob_list, "Predicted_probability_list.RDS")

# Save important variables
write.csv(impvar_list, "impvar_model.csv")
