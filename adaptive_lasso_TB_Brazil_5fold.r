set.seed(1005)
require(glmnet, "/projects/sunlab/R.lib")
#library(ROCR,"/projects/sunlab/R.lib")
#library(pROC,"/projects/sunlab/R.lib")
######################################################


models<-list()
pred_data_list<-list()
pred_prob_list<-list()
impvar_list<-list()

#######################################################
dat_cv<-readRDS("/projects/sunlab/Students_Work/Pradeep_work/TB_modelling/Brazil/data_TB_Brazil_methylation.RDS")

#### fit ridge regression. 
#cv_ridge <- cv.glmnet(as.matrix(dat_cv[,-1]), dat_cv[,1], family = "binomial",type.measure="class", alpha = 0,standardize=T)

#saveRDS(cv_ridge,"cv_ridge.RDS")

cv_ridge<-readRDS("/projects/sunlab/Students_Work/Pradeep_work/TB_modelling/Brazil/adaptive_lasso/cv_ridge.RDS")

ridge_coefs <- as.vector(coef(cv_ridge, s = "lambda.min"))[-1] ## remove intercept

weights <- 1 / abs(ridge_coefs + 1e-5)  # Add small value to avoid division by zero

foldid<-sample(1:5,size=nrow(dat_cv),replace=TRUE) 

saveRDS(foldid,"foldid.RDS")

for(i in 1:5){

  sample_index<-which(foldid==i)

  dat_train<-dat_cv[-sample_index,]
  
  dat_test<-dat_cv[sample_index,]

  model.cv<-cv.glmnet(as.matrix(dat_train[,-1]), dat_train[,1],family = "binomial",type.measure="class", alpha = 1,standardize=T,penalty.factor = weights)

  pred = predict(model.cv, newx = as.matrix(dat_test[,-1]),s='lambda.min',type="class")
  
  colnames(pred)<-"Predicted_Class"
  
  pred_data<-merge(pred,dat_test[,1,drop=F],by=0)
  
  pred_data_list[[i]]<-pred_data
  
  pred = predict(model.cv, newx = as.matrix(dat_test[,-1]),s='lambda.min',type="response")
  
  colnames(pred)<-"Predicted_Probability"
  
  pred_score<-merge(pred,dat_test[,1,drop=F],by=0)
  
  pred_prob_list[[i]]<-pred_score
  
  
    
  models[[i]]<-model.cv
  
  impvar<-coef(model.cv,s=model.cv$lambda.min)

  impvar<-data.frame(as.matrix(impvar[impvar[,1]!=0,]))
  
  #impvar_counts[i] <- nrow(impvar)  
  
  colnames(impvar)<-paste0("coef_","model",i)
  
  impvar$Var<-rownames(impvar)
  
  impvar_list[[i]]<-impvar

}

saveRDS(foldid,"foldid.RDS")
saveRDS(models,"Models_ResultCVenet_BE_Methyl450Epic_combinedMale.RDS")
saveRDS(pred_data_list,"predicted_data_list.RDS")
saveRDS(pred_prob_list,"Predicted_probability_list.RDS")


merged_df <- Reduce(function(x, y) merge(x, y, by = "Var", all = TRUE),impvar_list)
merged_df$non_na_count <- apply(merged_df[, -1], 1, function(x) sum(!is.na(x)))
write.csv(merged_df,"impvar_all_models.csv")