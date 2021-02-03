# Evaluation using CORA dataset
# dataset available at:
# https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
# Documentation available at:
# https://relational.fit.cvut.cz/dataset/CORA
# https://linqs.soe.ucsc.edu/data

# Here we use modified Cora dataset found at https://github.com/thunlp/CANE/tree/master/datasets/cora

# classes:
# Case_Based
# Genetic_Algorithms
# Neural_Networks
# Probabilistic_Methods
# Reinforcement_Learning
# Rule_Learning
# Theory

###########################
#### Loading text data #### 
###########################


data_filename <- 'dataset/cora_modified/data.txt'
group_filename <- 'dataset/cora_modified/group.txt'
data.group <- readLines(group_filename)
data.group <- as.matrix(data.group)
data.group <- factor(data.group)
empty <- which(data.group == '') # Get index of unknown groups
data.group <- data.group[-empty] # Removing unknown groups data
data.text <- readLines(data_filename, encoding = "utf8")
# Removing documents with unknown groups index
data.text <- data.text[-empty] 
# Split each document as a list of words and set to lowercase.
data.docsplit <- strsplit(tolower(data.text), ' ') 


#############################################
#### Extract document embeddings vectors ####
#############################################

# Required packages
library('doc2vec')
source('src/modules.R')
source('src/baseline.R')


# load word embedding vectors retrained on Cora dataset
word_embedding <- load_word_embedding_vector("resources/glove_cora.txt", nrows = NULL)

# Compute document embedding vectors
doc_emb_isobar <- sapply( 1:length(data.docsplit), 
                       function(doc_id) document_embedding(word_embedding, data.docsplit, doc_id, methode = "iso") )

# doc2vec method
x <- data.frame(doc_id = 1:length(data.text), text = data.text)
doc_emb_PVDM <- t(as.matrix(paragraph2vec(x, type="PV-DM", dim=300), which = "docs"))
doc_emb_PVDBOW <- t(as.matrix(paragraph2vec(x, type="PV-DBOW", dim=300), which = "docs"))

####################
#### Evaluation ####
####################

## Classification task with 2 methods: Naive Bayesian and SVM.

# Required packages
library(MASS)
library(e1071)
source('src/evaluation_functions.R')


# Training data percentage 
training_amount <- c(0.1, 0.2, 0.3, 0.4, 0.5)
out_csv <- data.frame(X1 = c("Training %", "Naive Bayes isobar", "Naive Bayes PVDM", "Naive Bayes PVDBOW", "SVM isobar", "SVM PVDM", "SVM PVDBOW"))

for(ta in training_amount){
  
  print(paste("training amount: ", 100*training_amount[ta], "% ; test amount:", 100*(1-training_amount[ta]), "%"))
  
  prf_nb_isobar <- c()
  prf_svm_isobar <- c()
  
  prf_nb_tfidfbar <- c()
  prf_svm_tfidfbar <- c()

  prf_nb_PVDM <- c()
  prf_svm_PVDM <- c()

  prf_nb_PVDBOW <- c()
  prf_svm_PVDBOW <- c()

  
  for(i in 1:10){
    svMisc::progress(i, max.value = 10 )
    # génération des même indices pour que les données soient comparées sur les même exemples
    ind <- generate_train_test_ind(doc_emb_isobar, data.group, training_amount = ta ) 
    
    ###################
    ## Vecteurs isobar ##
    dataset <- assign_ind(doc_emb_isobar, data.group, ind)
    
    # Bayésien Naïf
    nb_isobar <- naiveBayes(y ~ ., data = dataset$train)
    pred_nb_isobar <- predict(nb_isobar, dataset$test)
    prf_nb_isobar <- c(prf_nb_isobar, prf(dataset$test$y, pred_nb_isobar))
    
    # SVM
    svm_model_isobar = svm(y ~ ., data = dataset$train )
    pred_svm_isobar <- predict(svm_model_isobar, dataset$test )
    prf_svm_isobar <- c(prf_svm_isobar, prf(dataset$test$y, pred_svm_isobar))
    
    ###################
    ## Vecteurs PVDM ##
    dataset <- assign_ind(doc_emb_PVDM, data.group, ind)
    
    # Bayésien Naïf
    nb_PVDM <- naiveBayes(y ~ ., data = dataset$train)
    pred_nb_PVDM <- predict(nb_PVDM, dataset$test)
    prf_nb_PVDM <- c(prf_nb_PVDM, prf(dataset$test$y, pred_nb_PVDM))
    
    # SVM
    svm_model_PVDM = svm(y ~ ., data = dataset$train )
    pred_svm_PVDM <- predict(svm_model_PVDM, dataset$test )
    prf_svm_PVDM <- c(prf_svm_PVDM, prf(dataset$test$y, pred_svm_PVDM))
    
    
    ###################
    ## Vecteurs PVBOW ##
    dataset <- assign_ind(doc_emb_PVDBOW, data.group, ind)
    
    # Bayésien Naïf
    nb_PVDBOW <- naiveBayes(y ~ ., data = dataset$train)
    pred_nb_PVDBOW <- predict(nb_PVDBOW, dataset$test)
    prf_nb_PVDBOW <- c(prf_nb_PVDBOW, prf(dataset$test$y, pred_nb_PVDBOW))
    
    # SVM
    svm_model_PVDBOW = svm(y ~ ., data = dataset$train )
    pred_svm_PVDBOW <- predict(svm_model_PVDBOW, dataset$test )
    prf_svm_PVDBOW <- c(prf_svm_PVDBOW, prf(dataset$test$y, pred_svm_PVDBOW))
  }  
  
  mean_prf_nb_isobar <- mean_prf(prf_nb_isobar)
  mean_prf_svm_isobar <- mean_prf(prf_svm_isobar)
  
  mean_prf_nb_PVDM <- mean_prf(prf_nb_PVDM)
  mean_prf_svm_PVDM <- mean_prf(prf_svm_PVDM)
  
  mean_prf_nb_PVDBOW <- mean_prf(prf_nb_PVDBOW)
  mean_prf_svm_PVDBOW <- mean_prf(prf_svm_PVDBOW)
  
  
  
  print("naive bayes")
  print(paste("isobar:", mean_prf_nb_isobar$f1_score))
  print(paste("PV-DM:", mean_prf_nb_PVDM$f1_score))
  print(paste("PV-DBOW:", mean_prf_nb_PVDBOW$f1_score))
  
  print("SVM")
  print(paste("isobar:", mean_prf_svm_isobar$f1_score))
  print(paste("PV-DM:", mean_prf_svm_PVDM$f1_score))
  print(paste("PV-DBOW:", mean_prf_svm_PVDBOW$f1_score))
  
  
  new_col <- 100 * c(ta,
                     mean_prf_nb_isobar$f1_score,
                     mean_prf_nb_PVDM$f1_score,
                     mean_prf_nb_PVDBOW$f1_score,
                     mean_prf_svm_isobar$f1_score,
                     mean_prf_svm_PVDM$f1_score,
                     mean_prf_svm_PVDBOW$f1_score
               )
  
  out_csv <- cbind(out_csv, new_col)
}  
write.table(out_csv, file="results.csv", col.names=FALSE)


