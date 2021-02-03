#' generate_train_test
#'
#' @param x une matrice DxN qui contient les features où D est la dimension d'un
#'   vecteur observation et N est le nombre d'observations
#' @param y le vecteur de labels des observations x. Le vecteur y est de taille
#'   N.
#' @param training_amount un scalaire compris entre 0 et 1 définissant le
#'   pourcentage de données utilisés dans l'entrainement
#'
#' @return une liste de deux dataframe: train et test. train et test contiennent
#'   respectivement les données (features + labels) pour le test et
#'   l'entrainement. Les noms de colonne des dataframe sont y pour les labels et
#'   x pour les features.
generate_train_test <- function(x, y, training_amount){

  ind <- generate_train_test_ind(x, y, training_amount)
  data <- assign_ind(x, y, ind)
  
  return( list(train = data$train, test = data$test))
}

generate_train_test_ind <- function(x, y, training_amount){
  nobs <- length(y)
  n_sample_train <- round(training_amount * nobs)
  n_sample_test <- nobs - n_sample_train
  training_ind <- sample(nobs, n_sample_train)
  test_ind <- setdiff(1:nobs, training_ind)
  return(list(train = training_ind, test = test_ind))
}

assign_ind <- function(x, y, ind){
  training_x <- x[, ind$train]
  test_x <- x[, ind$test]
  training_y <- y[ind$train]
  test_y <- y[ind$test]
  
  train <- data.frame(x = t(training_x), y = training_y)
  test <- data.frame(x = t(test_x), y = test_y)
  return(list(train = train, test = test))
}



#' Title
#'
#' @param truth vecteur contenant les labels des individus test.
#' @param pred vecteur contenant les prédictions sur les individus test.
#'
#' @return liste de scalaires: precision, recall et f1_score.
prf <- function(truth, pred){
  cont_tab <- table(truth, pred)
  precision <- mean( diag(cont_tab) / colSums(cont_tab), na.rm = TRUE ) 
  recall <- mean( diag(cont_tab) / rowSums(cont_tab), na.rm = TRUE ) 
  f1_score <- 2 * (precision * recall) / (precision + recall) 
  return( list(precision = precision, recall = recall, f1_score = f1_score, tab_err = cont_tab))
}

#' Title
#'
#' @param x est une liste de prf concaténés de la manière suivante: c(prf1, prf2)
#'
#' @return 
mean_prf <- function(x){
  precision_ind <- which(names(x) == "precision")
  recall_ind <- which(names(x) == "recall")
  f1_score_ind <- which(names(x) == "f1_score")
  mean_precision <- mean(x[precision_ind]$precision)
  mean_recall <- mean(x[recall_ind]$recall)
  mean_f1_score <- mean(x[f1_score_ind]$f1_score)
  return( list( precision = mean_precision, recall = mean_recall, f1_score = mean_f1_score))
}