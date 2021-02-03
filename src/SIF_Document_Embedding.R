# Implémentation méthode Document Embedding à partir du papier: "A Simple But Tough to Beat Baseline for Sentence Embeddings", Arora et. al

source('src/modules.R')

SIF_document_embedding <- function(D, word_embedding, D_proba_w){
  require(MASS)

  # Create Hash table to increase speed (100 times faster than vectors)
  
  print("Try loading word probability hash table ...")
  stderr_pw <- try( load('../ressources/pw_hash.var' ), silent = TRUE )
  if(stderr_pw != 'pw_hash'){
    print("Compute word probabilities ...")
    p_w <- word_proba(D_proba_w)
    D_words <- names(word_proba(unlist(D)))
    p_w <- p_w[which(names(p_w) %in% D_words)]
    print("Store in a hash table ...")
    pw_hash <- new.env(hash = TRUE)
    for(n in names(p_w)){
      pw_hash[[n]] <- p_w[n]
    }
    print("Save ...")
    save(list = 'pw_hash', file = '../ressources/pw_hash.var')
  }
  
  print("Try loading embedding hash table ...")
  stderr_vw <- try( load('../ressources/vw_hash.var' ), silent = TRUE )
  if(stderr_vw != 'vw_hash'){
    v_w <- word_embedding$vec
    v_vocab <- word_embedding$vocabulary
    print("Store in a hash table ...")
    vw_hash <- new.env(hash = TRUE)
    for(n in 1:length(v_vocab)){
      vw_hash[[ v_vocab[n] ]] <- v_w[n, ]
    }
    print("Save ...")
    save(list = 'vw_hash', file = '../ressources/vw_hash.var')
  }
  
  
  v_vocab <- word_embedding$vocabulary
  N <- dim(word_embedding$vec)[2]
  a <- 0.001
  
  
  # Start algorithm
  print("Start SIF weighting algorithm ...")
  
  # Note: hash table version, would be greatly improved if vw_hash contains the additional vectors from D
  v_d <- matrix(0, nrow = length(D), ncol = N)
  i <- 1
  print("First step: compute v_d")
  for(doc in D){
    d <- unlist(doc)
    sum_vd <- numeric(length = N)
    # line below not needed if vw_hash contains all vectors from D
    d <- d[d %in% v_vocab]
    norm <- 0
    for(w_i in d){
      norm <- norm + ( a / (a + pw_hash[[w_i]]) )
      sum_vd <- sum_vd + ( a / (a + pw_hash[[w_i]]) ) * as.numeric(vw_hash[[w_i]])
    }
    v_d[i, ] <- sum_vd / norm
    i <- i + 1
  }
  
  print("Second step: remove first component v_d projection ...")
  X <- v_d
  X_svd <- svd(X)
  u <- X_svd$u[1, ]
  for(j in 1:length(D)){
    v_d[j, ] <- v_d[j, ] - u %*% t(u) %*% v_d[j, ]
  }
  v_d <- t(v_d)
  # End algorithm
  
  print("Save SIF document embedding vectors ...")
  saveRDS(v_d, file = '../ressources/SIF_vectors.Rda')
  return(v_d)
}





