source('src/modules.R')

weight_embedding <- function(corpus, ponderation, k = 2 , b = 0.75){
  require(udpipe)

  # description: calculer les ponderation tf idf et okapi
  # input:
  # - corpus
  # - ponderation: type de ponderation : tf-idf ou okapi
  # - k et b: les paramètres de Okapi
  # output: selon le type de ponderation, les moyennes des tf-idf ou Okapi par termes
  corpus.preprocess <- tolower(corpus)
  x <- document_term_frequencies(x = corpus.preprocess, split = " ")
  x <- document_term_frequencies_statistics(x, k, b)

  if (ponderation == "tfidf"){
    return(x[, c("doc_id", "term", "idf")])
  }else if (ponderation == "okapi"){
    return(x[, c("doc_id", "term","bm25")])
  }else message("Choose weighting scheme. tfidf/okapi")
}


document_embedding <- function(embedding, corpus, id, methode = "iso", weights = NA){
  # description: calculer le barycentre des vecteurs de word embedding
  # input: liste de mots, vecteur de word embedding, methode = c("barycentre","tfidf","okapi")
  # ouput: la moyenne
  
  doc_id <- paste0('doc', as.character(id))
  document <- corpus[[id]]

  word_vec <- embedding$vec
  vocabulary <- embedding$vocabulary

  #tokens
  mots <- unlist(strsplit(document, split = " "))
  
  #trouver l'index des mots entrés dans les vecteurs des mots embedding
  ind_mots <- na.omit(match(mots , vocabulary))

  if (methode == "iso") {
    barycentre <- mapply(mean, word_vec[ind_mots, ])
    return (barycentre)
    
  }else if (methode == "tfidf") {
    
    # words_in_vec <- as.vector(word_vec[ind_mots,1])
    # pond <- dtm_tfidf[num_doc , match(words_in_vec[,1],colnames(dtm_tfidf))]
    # tfidf <- colSums(pond * as.matrix(word_vec[ind_mots,-1]))/as.numeric(length(pond))
    
    words_in_vec <- as.vector( vocabulary[ind_mots] )
    ind <- which(weights$doc_id == doc_id)
    w <- weights[ind, c('term', 'idf')]
    pond <- as.numeric( as.matrix( w[match(words_in_vec, w$term), "idf"] ) )
    Z <- sum(pond)
    tfidf <- colSums(pond * as.matrix( word_vec[ind_mots, ] ) ) / Z
    return (tfidf)
    
  }else if (methode == "okapi"){
    words_in_vec <- as.vector( vocabulary[ind_mots] )
    w <- weights[which(weights$doc_id == doc_id), c('term', 'bm25')]
    pond <- as.numeric(as.matrix(w[match(words_in_vec, w$term), "bm25"]))
    Z <- sum(pond)
    okapi <- colSums(pond * as.matrix(word_vec[ind_mots, ])) / Z
    return (okapi)
  }
}




