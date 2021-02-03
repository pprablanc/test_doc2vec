# Document Embedding weighted with topic models

tm_embedding_main_topic <- function(corpus, word_embedding, k_topics = 10, method = 'LDA'){
  require(text2vec)
  docsplit <- strsplit(tolower(corpus), ' ') 
  
  # Topic modeling preprocessing
  print("Topic modeling preprocessing ...")
  tokens <- word_tokenizer(tolower(corpus))
  it <- itoken(tokens)
  vocab <- create_vocabulary(it, stopword = tm::stopwords("english"))
  vocab <- prune_vocabulary(vocab, term_count_min = 10) # filtrage freq(mot) < 5
  vectorizer <- vocab_vectorizer(vocab)
  dtm <- create_dtm(it, vectorizer)
  
  # Topic modeling
  print('Topic modeling ...')
  if(method == 'LDA'){
    require(topicmodels)
    lda_model <- topicmodels::LDA(dtm, k = k_topics)
    doc_topic <- lda_model@gamma
    word_topic <- topicmodels::posterior(lda_model)$terms
    topic_terms <- lda_model@terms
  }else if(method == 'LSA'){
    # Mauvaise idée avec LSA car on n'a pas de proportions dans les matrices, mais des valeurs négatives et positives
    model_tfidf = TfIdf$new()
    dtm_tfidf = model_tfidf$fit_transform(dtm)
    lsa = text2vec::LatentSemanticAnalysis$new(k_topics)
    doc_topic <- lsa$fit_transform(dtm_tfidf)
    word_topic <- lsa$components
    topic_terms <- colnames(word_topic)
  }else if(method == 'NMF'){
    # require(NMF)
    print('La méthode prend trop de temps pour le moment. Arret de la fonction.')
    # model_tfidf = TfIdf$new()
    # dtm_tfidf = model_tfidf$fit_transform(dtm)
    # topic_nmf <- nmf(dtm_tfidf, n_topics)
    return(NULL)
  }
  
  # get most significant topic per document
  prin_topic <- apply(doc_topic, 1, function(x) which.max(x) )
  
  print("Try loading embedding hash table ...")
  stderr_vw <- try( load('../ressources/vw_hash.var' ), silent = TRUE )
  if(stderr_vw != 'vw_hash'){
    print('Cannot load word embedding hash table')
  }

  # Weight each word vector with their most significant word per topic coefficient
  print("Construct document vectors ...")
  N <- dim(word_embedding$vec)[2]
  v_vocab <- word_embedding$vocabulary
  v_d <- matrix(0, nrow = length(docsplit), ncol = N)
  for(i in 1:length(docsplit)){
    svMisc::progress(i, max.value = length(docsplit) )
    d <- unlist(docsplit[[i]])
    sum_vd <- numeric(length = N)
    norm <- 0
    # line below not needed if vw_hash contains all vectors from docsplit
    d <- d[d %in% v_vocab & d %in% topic_terms]
    ind_wt <- match(d, topic_terms)
    W <- word_topic[prin_topic[i], ind_wt]
    W <- exp(W) / sum(exp(W))
    
    for(w_j in d){
      sum_vd <- sum_vd + W[w_j] * as.numeric( vw_hash[[w_j]] )
    }
    v_d[i, ] <- sum_vd
  }
  return(t(v_d))
}


tm_embedding <- function(corpus, word_embedding, k_topics = 10, method = 'LDA'){
  require(text2vec)
  docsplit <- strsplit(tolower(corpus), ' ') 
  
  # Topic modeling preprocessing
  print("Topic modeling preprocessing ...")
  tokens <- word_tokenizer(tolower(corpus))
  it <- itoken(tokens)
  vocab <- create_vocabulary(it, stopword = tm::stopwords("english"))
  vocab <- prune_vocabulary(vocab, term_count_min = 10) # filtrage freq(mot) < 5
  vectorizer <- vocab_vectorizer(vocab)
  dtm <- create_dtm(it, vectorizer)
  
  # Topic modeling
  print('Topic modeling ...')
  if(method == 'LDA'){
    require(topicmodels)
    lda_model <- topicmodels::LDA(dtm, k = k_topics)
    doc_topic <- lda_model@gamma
    word_topic <- topicmodels::posterior(lda_model)$terms
    topic_terms <- lda_model@terms
  }else if(method == 'LSA'){
    model_tfidf = TfIdf$new()
    dtm_tfidf = model_tfidf$fit_transform(dtm)
    lsa = LatentSemanticAnalysis$new(k_topics)
    doc_topic <- lsa$fit_transform(dtm_tfidf)
    word_topic <- lsa$components
    topic_terms <- colnames(word_topic)
  }else if(method == 'NMF'){
    # require(NMF)
    # print('La méthode prend trop de temps pour le moment. Arret de la fonction.')
    # model_tfidf = TfIdf$new()
    # dtm_tfidf = model_tfidf$fit_transform(dtm)
    # topic_nmf <- nmf(as.matrix(dtm_tfidf), k_topics)
    # return(NULL)
  }
  
  print("Try loading embedding hash table ...")
  stderr_vw <- try( load('../ressources/vw_hash.var' ), silent = TRUE )
  if(stderr_vw != 'vw_hash'){
    print('Cannot load word embedding hash table')
  }
  
  # Weight each word vector with their most significant word per topic coefficient
  print("Construct document vectors ...")
  N <- dim(word_embedding$vec)[2]
  v_vocab <- word_embedding$vocabulary
  v_d <- matrix(0, nrow = length(docsplit), ncol = N)
  for(i in 1:length(docsplit)){
    svMisc::progress(i, max.value = length(docsplit) )
    d <- unlist(docsplit[[i]])
    sum_vd <- numeric(length = N)
    
    # line below not needed if vw_hash contains all vectors from docsplit
    d <- d[d %in% v_vocab & d %in% topic_terms]
    
    # get word topic vectors
    ind_wt <- match(d, topic_terms)
    word_topic_submat <- word_topic[, ind_wt]
    
    # compute product of word topic vectors and document topic vector
    doc_word_prod <- doc_topic[i, ] * word_topic_submat
    
    # get max of doc_word_prod for each word
    W <- apply(doc_word_prod, 2, function(x) max(x))
    W <- exp(W) / sum(exp(W))
    
    for(w_j in d){
      sum_vd <- sum_vd + W[w_j] * as.numeric( vw_hash[[w_j]] )
    }
    v_d[i, ] <- sum_vd
  }
  return(t(v_d))
}






