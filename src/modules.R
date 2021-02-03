# other functions

load_word_embedding_vector <- function(filename, nrows, header = FALSE , stringsAsFactors = FALSE, sep = " "  ,quote= "", skip = 1){
  if(!is.null(nrows)){
    word_vec <- read.csv(filename, header = header ,stringsAsFactors = FALSE,
                         sep = sep ,quote= "", nrows = nrows,  skip = skip)
  }else{
    word_vec <- read.csv(filename, header = header ,stringsAsFactors = FALSE,
                         sep = sep ,quote= "",  skip = skip)
  }

  names(word_vec) <- NULL
  return( list( vec = word_vec[, -1], vocabulary = word_vec[, 1]) )
}

word_proba <- function(documents){
  require(tm)
  
  # list of preprocessing functions
  # ctrl <- list(tolower, stemming = SnowballC::wordStem, language = 'en')
  ctrl <- list(tolower, wordLengths = c(1, Inf))
  
  # compute term frequency
  tf <- termFreq(documents, control = ctrl)
  # low_freq_term <- which(tf < freq_min)
  # tf <- tf[-low_freq_term]
  
  # divide by the total number of terms to get term probability
  p_w <- tf / sum(tf)

  # barplot(sort(p_w, decreasing = TRUE)[1:50])
  return(p_w)
}

