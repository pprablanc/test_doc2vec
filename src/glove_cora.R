library(text2vec)
source('modules.R')

data_filename <- "../dataset/cora_modified/data.txt"
group_filename <- '../dataset/cora_modified/group.txt'

data.group <- readLines(group_filename)
data.group <- as.matrix(data.group)
data.group <- factor(data.group)
empty <- which(data.group == '') # On relève les indices des cases vides
data.group <- data.group[-empty] # On enlève les données vides

#charger les documents
data.text <- readLines(data_filename, encoding = "utf8")
# On enlève les données documents correspondant aux groupes non renseignés
data.text <- data.text[-empty]
#minuscule
data.text <- tolower(data.text)

# text8 <- readLines('../dataset/text8/text8', n = 1, encoding = "utf8", warn = FALSE)
# data.text <- c(data.text, text8)

#créer iterateur
tokens <- space_tokenizer(data.text)

#créer vocabulaire
it <- itoken(tokens, progresbar = FALSE)
vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, term_count_min = 10L)

#utilise le vocab
vectorizer <- vocab_vectorizer(vocab)
#fenetre de 5 mots
tcm <- create_tcm(it, vectorizer, skip_grams_window = 3L)

# load pre-traind vectors
u <- load_word_embedding_vector('../dataset/glove.6B/glove.6B.300d.txt', nrows = 50000)
word_vectors_size <- dim(u$vec)[2]
vocab_size <- length(u$vocabulary)
b_u <- runif(vocab_size, -0.5, 0.5)
v <-  matrix(runif(vocab_size * word_vectors_size, -0.5, 0.5),
             nrow = vocab_size,
             ncol = word_vectors_size)
b_v <- runif(vocab_size, -0.5, 0.5)

#glove
glove <- GlobalVectors$new(word_vectors_size = 300, 
                           vocabulary = u$vocabulary, 
                           x_max = 100, 
                           initial = list(w_i = as.matrix(u$vec), w_j = v, b_i = b_u, b_j = b_v))
target_word <- glove$fit_transform(tcm, n_iter = 100)

context_word <- glove$components

word_vectors <- target_word + t(context_word)
rm(context_word)

words <- colnames(tcm)
ind <- which(! (words %in% u$vocabulary) )
missing_vectors <- word_vectors[ind, ]
new_embedding <- rbind(word_vectors, missing_vectors)

write.table(new_embedding, file = "../ressources/glove_cora.txt", col.names = FALSE, quote = FALSE)
