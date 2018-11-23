from keras.preprocessing.text import Tokenizer


samples = ['The cat sat on the mat.', 'The dog ate my homework.']

#Creates a tokenizer, configured to only take into account the 1,000 most common words
tokenizer = Tokenizer(num_words=1000)

#builds the word index
tokenizer.fit_on_texts(samples)

#Turns strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
