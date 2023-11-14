#importing libraries
from keras.preprocessing import text
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.utils import pad_sequences
import numpy as np
import pandas as pd


#taking random sentences as data
data = """Deep learning (also known as deep structured learning) is part of a broade
Deep-learning architectures such as deep neural networks, deep belief networks, deep
"""
dl_data = data.split()


#tokenization
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(dl_data)
word2id = tokenizer.word_index
word2id['PAD'] = 0
id2word = {v:k for k, v in word2id.items()}
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in dl_data]
vocab_size = len(word2id)
embed_size = 100
window_size = 2
print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])



#generating (context word, target/label word) pairs
def generate_context_word_pairs(corpus, window_size, vocab_size):
  context_length = window_size*2
  for words in corpus:
    sentence_length = len(words)
    for index, word in enumerate(words):
      context_words = []
      label_word = []
      start = index - window_size
      end = index + window_size + 1

      context_words.append([words[i]
                            for i in range(start, end)
                            if 0 <= i < sentence_length
                            and i != index])
      label_word.append(word)
      x = pad_sequences(context_words, maxlen=context_length)
      y = to_categorical(label_word, vocab_size)
      yield (x, y)

i = 0
for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
  if 0 not in x[0]:
      # print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2wor

    if i == 10:
      break
    i += 1

                                    

#model building
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
cbow = Sequential()
cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size * 2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(vocab_size, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print(cbow.summary())
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# SVG(model_to_dot(cbow, show_shapes=True, show_layer_names=False, rankdir='TB').cre
                   
for epoch in range(1, 6):
  loss = 0.
  i = 0
  for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
    i += 1
    loss += cbow.train_on_batch(x, y)
    if i % 100000 == 0:
      print('Processed {} (context, word) pairs'.format(i))
  print('Epoch:', epoch, '\tLoss:', loss)
  print()

weights = cbow.get_weights()[0]
weights = weights[1:]
print(weights.shape)
print(pd.DataFrame(weights, index=list(id2word.values())[1:]).head())

from sklearn.metrics.pairwise import euclidean_distances

# Assuming you have already defined 'weights', 'word2id', and 'id2word'
distance_matrix = euclidean_distances(weights)

print(distance_matrix.shape)

similar_words = {}

# Assuming 'search_terms' is a list of terms you want to find similar words for
search_terms = ['deep']

for search_term in search_terms:
    if search_term in word2id:
        idx = word2id[search_term]
        similar_words[search_term] = [id2word[i] for i in distance_matrix[idx].argsort()]
    else:
        print(f"The search term '{search_term}' is not in the word2id dictionary.")

# Now 'similar_words' should contain similar words for the search terms
print(similar_words)
