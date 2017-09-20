import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequencesuences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for i in range(test_set.num_items):
    
        prob=float("-inf")
        max_word =  None
        
        word_probabilities = {}
        
        sequences, lengths = test_set.get_item_Xlengths(i)
        for word, model in models.items():
            try:
                word_probabilities[word] = model.score(sequences, lengths)
            except Exception as e:
                word_probabilities[word] = float("-inf")
            
            if word_probabilities[word] > prob:
                prob, max_word = word_probabilities[word], word
                
        probabilities.append(word_probabilities)
        guesses.append(max_word)
        
    return probabilities, guesses
