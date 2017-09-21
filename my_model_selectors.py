import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * L + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_bscore = None
        modell = None

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                L = model.score(self.X, self.lengths)
                feat = model.n_features
                p = (model.startprob_.size - 1) + (model.transmat_.size - 1) + model.means_.size + model.covars_.diagonal().size
                bscore = (-2 * L + p * math.log(len(self.sequences)))
                if min_bscore is None or min_bscore > bscore:
                    min_bscore = bscore
                    modell = model
            except:
                pass

        return modell


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = None
        modell = None
        other_words = list(self.words)
        other_words.remove(self.this_word)

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                score = model.score(self.X, self.lengths)

                all_score = 0.0
                for w in other_words:
                    X, lengths = self.hwords[w]
                    all_score = all_score+model.score(X, lengths)
                dic_score =  score - (all_score / (len(self.words)))
                if best_score is None or best_score < dic_score:
                    best_score = dic_score
                    modell = model
            except:
                    pass

        return modell

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
        n_splits : int, default=3
        Number of folds. Must be at least 2.
    '''
    
    n_splits=3
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score= None
        best_model = None
        n_c=None
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                fmodel=None
                L=None
                
                if (len(self.sequences) >= 2):
                    #n_splits=SelectorCV.n_splits
                    scores = []
                    n_splits = min(len(self.sequences),3)
                    model, L = None, None
                    split_method = KFold(random_state=self.random_state, n_splits=n_splits)
                    
                    for train_index, test_index in split_method.split(self.sequences):
                        
                        X_train, lengths_train = combine_sequences(train_index, self.sequences)
                        
                        X_test, lengths_test = combine_sequences(test_index, self.sequences)
                        
                        model =GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        L = model.score(X_test, lengths_test)
                        scores.append(L)
                    avg = np.average(scores) if len(scores) > 0 else float('-inf') 
                   
                    
                else:
                    fmodel = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    avg = fmodel.score(self.X, self.lengths)
  
                if best_score is None or avg > best_score:
                    best_score = avg
                    if fmodel is None:
                        fmodel = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                             random_state=self.random_state,verbose=False).fit(self.X, self.lengths)               
                    best_model = fmodel
            except:
                pass
        return best_model
