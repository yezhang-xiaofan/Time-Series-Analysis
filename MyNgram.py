
#This file override Ngram class
from itertools import chain

from math import log

from nltk.probability import (ConditionalProbDist, ConditionalFreqDist,
                              SimpleGoodTuringProbDist)
from nltk.util import ingrams
from nltk.model.api import ModelI
from nltk.model import NgramModel

def _estimator(fdist, bins):
    """
    Default estimator function using a SimpleGoodTuringProbDist.
    """
    # can't be an instance method of NgramModel as they
    # can't be pickled either.
    return SimpleGoodTuringProbDist(fdist)

class MyNgramModel(NgramModel):
    """
    A processing interface for assigning a probability to the next word.
    """
    
    def __init__(self, n, train, pad_left=True, pad_right=False,estimator=None, *estimator_args, **estimator_kwargs):
        super(MyNgramModel,self).__init__(n,train,pad_left,pad_right,estimator,*estimator_args, **estimator_kwargs)
        assert(isinstance(pad_left, bool))
        assert(isinstance(pad_right, bool))
        
        self._n = n
        self._lpad = ('',) * (n - 1) if pad_left else ()
        self._rpad = ('',) * (n - 1) if pad_right else ()

        if estimator is None:
            estimator = _estimator

        self._cfd = ConditionalFreqDist()
        self._ngrams = set()
        
            
        # If given a list of strings instead of a list of lists, create enclosing list
        if (train is not None) and isinstance(train[0], basestring):
            train = [train]

        for sent in train:
            for ngram in ingrams(chain(self._lpad, sent, self._rpad), n):
                self._ngrams.add(ngram)
                context = tuple(ngram[:-1])
                token = ngram[-1]
                self._cfd[context].inc(token)

        if not estimator_args and not estimator_kwargs:
            self._model = ConditionalProbDist(self._cfd, estimator, len(self._cfd))
        else:
            self._model = ConditionalProbDist(self._cfd, estimator, *estimator_args, **estimator_kwargs)

        # recursively construct the lower-order models
        self._backoff = None
        if n > 1:
            self._backoff = MyNgramModel(n-1, train, pad_left, pad_right, estimator, *estimator_args, **estimator_kwargs)
        
            if self._backoff is not None:
                self._backoff_alphas = dict()
    
            # For each condition (or context)
                for ctxt in self._cfd.conditions():
                    pd = self._model[ctxt] # prob dist for this context
                    backoff_ctxt = ctxt[1:]
                    backoff_total_pr = 0
                    total_observed_pr = 0
                    for word in self._cfd[ctxt].keys(): # this is the subset of words that we OBSERVED                    
                        backoff_total_pr += self._backoff.prob(word,backoff_ctxt) 
                        total_observed_pr += pd.prob(word)        
                    assert total_observed_pr <= 1 and total_observed_pr > 0
                    assert backoff_total_pr <= 1 and backoff_total_pr > 0
                    alpha_ctxt = (1.0-total_observed_pr) / (1.0-backoff_total_pr)
        
                    self._backoff_alphas[ctxt] = alpha_ctxt
                   
# Updated _alpha function, discarded the _beta function
    def _alpha(self, tokens):
    
        if tokens in self._backoff_alphas:
            return self._backoff_alphas[tokens]
        else:
            return 1
    