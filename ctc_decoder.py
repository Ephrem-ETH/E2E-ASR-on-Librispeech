"""
Author: Awni Hannun

This is an example CTC decoder written in Python. The code is
intended to be a simple example and is not designed to be
especially efficient.

The algorithm is a prefix beam search for a model trained
with the CTC loss function.

For more details checkout either of these references:
  https://distill.pub/2017/ctc/#inference
  https://arxiv.org/abs/1408.2873

"""

import numpy as np
import math
import collections
import re
from utils.word_to_characters import lexicon_dic
import arpa
import editdistance
#from pynlpl.lm import lm
with open('char_map.txt','r') as f:
    alphabet = []
    for char in f:
      char = char.split()
      alphabet.append(char[0])
with open('conf/keywords', 'r') as f:
    keywords = []
    for line in f:
        word = line.lower().split()
        keywords.append(word)
      
NEG_INF = -float("inf")
lexicon_dict = lexicon_dic() 

#load the language models
lm_models = arpa.loadf("mydata/local/3-gram.arpa")
# lm_models = lm.ARPALanguageModel("/home/emekonnen/mydata/E2E-ASR-pytorch/mydata/data/local/lm/3-gram.arpa")

lm = lm_models[0]

# Computing the logarithmic probability 
def compute_log_probs(trigrams):
  total_probs=0
  for i,tri in enumerate(trigrams):
    try:
      total_probs += lm.log_p(" ".join(tri))
    except KeyError:
      pass
  return total_probs

# Computing the raw probability of tri-gram
def compute_probs(trigrams):
  total_probs=0
  for i,tri in enumerate(trigrams):
    try:
      total_probs += lm.p(" ".join(tri))
    except KeyError:
      pass
  return total_probs

# This function is mainly to find the closest word from the lexicon 
# for not found words of the decoded output in the lexicon
def compute_similarity(decoded, word):

  min = 20 # assuming the maximum length of the keyword is 20
  max_prob = 0
  close_word = word
  close_word_prob = 0
  
 
  for lex_word in list(lexicon_dict.keys()):
    distance = editdistance.eval(str(word), str(lex_word.lower()))
    # print(f" {keyword} -> {preds} distance:{distance}")
    if distance < min:
      min = distance
      word_prob = lm.p(lex_word) 
      close_word = lex_word
    elif distance == min:      
      trigram1 = [x.upper() for x in decoded]
      trigram1.append(lex_word)
      try:
        if len(trigram1) >=3:
          # trigram_lex_word = [(trigram1[i],trigram1[i+1],trigram1[i+2]) for i in range(len(trigram1)-2)]
          trigram_lex_word = [trigram1[-3],trigram1[-2],trigram1[-1]]
          # word_prob = compute_probs(trigram_lex_word)
          word_prob = lm.p(" ".join(trigram_lex_word))
        trigram2 = [x.upper()  for x in decoded]
        trigram2.append(close_word)
        if len(trigram2) >=3:
          # trigram_close_word = [(trigram2[i],trigram2[i+1],trigram2[i+2]) for i in range(len(trigram2)-2)]
          trigram_close_word = [trigram2[-3],trigram2[-2],trigram2[-1]]
          # close_word_prob = compute_probs(trigram_close_word)
          close_word_prob = lm.p(" ".join(trigram_close_word))
      except KeyError:
        pass

      if word_prob > close_word_prob:
        close_word = lex_word

  return close_word.lower()

# Searching words of the decoded output that are not found in the lexicon dictionary 
# and finding the closest word from the lexicon and replacing it
def lexicon_search(best_beam):
    decoded = []
    for  word in best_beam:
        if not word.upper() in list(lexicon_dict.keys()):
            
            l_word = compute_similarity(decoded,word)
            decoded.append(l_word.lower())
        else:
            decoded.append(word)
    return decoded

def make_new_beam():
    fn = lambda : (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)

def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max)
                       for a in args))
    return a_max + lsp

def decode(probs, beam_size=10, blank=0, alpha= 0.3):
    """
    Performs inference for the given output probabilities.

    Arguments:
      probs: The output probabilities (e.g. log post-softmax) for each
        time step. Should be an array of shape (time x output dim).
      beam_size (int): Size of the beam to use during inference.
      blank (int): Index of the CTC blank label.

    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    T, S = probs.shape
    W = lambda l: re.findall(r'\w+[\s|>]', l)
    #print("len {0},{1}".format(len(alphabet),S))
    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(T): # Loop over time

        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()
        #pruned_alphabet = [alphabet[i] for i in np.where(probs[t] > -55.00 )[0]]
        for s in range(S): # Loop over vocab
        #for c in pruned_alphabet:
            #s = alphabet.index(c)
            p = probs[t, s]

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beam: # Loop over beam

                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
                  n_p_b, n_p_nb = next_beam[prefix]
                  n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                  next_beam[prefix] = (n_p_b, n_p_nb)
                  continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (alphabet[s],)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if alphabet[s] != end_t:
                  n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                                    
                    #sample code to score the prefix by LM 
                     
                else:
                  # We don't include the previous probability of not ending
                  # in blank (p_nb) if s is repeated at the end. The CTC
                  # algorithm merges characters not separated by a blank.
                  n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
                if len(n_prefix) > 1 and alphabet[s] == '>':
                      last_word = "".join(n_prefix).split(">")[-2]   
                                
                      #print(last_word)
                      if len(last_word) > 0  and last_word.upper()  in list(lexicon_dict.keys()):
                        
                        words = ("".join(n_prefix).replace(">"," ")).strip().split()
                        if len(words) >= 3:
                            trigrams = [(words[i],words[i+1],words[i+2]) for i in range(len(words)-2)]
                            log_p = compute_log_probs(trigrams)
                            lm_prob = alpha * log_p                
                            n_p_nb = logsumexp(n_p_nb,(p_nb + p) + lm_prob, (p_b + p) + lm_prob) 
                       
                     
                      else:
                        n_p_w = NEG_INF
                        n_p_nb = logsumexp(n_p_nb, p_nb + p + n_p_w, p_b + p + n_p_w)
                next_beam[n_prefix] = (n_p_b, n_p_nb)
                #print("n_p_b {0}, n_p_nb {1}".format(n_p_b,n_p_nb))

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if alphabet[s] == end_t:
                  n_p_b, n_p_nb = next_beam[prefix]
                  n_p_nb = logsumexp(n_p_nb, p_nb + p)
                  next_beam[prefix] = (n_p_b, n_p_nb)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                key=lambda x : logsumexp(*x[1]),
                #key=lambda x : logsumexp(*tuple(i + ((len(W(''.join(x[0]))) + 1) * beta) for i in list(x[1]))) ,
                reverse=True)
        beam = beam[:beam_size]

    best = beam[0]
    best_beam = "".join(best[0]).split(">")
    best_pred = lexicon_search(best_beam)
    
    return best_pred, -logsumexp(*best[1])

