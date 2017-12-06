import numpy as np
import re

class ColoredWeightedDoc(object):
    def __init__(self, doc, feature_names, coefs, token_pattern=r"(?u)\b\w\w+\b", binary = False):
        self.doc = doc
        self.feature_names = feature_names
        self.coefs = coefs
        self.binary = binary
        self.tokenizer = re.compile(token_pattern)
        self.abs_ranges = np.linspace(0, max([abs(coefs.min()), abs(coefs.max())]), 8)
    def _repr_html_(self):
        html_rep = ""
        tokens = self.doc.split(" ") 
        if self.binary:
            seen_tokens = set()       
        for token in tokens:
            vocab_tokens = self.tokenizer.findall(token.lower())
            if len(vocab_tokens) > 0:
                vocab_token = vocab_tokens[0]
                try:
                    vocab_index = self.feature_names.index(vocab_token)
                    
                    if not self.binary or vocab_index not in seen_tokens:
                        
                        if self.coefs[vocab_index] > 0: # positive word
                            for i in range(1, 7):
                                if self.coefs[vocab_index] < self.abs_ranges[i]:
                                    break
                            html_rep = html_rep + "<font size = " + str(i) + ", color=blue> " + token + " </font>"
                        
                        elif self.coefs[vocab_index] < 0: # negative word
                            for i in range(1, 7):
                                if self.coefs[vocab_index] > -self.abs_ranges[i]:
                                    break
                            html_rep = html_rep + "<font size = " + str(i) + ", color=red> " + token + " </font>"
                        
                        else: # neutral word
                            html_rep = html_rep + "<font size = 1, color=gray> " + token + " </font>"
                        
                        if self.binary:    
                            seen_tokens.add(vocab_index)
                    
                    else: # if binary and this is a token we have seen before
                        html_rep = html_rep + "<font size = 1, color=gray> " + token + " </font>"
                except: # this token does not exist in the vocabulary
                    html_rep = html_rep + "<font size = 1, color=gray> " + token + " </font>"
            else:
                html_rep = html_rep + "<font size = 1, color=gray> " + token + " </font>"
        return html_rep

class ColoredWeightedDocBigram(object):
    def __init__(self, doc, feature_names, coefs, token_pattern=r"(?u)\b\w\w+\b", binary = False):
        self.doc = doc
        self.feature_names = feature_names
        self.coefs = coefs
        self.binary = binary
        self.tokenizer = re.compile(token_pattern)
        self.abs_ranges = np.linspace(0, max([abs(coefs.min()), abs(coefs.max())]), 8)
        #self.ngram_size = 2 # hard-coded
    def _repr_html_(self):
        html_rep = ""
        tokens = self.doc.split(" ")
        if self.binary:
            seen_tokens = set()

        for i in range(len(tokens)-1):
            t1 = tokens[i]
            t2 = tokens[i+1] # bigram
            token = t1 + " " + t2
            vocab_tokens = self.tokenizer.findall(token.lower())
            if len(vocab_tokens) == 2: # hard-coded
                vocab_token = vocab_tokens[0] + " " + vocab_tokens[1]
                try:
                    vocab_index = self.feature_names.index(vocab_token)
                    
                    if not self.binary or vocab_index not in seen_tokens:
                        
                        if self.coefs[vocab_index] > 0: # positive word
                            for i in range(1, 7):
                                if self.coefs[vocab_index] < self.abs_ranges[i]:
                                    break
                            html_rep = html_rep + "<font size = " + str(i) + ", color=blue> [" + token + "] </font>"
                        
                        elif self.coefs[vocab_index] < 0: # negative word
                            for i in range(1, 7):
                                if self.coefs[vocab_index] > -self.abs_ranges[i]:
                                    break
                            html_rep = html_rep + "<font size = " + str(i) + ", color=red> [" + token + "] </font>"
                        
                        else: # neutral word
                            html_rep = html_rep + "<font size = 1, color=gray> [" + token + "] </font>"
                        
                        if self.binary:    
                            seen_tokens.add(vocab_index)
                    
                    else: # if binary and this is a token we have seen before
                        html_rep = html_rep + "<font size = 1, color=gray> [" + token + "] </font>"
                except: # this token does not exist in the vocabulary
                    html_rep = html_rep + "<font size = 1, color=gray> [" + token + "] </font>"
            else:
                html_rep = html_rep + "<font size = 1, color=gray> [" + token + "] </font>"
        return html_rep
