import os.path
import sys
import random
from operator import itemgetter
from collections import defaultdict
import math

#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            corpus.append(sentence)
            if i % 1000 == 0:
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        return corpus
    else:
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script

# Preprocess the corpus
def preprocess(corpus):
    UNK = "UNK"     # Unknown word token
    start = "<s>"   # Start-of-sentence token
    end = "</s>"    # End-of-sentence-token

    # Find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
        for word in sen:
            freqDict[word] += 1

    # Replace rare words with UNK
    for sen in corpus:
        for i in range(len(sen)):
            if freqDict[sen[i]] < 2:
                sen[i] = UNK

    # Bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    
    return corpus

def preprocessTest(vocab, corpus):
    UNK = "UNK"     # Unknown word token
    start = "<s>"   # Start-of-sentence token
    end = "</s>"    # End-of-sentence-token

    # Replace test words that were unseen in the training with UNK
    for sen in corpus:
        for i in range(len(sen)):
            if sen[i] not in vocab:
                sen[i] = UNK
    
    # Bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)

    return corpus

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    def __init__(self, corpus):
        print("""Your task is to implement four kinds of n-gram language models:
      a) an (unsmoothed) unigram model (UnigramModel)
      b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
      c) an unsmoothed bigram model (BigramModel)
      d) a bigram model smoothed using linear interpolation smoothing (SmoothedBigramModelLI)
      """)
    
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return ["mary", "had", "a", "little", "lamb", "."]
    
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        return 0.0
    
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)
            stringGenerated = str(prob) + " " + " ".join(sen) 
            print(stringGenerated, file=filePointer)

# Unigram language model
class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
    
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
    
    def prob(self, word):
        return self.counts[word] / self.total
    
    def generateSentence(self):
        sen = [start]
        while True:
            word = self.draw()
            sen.append(word)
            if word == end:
                break
        return sen
    
    def draw(self):
        rand = random.random()
        for word in list(self.counts.keys()):
            rand -= self.prob(word)
            if rand <= 0.0:
                return word

    def getSentenceProbability(self, sen):
        prob = 1.0
        for word in sen[1:]:  # skip the start token
            prob *= self.prob(word)
            if prob == 0:
                prob = 1e-10  # to avoid log(0)
        return prob
    
    def getCorpusPerplexity(self, corpus):
        log_prob = 0.0
        word_count = 0
        for sen in corpus:
            word_count += len(sen) - 1  # exclude the start token
            prob = self.getSentenceProbability(sen)
            log_prob += math.log(prob)
        return math.exp(-log_prob / word_count)

# Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.vocab_size = 0
        self.train(corpus)
    
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
        self.vocab_size = len(self.counts)
    
    def prob(self, word):
        return (self.counts[word] + 1) / (self.total + self.vocab_size)
    
    def generateSentence(self):
        sen = [start]
        while True:
            word = self.draw()
            sen.append(word)
            if word == end:
                break
        return sen
    
    def draw(self):
        rand = random.random()
        for word in list(self.counts.keys()):
            rand -= self.prob(word)
            if rand <= 0.0:
                return word

    def getSentenceProbability(self, sen):
        prob = 1.0
        for word in sen[1:]:  # skip the start token
            prob *= self.prob(word)
        return prob
    
    def getCorpusPerplexity(self, corpus):
        log_prob = 0.0
        word_count = 0
        for sen in corpus:
            word_count += len(sen) - 1  # exclude the start token
            prob = self.getSentenceProbability(sen)
            log_prob += math.log(prob)
        return math.exp(-log_prob / word_count)

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def __init__(self, corpus):
        self.bigrams = defaultdict(lambda: defaultdict(float))
        self.unigrams = defaultdict(float)
        self.train(corpus)
    
    def train(self, corpus):
        for sen in corpus:
            for i in range(len(sen) - 1):
                self.unigrams[sen[i]] += 1.0
                self.bigrams[sen[i]][sen[i + 1]] += 1.0
    
    def prob(self, prev_word, word):
        if self.unigrams[prev_word] == 0:
            return 0
        return self.bigrams[prev_word][word] / self.unigrams[prev_word]
    
    def generateSentence(self):
        sen = [start]
        while True:
            word = self.draw(sen[-1])
            sen.append(word)
            if word == end:
                break
        return sen
    
    def draw(self, prev_word):
        rand = random.random()
        for word in list(self.bigrams[prev_word].keys()):
            rand -= self.prob(prev_word, word)
            if rand <= 0.0:
                return word

    def getSentenceProbability(self, sen):
        prob = 1.0
        for i in range(len(sen) - 1):
            prob *= self.prob(sen[i], sen[i + 1])
            if prob == 0:
                prob = 1e-10  # to avoid log(0)
        return prob
    
    def getCorpusPerplexity(self, corpus):
        log_prob = 0.0
        word_count = 0
        for sen in corpus:
            word_count += len(sen) - 1
            prob = self.getSentenceProbability(sen)
            log_prob += math.log(prob)
        return math.exp(-log_prob / word_count)

# Smoothed bigram language model (use linear interpolation for smoothing, set lambda1 = lambda2 = 0.5)
class SmoothedBigramModel(LanguageModel):
    def __init__(self, corpus):
        self.bigrams = defaultdict(lambda: defaultdict(float))
        self.unigrams = defaultdict(float)
        self.total_unigrams = 0.0
        self.train(corpus)
    
    def train(self, corpus):
        for sen in corpus:
            for i in range(len(sen) - 1):
                self.unigrams[sen[i]] += 1.0
                self.bigrams[sen[i]][sen[i + 1]] += 1.0
                self.total_unigrams += 1.0
    
    def prob(self, prev_word, word):
        bigram_prob = self.bigrams[prev_word][word] / self.unigrams[prev_word] if self.unigrams[prev_word] > 0 else 0.0
        unigram_prob = self.unigrams[word] / self.total_unigrams
        return 0.5 * bigram_prob + 0.5 * unigram_prob
    
    def generateSentence(self):
        sen = [start]
        while True:
            word = self.draw(sen[-1])
            sen.append(word)
            if word == end:
                break
        return sen
    
    def draw(self, prev_word):
        rand = random.random()
        for word in list(self.unigrams.keys()):
            rand -= self.prob(prev_word, word)
            if rand <= 0.0:
                return word

    def getSentenceProbability(self, sen):
        prob = 1.0
        for i in range(len(sen) - 1):
            prob *= self.prob(sen[i], sen[i + 1])
        return prob
    
    def getCorpusPerplexity(self, corpus):
        log_prob = 0.0
        word_count = 0
        for sen in corpus:
            word_count += len(sen) - 1
            prob = self.getSentenceProbability(sen)
            log_prob += math.log(prob)
        return math.exp(-log_prob / word_count)

# Sample class for an unsmoothed unigram probability distribution
class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)

    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0

    def prob(self, word):
        return self.counts[word] / self.total

    def draw(self):
        rand = random.random()
        for word in list(self.counts.keys()):
            rand -= self.prob(word)
            if rand <= 0.0:
                return word

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')
    
    vocab = set()
    for sen in trainCorpus:
        for word in sen:
            vocab.add(word)
    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)

    # Unsmoothed Unigram Model
    unigramModel = UnigramModel(trainCorpus)
    unigramModel.generateSentencesToFile(20, 'unigram_output.txt')
    print("Unigram Model Perplexity (positive test):", unigramModel.getCorpusPerplexity(posTestCorpus))
    print("Unigram Model Perplexity (negative test):", unigramModel.getCorpusPerplexity(negTestCorpus))

    # Smoothed Unigram Model
    smoothedUnigramModel = SmoothedUnigramModel(trainCorpus)
    smoothedUnigramModel.generateSentencesToFile(20, 'smooth_unigram_output.txt')
    print("Smoothed Unigram Model Perplexity (positive test):", smoothedUnigramModel.getCorpusPerplexity(posTestCorpus))
    print("Smoothed Unigram Model Perplexity (negative test):", smoothedUnigramModel.getCorpusPerplexity(negTestCorpus))

    # Unsmoothed Bigram Model
    bigramModel = BigramModel(trainCorpus)
    bigramModel.generateSentencesToFile(20, 'bigram_output.txt')
    print("Bigram Model Perplexity (positive test):", bigramModel.getCorpusPerplexity(posTestCorpus))
    print("Bigram Model Perplexity (negative test):", bigramModel.getCorpusPerplexity(negTestCorpus))

    # Smoothed Bigram Model with Linear Interpolation
    smoothedBigramModelLI = SmoothedBigramModel(trainCorpus)
    smoothedBigramModelLI.generateSentencesToFile(20, 'smooth_bigram_li_output.txt')
    print("Smoothed Bigram Model LI Perplexity (positive test):", smoothedBigramModelLI.getCorpusPerplexity(posTestCorpus))
    print("Smoothed Bigram Model LI Perplexity (negative test):", smoothedBigramModelLI.getCorpusPerplexity(negTestCorpus))