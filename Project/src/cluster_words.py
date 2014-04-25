import os, sys
lib_path = os.path.abspath('./ref/word-clustering-master')
sys.path.append(lib_path)
import monoMain as cls
import pickle

def cluster(inputFileName, numClusInit):
	"""
		Input : inputFileName to Corpus (Text data - String)
		Output : List of clusters of words
	"""

	# Read the input file and get word counts
	wordDict, bigramDict = cls.readInputFile(inputFileName)

	lang = cls.Language(wordDict, bigramDict, numClusInit, 1)  
	cls.runOchClustering(lang)

	# Return the clusters
	clusters = {}
	for clus, wordList in lang.wordsInClusDict.iteritems():
		clusters[clus] = wordList

	return clusters

if __name__ == "__main__":
	# Run the main function
	
	input_filename = 'dataset/corpus.txt' # 101339 unique words
	output_filename = 'clusters.pkl'
	clusters = cluster(input_filename, 20)

	with open(output_filename, 'wb') as output:
		pickle.dump(clusters, output, pickle.HIGHEST_PROTOCOL)	
	
	"""
	d = eval(open('dataset/clusters.txt').read())
	num_words = 0
	for cluster_index, list_words in d.items():
		num_words += len(list_words)
	print num_words
	"""

#print cluster('./ref/word-clustering-master/example_input.txt', 5)
#{0: ['.', 'park', 'sentence', 'short'], 1: ['also', 'two', 'has', 'sentences', 'it', 'but'], 2: ['one', 'this', 'with', 'third', 'document', 'another', 'is', 'here', 'and', 'a'], 3: ['chased', 'program', 'by', 'only', 'was', 'cares', 'about', 'spaces'], 4: ['dog', 'in', 'the', 'cat', 'ran']}
