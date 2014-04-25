import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

# Column of Interest
NEG = 12
HET = 13

def process_dataset(doc_topic_weights_filename, annotated_data_filename, k, column_of_interest):
	"""
		Process the dataset from the given files,
		and return a featurized array of X and the normalized array of Y.
		k is the size of the window to concatenate turns
	"""

	topic_distributions = np.loadtxt(open(doc_topic_weights_filename,"rb"), delimiter=",",skiprows=1)[:,1:]
	annotations = np.loadtxt(open(annotated_data_filename, "rb"), delimiter=",",skiprows=1, dtype='string')[:,column_of_interest]

	last_index = topic_distributions.shape[0]
	num_topics = topic_distributions.shape[1]

	X = np.ndarray((last_index, num_topics * k))
	Y = np.zeros(last_index)

	topic_to_index = {}
	topic_index = 0

	for i in range(last_index - k):
		target_index = i + k / 2

		x = topic_distributions[i:i+k].reshape(-1)
		annotation = annotations[target_index]

		# Set i'th element of X and Y
		X[i] = x		
		if annotation in topic_to_index:
			Y[i] = topic_to_index[annotation]
		else:
			topic_to_index[annotation] = topic_index
			Y[i] = topic_index
			topic_index += 1

	return (X,Y)

def test_topic_distribution(doc_topic_weights_filename, annotated_data_filename, k, train_prop, num_repeat, column_of_interest):
	(X, Y) = process_dataset(doc_topic_weights_filename, annotated_data_filename, k, column_of_interest)
	num_train = int(X.shape[0] * train_prop)

	# We repeat the experiments and report the average
	scores = []
	for i in range(num_repeat):
		print "Iteration: %d" % i
		rng = np.random.RandomState(i)
		indices = np.arange(len(X))
		rng.shuffle( indices )

		# Divide the set into train and test sets
		X_train = X[indices[:num_train]]
		Y_train = Y[indices[:num_train]]

		X_test = X[indices[num_train+1:]]
		Y_test = Y[indices[num_train+1:]]

		# Build a classifier
		clf = LogisticRegression().fit(X_train, Y_train)

		# Make prediction
		predicted_labels = clf.predict(X_test)

		# Report the accuracy
		true_labels = Y_test
		score = f1_score(predicted_labels, true_labels)	
		scores.append( score )

	return sum(scores) / len(scores)

if __name__ == "__main__":

	m4_doc_topic_weights_filename = './M4/DocTopicWeights.csv'
	block_doc_topic_weights_filename = './BlockHMM/DocTopicWeights.csv'
	lda_doc_topic_weights_filename = './LDA/DocTopicWeights.csv'

	annotated_data_filename = './AnnotatedData_NoContent.csv'
	train_prop = 0.8

	doc_topic_weights_filenames = [lda_doc_topic_weights_filename, block_doc_topic_weights_filename, m4_doc_topic_weights_filename ]
	num_repeat = 10


	(EXPERIMENT1_2, EXPERIMENT1_1) = (1,1)

	if EXPERIMENT1_2:
	
		print "=============== Experiment 1.1: NEGOTATION ANNOTATION IDENTIFICATION ==============="
		# Test for each dataset and for different k's
		ks = [1, 3, 6, 9, 12]
		f1_scores_for_each_dataset = []
		for doc_topic_weights_filename in doc_topic_weights_filenames:
			
			f1_scores = []		
			for k in ks:
				print "Calculating the results for the dataset %s with k = %d ... " % (doc_topic_weights_filename, k)
				score = test_topic_distribution(doc_topic_weights_filename, annotated_data_filename, k, train_prop, num_repeat, NEG)
				f1_scores.append(score)
			f1_scores_for_each_dataset.append( f1_scores )

		print f1_scores_for_each_dataset



	if EXPERIMENT1_1:
		print "=============== Experiment 1.2: Heteroglossia ANNOTATION IDENTIFICATION ==============="

		# Test for each dataset and for different k's
		f1_scores_for_each_dataset = []
		ks = [1, 3, 6, 9, 12, 15]

		for doc_topic_weights_filename in doc_topic_weights_filenames:
			
			f1_scores = []		
			for k in ks:
				print "Calculating the results for the dataset %s with k = %d ... " % (doc_topic_weights_filename, k)
				score = test_topic_distribution(doc_topic_weights_filename, annotated_data_filename, k, train_prop, num_repeat, HET)
				f1_scores.append(score)
			f1_scores_for_each_dataset.append( f1_scores )

		print f1_scores_for_each_dataset


