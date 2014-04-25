import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

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

	return (X,Y, topic_to_index)

def test_topic_distribution(doc_topic_weights_filename, annotated_data_filename, k, test_indice, train_prop, column_of_interest):
	(X, Y, topic_to_index) = process_dataset(doc_topic_weights_filename, annotated_data_filename, k, column_of_interest)
	num_train = int(X.shape[0])

	# We repeat the experiments and report the average
	scores = []

	# Divide the set into train and test sets
	X_train = X[:num_train]
	Y_train = Y[:num_train]

	test_start_index = test_indice[0]
	test_end_index = test_indice[1]
	X_test = X[test_start_index:test_end_index]
	Y_test = Y[test_start_index:test_end_index]

	# Build a classifier
	clf = LogisticRegression().fit(X_train, Y_train)

	# Make prediction
	predicted_labels = clf.predict(X_test)

	# Report the accuracy
	true_labels = Y_test
	score = f1_score(predicted_labels, true_labels)	
	print "---------------- %s --------------------"  % str(test_indice)
	print classification_report(predicted_labels, true_labels)	
	print topic_to_index
	print "-----------------------------------------" 
	return score

if __name__ == "__main__":

	m4_doc_topic_weights_filename = './M4/DocTopicWeights.csv'
	block_doc_topic_weights_filename = './BlockHMM/DocTopicWeights.csv'
	lda_doc_topic_weights_filename = './LDA/DocTopicWeights.csv'

	annotated_data_filename = './AnnotatedData_NoContent.csv'
	train_prop = 0.8

	doc_topic_weights_filenames = [block_doc_topic_weights_filename ]
	test_indices = [(0,47), (48, 97), (98, 171), (172, 204), (205, 296), (297, 372), (373, 458), (459, 590), (591, 762), (763, 819), (820, 875), (876, 964)]

	print "=============== Experiment : Heteroglossia ANNOTATION IDENTIFICATION ==============="

	# Test for each dataset and for different k's
	f1_scores_for_each_dataset = []
	k = 1

	for doc_topic_weights_filename in doc_topic_weights_filenames:
		
		f1_scores = []		
		for test_indice in test_indices:
			print "Calculating the results for the dataset %s with test_indice = %s ... " % (doc_topic_weights_filename, str(test_indice))
			score = test_topic_distribution(doc_topic_weights_filename, annotated_data_filename, k, test_indice, train_prop, HET)
			f1_scores.append(score)
		f1_scores_for_each_dataset.append( f1_scores )

	print f1_scores_for_each_dataset


