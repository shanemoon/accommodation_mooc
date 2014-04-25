import numpy as np
import ast

def get_top_ranked_words_for_each_topic(word_weights_filename, n):

	num_topics = 5
	words = []
	topic_weights = []

	index = 0
	for line in open(word_weights_filename):
		split_line = line.split(',')
		try:
			topic_weight = [ast.literal_eval(e.strip()) for e in split_line[-num_topics:]]
			topic_weights.append( topic_weight )

			word = ''.join(split_line[0:-num_topics])
			words.append( word )
		except:
			pass

	words = np.array( words )
	topic_weights = np.array( topic_weights )
	
	top_ranked_words_for_each_topic = []

	for topic in range(num_topics):
		weights_per_topic = topic_weights[:,topic]
		top_indices = np.argsort(weights_per_topic)[-n:]
		top_ranked_words_for_each_topic.append( words[top_indices] )

	return top_ranked_words_for_each_topic

if __name__ == "__main__":

	m4_topic_word_weights = './M4/TopicWordWeights.csv'
	block_topic_word_weights = './BlockHMM/TopicWordWeights.csv'
	lda_topic_word_weights = './LDA/TopicWordWeights.csv'


	topic_word_weights_filenames = [block_topic_word_weights, m4_topic_word_weights ]

	n = 15

	for word_weights_filename in topic_word_weights_filenames:
		top_ranked_words_for_each_topic = get_top_ranked_words_for_each_topic(word_weights_filename, n)

		print "model: %s" % word_weights_filename

		# Printing in LaTex form
		for i in range(n):
			line = ""
			for top_ranked_words_per_topic in top_ranked_words_for_each_topic:
				line += top_ranked_words_per_topic[i] + ' & '
			line = line[:-2] + " \\\\"
			print line

