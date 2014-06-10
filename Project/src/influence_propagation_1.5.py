import numpy as np
import matplotlib.pyplot as plt
import csv
import random, string

def extract_conversations(b, a, posts):
	"""
		Extract a set of exchanges between b and a in a given post

		Our definition of a conversational exchange:
			- 'a' makes a post
			- After a while, in the same thread, 'b' makes a post (a reply)
			=> We define this as a conversational exchange between a and b
	"""
	a_utters = []
	b_replies = []

	# The initial target is 'a', because we assume that 'a' initiates the conversation
	target = a

	for post in posts:
		# Each post is a tuple of:
		(timestamp, author, text) = post
		text = text.lower()

		if author == target:
			# Append to the set of utterances,
			# and switch the target
			if target == a:
				if len(a_utters) == len(b_replies):
					a_utters.append( text )
				elif len(a_utters) > len(b_replies):
					a_utters[-1] = text
				target = b
			elif target == b:
				b_replies.append( text )
				target = a

	if target == b:
		# This means we were not able to find
		# b's reply to a's utterance.
		# Therefore, we remove a's last utterance from the set
		a_utters.pop()

	if len(a_utters) != len(b_replies):
		print len(a_utters), len(b_replies)

	return (b_replies, a_utters)

		

def crawl_conversation_set(b, a, data):
	"""
		We return the set S(b,a) of exchanges across the entire corpus 
		to determine the coordination of b towards a

		Specifically it returns:
			- E(u_b -> u_a) : a set of b's replies to a's utterances
			- E(u_a) : a set of a's utterances 
	"""

	# Read the input structured data
	(user_to_threads_ids, thread_id_to_posts) = data

	# List of threads that a and b participated in
	a_threads = user_to_threads_ids[a]
	b_threads = user_to_threads_ids[b]

	# We are interseted in the threads that both a and b particpated in together
	common_threads = set(a_threads) & set(b_threads)

	# We now crawl the conversational sets between b and a.
	a_utters = []
	b_replies = []

	for thread in common_threads:
		# Retreive all the posts in this thread
		posts_in_thread = thread_id_to_posts[thread]

		# Extract convesrational exchanges from this thread (a set of posts)
		(b_replies_in_thread, a_utters_in_thread) = extract_conversations(b, a, posts_in_thread)
		
		# Extend to the global list
		a_utters.extend( a_utters_in_thread )
		b_replies.extend( b_replies_in_thread )

	return (b_replies, a_utters)


def structure_data(filename):
	"""
		We struct the input data so that we can easily query for
		specific elements
	"""

	# These are what we are going to return
	user_to_threads_ids = {}
	thread_id_to_posts = {}

	with open(filename, 'rb') as f:
		reader = csv.reader(f)
		# Skip the first row (column names)
		reader.next()

		# Iterate through each row
		for row in reader:
			# These are the only elements that we are interested in
			(parent_id, post_author, post_id, post_text, timestamp, votes) = row[2], row[4], row[5], row[7], int(row[13]), int(row[16])
			parent_id = post_id if parent_id == '' else parent_id

			# If this author is new to us, we add this author to user_to_threads_ids
			# Then we add this thread_id to this dictionary
			if post_author not in user_to_threads_ids:
				user_to_threads_ids[post_author] = [ parent_id ]
			else:
				user_to_threads_ids[post_author].append( parent_id )

			# If this is a new therad, we add this to threads list
			# Then we add this post as a tuple to this dictionary
			if parent_id not in thread_id_to_posts:
				thread_id_to_posts[parent_id] = [ (timestamp, post_author, post_text) ]
			else:
				thread_id_to_posts[parent_id].append( (timestamp, post_author, post_text) )			

	# Return the sturcutred data in a tuple
	structured_data = (user_to_threads_ids, thread_id_to_posts)
	return structured_data


def measure_accommodation(b_replies, a_utters, markers):
	"""
		Measure accommodation of b towards a
		Markers are assumed to be from the same brown cluster (or exhibits high similarity)
		Return: (C_b_a)^m for all markers
	"""
	# C[marker] = accommodation_score
	# c = P( E(u_b -> u_a) | E(u_a) ) - P( E(u_b -> u_a) )

	C = {}
	E_b_given_a = {}    # Counting  E(u_b -> u_a) | E(u_a)
	E_b         = {}    # Counting  E(u_b -> u_a)
	E_a         = {}    # Counting  E(u_a)

	# I MAY NEED TO MODIFY HOW TO COUNT NUM_UTTERS
	num_utters = len(b_replies)

	for marker in markers:
		# Initialize
		E_b[marker] = 0
		E_a[marker] = 0
		E_b_given_a[marker] = 0

		for i in range(num_utters):
			a_utter = a_utters[i]
			b_reply = b_replies[i]

			if marker in b_reply:
				E_b[marker] = E_b.get(marker, 0) + 1

			if marker in a_utter:
				E_a[marker] = E_a.get(marker, 0) + 1

			if marker in a_utter and marker in b_reply:
				E_b_given_a[marker] = E_b_given_a.get(marker, 0) + 1

		# Smooth out the results
		C[marker] = float(E_b_given_a[marker]) / (E_a[marker]+1) - float(E_b[marker]) / (num_utters+1)

	# Aggregaited result
	C['aggregiated'] = float( sum( [ E_b_given_a[marker] for marker in markers] )) / \
						( sum( [ E_a[marker] for marker in markers] ) + 1 ) - \
						float(sum( [ E_b[marker] for marker in markers] )) / \
							(num_utters * len(markers) + 1)
	
	return (C, E_b_given_a, E_b, E_a)

def get_batch_accommodation(B, A, data, markers, num_E_b_given_a, num_E_a, num_E_b, num_utters, task_label):
	"""
		Return: 
	"""
	(E_b_given_a_set, E_b_set, E_a_set) = {}, {}, {}

	b_replies_set = []
	A_utters_set = []
	for b in B:
		for a in A:
			(b_replies, a_utters) = crawl_conversation_set(b, a, data)
			b_replies_set.extend( b_replies )
			A_utters_set.extend( a_utters )

	(C_b_a, E_b_given_a, E_b, E_a)  = measure_accommodation( b_replies_set, A_utters_set, markers)
	num_E_b_given_a[task_label] = num_E_b_given_a.get(task_label, 0) + float( sum( [ E_b_given_a[marker] for marker in markers] )) 
	num_E_a[task_label] = num_E_a.get(task_label, 0) + float( sum( [ E_a[marker] for marker in markers] )) 
	num_E_b[task_label] = num_E_b.get(task_label, 0) + float( sum( [ E_b[marker] for marker in markers] )) 		
	num_utters[task_label] = num_utters.get(task_label, 0) + len(b_replies_set) * len(markers)
	return (C_b_a, num_E_b_given_a, num_E_a, num_E_b, num_utters)


def get_frequent_clusters(filename):
	"""
		Load the cluster of words, and do frequency analysis for each cluster
	"""

	# Load the cluster: see word2vec for how I obtained a cluster of words
	# Cluster that we use have the following configuration:
	# -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -classes 2000
	word_to_cluster = {}
	cluster_to_words = {}
	cluster_to_freq = {}

	for line in open('./dataset/classes_2000.sorted.txt'):
		(word, cluster) = line.split()
		cluster_index = int(cluster)
		word_to_cluster[word] = cluster_index
		if cluster_index in cluster_to_words:
			cluster_to_words[cluster_index].append( word )
		else:
			cluster_to_words[cluster_index] = [ word ]
	
	# Read the .CSV file and measure the frequency of word that also appears on the cluster
	with open(filename, 'rb') as f:
		reader = csv.reader(f)
		# Skip the first row (column names)
		reader.next()

		# Iterate through each row
		i = 0
		for row in reader:
			# These are the only elements that we are interested in
			words_in_post = [word.lower().strip(string.punctuation) for word in row[7].split()]
			for word in words_in_post:
				if word in word_to_cluster:
					cluster = word_to_cluster[word]
					cluster_to_freq[cluster] = cluster_to_freq.get(cluster, 0) + 1

	# Sort the clusters by word frequency in the .csv dataset
	cluster_to_freq_sorted = []
	for index, freq in cluster_to_freq.items():
		cluster_to_freq_sorted.append( (freq, index) )
	cluster_to_freq_sorted.sort(reverse=True)

	# Return
	return	word_to_cluster, cluster_to_words, cluster_to_freq, cluster_to_freq_sorted

if __name__ == "__main__":
	# Run the main function
	filename = "dataset/cleanEnglish.csv"

	# Parse the original .csv file and get the structured data 
	data = structure_data(filename)

	# Define users, authors, our annotated leaders and non-leaders
	authors = data[0].keys()   # 3633 students
	leaders = ['Alain Rouleau', 'Chrissie Nyssen', 'John Routh', 'Alexander Falgui', 'Rae Bezer', 'Alan C. Orrick', 'Tom Enos', 'Ashwini Kempraj'] 
	non_leaders = ['Sandy Quaglieri', 'Obasi Eric Ojong', 'Ana', 'Barry Andersen', 'Anne Benissan', 'Artyom Vecherov', 'Bradley S. Walter']	

	INSTRUCTORS = ["Paul Gries", "Jen Campbell"]
	TAS = ["Jen Lee", "ETIENNE PAPEGNIES", "Nitish Mittal", "Kevin Eugensson"]
	STAFFS = ["jonathan lung"]

	users = list(set(authors) - set(leaders) - set('Anonymous') - set(INSTRUCTORS) - set(TAS) - set(STAFFS))


	# Define lexemes to measure accommodation on
	# The Echoes of Paper uses 451 lexemes total from the LWIC-derived 8 categories including articles, quantifiers, etc.
	# However, we take a different approach. We are interested in language accommodation for a group of words
	# that are semantically similar. Given a list of 'options' of words that a person could have chosen,
	# we are intereseted in finding out whether the language accommodation can affect this choice of words.

	# Obtain the word cluster and do frequency analysis to choose these lexemes. For now we use the most popular clusters
	(word_to_cluster, cluster_to_words, cluster_to_freq, cluster_to_freq_sorted) = get_frequent_clusters(filename)

	USE_HANDPICKED_LIST = 1
	if USE_HANDPICKED_LIST:
		selected_list = [1058, 707, 1171, 1928, 9, 1258, 619, 1966, 317, 79, 53, 1270, 1783, 1528, 825, 154, 591, 957, 100]
		most_frequent_clusters_and_words = [cluster_to_words[cluster] for cluster in selected_list]
	else:
		most_frequent_clusters_and_words = [cluster_to_words[pair[1]] for pair in cluster_to_freq_sorted[:20]]

	# Batch Test Run to measure accommodation for each cluster
	# We summarize the results in the following dictionaries of ( cluster_index, aggregiated_accommodation ) pairs
	target_leaders = {}
	target_nonleaders = {}
	speaker_leaders = {}
	speaker_nonleaders = {}

	num_E_b_given_a = {}
	num_E_a = {}
	num_E_b = {}
	num_utters = {}

	i = -1
	for markers in most_frequent_clusters_and_words:
		i += 1
		# 1. 
		# Target: Leaders (Group) = A
		# Speaker: Users (Individual) = B
		print "Target: Leaders"
		(C_b_a, num_E_b_given_a, num_E_a, num_E_b, num_utters)	= \
		get_batch_accommodation(users, leaders, data, markers, num_E_b_given_a, num_E_a, num_E_b, num_utters, 'target_leaders')
		target_leaders[ cluster_to_freq_sorted[i][1] ] = C_b_a['aggregiated']

		# 2. 
		# Target: Non-Leaders (Group) = A
		# Speaker: Users (Individual) = B
		print "Target: Non-Leaders"
		(E_b_given_a_set, E_b_set, E_a_set) = {}, {}, {}
		(C_b_a, num_E_b_given_a, num_E_a, num_E_b, num_utters)	= \
		get_batch_accommodation(users, non_leaders, data, markers, num_E_b_given_a, num_E_a, num_E_b, num_utters, 'target_nonleaders')
		target_nonleaders[ cluster_to_freq_sorted[i][1] ] = C_b_a['aggregiated']		

		# 3. 
		# Speaker: Leaders (Group)
		# Target: Users (Individual)
		print "Speaker: Leaders"
		(C_b_a, num_E_b_given_a, num_E_a, num_E_b, num_utters)	= \
		get_batch_accommodation(leaders, users, data, markers, num_E_b_given_a, num_E_a, num_E_b, num_utters, 'speaker_leaders')
		speaker_leaders[ cluster_to_freq_sorted[i][1] ] = C_b_a['aggregiated']		

		# 4. 
		# Speaker: Non-Leaders (Group)
		# Target: Users (Individual)
		print "Speaker: Non-Leaders"
		(C_b_a, num_E_b_given_a, num_E_a, num_E_b, num_utters)	= \
		get_batch_accommodation(non_leaders, users, data, markers, num_E_b_given_a, num_E_a, num_E_b, num_utters, 'speaker_nonleaders')
		speaker_nonleaders[ cluster_to_freq_sorted[i][1] ] = C_b_a['aggregiated']

target_leaders['aggregiated'] = num_E_b_given_a['target_leaders']  / (num_E_a['target_leaders'] + 1) - \
								num_E_b['target_leaders'] / (num_utters['target_leaders'] + 1)
target_nonleaders['aggregiated'] = num_E_b_given_a['target_nonleaders']  / (num_E_a['target_nonleaders'] + 1) - \
								num_E_b['target_nonleaders'] / (num_utters['target_nonleaders'] + 1)
speaker_leaders['aggregiated'] = num_E_b_given_a['speaker_leaders']  / (num_E_a['speaker_leaders'] + 1) - \
								num_E_b['speaker_leaders'] / (num_utters['speaker_leaders'] + 1)
speaker_nonleaders['aggregiated'] = num_E_b_given_a['speaker_nonleaders']  / (num_E_a['speaker_nonleaders'] + 1) - \
								num_E_b['speaker_nonleaders'] / (num_utters['speaker_nonleaders'] + 1)

print "========================================= Summary ========================================="
print "We expect that the accommodation of (1) target_leaders is higher than (2) target_nonleaders"
print target_leaders['aggregiated'], target_leaders
print target_nonleaders['aggregiated'], target_nonleaders

print "We expect that the accommodation of (1) speaker_leaders is smaller than (2) speaker_nonleaders"
print speaker_leaders['aggregiated'], speaker_leaders
print speaker_nonleaders['aggregiated'], speaker_nonleaders