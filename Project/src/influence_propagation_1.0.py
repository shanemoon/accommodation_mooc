import numpy as np
import matplotlib.pyplot as plt
import csv
import random

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
		Return: (C_b_a)^m for all markers
	"""
	# C[marker] = accommodation_score
	# c = P( E(u_b -> u_a) | E(u_a) ) - P( E(u_b -> u_a) )

	C = {}
	E_b_given_a = {}    # Counting  E(u_b -> u_a) | E(u_a)
	E_b         = {}    # Counting  E(u_b -> u_a)
	E_a         = {}    # Counting  E(u_a)
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
						float(sum( [ E_a[marker] for marker in markers] )) / \
							(num_utters * 3 + 1)
	
	return (C, E_b_given_a, E_b, E_a)

if __name__ == "__main__":
	# Run the main function
	filename = "cleanEnglish.csv"
	data = structure_data(filename)

	# Define lexemes to measure accommodation on
	# The Echoes of Paper uses 451 lexemes total from the LWIC-derived 8 categories including articles, quantifiers, etc.
	# Because we do not have access to these lexemes yet, we perform a mini accommodation test just on English articles (`a', `an', `the')
	markers = ['a', 'an', 'the']

	# Batch Test Run
	authors = data[0].keys()   # 3633 students
	leaders = ['Alain Rouleau', 'Chrissie Nyssen', 'John Routh', 'Alexander Falgui', 'Rae Bezer', 'Alan C. Orrick', 'Tom Enos', 'Ashwini Kempraj'] 
	non_leaders = ['Sandy Quaglieri', 'Obasi Eric Ojong', 'Ana', 'Barry Andersen', 'Anne Benissan', 'Artyom Vecherov', 'Bradley S. Walter']	

	INSTRUCTORS = ["Paul Gries", "Jen Campbell"]
	TAS = ["Jen Lee", "ETIENNE PAPEGNIES", "Nitish Mittal", "Kevin Eugensson"]
	STAFFS = ["jonathan lung"]

	users = list(set(authors) - set(leaders) - set(non_leaders) - set('Anonymous') - set(INSTRUCTORS) - set(TAS) - set(STAFFS))

	# Measure accommodation 
	
	# 1. 
	# Target: Leaders (Group)
	# Speaker: Users (Individual)
	print "Target: Leaders"
	leader_C_b_A = 0
	(E_b_given_a_set, E_b_set, E_a_set) = {}, {}, {}

	b_replies_set = []
	A_utters_set = []
	for b in users:
		for a in leaders:
			(b_replies, a_utters) = crawl_conversation_set(b, a, data)
			b_replies_set.extend( b_replies )
			A_utters_set.extend( a_utters )

	(C_b_a, E_b_given_a, E_b, E_a)  = measure_accommodation( b_replies_set, A_utters_set, markers)
	print C_b_a


	# 2. 
	# Target: Non-Leaders (Group)
	# Speaker: Users (Individual)
	print "Target: Non-Leaders"
	leader_C_b_A = 0
	(E_b_given_a_set, E_b_set, E_a_set) = {}, {}, {}

	b_replies_set = []
	A_utters_set = []
	for b in users:
		for a in non_leaders:
			(b_replies, a_utters) = crawl_conversation_set(b, a, data)
			b_replies_set.extend( b_replies )
			A_utters_set.extend( a_utters )

	(C_b_a, E_b_given_a, E_b, E_a)  = measure_accommodation( b_replies_set, A_utters_set, markers)
	print C_b_a	


	# 1. 
	# Speaker: Leaders (Group)
	# Target: Users (Individual)
	print "Speaker: Leaders"
	leader_C_b_A = 0
	(E_b_given_a_set, E_b_set, E_a_set) = {}, {}, {}

	b_replies_set = []
	A_utters_set = []
	for b in leaders:
		for a in users:
			(b_replies, a_utters) = crawl_conversation_set(b, a, data)
			b_replies_set.extend( b_replies )
			A_utters_set.extend( a_utters )

	(C_b_a, E_b_given_a, E_b, E_a)  = measure_accommodation( b_replies_set, A_utters_set, markers)
	print C_b_a


	# 2. 
	# Speaker: Non-Leaders (Group)
	# Target: Users (Individual)
	print "Speaker: Non-Leaders"
	leader_C_b_A = 0
	(E_b_given_a_set, E_b_set, E_a_set) = {}, {}, {}

	b_replies_set = []
	A_utters_set = []
	for b in non_leaders:
		for a in users:
			(b_replies, a_utters) = crawl_conversation_set(b, a, data)
			b_replies_set.extend( b_replies )
			A_utters_set.extend( a_utters )

	(C_b_a, E_b_given_a, E_b, E_a)  = measure_accommodation( b_replies_set, A_utters_set, markers)
	print C_b_a	

