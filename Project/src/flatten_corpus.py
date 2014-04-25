from influence_propagation import structure_data
import csv

filename = "dataset/cleanEnglish.csv"
corpus_filename = "dataset/corpus.txt"

entire_text = ""
with open(filename, 'rb') as f:
	reader = csv.reader(f)
	# Skip the first row (column names)
	reader.next()

	# Iterate through each row
	for row in reader:
		# These are the only elements that we are interested in
		post_text = row[7]
		entire_text += post_text + '\n'


f = open(corpus_filename, 'w')
f.write( entire_text )