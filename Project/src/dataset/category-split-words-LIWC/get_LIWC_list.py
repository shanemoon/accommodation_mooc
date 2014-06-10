import string
d = {}

with open("cat.txt") as f:
    for line1 in f:
        (key, val) = line1.split()
        d[int(key)] = val

cat_to_words = {}

# Quantifier, Conjunction, Indef. Pronoun, Adverb, Article, Aux. Verb, Pers. Pron., Preposition
for category in [20,18,9,16,10,12,3,17]:
	with open("words.txt") as f1:
		for line in f1:
			line_splited = (line.replace("\r\n","").split("\t"))
			catList = line_splited[1:]
			word = line_splited[0].strip(string.punctuation)

			if str(category) in (cat for i, cat in enumerate(catList)):
				if d[category] in cat_to_words:
					cat_to_words[d[category]].append( word )
				else:
					cat_to_words[d[category]] = [word]
print cat_to_words.values()