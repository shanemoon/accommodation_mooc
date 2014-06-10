d = {}
with open("cat.txt") as f:
    for line1 in f:
          (key, val) = line1.split()
          d[int(key)] = val
print len(d)
for key in d:
        print key, 'corresponds to', d[key]
        file = open(d[key], 'w+')
        with open("words.txt") as f1:
            for line in f1:
               wordList=(line.replace("\r\n","").split("\t"))[1:]
               word = str(key)
               print wordList
               print word
               if word in (w for i, w in enumerate(wordList)):
                  print "yes"
                  file.write("".join(line.split("\t")[:1])+"\n")

