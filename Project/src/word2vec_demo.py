import word2vec
import os

path_dataset = os.path.abspath('dataset/text8')
path_clusters = os.path.abspath('dataset/text8.clusters')

word2vec.word2clusters(path_dataset, path_clusters, 100)
clusters = word2vec.load_clusters(path_clusters)

print clusters