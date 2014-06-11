# Graph test
# Download: http://graph-tool.skewed.de/download

from graph_tool.all import *
import numpy as np

users = ['Shane', 'Moon', 'Seungwhan']
num_users = len(users)
matrix = np.array( [[0,0,1], [0,1,1], [1,0,0]] )

g = Graph()
vertices = []

# Set up vertices
for i in range(num_users):
	name = users[i]
	v = g.add_vertex()
	v.text = name

	# Append to the list
	vertices.append( v )

# Draw edges
for i in range(num_users):
	for j in range(num_users):
		if matrix[i,j] > 0:
			g.add_edge( vertices[j], vertices[i] )

# Draw the graph
graph_draw(g, vertex_text=g.vertex_text, vertex_font_size=18, output_size=(200, 200), output="graph_test.png")