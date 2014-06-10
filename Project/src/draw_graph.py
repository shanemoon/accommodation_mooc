#!/usr/bin/env python
import numpy as np
from pylab import *

def show_bars(groups, groups_means, xtick_labels, y_label, title, output_filename, groups_std=[]):
	"""
		groups: ['group1', 'group2']
		groups_means: [[5, 6, 7], [3, 4, 5]]
		groups_std: [[0.1,0.1,0.1], [0.1,0.2,0.1]]
	"""
	N = len(groups_means[0])
	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars

	colors = ['r', 'b', 'k', 'g']

	groups_rects = []
	fig, ax = plt.subplots()

	for i in range(len(groups)):
		means = groups_means[i]
		#stds = groups_std[i]
	
		rects = ax.bar(ind+width*i, means, width, color=colors[i % len(colors)]) #, yerr=menStd
		groups_rects.append( rects )

	# add some
	ax.set_ylabel( y_label )
	ax.set_title(title)
	ax.set_xticks(ind+width)
	ax.set_xticklabels( xtick_labels )
	ax.legend( [rect[0] for rect in groups_rects], groups )

	def autolabel(rects):
	    # attach some text labels
	    for rect in rects:
	        height = rect.get_height()
	        #ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height), ha='center', va='bottom')

	for rects in groups_rects:
		autolabel(rects)

	savefig(output_filename)
	clf()

show_bars(['Men', 'Women'], [(20, 35, 30, 35, 27), (25, 32, 34, 20, 25)], ('G1', 'G2', 'G3', 'G4', 'G5'), 'Scores', 'Scores by group and gender', 'test.eps')