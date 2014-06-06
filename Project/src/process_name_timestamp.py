# We process the data so that we can have more insight into how users flow in and out 
# from the online course community.

import numpy as np
import matplotlib.pyplot as plt


f = open("./dataset/name_timestamp_only.csv","rb")

author_name_to_id = {}
ANONYMOUS = ["Anonymous"]
INSTRUCTORS = ["Paul Gries", "Jen Campbell"]
TAS = ["Jen Lee", "ETIENNE PAPEGNIES", "Nitish Mittal", "Kevin Eugensson"]
STAFFS = ["jonathan lung"]
BLACKKLIST = []
BLACKKLIST.extend(ANONYMOUS)
BLACKKLIST.extend(INSTRUCTORS)
BLACKKLIST.extend(TAS)
BLACKKLIST.extend(STAFFS)

list_post_authors = []
list_post_author_ids = []
list_post_timestamps = []
start_timestamp = 1376875130

last_author_index = -1
num_good_data = 0
for line in f:	
	# Go through each line
	data = line.split(',')

	if len(data) == 2:
		# Sanity check that the data composes of two columns. 
		# Some of the data are corrupted, missing values, etc.
		author_name = data[0]
		timestamp_str = data[1]

		if author_name != "" and author_name not in BLACKKLIST:
			try:
				timestamp = int(timestamp_str)
				if timestamp > start_timestamp:
					# Finally, this should be good data
					if author_name not in author_name_to_id:
						last_author_index += 1
						author_name_to_id[author_name] = last_author_index

					list_post_author_ids.append( author_name_to_id[author_name] )
					list_post_authors.append( author_name )
					list_post_timestamps.append( timestamp -  start_timestamp)
			except:
				# If timestamp cannot be casted as int, it's a trash datapoint
				pass

num_good_data = len(list_post_timestamps)
num_authors = last_author_index

num_cohorts = 10
num_timesteps = 15
groups = np.zeros((num_cohorts, num_timesteps))

size_timestep = max(list_post_timestamps) / num_timesteps
print max(list_post_timestamps)
size_cohort = num_authors / num_cohorts

# Now go through the list again, and categorize them into cohorts and timesteps
i = 0
for i in range(num_good_data):
	author_id = list_post_author_ids[i]
	timestamp = list_post_timestamps[i]

	# Author cohort
	author_cohort = author_id / size_cohort
	if author_cohort == num_cohorts:
		# We just put the rest of the authors in the last cohort
		author_cohort -= 1
	elif author_cohort > num_cohorts:
		print "something doesn't seem right"
	else:
		pass

	# Time group
	time_group = timestamp / size_timestep
	if time_group == num_timesteps:
		time_group -= 1
	elif time_group > timestamp / size_timestep:
		print "something doesn't seem right"		
	else:
		pass
	groups[author_cohort][time_group] = groups[author_cohort][time_group] + 1


# Now we visualize it
ind = np.linspace(0, 1, num=num_timesteps, endpoint=False)
width = 1.0 / num_timesteps
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#6699FF', '#CC6600', 'w']

top = np.zeros(num_timesteps)
for i in range(num_timesteps):
	top[i] = sum(groups[:, i])
total = top

for i in range(num_cohorts):
	p = plt.bar(ind, top / total, width, color=colors[i % len(colors)])
	top = top - groups[i,:]


plt.title('%% of posts by %d authors grouped into %d sequential cohorts' % (num_authors, num_cohorts))
plt.xlabel('Time (Normalized)')
plt.ylabel("% of posts by each cohort")
plt.show()

# Save the author names in the order of their first appearance
open("./dataset/author_names_in_order","w").write(str(list_post_authors))
