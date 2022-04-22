import numpy as np

# TODO(Colton): Code here to import csv file and store it as list of dictionaires
# Meals = [{"name": "Pizza", "time": 5}, {}, {}]

def CompileMealList():

	numCols = 20
	skipRows = 2

	# simpler titles
	titles = ["name",
			  "tags",
			  "time",
			  "cal",
			  "cost",
			  "carbs",
			  "protein",
			  "fat",
			  "sugar",
			  "satfat",
			  "fiber",
			  "sodium",
			  "vita",
			  "vitc",
			  "vitd",
			  "vite",
			  "vitb12",
			  "calcium",
			  "iron",
			  "potassium"]

	# skip rows 1 and 2
	texts = np.loadtxt(open("meal.tsv", "rb"), dtype='str', delimiter="\t", usecols = (0,1), skiprows=skipRows)
	nums = np.loadtxt(open("meal.tsv", "rb"), delimiter="\t", usecols=(range(2,numCols)), skiprows=skipRows)  # use this when sheet full

	meals = []
	for text,num in zip(texts,nums):
		entry = {}

		# First 2 columns are strings
		entry["name"] = text[0]
		entry["tags"] = text[1].replace(" ", "").split(',')

		# The rest is numeric data
		for i in range(2,numCols):
			entry[titles[i]] = num[i-2]

		meals.append(entry)

	return meals

# Main
meals = CompileMealList()
print(meals)

