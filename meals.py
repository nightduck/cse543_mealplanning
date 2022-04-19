import numpy as np

# TODO(Colton): Code here to import csv file and store it as list of dictionaires
# Meals = [{"name": "Pizza", "time": 5}, {}, {}]

def CompileMealList():
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
	texts = np.loadtxt(open("meal.tsv", "rb"), dtype='str', delimiter="\t", usecols = (0,1))
	# print(text)
	# nums = np.loadtxt(open("meal.tsv", "rb"), delimiter="\t", usecols=(2,19)) use this when sheet full
	nums = np.loadtxt(open("meal.tsv", "rb"), delimiter="\t", usecols=(range(2,13)))

	meals = []
	for text,num in zip(texts,nums):
		#print(t,n)
		entry = {}

		entry["name"] = text[0]
		entry["tags"] = text[1].replace(" ", "").split(',')
		for i in range(2,13):
			entry[titles[i]] = num[i-2]

		meals.append(entry)

	#print(meals)
	return meals

# Main
Meals = CompileMealList()
