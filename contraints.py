# Define constraints as predicates.
# At least 2500 cal
# No more than $800
# Etc

def sodium_limit(in_list):
    return sum([x * meal.Meals[i]["sodium"] for i, x in enumerate(in_val)]) < 3000


# Bundle them all up at the end
Constraints = [sodium_limit, budget_limit]