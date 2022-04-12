import meals
import numpy as np

#TODO: How to format this list, so DNC is used
in_val = [None] * len(meals.Meals)
#x1 = in_val[0] = number of sesame bagels
#x2 = in_val[1] = number of protein shakes
# ... etc

def cost(in_val):
    return sum([x * meal.Meals[i]["cost"] for i, x in enumerate(in_val)])

def time(in_val):
    return sum([x * meal.Meals[i]["time"] for i, x in enumerate(in_val)])

def entropy(in_val):
    # TODO(Oren Bell): return variety of diet
    return None

def objective_fn(in_val, a, b, c):
    return a*cost(in_val) + b*time(in_val) - c*entropy(in_val)


# Define constraints as predicates.
# At least 2500 cal
# No more than $800

def sodium_limit(in_list):
    return sum([x * meal.Meals[i]["sodium"] for i, x in enumerate(in_val)]) < 3000

# Main function
# Inputs: l, meals.Meals, and constrains
# TODO: Branch and bound here, with that relaxation thing whatchamacallit
# TODO: Assess predicates, and if any are false, trim that branch of searching

# TODO: Some pseudocode 
