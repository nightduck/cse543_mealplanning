import meals
import constraints
import numpy as np

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


# Main function
# Inputs: in_val, meals.Meals, and constraints.Constraints
# TODO: Branch and bound here, with that relaxation thing whatchamacallit
# TODO: Assess constraints, and if any are false, trim that branch of searching

# TODO: Some pseudocode for the general algorithm
