import meals
import constraints
import math

in_val = [(0, 31)] * len(meals.Meals)
#x1 = in_val[0] = number of sesame bagels (as a range of two numbers)
#x2 = in_val[1] = number of protein shakes
# ... etc

def cost(in_val):
    return sum([n * meal.Meals[i]["cost"] for i, n in enumerate(in_val)])

def cost_der(in_val):
    return [meal.Meals[i]["cost"] for i, n in enumerate(in_val)]

def time(in_val):
    return sum([n * meal.Meals[i]["time"] for i, n in enumerate(in_val)])

def time_der(in_val):
    return [meal.Meals[i]["time"] for i, n in enumerate(in_val)]

def entropy(in_val):
    occurence = {}
    for i, n in enumerate(in_val):
        meal = meals.Meals[i]
        for t in meal["tags"]:
            if not t in occurence:
                occurence[t] = 0
            occurence[t] += n * meals.Meals[i]["calories"] / len(meal["tags"])

    variance = 0
    for v in occurence:
        variance -= v * math.log(v)

def entropy_der(in_val):
    return np.zeros(len(in_val))

def entropy_hes(in_val):
    return np.zeros((len(in_val), len(in_val)))

def objective_fn(in_val, a, b, c):
    return a*cost(in_val) + b*time(in_val) - c*entropy(in_val)

def obj_der(in_val, a, b, c):
    return a*cost_der(in_val) + b*time_der(in_val) - c*entropy_der(in_val)

def obj_hes(in_val, a, b, c):
    return - c*entropy_hes(in_val)

# TODO(Oren): Nonlinear optimizer, given optimization fn, list of equality constraints, and list
# of inequality constraints, return an optimal solution. This meants constraints can't be
# expressed in the form of predicates
def kkt(obj_fn, eq_constraints, ineq_constraints, num_inputs):
    # TODO(Oren): Need derivatives as inputs to compute lagrangian multipliers?
    solution = [0.5] * num_inputs
    return (obj_fn(solution), solution)

# Main function
# Inputs: in_val, meals.Meals, and constraints.Constraints
# TODO: Branch and bound here, with that relaxation thing whatchamacallit
# TODO: Assess constraints, and if any are false, trim that branch of searching

# TODO: Some pseudocode for the general algorithm
# Input: in_val is list of tuples, with each tuple representing a range of integer values that
#        particular input could be, eg [(8, 15), (0, 31), (0, 31)] means
#        8 <= x1 <= 15,  0 <= x2 <= 32,  0 <= x3 <= 32
#
# When performing a branch, take an arbitrary input, and cut its range in half, creating two
# new nodes in the branch and bound graph. So [(8, 15), (0, 31), (0, 31)] could become
# [(8, 15), (0, 15), (0, 31)]  and  [(8, 15), (16, 31), (0, 31)]
#
# When each tuple is two of the same numbers (eg 4 <= xi <= 4), then xi = 4
# Test for this case with (returns T/F):    all([x == y for x,y in in_val])
# Simplify list and remove tuples with:     new_list = [x for x,y in in_val]

