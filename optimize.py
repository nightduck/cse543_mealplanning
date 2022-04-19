import meals
import contraints
import math
import numpy as np


tag_index = {}
in_val_buffer = []
occurances_buffer = None

def cost(in_val):
    return sum([n * meal.Meals[i]["cost"] for i, n in enumerate(in_val)])

def cost_der(in_val):
    return [meal.Meals[i]["cost"] for i, n in enumerate(in_val)]

def time(in_val):
    return sum([n * meal.Meals[i]["time"] for i, n in enumerate(in_val)])

def time_der(in_val):
    return [meal.Meals[i]["time"] for i, n in enumerate(in_val)]

def tag_occurance(in_val):
    # Litle buffering to not repeat computation for function, derivative, and hessian
    if in_val == in_val_buffer:
        return occurances_buffer

    # TODO(Oren Bell): This is a sparse matrix operation, so there's a faster way to do it
    # Multiple 1xn vector by nxm matrix to get 1xm vector
    occurances = np.matmul(in_val, coefs)
    total = np.sum(occurances)
    occurances = occurances / total


    # occurances = np.ndarray(len(tag_index))

    # for i, n in enumerate(in_val):
    #     meal = meals.Meals[i]
    #     for t in meal["tags"]:
    #         j = tag_index[t]
    #         occurances[j] += n * coefs[i, j]
    
    occurances_buffer = occurances
    return occurances

def entropy(in_val):
    occurances = tag_occurance(in_val)

    variance = 0
    for p in occurances:
        if p != 0:
            variance -= p * math.log(p)

    return variance

def entropy_der(in_val):
    occurances = tag_occurance(in_val)
    total = np.sum(occurances)

    gradient = np.zeros(len(in_val))

    for i in range(len(in_val)):

        # Compute summatation at this entry in vector
        result = 0
        for j, p in enumerate(occurances):
            coef = coefs[i, j]
            if coef != 0:   # If coef is zero, then dp/dx is zero
                if p == 0:  # If p is zero anyways, then x is zero, and dp/dx is inf
                    assert(in_val[i] == 0)
                    result = math.inf
                else:
                    result -= coef + coef * math.log(p)
        gradient[i] = result / total
    
    return gradient

def entropy_hes(in_val):
    occurances = tag_occurance(in_val)
    total = np.sum(occurances)

    hessian = np.zeros((len(in_val), len(in_val)))

    for a, xa in enumerate(in_val):
        for b, xb in enumerate(in_val):

            # Compute sumation at this entry in matrix
            result = 0
            for j, p in enumerate(occurances):
                coef_aj = coefs[a, j]
                coef_bj = coefs[b, j]
                if coef_aj != 0 and coef_bj != 0:
                    if p == 0:
                        result = -math.inf
                    else:
                        result -= coef_aj * coef_bj / p

            hessian[a,b] = result

    return hessian / total

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


counter = 0
for m in meals.Meals:
    for t in m["tags"]:
        if t not in tag_index:
            tag_index[t] = counter
            counter += 1

coefs = np.ndarray((len(meals.Meals), len(tag_index)))

for i, meal in enumerate(meals.Meals):
    for t in meal["tags"]:
        j = tag_index[t]
        coefs[i, j] += meal["cal"] / len(meal["tags"])



#in_val = [(0, 31)] * len(meals.Meals)
#x1 = in_val[0] = number of sesame bagels (as a range of two numbers)
#x2 = in_val[1] = number of protein shakes
# ... etc