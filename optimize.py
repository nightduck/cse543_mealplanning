import meals
import contraints
import math
import numpy as np
import heapq
from scipy.optimize import Bounds, LinearConstraint, minimize
from functools import partial


tag_index = {}
in_val_buffer = []
occurances_buffer = None

def cost(in_val):
    return sum([n * meals.Meals[i]["cost"] for i, n in enumerate(in_val)])

def cost_der(in_val):
    return [meals.Meals[i]["cost"] for i, n in enumerate(in_val)]

def time(in_val):
    return sum([n * meals.Meals[i]["time"] for i, n in enumerate(in_val)])

def time_der(in_val):
    return [meals.Meals[i]["time"] for i, n in enumerate(in_val)]

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
            try:
                variance -= p * math.log(p)
            except Exception as err:
                print(p)
                raise err

    return variance

def entropy_der(in_val):
    occurances = tag_occurance(in_val)
    total = np.sum(occurances)

    gradient = np.zeros(len(in_val))

    for i in range(len(in_val)):

        # Compute summatation at this entry in vector
        result = 0
        for j, p in enumerate(occurances):
            coef = coefs[i, j] / total
            if coef != 0:   # If coef is zero, then dp/dx is zero
                if p == 0:  # If p is zero anyways, then x is zero, and dp/dx is inf
                    assert(in_val[i] == 0)
                    result = math.inf
                else:
                    result -= coef + coef * math.log(p / total)
        gradient[i] = result
    
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
                coef_aj = coefs[a, j] / total
                coef_bj = coefs[b, j] / total
                if coef_aj != 0 and coef_bj != 0:
                    if p == 0:
                        result = -math.inf
                    else:
                        result -= coef_aj * coef_bj / p / total

            hessian[a,b] = result

    return hessian

def objective_fn(a, b, c, in_val):
    return np.multiply(a, cost(in_val)) + np.multiply(b, time(in_val)) #- np.multiply(c, entropy(in_val))

def obj_der(a, b, c, in_val):
    return np.multiply(a, cost_der(in_val)) + np.multiply(b, time_der(in_val)) #- np.multiply(c, entropy_der(in_val))

def obj_hes(a, b, c, in_val):
    return np.zeros((len(in_val), len(in_val)))
    #return - c*entropy_hes(in_val)

# TODO(Oren): Nonlinear optimizer, given optimization fn, list of equality constraints, and list
# of inequality constraints, return an optimal solution. This meants constraints can't be
# expressed in the form of predicates
def relaxed_optimization(a, b, c, constraints, bounds):    
    res = minimize(partial(objective_fn, a, b, c), [8] * 31, method="trust-constr",
            jac=partial(obj_der, a, b, c), hess=partial(obj_hes, a, b, c),
            constraints=constraints + [bounds], options={'verbose': 1})

    return (objective_fn(a, b, c, res.x), res.x)

# Main function
# Inputs: in_val, meals.Meals, and constraints.Constraints
# TODO: Branch and bound here, with that relaxation thing whatchamacallit
# branch: create a set of children, which are just represented as an additional constraint, setting
# a certain meal to be used a certain number of times.
def branch(split_value, index):
    # first will be <= 0 if x[index] <= split_value, second will be <= 0 if x[index] > split_value
    return [[lambda x: x[index]-split_value], [lambda x: split_value-x[index]]]

# return true iff solution is entirely integers
def is_integer(solution):
    return all([abs(round(x)-x) <= 1e-3 for x in solution])

# return first index of a non-integer value
def find_noninteger(solution):
    for i in range(len(solution)):
        if abs(round(solution[i])-solution[i]) > 1e-3:
            return i
    return -1

# run branch and bound
def branch_and_bound(obj_fn, eq_constraints, ineq_constraints, num_inputs):
    base_solution = kkt(obj_fn, eq_constraints, ineq_constraints, num_inputs)
    # initial "minimum" value -> infinity (beaten by any valid solution)
    best_value = math.inf
    best_solution = None
    # i is just some metadata about how many nodes were explored
    i = 1
    # base constraints for the root of the tree
    # as we branch, we add additional constraints (e.g. num bagels > 3, num pasta < 4, etc)
    bb_heap = [(eq_constraints, ineq_constraints)]
    # while some branches have yet to be explored
    while len(bb_heap) > 0:
        eq_cons, ineq_cons = heapq.heappop(bb_heap)
        soln = kkt(obj_fn, eq_cons, ineq_cons, num_inputs)
        score = obj_fn(soln)
        # even relaxed problem doesn't beat our best integer score so far
        if score > best_value:
            pass
        # solution is integer-valued and has beaten the best integer-valued score so far
        elif is_integer(soln) and score < best_value:
            best_value = score
            best_solution = soln
        # need to branch , choose unconstrained index to branch on
        else:
            # find index to branch on, create new inequality constraints
            index = find_noninteger(soln)
            new_children = branch(soln[index], index)
            # create new node with updated constraints
            for child in new_children:
                heapq.heappush(bb_heap, (eq_cons, child.extend(ineq_cons)))
                i=i+1
    # if it returns None, it never found an integer solution. Hopefully shouldn't happen.
    print("Total nodes explored: " + i)
    return best_solution


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

def example_call_to_relaxed_optimize():
    # If you want to bound a variable, add a constraint. Constraints are modelled as follows
    # lower <= a*x0 + b*x1 + ... z*x26 <= upper
    # but in matrix form
    # 
    # |0|    | 1 0 0 |    |32|
    # |5| <= | 0 1 0 | <= | 5|
    # |4|    | 0 0 1 |    | 8|
    #
    # The above means x0 is unbounded (between 0 and 32), x1 == 5, and x2 is between 4 and 8
    
    # This is expressed as a linear constraint. First argument is the identity matrix.
    # Second and third arguments are lower and uppers bounds on each variable
    bounds = LinearConstraint(np.identity(3), [0,5,4], [32,5,8])

    # Equality constraints here are modelled as two opposing inequality constraints
    
    # Tri will provide an array of these LinearConstraint's
    cals = [m["cal"] for m in meals.Meals]
    linear_constraints = [LinearConstraint(cals, [17500], [np.inf])]

    # Redefine bounds for the whole input space
    lower_bounds = [0]*len(meals.Meals)
    upper_bounds = [16]*len(meals.Meals)
    lower_bounds[3] = 4     # Artifically constrain this variable
    upper_bounds[3] = 8
    bounds = LinearConstraint(np.identity(len(meals.Meals)), lower_bounds, upper_bounds)

    # We specify the objective fn, the jacobian, the hessian, the 3 preference coefficients,
    # Tri's linear constraints, and your bounds
    a = 5
    b = 1
    c = 0.1
    value, solution = relaxed_optimization(a, b, c,
        linear_constraints, bounds)

    print("Relaxed solution of %f at %s" % (value, str(solution)))


counter = 0
for m in meals.Meals:
    for t in m["tags"]:
        if t not in tag_index:
            tag_index[t] = counter
            counter += 1

coefs = np.ndarray((len(meals.Meals), len(tag_index)), dtype=np.float64)

for i, meal in enumerate(meals.Meals):
    for t in meal["tags"]:
        j = tag_index[t]
        coefs[i, j] += meal["cal"] / len(meal["tags"])

example_call_to_relaxed_optimize()