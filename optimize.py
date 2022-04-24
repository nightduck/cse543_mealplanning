import meals
import constraints
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


def pretty_print_solution(arr, integer=True):
    for i, v in enumerate(arr):
        name = meals.Meals[i]["name"]
        if integer and v >= 0.5:
            print(f"{name} : {int(v)}")
        elif not integer and round(v, 3)>0:
            print(f"{name} : {round(v, 3)}")            

# TODO(Oren): Nonlinear optimizer, given optimization fn, list of equality constraints, and list
# of inequality constraints, return an optimal solution. This meants constraints can't be
# expressed in the form of predicates
def relaxed_optimization(a, b, c, constraints, bounds):    
    res = minimize(partial(objective_fn, a, b, c), [8] * len(meals.Meals), method="trust-constr",
            jac=partial(obj_der, a, b, c), hess=partial(obj_hes, a, b, c),
            constraints=constraints, bounds=bounds, options={'verbose': 0})

    return (objective_fn(a, b, c, res.x), res.x, res.constr_violation)

# Main function
# Inputs: in_val, meals.Meals, and constraints.Constraints
# TODO: Branch and bound here, with that relaxation thing whatchamacallit
# branch: create a set of children, which are just represented as an additional constraint, setting
# a certain meal to be used a certain number of times.
def branch(bounds, split_value, index):
    # first will be <= 0 if x[index] <= split_value, second will be <= 0 if x[index] > split_value
    # TODO: rather than always add additional inequality bounds, check list of bounds to add equality bound as needed
    # also, check for any conflicting bounds
    right_lower = np.copy(bounds.lb)
    right_upper = np.copy(bounds.ub)
    # set upper bound of left branch to floor of found value
    bounds.ub[index] = math.floor(split_value)
    # set lower bound of right branch to ceiling of found value
    right_lower[index] = math.ceil(split_value)
    if right_lower[index] >= right_upper[index]:
        right_lower[index] = right_upper[index]
    if bounds.lb[index] >= bounds.ub[index]:
        bounds.lb[index] = bounds.ub[index]
         # special case
    # elif bounds.lb[index] >= bounds.ub[index]:
        # also special case
    # make two bounds constraints: old_lower <= variable < floor, ceil < variable <= old_upper
    # (lower branch = old constraint modified in-place, upper branch = new constraint)
    return [bounds, Bounds(right_lower, right_upper)]

# return true iff solution is entirely integers
def is_integer(solution):
    return all([abs(round(x)-x) <= 5e-3 for x in solution])

# return first index of a non-integer value
def find_noninteger(solution):
    for i in range(len(solution)):
        if abs(round(solution[i])-solution[i]) > 1e-3:
            return i
    return -1

def ex_opt(x):
    return -5*x[0]-8*x[1]

def ex_opt_scorer(a,b,c,x):
    return ex_opt(x)

def ex_jac(x):
    return np.array([-5,-8], dtype=np.float64)

def ex_hess(x):
    return np.array([[0,0],[0,0]], dtype=np.float64)

def ex_test_opt(a,b,c,cons,bds):
    res = minimize(ex_opt, [0] * 2, method="trust-constr",
            jac=ex_jac, hess=ex_hess,
            constraints=cons, bounds=bds)

    return (ex_opt(res.x), res.x, res.constr_violation)

def to_int(x):
    return [round(a) for a in x]
# run branch and bound
def branch_and_bound(relaxed_method, obj_fn, a, b, c, constraints, bounds):
    # initial "minimum" value -> infinity (beaten by any valid solution)
    best_value = np.inf
    best_solution = None
    # i is just some metadata about how many nodes were explored
    i = 1
    # base constraints for the root of the tree
    # as we branch, we add additional constraints (e.g. num bagels > 3, num pasta < 4, etc)
    # dummy score to start off with
    bb_heap = [(0,0, bounds)]
    # while some branches have yet to be explored
    while len(bb_heap) > 0:
        _, _, bds = heapq.heappop(bb_heap)
        score, solution, violation = relaxed_method(a,b,c,constraints,bds)
        # either unsatisfiable constraints, or even the best available score here is worse than the best we've found so far
        if violation > 1e-3 or score > best_value:
            pass
        # solution is integer-valued (bottom of branch)
        elif is_integer(solution):
            # if new best value, or if none previously existed, update scores and solution
            intsol = to_int(solution)
            intscore = obj_fn(a,b,c,intsol)
            if best_value is None or intscore < best_value:
                best_solution = intsol
                best_score = intscore
        # need to branch , choose unconstrained index to branch on
        else:
            # find index to branch on, create new inequality constraints
            index = find_noninteger(solution)
            new_children = branch(bds, solution[index], index)
            # create new node with updated constraints
            # min-heap sorted by score of solution, tie-broken by index
            for child in new_children:
                heapq.heappush(bb_heap, (score, i, child))
                i=i+1
    # if it returns None, it never found an integer solution. Hopefully shouldn't happen.
    print(f"Total nodes explored: {i}")
    return best_score, best_solution


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
    bounds = Bounds([0,5,4], [32,5,8])
    # Equality constraints here are modelled as two opposing inequality constraints
    
    # Tri will provide an array of these LinearConstraint's
    cals = [m["cal"] for m in meals.Meals]
    linear_constraints = [LinearConstraint(cals, [17500], [np.inf])]

    # Redefine bounds for the whole input space
    lower_bounds = [0]*len(meals.Meals)
    upper_bounds = [16]*len(meals.Meals)
    lower_bounds[3] = 4     # Artifically constrain this variable
    upper_bounds[3] = 8
    bounds = Bounds(lower_bounds, upper_bounds)

    # We specify the objective fn, the jacobian, the hessian, the 3 preference coefficients,
    # Tri's linear constraints, and your bounds
    a = 5
    b = 1
    c = 0.1
    value, solution, violation = relaxed_optimization(a, b, c,
        linear_constraints, bounds)

    print("Relaxed solution of %f at about" % (value))
    pretty_print_solution(solution, integer=False)

def example_call_to_branch_and_bound():
    constraint_matrix = np.array([[5,9],[1,1]], dtype=np.float64)
    lower = np.array([0,0], dtype=np.float64)
    upper = np.array([45,6], dtype=np.float64)
    constraints = LinearConstraint(constraint_matrix, lower, upper)
    bounds = Bounds([0,0], [6,6])
    a = 1
    b = 1
    c = 1
    value, solution = branch_and_bound(ex_test_opt, ex_opt_scorer, a,b,c,constraints, bounds)
    print(f"Best b-b solution is {solution}, with score {value}.")


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

example_call_to_branch_and_bound()