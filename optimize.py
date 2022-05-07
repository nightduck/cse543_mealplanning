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
    # TODO(Oren Bell): This is a sparse matrix operation, so there's a faster way to do it
    # Multiple 1xn vector by nxm matrix to get 1xm vector
    occurances = np.matmul(in_val, coefs)
    total = np.sum(occurances)
    occurances = occurances / total
    
    return occurances

def entropy(in_val):
    #in_val = np.reshape(input, (1,36))
    cal_sum = np.matmul(cals, in_val)
    if cal_sum == 0:    # Special case taking the limit
        return math.log(1 / len(in_val))
    occurances = np.matmul(in_val, coefs) / cal_sum

    variance = 0
    for p in occurances:
        if p > 0:
            variance -= p * math.log(p)

    return variance

def entropy_der(in_val):
    cal_sum = np.matmul(cals, in_val)

    gradient = np.zeros(len(in_val))
    for i, x in enumerate(in_val):
        lnx = math.log(x) if x > 0 else -1e82
        b_i = coefs[i,:] / cal_sum
        dx = - sum(b_i * lnx + b_i)
        gradient[i] = dx

    #assert(sum(gradient * [m["cal"] * n for m, n in zip(meals.Meals, in_val)]) == 0)
    return gradient

def entropy_hes(in_val):
    cal_sum = np.matmul(cals, in_val)
    hessian = np.identity(len(in_val))

    for i, v in enumerate(in_val):
        v = v if v > 0 else 1e-9
        hessian[i,i] = - sum(coefs[i,:]) / (v * cal_sum)

    return hessian

def objective_fn(a, b, c, in_val):
    return np.multiply(a, cost(in_val)) + np.multiply(b, time(in_val)) - np.multiply(c, entropy(in_val))

def obj_der(a, b, c, in_val):
    return np.multiply(a, cost_der(in_val)) + np.multiply(b, time_der(in_val)) - np.multiply(c, entropy_der(in_val))

def obj_hes(a, b, c, in_val):
    #return np.zeros((len(in_val), len(in_val)))
    return - c*entropy_hes(in_val)


def pretty_print_solution(arr, integer=True):
    if any([a == np.inf or a == None for a in arr]):
        print("No solution exists")
        return

    cost = 0
    time = 0
    for i, v in enumerate(arr):
        name = meals.Meals[i]["name"]
        if integer and v >= 0.5:
            print(f"{name} : {int(v)}")
            cost += int(v) * meals.Meals[i]["cost"]
            time += int(v) * meals.Meals[i]["time"]
        elif not integer and round(v, 3)>0:
            print(f"{name} : {round(v, 3)}")
            cost += round(v, 3) * meals.Meals[i]["cost"]
            time += round(v, 3) * meals.Meals[i]["time"]

    print("Time:", round(time), "min")
    print("Cost: $", round(cost, 2))

# Nonlinear optimizer, given optimization fn, list of equality constraints, and list
# of inequality constraints, return an optimal solution. This meants constraints can't be
# expressed in the form of predicates
def relaxed_optimization(a, b, c, constraints, bounds, primer=[1]*len(meals.Meals)):
    res = minimize(partial(objective_fn, a, b, c), primer, method="SLSQP",
            jac=partial(obj_der, a, b, c), #hess=partial(obj_hes, a, b, c),
            constraints=constraints, bounds=bounds, options={'maxiter': 100000})

    solution = np.around(res.x, 3)
    return (objective_fn(a, b, c, solution), solution, 0 if res.success else 0)

def multiplier_method(a, b, c, constraints, bounds, primer=[1]*len(meals.Meals)):
    solution = primer
    # scaling_factor = 1
    # gradient = obj_der(a, b, c, solution)

    # while scaling_factor > 0.001:
    #     norm = gradient / np.linalg.norm(gradient) * scaling_factor

    #     solution -= norm

    #     new_gradient = obj_der(a,b,c,solution)
    #     old_unit = gradient / np.linalg.norm(gradient)
    #     new_unit = new_gradient / np.linalg.norm(new_gradient)
    #     if np.arccos(np.dot(new_unit, old_unit)) > np.pi / 2:
    #         scaling_factor *= 0.9
    #     gradient = new_gradient

    lambdas = np.zeros(2*sum([len(con.lb) for con in constraints]) + 2*len(meals.Meals))
    constraint_matrix = np.zeros((lambdas.shape[0], len(meals.Meals)))
    constraint_constants = np.zeros(lambdas.shape[0])
    ck = 1
    scaling_factor = 1

    i = 0
    for con in constraints:
        constraint_matrix[i:i+len(con.lb),:] = con.A
        constraint_constants[i:i+len(con.lb)] = con.ub
        i += len(con.lb)

        constraint_matrix[i:i+len(con.lb),:] = np.multiply(con.A, -1)
        constraint_constants[i:i+len(con.lb)] = np.multiply(con.lb, -1)
        i += len(con.lb)
    
    constraint_matrix[i:i+len(bounds.ub),:] = np.identity(len(bounds.ub))
    constraint_constants[i:i+len(bounds.ub)] = bounds.ub
    i += len(bounds.ub)
    constraint_matrix[i:i+len(bounds.lb),:] = -1 * np.identity(len(bounds.lb))
    constraint_constants[i:i+len(bounds.lb)] = np.multiply(bounds.lb, -1)
    i += len(bounds.lb)

    # Update the lambdas via gradient descent (their derivative is just their constraint function)
    violation = 0
    for i, con in enumerate(constraint_matrix):
        g = np.matmul(con, solution) - constraint_constants[i]
        if g > 0:       # Constraint is active. Update lambda
            lambdas[i] += g
            violation += g
        else:           # Constraint in inactive, disable lambda
            lambdas[i] = 0
    old_violation = violation

    gradient = obj_der(a,b,c,solution) + np.matmul(lambdas, constraint_matrix)

    solution_exists = False
    while scaling_factor > 1e-4 or (violation > 1e-3 and solution_exists):
        norm = gradient / np.linalg.norm(gradient) * scaling_factor

        solution -= norm

        # Update the lambdas via the multiplier method
        violation = 0
        for i, con in enumerate(constraint_matrix):
            g = np.matmul(con, solution) - constraint_constants[i]
            if g > 0:       # Constraint is active. Update lambda
                lambdas[i] += g
                lambdas[i] = max(lambdas[i], 1e128)
                violation += g
            else:           # Constraint in inactive, disable lambda
                lambdas[i] = 0

        # if (old_violation == 0 or violation / old_violation > 0.75) and ck < 1e128:
        #     ck *= 1.5
        old_violation = violation
        if violation == 0:
            solution_exists = True

        new_gradient = obj_der(a,b,c,solution) + np.matmul(lambdas, constraint_matrix)

        old_unit = gradient / np.linalg.norm(gradient)
        new_unit = new_gradient / np.linalg.norm(new_gradient)
        if np.arccos(np.clip(np.dot(new_unit, old_unit),-1,1)) > np.pi / 2:
            scaling_factor *= 0.9 + (np.tanh(np.sqrt(violation)) / 10)  # Small violation means the change is 0.9,
                                                                # large violation means the change is 1 (same)
        gradient = new_gradient

    return objective_fn(a,b,c, solution), solution, violation

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
    # Return the one with the narrowest search space last, so it get processed first
    if right_upper[index] - split_value > split_value - bounds.lb[index]:
        return [Bounds(right_lower, right_upper), bounds]
    else:
        return [bounds, Bounds(right_lower, right_upper)]

# Work as above except it always splits the bounds in half, rather that using the solutions as
# a recommended split point. This avoids the tendency to run in O(n) time when chasing constraints
def log_branch(bounds, index):
    split_value = (bounds.ub[index] - bounds.lb[index]) / 2 + bounds.lb[index]
    branch(bounds, split_value, index)

def convert_constraints_to_dict(con):
    def ineq_fn(A, b, x):
        # Turn linear constraints from matrices to functions
        return np.matmul(A, x) - b

    # Should output scalars, so address any multi-row constraints
    con_upper = [{"type": "ineq",
                "fun": partial(ineq_fn, np.multiply(c.A, -1), c.ub[0] * -1),
                "jac" : lambda x : np.multiply(c.A, -1)}
                for c in con if c.ub[0] != np.inf]
    con_lower = [{"type": "ineq", "fun": partial(ineq_fn, c.A, c.lb[0]), "jac" : lambda x : c.A} for c in con]

    return con_upper + con_lower

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
    #constraints = convert_constraints_to_dict(constraints)

    # initial "minimum" value -> infinity (beaten by any valid solution)
    best_score = np.inf
    best_solution = [np.inf] * len(meals.Meals)
    # i is an index to track previously calculated solutions, which can be used as initial
    # conditions to accelerate the computation of their children
    i = 1
    precomp_solutions = [[1]*len(meals.Meals)]
    # base constraints for the root of the tree
    # as we branch, we add additional constraints (e.g. num bagels > 3, num pasta < 4, etc)
    # dummy score to start off with
    bb_heap = [(0, 0, bounds)]
    # while some branches have yet to be explored
    while len(bb_heap) > 0:
        _, j, bds = bb_heap.pop()
        primer = precomp_solutions[j]

        score, solution, violation = relaxed_method(a,b,c,constraints,bds,primer=primer)
        #print("%f at %s" % (score, str(solution)))
        # either unsatisfiable constraints, or even the best available score here is worse than the best we've found so far
        if violation > 1e-3 or score > best_score:
            pass
        # solution is integer-valued (bottom of branch)
        elif is_integer(solution):
            # if new best value, or if none previously existed, update scores and solution
            intsol = to_int(solution)
            intscore = obj_fn(a,b,c,intsol)
            if intscore < best_score:
                best_solution = intsol
                best_score = intscore
        # need to branch , choose unconstrained index to branch on
        else:
            precomp_solutions.append(solution)

            # find index to branch on, create new inequality constraints
            index = find_noninteger(solution)
            new_children = branch(bds, solution[index], index)
            # create new node with updated constraints
            # min-heap sorted by score of solution, tie-broken by index
            for child in new_children:
                bb_heap.append((score, i, child))
                j += 1
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

coefs = np.zeros((len(meals.Meals), len(tag_index)), dtype=np.float64)

for i, meal in enumerate(meals.Meals):
    for t in meal["tags"]:
        j = tag_index[t]
        coefs[i, j] += meal["cal"] / len(meal["tags"])

cals = np.array([m["cal"] for m in meals.Meals])

if __name__ == "__main__":
    # example_call_to_relaxed_optimize()
    # example_call_to_branch_and_bound()
    # a = np.ndarray(36)
    # a.fill(2)
    # entropy_der(a)
    _, result, _ = multiplier_method(0, 1, 0, [constraints.define_constraint("cal", lower_bound=10500, upper_bound=np.inf)],
            Bounds([0]*len(meals.Meals), [6]*len(meals.Meals)))
    pretty_print_solution(result, integer=False)