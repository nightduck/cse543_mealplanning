import meals
import constraints
import math
import heapq

in_val = [(0, 31)] * len(meals.Meals)
#x1 = in_val[0] = number of sesame bagels (as a range of two numbers)
#x2 = in_val[1] = number of protein shakes
# ... etc

def cost(in_val):
    return sum([n * meal.Meals[i]["cost"] for i, n in enumerate(in_val)])

def time(in_val):
    return sum([n * meal.Meals[i]["time"] for i, n in enumerate(in_val)])

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

def objective_fn(in_val, a, b, c):
    return a*cost(in_val) + b*time(in_val) - c*entropy(in_val)

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
        else if is_integer(soln) and score < best_value:
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

