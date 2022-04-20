from scipy.optimize import Bounds, LinearConstraint, minimize
import numpy as np
import optimize
from functools import partial

#print(optimize.tag_index)
# print(optimize.coefs.shape, ": ")
# print(optimize.coefs)

rng = np.random.default_rng()
in_val = rng.integers(0, 3, size=31)

# print(in_val)
# print(optimize.tag_occurance(in_val))
# print(optimize.tag_index)

# print(optimize.entropy(in_val))
# print(optimize.entropy_der(in_val))
# print(optimize.entropy_hes(in_val))
# print(optimize.entropy_hes(in_val).shape)

bounds = Bounds([0]*len(optimize.meals.Meals), [np.inf]*len(optimize.meals.Meals))

fat = [m["fat"] for m in optimize.meals.Meals]
linear_constraint = LinearConstraint(fat, [400], [900])

grad = optimize.entropy_der(in_val)

a = 2
b = 1
c = 1

res = minimize(partial(optimize.objective_fn, a, b, c), [0] * 31, method="trust-constr",
        jac=partial(optimize.obj_der, a, b, c), hess=partial(optimize.obj_hes, a, b, c),
        constraints=[linear_constraint], options={'verbose': 1},
        bounds=bounds)

print(res.x)