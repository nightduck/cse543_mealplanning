from optimize import *
import datetime

bounds = Bounds([0]*len(meals.Meals), [128]*len(meals.Meals))


# No constraints
print("Under no constraints")
start = datetime.datetime.now()
a = 1
b = 0.25
c = 1
con = []
value, solution = branch_and_bound(relaxed_optimization, objective_fn, a, b, c, con, bounds)
pretty_print_solution(solution)
print(datetime.datetime.now() - start)
print()

# Only calorie constraint
print("Under calorie constraints")
start = datetime.datetime.now()
con.append(constraints.define_constraint("cal", lower_bound=10500, upper_bound=np.inf))
value, solution = branch_and_bound(relaxed_optimization, objective_fn, a, b, c, con, bounds)
pretty_print_solution(solution)
print(datetime.datetime.now() - start)
print()


# Add satfat and sugar constraints
print("Under sugar and saturated fat constraints")
start = datetime.datetime.now()
con.append(constraints.define_constraint("sugar", lower_bound=0, upper_bound=420))
con.append(constraints.define_constraint("satfat", lower_bound=0, upper_bound=168))
value, solution = branch_and_bound(relaxed_optimization, objective_fn, a, b, c, con, bounds)
pretty_print_solution(solution)
print(datetime.datetime.now() - start)
print()

# TODO: Add macro nutrients

# Add vitamin a, c, and d
print("Under Vitamin A, C, and D constraints")
start = datetime.datetime.now()
con.append(constraints.define_constraint("vita", lower_bound=21000, upper_bound=140000))
con.append(constraints.define_constraint("vitc", lower_bound=630, upper_bound=np.inf))
con.append(constraints.define_constraint("vitd", lower_bound=4200, upper_bound=28000))
value, solution = branch_and_bound(relaxed_optimization, objective_fn, a, b, c, con, bounds)
pretty_print_solution(solution)
print(datetime.datetime.now() - start)
print()

# Add fiber and sodium constraints
print("Under fiber and sodium constraints")
start = datetime.datetime.now()
con.append(constraints.define_constraint("fiber", lower_bound=147, upper_bound=266))
con.append(constraints.define_constraint("sodium", lower_bound=3500, upper_bound=19320))
value, solution = branch_and_bound(relaxed_optimization, objective_fn, a, b, c, con, bounds)
pretty_print_solution(solution)
print(datetime.datetime.now() - start)
print()

# Add all other vitamins and minerals
print("Under all other constraints")
start = datetime.datetime.now()
con.append(constraints.define_constraint("vite", lower_bound=150, upper_bound=10500))
con.append(constraints.define_constraint("vitb12", lower_bound=16, upper_bound=np.inf))
con.append(constraints.define_constraint("calcium", lower_bound=7000, upper_bound=17500))
con.append(constraints.define_constraint("iron", lower_bound=126, upper_bound=315))
con.append(constraints.define_constraint("potassium", lower_bound=23800, upper_bound=np.inf))
value, solution = branch_and_bound(relaxed_optimization, objective_fn, a, b, c, con, bounds)
pretty_print_solution(solution)
print(datetime.datetime.now() - start)
print()

# TODO: Body building diet

# TODO: Time is valuable diet