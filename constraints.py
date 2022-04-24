"""
Define nutrition constraints using scipy linear constraints
https://www.fda.gov/food/new-nutrition-facts-label/daily-value-new-nutrition-and-supplement-facts-labels
5% DV or less of a nutrient per serving is considered low.
20% DV or more of a nutrient per serving is considered high.
As we considering weekly value, lower bound and upper bound are multiplied by 7.
For macros, carbs: 45–65% of total calories, fats: 20–35% of total calories, proteins: 10–35% of total calories
"""

import numpy as np
import meals
from scipy.optimize import LinearConstraint
from functools import reduce


def define_macro_constraint(nutrient: str, lower_percentage=0.45, upper_percentage=0.65):
    # check if nutrient is in each meal
    for m in meals.Meals:
        if nutrient not in m.keys():
            m[nutrient] = 0

    # e.g. for carbs, 45% cal <= carbs <= 65% cal
    # convert to grams to cal
    macro_constraint = [(m[nutrient] * 7.71618) for m in meals.Meals]
    lower_macro_constraint = [(m['cal'] * lower_percentage) for m in meals.Meals]
    upper_macro_constraint = [(m['cal'] * upper_percentage) for m in meals.Meals]

    # 45% cal <= carbs ( or 0 <= -45% cal + carbs)
    macro_constraint_1 = [(-lower_macro_constraint[i] + macro_constraint[i]) for i in range(len(meals.Meals))]

    # carbs <= 65% cal ( or 0 <= -carbs + 65% cal)
    macro_constraint_2 = [(-macro_constraint[i] + upper_macro_constraint[i]) for i in range(len(meals.Meals))]

    final_macro_constraint = [LinearConstraint(macro_constraint_1, [0.0], [np.inf]),
                              LinearConstraint(macro_constraint_2, [0.0], [np.inf])]
    return final_macro_constraint


def define_constraint(nutrient: str, lower_bound=0.0, upper_bound=np.inf):
    # check if nutrient is in each meal
    for m in meals.Meals:
        if nutrient not in m.keys():
            m[nutrient] = 0

    constraint = [m[nutrient] for m in meals.Meals]
    return LinearConstraint(constraint, [lower_bound], [upper_bound])


def get_nutrition_constraints():
    cal_constraint = define_constraint("cal", lower_bound=10500, upper_bound=np.inf)                # in cal
    carbs_constraint = define_constraint("carbs", lower_bound=1828.75, upper_bound=2310)            # in g
    protein_constraint = define_constraint("protein", lower_bound=332.5, upper_bound=420)           # in g
    fat_constraint = define_constraint("fat", lower_bound=518.7, upper_bound=655.2)                 # in g
    sugar_constraint = define_constraint("sugar", lower_bound=332.5, upper_bound=420)               # in g
    satfat_constraint = define_constraint("satfat", lower_bound=133, upper_bound=168)               # in g
    fiber_constraint = define_constraint("fiber", lower_bound=186.2, upper_bound=235.2)             # in g
    sodium_constraint = define_constraint("sodium", lower_bound=15295, upper_bound=19320)           # in mg
    vita_constraint = define_constraint("vita", lower_bound=19950, upper_bound=25200)               # in IU
    vitc_constraint = define_constraint("vitc", lower_bound=598.5, upper_bound=756)                 # in mg
    vitd_constraint = define_constraint("vitd", lower_bound=5320, upper_bound=6720)                 # in IU
    vite_constraint = define_constraint("vite", lower_bound=148.96, upper_bound=188.16)             # in IU
    vitb12_constraint = define_constraint("vitb12", lower_bound=15.96, upper_bound=20.16)           # in mcg
    calcium_constraint = define_constraint("calcium", lower_bound=8645, upper_bound=10920)          # in mg
    iron_constraint = define_constraint("iron", lower_bound=119.7, upper_bound=151.2)               # in mg
    potassium_constraint = define_constraint("potassium", lower_bound=31255, upper_bound=39480)     # in mg

    linear_constraints = [cal_constraint, carbs_constraint, protein_constraint, fat_constraint, sugar_constraint,
                          satfat_constraint, fiber_constraint, sodium_constraint, vita_constraint, vitc_constraint,
                          vitd_constraint, vite_constraint, vitb12_constraint, calcium_constraint, iron_constraint,
                          potassium_constraint]
    return linear_constraints


def get_macro_constraints():
    carbs_macro_constraint = define_macro_constraint("carbs", lower_percentage=0.45, upper_percentage=0.65)
    fat_macro_constraint = define_macro_constraint("fat", lower_percentage=0.25, upper_percentage=0.35)
    protein_macro_constraint = define_macro_constraint("protein", lower_percentage=0.10, upper_percentage=0.35)
    macro_constraints = [carbs_macro_constraint, fat_macro_constraint, protein_macro_constraint]
    macro_constraints = reduce(lambda x, y: x+y, macro_constraints)
    return macro_constraints


Constraints = get_nutrition_constraints() + get_macro_constraints()

if __name__ == "__main__":
    print(Constraints)
