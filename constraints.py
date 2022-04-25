"""
Define nutrition constraints using scipy linear constraints
https://ods.od.nih.gov/factsheets/list-all/
https://www.hsph.harvard.edu/nutritionsource/carbohydrates/
https://www.hsph.harvard.edu/nutritionsource/salt-and-sodium/
As we considering weekly value, lower bound and upper bound are multiplied by 7.
For macros, carbs: 45–65% of total calories, fats: 20–35% of total calories, proteins: 10–35% of total calories
"""

import numpy as np
import meals
from scipy.optimize import LinearConstraint


def define_constraint(nutrient: str, lower_bound=0.0, upper_bound=np.inf):
    # check if nutrient is in each meal
    for m in meals.Meals:
        if nutrient not in m.keys():
            m[nutrient] = 0

    constraint = [m[nutrient] for m in meals.Meals]
    return LinearConstraint(constraint, [lower_bound], [upper_bound])


def get_nutrition_constraints():
    cal_constraint = define_constraint("cal", lower_bound=10500, upper_bound=np.inf)                # in cal
    sugar_constraint = define_constraint("sugar", lower_bound=0, upper_bound=252)                   # in g
    satfat_constraint = define_constraint("satfat", lower_bound=0, upper_bound=200)                 # in g
    fiber_constraint = define_constraint("fiber", lower_bound=105, upper_bound=np.inf)              # in g
    sodium_constraint = define_constraint("sodium", lower_bound=10500, upper_bound=np.inf)          # in mg
    vita_constraint = define_constraint("vita", lower_bound=21000, upper_bound=70000)               # in IU
    vitc_constraint = define_constraint("vitc", lower_bound=630, upper_bound=14000)                 # in mg
    vitd_constraint = define_constraint("vitd", lower_bound=4200, upper_bound=28000)                # in IU
    vite_constraint = define_constraint("vite", lower_bound=150, upper_bound=10500)                 # in IU
    vitb12_constraint = define_constraint("vitb12", lower_bound=16, upper_bound=np.inf)             # in mcg
    calcium_constraint = define_constraint("calcium", lower_bound=7000, upper_bound=17500)          # in mg
    iron_constraint = define_constraint("iron", lower_bound=56, upper_bound=315)                    # in mg
    potassium_constraint = define_constraint("potassium", lower_bound=23800, upper_bound=np.inf)    # in mg

    linear_constraints = [cal_constraint, sugar_constraint,
                          satfat_constraint, fiber_constraint, sodium_constraint, vita_constraint, vitc_constraint,
                          vitd_constraint, vite_constraint, vitb12_constraint, calcium_constraint, iron_constraint,
                          potassium_constraint]
    return linear_constraints


def get_macro_constraints(carb=0.60, fat=0.3, protein=0.10, tolerance=0.05):
    carb_cal = np.array([(m["carbs"] * 4) for m in meals.Meals])
    fat_cal = np.array([(m["fat"] * 9) for m in meals.Meals])
    protein_cal = np.array([(m["protein"] * 4) for m in meals.Meals])

    cal = np.array([(m["cal"]) for m in meals.Meals])

    carb_l = (carb - tolerance) * cal
    carb_u = (carb + tolerance) * cal
    fat_l = (fat - tolerance) * cal
    fat_u = (fat + tolerance) * cal
    protein_l = (protein - tolerance) * cal
    protein_u = (protein + tolerance) * cal

    arr = np.ndarray((6,len(meals.Meals)))
    arr[0,:] = carb_cal - carb_l
    arr[1,:] = carb_u - carb_cal
    arr[2,:] = fat_cal - fat_l
    arr[3,:] = fat_u - fat_cal
    arr[4,:] = protein_cal - protein_l
    arr[5,:] = protein_u - protein_cal

    return LinearConstraint(arr, [0]*6, [np.inf]*6)


Constraints = get_nutrition_constraints() + [get_macro_constraints()]

if __name__ == "__main__":
    print(Constraints)
