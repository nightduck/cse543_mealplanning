# Define constraints as predicates
# https://www.fda.gov/food/new-nutrition-facts-label/daily-value-new-nutrition-and-supplement-facts-labels
# 5% DV or less of a nutrient per serving is considered low.
# 20% DV or more of a nutrient per serving is considered high.

def prep_time_constraint(in_list):  # in minutes
    return 0 < sum([x * meal.Meals[i]["prep_time"] for i, x in enumerate(in_list)]) < 60


def calories_constraint(in_list):  # in cal
    return 1500 < sum([x * meal.Meals[i]["calories"] for i, x in enumerate(in_list)]) < 2500


def cost_constraint(in_list):  # in $
    return 0 < sum([x * meal.Meals[i]["cost"] for i, x in enumerate(in_list)]) < 10


def carbs_constraint(in_list):  # in grams
    return 261.25 < sum([x * meal.Meals[i]["carbs"] for i, x in enumerate(in_list)]) < 330


def protein_constraint(in_list):  # in grams
    return 47.5 < sum([x * meal.Meals[i]["protein"] for i, x in enumerate(in_list)]) < 60


def fat_constraint(in_list):  # in grams
    return 74.1 < sum([x * meal.Meals[i]["fat"] for i, x in enumerate(in_list)]) < 93.6


def sugar_constraint(in_list):  # in grams
    return 47.5 < sum([x * meal.Meals[i]["sugar"] for i, x in enumerate(in_list)]) < 60


def saturated_fat_constraint(in_list):  # in grams
    return 19 < sum([x * meal.Meals[i]["saturated_fat"] for i, x in enumerate(in_list)]) < 24


def fiber_constraint(in_list):  # in grams
    return 26.6 < sum([x * meal.Meals[i]["fiber"] for i, x in enumerate(in_list)]) < 33.6


def sodium_constraint(in_list):  # in mg
    return 2185 < sum([x * meal.Meals[i]["sodium"] for i, x in enumerate(in_list)]) < 2760


def vitamin_a_constraint(in_list):  # in IU
    return 2850 < sum([x * meal.Meals[i]["vitamin_a"] for i, x in enumerate(in_list)]) < 3600


def vitamin_c_constraint(in_list):  # in mg
    return 85.5 < sum([x * meal.Meals[i]["vitamin_c"] for i, x in enumerate(in_list)]) < 108


def vitamin_d_constraint(in_list):  # in IU
    return 760 < sum([x * meal.Meals[i]["vitamin_d"] for i, x in enumerate(in_list)]) < 960


def vitamin_e_constraint(in_list):  # in IU
    return 21.28 < sum([x * meal.Meals[i]["vitamin_e"] for i, x in enumerate(in_list)]) < 26.88


def vitamin_b12_constraint(in_list):  # in mcg
    return 2.28 < sum([x * meal.Meals[i]["vitamin_b12"] for i, x in enumerate(in_list)]) < 2.88


def calcium_constraint(in_list):  # in mg
    return 1235 < sum([x * meal.Meals[i]["calcium"] for i, x in enumerate(in_list)]) < 1560


def iron_constraint(in_list):  # in mg
    return 17.1 < sum([x * meal.Meals[i]["iron"] for i, x in enumerate(in_list)]) < 21.6


def potassium_constraint(in_list):  # in mg
    return 4465 < sum([x * meal.Meals[i]["potassium"] for i, x in enumerate(in_list)]) < 5640


# Bundle them all up at the end
Constraints = [sodium_constraint, cost_constraint]