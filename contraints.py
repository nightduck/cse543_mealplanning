# Define constraints as predicates
# https://www.fda.gov/food/new-nutrition-facts-label/daily-value-new-nutrition-and-supplement-facts-labels
# 5% DV or less of a nutrient per serving is considered low.
# 20% DV or more of a nutrient per serving is considered high.

def prep_time_constraint(in_list):  # in minutes
    return sum([x * meal.Meals[i]["time"] for i, x in enumerate(in_list)]) - 60


# keep total calories under 2500 per day (but no less than 1500)
def calories_constraint(in_list):  # in cal
    cal_sum = sum([x * meal.Meals[i]["cal"] for i, x in enumerate(in_list)])
    if cal_sum >= 1500:
        return cal_sum - 2500
    else:  # when cal_sum < 1500, cal_sum + 1500 always returns positive value which implies violated constraint
        return cal_sum + 1500


def cost_constraint(in_list):  # in $
    return sum([x * meal.Meals[i]["cost"] for i, x in enumerate(in_list)]) - 10


def carbs_constraint(in_list):  # in grams
    carbs_sum = sum([x * meal.Meals[i]["carbs"] for i, x in enumerate(in_list)])
    if carbs_sum >= 261.25:
        return carbs_sum - 330
    else:
        return carbs_sum + 261.25


def protein_constraint(in_list):  # in grams
    protein_sum = sum([x * meal.Meals[i]["protein"] for i, x in enumerate(in_list)])
    if protein_sum >= 47.5:
        return protein_sum - 60
    else:
        return protein_sum + 47.5


def fat_constraint(in_list):  # in grams
    fat_sum = sum([x * meal.Meals[i]["fat"] for i, x in enumerate(in_list)])
    if fat_sum >= 74.1:
        return fat_sum - 93.6
    else:
        return fat_sum + 74.1


def sugar_constraint(in_list):  # in grams
    sugar_sum = sum([x * meal.Meals[i]["sugar"] for i, x in enumerate(in_list)])
    if sugar_sum >= 47.5:
        return sugar_sum - 60
    else:
        return sugar_sum + 47.5


def saturated_fat_constraint(in_list):  # in grams
    satfat_sum = sum([x * meal.Meals[i]["satfat"] for i, x in enumerate(in_list)])
    if satfat_sum >= 19:
        return satfat_sum - 24
    else:
        return satfat_sum + 19


def fiber_constraint(in_list):  # in grams
    fiber_sum = sum([x * meal.Meals[i]["fiber"] for i, x in enumerate(in_list)])
    if fiber_sum >= 26.6:
        return fiber_sum - 33.6
    else:
        return fiber_sum + 26.6


def sodium_constraint(in_list):  # in mg
    sodium_sum = sum([x * meal.Meals[i]["sodium"] for i, x in enumerate(in_list)])
    if sodium_sum >= 2185:
        return sodium_sum - 2760
    else:
        return sodium_sum + 2185


def vitamin_a_constraint(in_list):  # in IU
    vita_sum = sum([x * meal.Meals[i]["vita"] for i, x in enumerate(in_list)])
    if vita_sum >= 2850:
        return vita_sum - 3600
    else:
        return vita_sum + 2850


def vitamin_c_constraint(in_list):  # in mg
    vitc_sum = sum([x * meal.Meals[i]["vitc"] for i, x in enumerate(in_list)])
    if vitc_sum >= 85.5:
        return vitc_sum - 108
    else:
        return vitc_sum + 85.5


def vitamin_d_constraint(in_list):  # in IU
    vitd_sum = sum([x * meal.Meals[i]["vitd"] for i, x in enumerate(in_list)])
    if vitd_sum >= 760:
        return vitd_sum - 960
    else:
        return vitd_sum + 760


def vitamin_e_constraint(in_list):  # in IU
    vite_sum = sum([x * meal.Meals[i]["vite"] for i, x in enumerate(in_list)])
    if vite_sum >= 21.28:
        return vite_sum - 26.88
    else:
        return vite_sum + 21.28


def vitamin_b12_constraint(in_list):  # in mcg
    vitb12_sum = sum([x * meal.Meals[i]["vitb12"] for i, x in enumerate(in_list)])
    if vitb12_sum >= 2.28:
        return vitb12_sum - 2.88
    else:
        return vitb12_sum + 2.28


def calcium_constraint(in_list):  # in mg
    calcium_sum = sum([x * meal.Meals[i]["calcium"] for i, x in enumerate(in_list)])
    if calcium_sum >= 1235:
        return calcium_sum - 1560
    else:
        return calcium_sum + 1235


def iron_constraint(in_list):  # in mg
    iron_sum = sum([x * meal.Meals[i]["iron"] for i, x in enumerate(in_list)])
    if iron_sum >= 17.1:
        return iron_sum - 21.6
    else:
        return iron_sum + 17.1


def potassium_constraint(in_list):  # in mg
    potassium_sum = sum([x * meal.Meals[i]["potassium"] for i, x in enumerate(in_list)])
    if potassium_sum >= 4465:
        return potassium_sum - 5640
    else:
        return potassium_sum + 4465


# Bundle them all up at the end
Constraints = [prep_time_constraint, calories_constraint, cost_constraint, carbs_constraint, protein_constraint,
               fat_constraint, sugar_constraint, saturated_fat_constraint, fiber_constraint, sodium_constraint,
               vitamin_a_constraint, vitamin_c_constraint, vitamin_d_constraint, vitamin_e_constraint,
               vitamin_b12_constraint, calcium_constraint, iron_constraint, potassium_constraint]
