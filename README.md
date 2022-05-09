# CSE 543 Meal Planning Optimization Problem

The input to this program cna be found in `meal.tsv`, which contains a dataset of accepable food items (from dietary supplements to snacks to entrees), along with their cost, prep time, and nutritional content. The bulk of computation can be found in the `optimize.py` module. Accessory modules include `meals.py`, which parses the dataset, and `constraints.py`, which provides API calls to define constraints for the problem, as well as default sets of constraints.

## Using

The only external requirements are numpy and scipy which can be installed with `pip install -r requirements`.

An example usage can be found in `experiments.py`. Run as is, this file will provide all the results listed in our project report. It can be run with `python3 experiments.py` to verify our results. Runtime should be about 1-2 minutes.
