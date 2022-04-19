import unittest
import optimize

#print(optimize.tag_index)
print(optimize.coefs.shape, ": ")
print(optimize.coefs)

rng = optimize.np.random.default_rng()
in_val = rng.integers(0, 3, size=31)

print(in_val)
print(optimize.tag_occurance(in_val))
print(optimize.tag_index)

print(optimize.entropy(in_val))
print(optimize.entropy_der(in_val))
print(optimize.entropy_hes(in_val))
print(optimize.entropy_hes(in_val).shape)