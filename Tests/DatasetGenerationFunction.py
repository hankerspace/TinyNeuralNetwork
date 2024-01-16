import random

# Computes the xor of a list of bits
def xor(list):
    sum = 0
    for i in list:
        sum += i
    return sum % 2

# Check if number is duffinian
# from https://rosettacode.org/wiki/Duffinian_numbers#Python
def factors(n):
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

is_relively_prime = lambda a, b: gcd(a, b) == 1
sigma_sum = lambda x: sum(factors(x))
is_duffinian = lambda x: is_relively_prime(x, sigma_sum(x)) and len(factors(x)) > 2

# Check if each integers in the list is duffinian
def is_duffinian_list(list):
    for i in list:
        if not is_duffinian(i):
            return False
    return True


# Generate a random list of bits and create a list of tuples (input, output) using the generationFunction
def random_dataset_of_bits(listSize, bits, generationFunction):
    new_list = []
    for i in range(listSize):
        random_list = [random.randrange(2) for _ in range(bits)]
        new_list.append((random_list, [generationFunction(random_list)]))
    return new_list

# Generate a random list of integers and create a list of tuples (input, output) using the generationFunction
def random_dataset_of_integers(listSize, integers, generationFunction, min=0, max=255):
    new_list = []
    for i in range(listSize):
        random_list = [random.randint(min, max) for _ in range(integers)]
        new_list.append((random_list, [generationFunction(random_list)]))
    return new_list
