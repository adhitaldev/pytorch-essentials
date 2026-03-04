## Plays with various python numeric concepts
import sys
import math
from math import inf
import decimal
from decimal import Decimal
print(decimal.__file__)

# Using infinity is great for min_max value
max_val = inf # inf comes from IEEE 754 and 
min_val = -inf

def inf_funcs():
    # math.inf is a floatign point infinity defined by IEEE 754
    print("----- math.inf COOLNESS --- ")
    print(f"inf > 1e308 {inf > 1e308}")
    print(f"inf > sys.maxsize {inf > sys.maxsize}")
    print(f"inf + 1 = {inf + 1}")
    print(f"inf * 2 = {inf * 2}")

def decimal_funcs():
    # Decimals are slow, exact and rules - driven
    # Base 10 arithmetic
    # Arbritrary precision, use controlled rounding and precision
    print("----- DECIMAL COOLNESS --- ")
    val = Decimal("19.99")
    tax = Decimal("0.0239")
    total = val * (Decimal("1") + tax)
    print(f"Calculating total vased on val:{val} and tax:{tax}")

def float_funcs():
    # Use float for science, graphics, ML, sensors, simulations, performance-critical math
    # Floats are fast, approximate, hardware friendly
    # IEEE-754 precision, base 2 representation
    # Default tolerance is 1e-9print("----- DECIMAL COOLNESS --- ")
    print("----- FLOAT COOLNESS --- ")
    a = 2.145
    b = 2.1452
    print(f"{a} and {b} are close - {math.isclose(a, b, rel_tol=0.0001)}")

def threetwoOrSixFour():
    print("----- sys.maxsize COOLNESS --- ")
    max_32 = 2 ** 31 - 1
    max_64 = 2 ** 63 - 1
    if sys.maxsize == max_32:
        print("32 bit system")
    elif sys.maxsize == max_64:
        print("64 bit system")
    else:
        print("Unknown system!")


# Run some cool math funcs
inf_funcs()
float_funcs()
decimal_funcs()
threetwoOrSixFour()
