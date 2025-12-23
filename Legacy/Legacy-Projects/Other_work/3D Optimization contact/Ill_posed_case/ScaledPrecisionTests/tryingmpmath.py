import mpmath
from decimal import Decimal as D
from pdb import set_trace


mpmath.mp.dps = 36

x = mpmath.mpf(2/7)  # Convert to high-precision float
y = D(2/7)
r1 = x**2 - 4*x + 4
r2 = y**2 - 4*y + 4


set_trace()

print(result)
print(result2)
