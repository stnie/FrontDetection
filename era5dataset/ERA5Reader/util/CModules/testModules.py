import numpy as np
import CalcVariables.example
import time



def equivalentPotentialTemp(t, q, PaPerLevel):
    # celsius temp
    ctemp = t-273.15

    # constants
    cp = 1.00482
    cw = 4.18674
    Rd = 287.05
    print(cp, cw, Rd, Rd/(cp*1000.0))

    # L value
    L = 2500.78-2.325734*ctemp

    #equivalent temp
    ctemp += q*(L/(cp+q*cw)) + 273.15

    # precalculate exponent
    cp *= 1000
    val = Rd/cp

    #equivalent potential temp (this is very slow! approx 3.5 seconds)
    # multiplication by 100 to get Pa out of hPa
    ctemp *= np.power(100000/PaPerLevel, val)

    return ctemp

levels = 1
lats = 721
lons = 1440
t = np.random.rand(levels, lats,lons)
q = np.random.rand(levels, lats,lons)
sp = np.random.rand(levels, lats,lons)


iters = 17
begin = time.time()
for iter in range(iters):
    ept_c = CalcVariables.example.ept(t,q,sp)
end = time.time()
print("C-Module:", end-begin, "s")
begin = time.time()
for iter in range(iters):
    ept_old = equivalentPotentialTemp(t,q,sp)
end = time.time()
print("Py-Module:", end-begin, "s")


print(t.dtype)
print(q.dtype)
print(sp.dtype)

diff = (np.linalg.norm(ept_c-ept_old))
print("Difference is:",diff)
if(diff>0):
    print(ept_c.dtype)
    print(ept_old.dtype)
