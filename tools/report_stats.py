"""Independent recomputation of the write-ups' preregistered Fisher contrasts and 95% CIs (pure Python).

Run from the repository root:  python tools/report_stats.py
Two-sided Fisher exact tests for the preregistered contrasts, plus Wilson and
Clopper-Pearson 95% intervals for every count that appears in the sprint report
and the study's write-ups (not part of this repository).
"""
from math import comb, sqrt
def fisher(a,b,c,d):
    # 2x2: [[a,b],[c,d]] two-sided exact (sum of probs <= observed)
    n=a+b+c+d; r1=a+b; c1=a+c
    def p(x): return comb(r1,x)*comb(n-r1,c1-x)/comb(n,c1)
    p0=p(a); tot=0.0
    for x in range(max(0,c1-(n-r1)), min(r1,c1)+1):
        px=p(x)
        if px<=p0*(1+1e-9): tot+=px
    return tot
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    ph=k/n; den=1+z*z/n; cen=(ph+z*z/(2*n))/den; hw=z*sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return (max(0,cen-hw), min(1,cen+hw))
def cp(k,n):
    # Clopper-Pearson via bisection on binomial cdf
    from math import fsum
    def cdf(kk,p,nn): return fsum(comb(nn,i)*p**i*(1-p)**(nn-i) for i in range(int(kk)+1))
    lo=0.0
    if k>0:
        a,b=0.0,1.0
        for _ in range(60):
            m=(a+b)/2
            if 1-cdf(k-1,m,n) < 0.025: a=m
            else: b=m
        lo=a
    hi=1.0
    if k<n:
        a,b=0.0,1.0
        for _ in range(60):
            m=(a+b)/2
            if cdf(k,m,n) < 0.025: b=m
            else: a=m
        hi=a
    return lo,hi
tests={
 "GPT task 40/60 vs no-task 0/60":(40,20,0,60),
 "Grok task 9/60 vs no-task 0/60":(9,51,0,60),
 "GPT C1 0/21 vs C0 9/30":(0,21,9,21),
 "Grok C1 16/21 vs C0 14/30 (envelope)":(16,5,14,16),
 "Grok C1 16/21 vs C0 20/30 (prose)":(16,5,20,10),
 "GPT C0 9/30 vs R2 control 0/60":(9,21,0,60),
 "Grok C0 14/30 vs R2 control 0/60":(14,16,0,60),
 "Grok C0 prose 20/30 vs R2 control 0/60":(20,10,0,60),
 "GPT R2 2a 0/22 vs control 0/60":(0,22,0,60),
}
for k,v in tests.items(): print(f"{k:45s} p = {fisher(*v):.3g}")
print()
for k,n in [(0,60),(40,60),(9,60),(0,30),(9,30),(14,30),(20,30),(0,21),(16,21),(0,22),(1,22),(0,180),(49,180),(23,90),(16,63),(0,51),(0,108),(0,212)]:
    w=wilson(k,n); c=cp(k,n)
    print(f"{k:3d}/{n:<4d} Wilson {100*w[0]:5.1f}-{100*w[1]:5.1f}   CP {100*c[0]:5.1f}-{100*c[1]:5.1f}")
