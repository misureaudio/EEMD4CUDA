import numpy as np
from scipy.interpolate import CubicSpline
import emd_ref as R

N=1024
t=np.arange(N,dtype=float)
x=np.sin(2*np.pi*0.05*t)+0.5*np.sin(2*np.pi*0.013*t+0.7)+0.1*np.random.default_rng(3).standard_normal(N)

mx=np.where((x[1:-1]>x[:-2])&(x[1:-1]>x[2:]))[0]+1
mn=np.where((x[1:-1]<x[:-2])&(x[1:-1]<x[2:]))[0]+1
print('n_max',len(mx),'n_min',len(mn))

# CPU envelope (max)
env_cpu,_=R.envelope(t,x,'max')
# scipy natural spline through maxima
cs=CubicSpline(t[mx],x[mx],bc_type='natural')
env_scipy=cs(t)
print('CPU env vs scipy natural : max|d| = %.3e'%np.max(np.abs(env_cpu-env_scipy)))

# ---- numpy replica of the GPU kernel's EXACT ops ----
def gpu_envelope_replica(t,x,want):
    if want=='max':
        k=np.where((x[1:-1]>x[:-2])&(x[1:-1]>x[2:]))[0]+1
    else:
        k=np.where((x[1:-1]<x[:-2])&(x[1:-1]<x[2:]))[0]+1
    tm=t[k].astype(float); ym=x[k].astype(float); p=len(k)
    H=np.diff(tm)
    nint=p-2
    d=2.0*(H[0:nint]+H[1:nint+1])
    l=np.zeros(nint); u=np.zeros(nint)
    l[1:]=H[1:nint]
    u[0:nint-1]=H[1:nint]
    r=6.0*((ym[2:nint+2]-ym[1:nint+1])/H[1:nint+1]-(ym[1:nint+1]-ym[0:nint])/H[0:nint])
    # Thomas (same as GPU _thomas)
    c_=np.zeros(nint); X=np.zeros(nint)
    c_[0]=(u[0]/d[0]) if nint>1 else 0.0
    X[0]=r[0]/d[0]
    for i in range(1,nint):
        c_[i]=(u[i]/(d[i]-l[i]*c_[i-1])) if i<nint-1 else 0.0
        X[i]=(r[i]-l[i]*X[i-1])/(d[i]-l[i]*c_[i-1])
    for i in range(nint-2,-1,-1):
        X[i]=X[i]-c_[i]*X[i+1]
    Mf=np.zeros(p); Mf[1:-1]=X
    # GPU-style eval: linear scan interval (i while t>tm[i+1]), clamp p-2
    env=np.empty(len(t))
    for ii,tt in enumerate(t):
        i=0
        while i<p-2 and tt>tm[i+1]:
            i+=1
        if i>=p-2: i=p-2
        if i<0: i=0
        h=tm[i+1]-tm[i]
        a=(tm[i+1]-tt)/h; b=(tt-tm[i])/h
        env[ii]=a*ym[i]+b*ym[i+1]+((a**3-a)*Mf[i]+(b**3-b)*Mf[i+1])*h*h/6.0
    return env

env_gpu_rep= gpu_envelope_replica(t,x,'max')
print('GPU-replica vs CPU env    : max|d| = %.3e'%np.max(np.abs(env_gpu_rep-env_cpu)))
print('GPU-replica vs scipy      : max|d| = %.3e'%np.max(np.abs(env_gpu_rep-env_scipy)))
# where is the max diff?
d=np.abs(env_gpu_rep-env_cpu)
i=np.argmax(d)
print('  argmax idx=%d  t=%s  cpu=%.6f rep=%.6f scipy=%.6f'%(i,t[i],env_cpu[i],env_gpu_rep[i],env_scipy[i]))
print('  is idx a max-knot?', i in mx, '  min-knot?', i in mn)
