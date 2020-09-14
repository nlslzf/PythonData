import numpy as np
a = np.array([2,3,4])
a
a.dtype
from sympy import init_session
init_session()
expr = integrate(cos(x),x)
expr
expr = Integrate(cos(x),x)
Derivative(expr,x,2)
b = np.array([1.2,3.5,5.1])
b
b,dtype
b.dtype
b = np.array([[1.5,2,4],[4,5,6]])
b
b.dtype
d.size
b.size
b.ndim
b.shape
b.itemsize
b.date
b.data
c = dtype([[1,2],[3,4]],dtype=complex)
c = np.array([[1,2],[3,4]],dtype=complex)
c
np.zeros((3,4))
np.zeros([3,4])
np.ones((2,5),dtype=np.int16)
np.empty((3,4))
np.arange(0,4,0.5)
np.pi
pi
pi.evalf()
pi.evalf(18)
pi.evalf(100)
np.linsapce(0,2,9)
np.linspace(0,2,9)
np.linspace(0,2*np.pi,100)
x = np.linspace(0,2*np.pi,100)
f = np.sin(x)
a = np.arange(6)
b = np.arange(12).reshape(4,3)
b
print(b)
b = np.arange(12).reshape(2,3,2)
b
print(np.arange(10000))
a
b
c = np.linspace(1,6,6)
c
a -c
a@c
a.dot(c)
a.dtype,name
a.dtype.name
a.dtype
a = np.random.random((2,))
a = np.random.random((2,3))
a
a.sum()
a.min()
a.max()
a.sum(axis=0)
a
a=cumsum(axis=0)
a.cumsum(axis=0)
np.exp(a)
np.sin(a)
np.sqrt(a)
a = np.range(10)**3
a = np.arange(10)**3
a
a[2]
a[2:5]
a[2:-1]
a[2:]
a[2:0]
a[2:-2]
a[:6:2]
a[:6:2]=-1000
a
a[::2]=-1000
a
for i in a:
    print (i**(1/3.))
for i in a:
    print(ok)
def f(x,y):
    return 10*x+y
b = np.fromfunction(f,(5,4),dtype=int)
b
b[2,3]
b[0:5],1
b[0:5,1]
b[1:3,:]
b[-1]
b[-1,:]
b[-1] == b[-1,:]
c = np.array([[[0,1,2],[10,12,15]],[100,101,102],[110,113,115]])
c
c = np.array([[[0,1,2],[10,12,15]],[[100,101,102],[110,113,115]]])
c
c.shape()
c.shape
c[1,...]
c[...,2]
for row in b:
    print(row)
for row in c:
    print(row)
for row in c.flat:
    print(row)
a = np.floor(10*np.random.random(3,4))
a = np.floor(10*np.random.random((3,4)))
a
a.shape
a.ravel
a.ravel()
a.flat
a.flat()
a.flat
a.ravel()
a.reshape(6,2)
a
a.reshape((6,2))
a.T
a
a.resize(2,6)
a
a = np.floor(10*np.random.random((2,2)))
b = np.floor(10*np.random.random((2,2)))
a
b
np.vstack((a,b))
np.hstack((a,b))
a
a = np.arange([4.,2.])
a = np.array([4.,2.])
a
a[:,np.newaxis]
a[:,np.newaxis].shape
a.shape
a[np.newaxis,:].shape
b = np.array((3.0,8.0))
b
np.column_stack(a,b)
np.column_stack((a,b))
np.row_stack((a,b))
np.r_[1:4,0,4]
np.array([1:2])
np.array([1,2])
a = np.floor(10*np.random.random((2,12)))
a
np.split(a,3)
np.hsplit(a,3)
np.hsplit(a,(3,4))
a
np.hsplit(a,(3,6))
a = np.arange(12)
a
b = a
b
b.shape = 3,4
b
a
id(a)
id(b)
b is a
c = a.view()
c
a
c is a
c.base is a
c.flags.owndata
a.flags.owndata
c.shapr
c.shape
c.resize(2,6)
c
a
c.reshape(6,2)
c
c
a
c[1,4]=1234
c
a
a.flags
c.flags
a[0,0]=1009
a
c
s = a[:,1:3]
a
s
s[:]
s[:] = 10
s
a
d = a.copy()
d
d.base isa
d.base is a
a = np.arange(12)**2
j = np.array([[3,4],[9.7]])
a[j]
j = np.array([[3,4],[9,7]])
a[j]
plaette = np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,255]])
plaette
image = np.array([0,2],[2,3])
image = np.array([[0,2],[2,3]])
image
plaette[image]
image = np.array([[[0,2],[2,2]],[[2,0],[2,3]]])
plaette[image]
time = np.linspace(20,145,5)
data = np.sin(np.arange(20)).reshape(5,4)
time
data
ind = data.argmax(axis=0)
ind
data.max(axis=0)
time_max = time[ind]
time_max
data.shape[1]
data.shape
a = np.arange(12).reshape(3,4)
a
 b = a >4
b
a[b]
a[b] = 0
a
%hist -f filename.py
a = np.array([2,3,4,5])
b = np.array([8,5,4])
c = np.array([5,4,6,8,3])
ax, bx, cx = np.ix_(a,b,c)
ax
ax.shape
bx.shape
cx.shape
c.shape
result = ax+bx8cx
result = ax+bx*cx
result
np.add(1,3)
def ufunc_reduce(ufct,*vectors):
    vs = np.ix(*vectors)
    r = ufct.identify
    for v in vs:
        r = ufct(r,v)
    return r
ufunc_reduce(np.add,a,b,c)
def ufunc_reduce(ufct,*vectors):
    vs = np.ix_(*vectors)
    r = ufct.identify
    for v in vs:
        r = ufct(r,v)
    return r
ufunc_reduce(np.add,a,b,c)
def ufunc_reduce(ufct,*vectors):
    vs = np.ix_(*vectors)
    r = ufct.identity
    for v in vs:
        r = ufct(r,v)
    return r
ufunc_reduce(np.add,a,b,c)
np.add.identity
np.identify()
np.identify(2)
np.identity(2)
np.identity()
np.identity(2)
np.identity(4)
np.eyes(4)
np.eye(4)
np.eye(3,4)
np.identity(4,3)
import numpy as np
a = np.array([[1.0,2.0],[3.0,4.0]])
a
print(a)
a.T
a.transpose()
np.linalg.inv(a)
 u = np.eye(2)
u
j = np.array([[0.0,-1.0],[1.0,0.0]])
j
j@j
j.dot(j)
np.trace(j)
np.trace(u)
u
y = np.array([[5.],[7.]])
y
np.linalg.solve(a,y)
a
np.linalg.eig(j)
 a
x = np.float32(1.0)
x
x.dtype
y = np.int_([1,2,4])
y
np.array([1,2,3],dtype='float')
d = np.array([1,2,3],dtype='float')
d.type
d.dtype
d.astype(float32)
d.astype(int8)
d.astype(int)
d.astype(np.float32)
np.float64(d)
d.dtype
d.dtype.name
np.power(100,8,dtype=np.int32)
np.power(100,8,dtype=np.int64)
np.iinfo(int)
np.iinfo(np.int)
np.iinfo(np.int32)
np.power(100,100,dtype=np.int64)
np.power(100,100,dtype=np.float64)
np.finfo
np.finfo()
np.finfo(np.longdouble)
x = np.array([[1,2.0],[0,0.1],(1+1j,3.)])
x
np.zeros((2,3))
np.linspace((1,10,1))
np.linspace(1,10,1)
np.linspace(1,10,3)
np.linspace(1,10,4)
np.linspace(1,10,4,dtype=int)
np.indices((3.3))
np.indices((3,3))
%hist -f filename.py
