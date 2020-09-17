import np as np
import numpy as np
from io import StringIO
data = u
data = u"1,2,3\n4,5,6"
np.genfromtxt(StringIO(data),delimiter=',')
ls
cat jkey,key
cat jkey.key
cd ..
cd jupyter/
ls
which bash
cel
free -m
data = u" 1 2 3\n 4 5 67\n890123 4"
np.genfromtxt(StringIO(data),delimiter=3)
data = u"  1  2  3\n  4  5 67\n890123  4"
np.genfromtxt(StringIO(data),delimiter=3)
data = u"123456789\n  4  7 9\n   4567 9"
np.genfromtxt(StringIO(data),delimiter=(4,3,2))
data = u"1, abd , 2\n 3, xxx, 4"
np.genfromtxt(StingIO(data),delimiter=',',dtype="|U5")
np.genfromtxt(StringIO(data),delimiter=',',dtype="|U5")
np.genfromtxt(StringIO(data),delimiter=',',dtype="|U5",autostrip=True)
daat = u"""#
# Skip me !
# Skip me too !
1, 3
3, 4
5, 6 #This is the third line of the data
7, 8
# And here comes the last line
9, 0
"""
daat
daat > test
fa = open('a.txt'.'w+')
fa = open('a.txt','w+')
fa.write(daat)
fa.close()
np.genfromtxt(StringIO(daat),comment='#',delimiter=',')
np.genfromtxt(StringIO(daat),comments='#',delimiter=',')
data = u"\n".join(str(i) for i in range(10))
data
data = u"\n".join(chr(i) for i in range(10))
data
data = u"\n".join(chr(96+i) for i in range(10))
data
data = u"\n".join(str(i) for i in range(10))
data
np.genfromtxt(StringIO(data),)
np.genfromtxt(StringIO(data),skip_header=1,skip_footer=1)
data = u"1 2 3\n4 5 6"
data
np.genfromtxt(StringIO(data),usecols=(0,-1))
np.genfromtxt(StringIO(data),usecols=(1:-1))
np.genfromtxt(StringIO(data),usecols=(1,-1))
np.genfromtxt(StringIO(data),usecols=(:))
np.genfromtxt(StringIO(data),usecols=(1,:))
np.genfromtxt(StringIO(data),usecols=(1,))
np.genfromtxt(StringIO(data),usecols=(1,2))
np.genfromtxt(StringIO(data),usecols=[1])
np.genfromtxt(StringIO(data),usecols=[1,:])
np.genfromtxt(StringIO(data),usecols=[1:])
np.genfromtxt(StringIO(data),usecols=[1,-1])
np.genfromtxt(StringIO(data),usecols=[:])
np.genfromtxt(StringIO(data),usecols=[1,:])
np.genfromtxt(StringIO(data),usecols=[1])
np.genfromtxt(StringIO(data),usecols=[1,2])
np.genfromtxt(StringIo(data),names='a,b,c',usecols=('a','c'))
np.genfromtxt(StringIO(data),names='a,b,c',usecols=('a','c'))
data = StringIO("So it goes\n#a b c\n1 2 3\n 4 5 6")
data
np.genfromtxt(data,skip_header=1,names=True)
data = StingIO("1 2 3\n 4 5 6")
data = StringIO("1 2 3\n 4 5 6")
np.genfromtxt(data,dtypr=(int,float,int))
np.genfromtxt(data,dtype=(int,float,int))
convertfunc = lambda x: float(x.strip(b"%"))/100.
data = u"1,2.3%,45.\n6,78.9%,0"
np.genfromtxt(StringIO(data),delimiter='',names=names,converters={1:convertfunc})
names = ['a','b','c']
np.genfromtxt(StringIO(data),delimiter='',names=names,converters={1:convertfunc})
data = u"N/A, 2, 3\n4,,???"
kwargs = dict(delimiter=',',dtype=int,names='a,b,c',missing_values={0:"N"})
kwargs = dict(delimiter=',',dtype=int,names='a,b,c',missing_values={0:"N/A",'b':0,2:"???"},filling_values={0:0,'b':0,2:-999})
np.gemfromtxt(SringIO(data),**kwargs)
np.genfromtxt(SringIO(data),**kwargs)
np.genfromtxt(StringIO(data),**kwargs)
x = np.arange(10,1,-1)
x
y = np.linspace(1,19,4)
y
x[np.array([3,3,1,8])]
z = x[np.array([3,3,1,8])]
z.base
z.base()
y = x
y.base
y.base()
np.base(y)
x.base
z.flags.owndata
y.flags.owndata
x.flags.owndata
y.base is a
y.base is x
z.base is x
z
x
y = x
y.base is x
y[-1]=100
y
x
y is a
y is x
x is y
a = x [1:3]
a
a.base
a.base is x
a
b = x[np.array[1:3]]
b = x[np.array([1:3])]
b = x[np.array([1:3:1])]
b = x[np.array([1,2,3])]
b
b.base
b.base isa
b.base is a
y = np.arange(35).reshape(5,7)
y
y[np.array([0,2,4]),np.array([0,1,2])]
b = y>20
b
b[:6]
b[:,6]
b[:3]
b[:,3]
y[b[:,5],1:3]
y[1:3,b[:,5]]
y[b[1:3,5]]
y[b[:,5].1:3]
y[b[:,5],1:3]
y[np.array([1,2,4]),1:3]
y[1:3,np.array([1,2,4])]
b
y
y[1,2]
y[b[:,5]]
x = np.arange(5)
x[:,np.newaxis] + x[np.newaxis,:]
x[:,np.newaxis]
z = np.arange(81).reshape(3,3,3,3)
z[1,...,2]
z
x = np.arange(0,50,10)
x
x[np.array([1,1,3,1])] += 1
x
x = np.arange(0,50,10)
x
x[1,1,3,1] +=1
x[[1,1,3,1]] +=1
x
x[(1,1,3,1)] +=1
x[[1,1,3,1]] +=1
x
indices = (1,1,1,1)
z[indices]
indices = (1,1,1,slice(1,2))
indices = (1,1,1,slice(0,2))
z[indices]
slice(0:4)
slice(0,4)
a = slice(0,4)
a
a.type
type(a)
z[[1,1,1]]
z[[1,1,1,:]]
z[[1,1,1]]
z[(1,1,1)]
%hist -f hist2.py
