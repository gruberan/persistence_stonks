import csv
import itertools
import re
from persistence_landscapes.linear_btree import Linear_BTree
import numpy as np
from copy import deepcopy

class LandscapeFunction:
    def __call__(self, x):
        return self.evaluate(x)
    def evaluate(self,x):
        """Returns the value of the function at input value x."""
        if (x < self.xmin) or (x > self.xmax):
            return 0
        elif self._cache == None:
            return self._evaluate(x)
        elif x in self._cache:
            return self._cache[x]
        else:
            y = self._evaluate(x)
            self._cache[x] = y
            return y
    def convert_to_fixed_lookup(self,mesh=None):
        """
        Converts to a fixed lookup landscape function.
        
        mesh (list of x values): Lookup points in the domain.
        """
        if mesh == None:
            mesh = self.get_xvalues()
        if self.__class__.__name__ == 'LandscapeFunction_Fixed_Lookup' and mesh == self.get_xvalues():
            return self
        else:
            return LandscapeFunction_Fixed_Lookup(mesh,[self.evaluate(x) for x in mesh])
    def _pairwise_iterator(self):
        """Iterates as (i, i+1)."""
        i, j = itertools.tee(self)
        next(j,None)
        return zip(i,j)
    def integrate(self,x_values=None):
        """
        Compute the integral of the function. If x_values are given, these will be the mesh of the integral. If not, the mesh is all corners of the landscape function.
        """
        integral = 0
        y2 = None
        if x_values == None:
            pwiter = self._pairwise_iterator()
        else:
            pwiter = zip(x_values,x_values[1:])
        for x1, x2 in pwiter:
            if not x1 == x2:
                if y2 == None:
                    y1 = self.evaluate(x1)
                else:
                    y1 = y2 # this enforces only n evaluations
                y2 = self.evaluate(x2)
                if not y1 == y2:
                    integral += 0.5*(x2 - x1)*(y2 + y1)
                else:
                    integral += x2*y1 - y1*x2
        return integral
    def plot(self):
        import matplotlib.pyplot as plt
        X = self.get_xvalues()
        Y = [self(x) for x in X]
        plt.plot(X,Y)        
    def __add__(self,other):
        return LandscapeFunction_LinearCombination([1.0,1.0],[self,other])
    def __sub__(self,other):
        return LandscapeFunction_LinearCombination([1.0,-1.0],[self,other])
    def __mul__(self,other):
        return LandscapeFunction_LinearCombination([np.double(other)],[self])
    def __truediv__(self,other):
        return LandscapeFunction_LinearCombination([1.0/np.double(other)],[self])
    def __matmul__(self,other): #Integral of product
        return abs(self - other).integrate()

class LandscapeFunction_Fixed_Lookup(LandscapeFunction):
    # Cannot interpolate between points
    def __init__(self, X, Y): #X does not have to be sorted
        self.xmin, self.xmax = min(X), max(X)
        self._cache = dict(zip(X,Y))
        self.__sorted_xvalues = None
    def get_xvalues(self):
        if not self.__sorted_xvalues: 
            return self._cache.keys()
        else:
            return self.__sorted_xvalues
    def _evaluate(self,x):
        # Convert to an interpolating landscape function.
        # This is inefficient. For optimal performance, do not evaluate a fixed lookup landscape function outside of its domain.
        if not self.__sorted_xvalues:
            self.__sorted_xvalues = [x for x in sorted(self._cache)]
        self = LandscapeFunction_Interpolating(self.__sorted_xvalues,[self._cache[x] for x in self.__sorted_xvalues],0)
        return self.evaluate(x)
    def __iter__(self):
        if not self.__sorted_xvalues:
            self.__sorted_xvalues = [x for x in sorted(self._cache)]
        for x in self.__sorted_xvalues:
            yield x
    def __matmul__(self,other): #add outside of domain
        integral = 0
        # Adds integral of other that is below the domain of self
        X = [x for x in other.get_xvalues() if x <= self.xmin]
        if len(X) > 1:
            integral += other.integrate(X)
        # Adds integral of other that is above the domain of self
        X = [x for x in other.get_xvalues() if x >= self.xmax]
        if len(X) > 1:
            integral += other.integrate(X)
        # Adds integral of |self-other| on the domain of self
        integral += abs(self - other).integrate(self.get_xvalues())
        return integral

class LandscapeFunction_Zero(LandscapeFunction):
    def __init__(self):
        self.xmin, self.xmax = 0, 0
        self._cache = None
    def get_xvalues(self):
        return []
    def _evaluate(self,x):
        return 0
    def __iter__(self):
        yield 0

class LandscapeFunction_Interpolating(LandscapeFunction):
    def __init__(self, X, Y, already_sorted=False, memoization='local'):
        """
        
        already_sorted (bool): indicates that the X values are already sorted.
        
        memoization (str): This flag can have several values:
            - 'None'    don't cache anything. Use for when you will only evaluate between x-values with no reason to expect repeated x-values.
            - 'Local'   cache value of function only on its x values. Use when function will be evaluated on its set of x-values frequently, and rarely in between. Recommended for landscape functions in ensembles.
            - 'All'     cache all x values after every evaluation.
        """
        self._data = Linear_BTree.from_list(X, Y, already_sorted)
        
        self.memoization = memoization
        if memoization == 'none':
            self._cache = None
        else:
            self._cache = dict(zip(X,Y))
        
        self.xmin, self.xmax = min(X), max(X)
        
        self.modified_since_abs = True
        
    def insert(self, x, y):
        self.modified_since_abs = True
        if self.xmax < x:
            self.xmax = x
        if self.xmin > x:
            self.xmin = x
        if not self.memoization == 'None':
            self._cache[x] = y
        self._data.insert(x, y, delay_update=False)

    def evaluate(self, x):
        if (x < self.xmin) or (x > self.xmax):
            return 0
        if not self.memoization == 'None' and x in self._cache:
            return self._cache[x]
        val = self._data.evaluate(x)
        if self.memoization == 'All':
            self._cache[x] = val
        return val
            
    def get_xvalues(self):
        return tuple(x for x in self)
        
    def get_xyvalues(self):
        if self._cache == None:
            X, Y = zip(*((x, self.evaluate(x)) for x in self))
        else:
            X, Y = list(self._cache.keys()), list(self._cache.values())
        return X, Y
    
    def __abs__(self):
        """Returns the pointwise absolute value function |f|(x) = |f(x)|, where f is self."""
        if not self.modified_since_abs:
            return self._abs
        self._abs = deepcopy(self)
        y2 = None
        for x1, x2 in self._pairwise_iterator():
            if y2 == None:
                y1 = self.evaluate(x1)
            else:
                y1 = y2 # this enforces only n evaluations
            y2 = self.evaluate(x2)
            if y2 < 0:
                if y1 > 0:
                    x0 = (y2*x1 - y1*x2)/(y2 - y1)
                    self._abs.insert(x0,0)
                    if not self.memoization == 'None':
                        self._abs._cache[x0] = 0
                self._abs.insert(x1,-y1)
                if not self.memoization == 'None':
                    self._abs._cache[x2] = -y2
            elif y1 < 0: # and y2 >= 0
                if y2 > 0:
                    x0 = (y2*x1 - y1*x2)/(y2 - y1)
                    self._abs.insert(x0,0)
                    if not self.memoization == 'None':
                        self._abs._cache[x0] = 0
                self._abs.insert(x1,-y1)
                if not self.memoization == 'None':
                    self._abs._cache[x1] = -y1
        self.modified_since_abs = False
        return self._abs
    
    def __iter__(self):
        """ Yields sorted x values. """
        for x in self._data:
            yield x

class LandscapeFunction_Ensemble(LandscapeFunction):
    def __init__(self):
        pass
    def __iter__(self):
        #Iterates through sorted x-values of member LandscapeFunctions
        from heapq import merge
        for x in merge(*self.landscape_functions):
            yield x

class LandscapeFunction_LinearCombination(LandscapeFunction_Ensemble):
    def __init__(self,coefficients,landscape_functions,no_cacheQ=None):
        self.coefficients, self.landscape_functions = coefficients, landscape_functions
        self.xmin, self.xmax = min(landscape_function.xmin for landscape_function in self.landscape_functions), max(landscape_function.xmax for landscape_function in self.landscape_functions)
        if no_cacheQ == None: # Disable this for one-time use ensembles
            self._cache = {} 
        else:
            self._cache = None
    def _evaluate(self,x):
        return sum(coefficient*(landscape_function.evaluate(x)) for coefficient, landscape_function in zip(self.coefficients,self.landscape_functions))
    def get_xvalues(self):
        return sorted(list(set(x for x in self)))
    def collapse(self):
        return LandscapeFunction_Interpolating(*zip(*((x,self.evaluate(x)) for x in self.get_xvalues())))
    def __abs__(self):
        return abs(self.collapse())
    def __iadd__(self,other):
        self.coefficients += [1]
        self.landscape_functions += [other]
        if not self._cache == None:
            for x in self._cache:
                self._cache[x] += other.evaluate(x)
        return self
    def __isub__(self,other):
        self.coefficients += [-1]
        self.landscape_functions += [other]
        if not self._cache == None:
          for x in self._cache:
              self._cache[x] -= other.evaluate(x)
        return self
    def __imul__(self,other):
        self.coefficients = [np.double(other)*coefficient for coefficient in self.coefficients]
        if not self._cache == None:
          for x in self._cache:
              self._cache[x] *= np.double(other)
        return self
    def __itruediv__(self,other):
        self.coefficients = [np.double(1.0/other)*coefficient for coefficient in self.coefficients]
        if not self._cache == None:
          for x in self._cache:
              self._cache[x] *= (1.0/other)
          return self

class LandscapeFunction_Product(LandscapeFunction_Ensemble):
    def __init__(self,L1,L2):
        self.L1, self.L2 = L1, L2
    def integrate(self,X=None):
        integral = 0
        y2 = None
        if X == None:
            pwiter = self._pairwise_iterator()
        else:
            pwiter = zip(X,X[1:])
        for x1, x2 in pwiter:
            if not x1 == x2:
                if y2 == None:
                    y1 = self.L1.evaluate(x1)
                    z1 = self.L2.evaluate(x1)
                else:
                    y1 = y2 # this enforces only n evaluations
                    z1 = z2
                y2 = self.L1.evaluate(x2)
                z2 = self.L2.evaluate(x2)
                if not y1 == y2:
                    if not z1 == z2:
                        integral += (1.0/6.0)*(x2 - x1)*(y1*(2*z1+z2) + y2*(z1+2*z2))
                    else:
                        integral += (1.0/6.0)*(x2 - x1)*(y1 + y2)*z1 
                else:
                    if not z1 == z2:
                        integral += (1.0/6.0)*(x2 - x1)*y1*(z1 + z2)
                    else:
                        integral += (x2 - x1)*y1*z1
        return integral
  
 
def inner_product(landscape_function1, landscape_function2,X=None):
    return abs(landscape_function1 - landscape_function2).integrate(X)

class Landscape:
    def __init__(self,landscape_functions):
        self.landscape_functions = landscape_functions
    def __getitem__(self,index):
        if index < len(self.landscape_functions):
            return self.landscape_functions[index]
        else:
            return LandscapeFunction_Zero()
    def __len__(self):
        return len(self.landscape_functions)
    def evaluate(self,x,maxrank=None):
        if maxrank == None:
            return [landscape_function.evaluate(x) for landscape_function in self.landscape_functions]
        elif maxrank < len(self.landscape_functions):
            return [landscape_function.evaluate(x) for landscape_function in self.landscape_functions[:maxrank]]
        else:
            return [landscape_function.evaluate(x) for landscape_function in self.landscape_functions] + [0]*(len(self.landscape_functions) - maxrank)
    def convert_to_fixed_lookup(self,mesh=None):
        """
        - WITH a mesh, returns a Landscape_Fixed_Lookup object containing LandscapeFunction_Fixed_Lookup objects all generated using the same mesh. Using a mesh is strongly recommended.
        - WITHOUT a mesh, returns a Landscape object containing LandscapeFunction_Fixed_Lookup objects which are hashed over their own possibly different xvalue sets.
        - Note that landscapes will be 0 valued outside of the mesh, so the mesh should be between xmin and xmax.
        """
        if mesh == None:
            return Landscape([landscape_function.convert_to_fixed_lookup(mesh) for landscape_function in self.landscape_functions])
        else:
            return Landscape_Fixed_Lookup([landscape_function.convert_to_fixed_lookup(mesh) for landscape_function in self.landscape_functions],mesh)
    def get_range(self):
        return (min(landscape_function.xmin for landscape_function in self.landscape_functions), max(landscape_function.xmax for landscape_function in self.landscape_functions))
    def get_mesh(self,num_bins):
        xmin, xmax = self.get_range()
        delta = (xmax - xmin)/num_bins
        return [xmin + delta*k for k in range(num_bins + 1)]
    def write(self,filename):
        LandscapeWriter.write(self,filename)
    def integrate(self):
        return sum([self[k].integrate() for k in range(len(self))])
    def __add__(self,other):
        return Landscape_LinearCombination([1.0,1.0],[self,other])
    def __sub__(self,other):
        return Landscape_LinearCombination([1.0,-1.0],[self,other])
    def __mul__(self,other):
        return Landscape_LinearCombination([np.double(other)],[self])
    def __truediv__(self,other):
        return Landscape_LinearCombination([1.0/np.double(other)],[self])
    def __abs__(self):
        return Landscape([abs(landscape_function) for landscape_function in self.landscape_functions])
    def __matmul__(self,other):
        return sum([self[k] @ other[k] for k in range(max(len(self),len(other)))])

class Landscape_Fixed_Lookup(Landscape):
    #Landscape_Fixed_Lookup objects are assumed to contain LandscapeFunction_Fixed_Lookup objects all generated over the same mesh.
    def __init__(self,landscape_functions,mesh):
        self.landscape_functions = landscape_functions
        self.mesh = mesh

class Landscape_LinearCombination(Landscape):
    def __init__(self,coefficients,landscapes):
        maxdepth = max([len(landscape) for landscape in landscapes])
        self.landscape_functions = [LandscapeFunction_LinearCombination(coefficients,[landscape[i] for landscape in landscapes]) for i in range(maxdepth)]
    def collapse(self):
        return Landscape([landscape_function.collapse() for landscape_function in self.landscape_functions]) 
    def __iadd__(self,other):
        for landscape_function in self.landscape_functions:
            landscape_function += other
        return self
    def __isub__(self,other):
        for landscape_function in self.landscape_functions:
            landscape_function -= other
        return self
    def __imul__(self,other):
        for landscape_function in self.landscape_functions:
            landscape_function *= other
        return self
    def __itruediv__(self,other):
        for landscape_function in self.landscape_functions:
            landscape_function /= other
        return self

def average(collection):
    if all([issubclass(entry.__class__,Landscape) for entry in collection]):
        return Landscape_LinearCombination([np.double(1)/np.double(len(collection))] * len(collection) , collection)
    if all([issubclass(entry.__class__,LandscapeFunction) for entry in collection]):
        return LandscapeFunction_LinearCombination([np.double(1)/np.double(len(collection))] * len(collection) , collection)


class Landscape_Reader:
    def __almostequal(x,y):
        EPSILON = 0.00000000001
        return abs(x-y) <= EPSILON
    def __from_Barcode(barcode): ## Generates exact landscape from barcodes, input as [birth time, death time] pairs
        def birth(a): return a[0]-a[1]
        def death(a): return a[0]+a[1]
        landscape_functions = []
        barcode = sorted(barcode,key= lambda x:(x[0],-x[1])) # sort primarily by 1st arg ascending, secondarily by 2nd arg descending
        barcode = [((p[0]+p[1])/2.0, (p[1]-p[0])/2.0) for p in barcode] # map to center, radius form
        while not len(barcode) == 0:
            L = [(birth(barcode[0]),0.0),barcode[0]]
            i = 1
            newbarcode = []
            while i < len(barcode):
                p = 1
                if (birth(barcode[i]) >= birth(L[-1])) and (death(barcode[i]) > death(L[-1])):
                    if birth(barcode[i]) < death(L[-1]):
                        pt = ((birth(barcode[i]) + death(L[-1]))/2.0, (death(L[-1]) - birth(barcode[i]))/2.0)
                        L.append(pt)
                        while (i+p < len(barcode)) and (Landscape_Reader.__almostequal(birth(pt),birth(barcode[i+p]))) and death(pt) <= death(barcode[i+p]):
                            newbarcode.append(barcode[i+p])
                            p = p + 1
                        newbarcode.append(pt)
                        while (i+p < len(barcode)) and (birth(pt) <= birth(barcode[i+p])) and (death(pt) >= death(barcode[i+p])):
                            newbarcode.append(barcode[i+p])
                            p = p + 1
                    else:
                        L.append((death(L[-1]),0.0))
                        L.append((birth(barcode[i]),0.0))
                    L.append(barcode[i])
                else:
                    newbarcode.append(barcode[i])
                i = i + p
            L.append((death(L[-1]),0.0))
            #remove duplicates from L
            seen = set()
            seen_add = seen.add
            L = [x for x in L if not (x in seen or seen_add(x))]
            landscape_functions.append(LandscapeFunction_Interpolating(*tuple(zip(*L)), already_sorted=True))
            barcode = newbarcode 
        return Landscape(landscape_functions)
    def __from_PointLists(landscape_pointlists):
        return Landscape([LandscapeFunction_Interpolating(*tuple(zip(*pointlist))) for pointlist in landscape_pointlists])
    def __read_bar_file(filename, ERRORMAX = 10):
        """
        Reads barcode data from .bar file, assumed created by perseus.
        Set ERRORMAX slightly over intended diameter of sample space to debug problems in .bar files.
        """
        data=[]
        ERRORMAX = 10
        with open(filename,'r') as barcodefile:
            barcodereader = csv.reader(barcodefile,delimiter=' ')
            for row in barcodereader:
                b,d = [np.double(x) for x in row]
                if b >= 0 and d >= 0 and b < ERRORMAX and d < ERRORMAX: #throw out infinite -1, 0 barcodes and anything that extends beyond sample diameter
                    data.append([b,d])
                else:
                    if b >= 0 and d >= 0:
                        print('ERRORMAX triggered')
        return Landscape_Reader.__from_Barcode(data)
    def __read_lan_file(filename):
        """
        Reads landscape data from .lan file.
        """
        data, current = [], None
        with open(filename,'r') as landscapefile:
            for line in landscapefile:
                newlandscape = re.compile("lambda_(\\d+)")
                newlandscapematches = newlandscape.findall(line)
                if len(newlandscapematches) > 0:
                    if not current == None:
                        data.append(current)
                    current = []
                    lasta = -1.0
                    continue
                newpoint = re.compile("([-\\d\\.e]+)\\s+([-\\d\\.e]+)")
                newpointmatches = newpoint.findall(line)
                if len(newpointmatches) > 0:
                    number = re.compile("(-*\\d\\.*\\d*)e*(-*[\\d]*)")
                    numbermatch = number.findall(newpointmatches[0][0])
                    if numbermatch[0][1] == '':
                        a = np.double(numbermatch[0][0])
                    else:
                        a = np.double(numbermatch[0][0])*10**np.double(numbermatch[0][1])
                    numbermatch = number.findall(newpointmatches[0][1])
                    if numbermatch[0][1] == '':
                        b = np.double(numbermatch[0][0])
                    else:
                        b = np.double(numbermatch[0][0])*10**np.double(numbermatch[0][1])
                    if not ( a == -1.0 or a == -0.5 or Landscape_Reader.__almostequal(a,lasta) ): # This disallows duplicate entries. Adjust EPSILON as necessary.
                        current.append([a,b])
                        lasta = a
            data.append(current)
        return Landscape_Reader.__from_PointLists(data)
    def read(filename):
        if filename[-3:] == 'bar':
            return Landscape_Reader.__read_bar_file(filename)
        elif filename[-3:] == 'lan':
            return Landscape_Reader.__read_lan_file(filename)
        else:
            return Landscape_Reader.__from_PointLists(filename)
    def read_fromlist(L):
        return Landscape_Reader.__from_Barcode([list(x) for x in L if not np.inf in x])

# TODO
#class Landscape_Writer: 
#    def write(landscape,filename):
#       pass