'''
Argument Naming Conventions:
    c - constant / scalar
    u, v - vector
    m - matrix
    p - point
    l - line
'''

import math as _math
import copy as _copy

# class Matrix:
#     '''Base class'''

#     def __init__(self, dataArray):
#         self.data = dataArray

#     def __str__(self):
#         '''Gets called using print()'''
#         return 'Matrix: {}'.format(self.data)

#     def __repr__(self):
#         '''Gets called when just using the instance name'''
#         return 'Matrix: {}'.format(self.data)

#     def __eq__(self, m):
#         '''Operator overloading of the == operator'''
#         return self.data == m.data

#     def __lt__(self, m):
#         # return all([i < j for i, j in zip(self.data, m.data)]) # comparing two vectors
#         return all([i < m for i in self.data])

OUTPUT_PREC = 3

class Vector:

    def __init__(self, coords):
        self.coords = coords
        self.dimension = len(self.coords)

    def __str__(self):
        return 'Vector: {}'.format(round(self, OUTPUT_PREC).coords)

    def __repr__(self):
        return 'Vector: {}'.format(round(self, OUTPUT_PREC).coords)

    def __getitem__(self, i):
        return self.coords[i]

    def __setitem__(self, i, val):
        self.coords[i] = val

    def __eq__(self, v):
        return self.coords == v.coords

    def __lt__(self, c):
        return all([i < c for i in self.data])

    def __add__(self, v):
        return add(self, v)

    def __sub__(self, v):
        return subtract(self, v)

    def __mul__(self, c):
        return multiply(self, c)

    def __neg__(self):
        '''Operator overloading of unary operator: -'''
        return Vector([-i for i in self.coords])

    def __pos__(self):
        '''Operator overloading of unary operator: +'''
        return self

    def __abs__(self):
        '''Operator overloading of unary operator: abs()'''
        return Vector([abs(i) for i in self.coords])

    def __round__(self, precision=0):
        return Vector([round(i, precision) for i in self.coords])

    def add(self, v):
        for i, j in enumerate(v.coords):
            self.coords[i] += j

    def subtract(self, v):
        for i, j in enumerate(v.coords):
            self.coords[i] -= j

    def multiply(self, c):
        for i in range(self.dimension):
            self.coords[i] *= c

    def dot(self, v):
        if self.dimension != v.dimension:
            raise ValueError('vectors have different dimensions')
        return dot(self, v)

    def magnitude(self):
        return _math.sqrt(sum(i ** 2 for i in self.coords))

    def normalize(self):
        if self.isZero(): # catching ZeroDivisionError will have the same effect
            raise ValueError('zero vector cannot be normalized')
        return multiply(self, 1 / self.magnitude())

    def angleBetween(self, v, mode='rad'):
        if self.isZero() or v.isZero():
            raise ValueError('zero vector has no angle')
        return angleBetween(self, v, mode)

    def isParallel(self, v):
        return self.isZero() or v.isZero() or isParallel(self, v)

    def isOrthogonal(self, v):
        return self.isZero() or v.isZero() or isOrthogonal(self, v)

    def projection(self, b):
        return projection(self, b)

    def orthogonalComponent(self, b):
        return orthogonalComponent(self, b)

    def cross(self, v):
        # if self.dimension != 3 or v.dimension != 3:
        #     raise ValueError('')
        return cross(self, v)

    # @property
    def isZero(self):
        # return all(i == 0 for i in self.coords)
        return approx(self.magnitude(), 0)

class Point(Vector):

    def __init__(self, *coords):
        super().__init__(coords)

    def __str__(self):
        return 'Point: {}'.format(round(self, OUTPUT_PREC).coords)

    def __repr__(self):
        return 'Point: {}'.format(round(self, OUTPUT_PREC).coords)

class Line:

    def __init__(self, normalVec, k=0, basepoint=None):
        self.normalVec = normalVec
        self.basepoint = basepoint if basepoint != None else self.computeBase(normalVec, k)
        self.k = k if basepoint == None else self.computeK(normalVec, basepoint)
        self.dimension = normalVec.dimension

    def __str__(self):
        return 'Line: {}x + {}y = {}'.format(*round(self.normalVec, OUTPUT_PREC).coords, round(self.k, OUTPUT_PREC))

    def __repr__(self):
        return 'Line: {}x + {}y = {}'.format(*round(self.normalVec, OUTPUT_PREC).coords, round(self.k, OUTPUT_PREC))

    def isParallel(self, l):
        return self.normalVec.isParallel(l.normalVec)

    def isSame(self, l):
        if self.normalVec.isZero():
            return approx(self.k, l.k) if l.normalVec.isZero() else False
        connection = Vector([self.basepoint.coords[1] - l.basepoint.coords[1], self.basepoint.coords[0] - l.basepoint.coords[0]])
        return connection.isOrthogonal(self.normalVec) and connection.isOrthogonal(l.normalVec)

    def intersection(self, l):
        if self.isSame(l):
            return 'two lines are the same!'
        elif self.isParallel(l):
            return 'there are no intersections'
        # consider A = 0
        A, B = self.normalVec.coords
        C, D = l.normalVec.coords
        return Point((D * self.k - B * l.k) / (A * D - B * C), (A * l.k - C * self.k) / (A * D - B * C))

    @staticmethod
    def computeBase(normalVec, k):
        return Point(0, k / normalVec.coords[1])

    @staticmethod
    def computeK(normalVec, basepoint):
        v = normalVec.orthogonalComponent().coords
        return v[0] / v[1] * (-basepoint.coords[0] + v[1] / v[0] * basepoint.coords[1])

class Plane:

    def __init__(self, normalVec, k=0, basepoint=None):
        self.normalVec = normalVec
        # self.basepoint = basepoint if basepoint != None else self.computeBase(normalVec, k)
        self.k = k if basepoint == None else self.computeK(normalVec, basepoint)
        self.dimension = normalVec.dimension

    def __str__(self):
        return 'Plane: {}x + {}y + {}z = {}'.format(*round(self.normalVec, OUTPUT_PREC).coords, round(self.k, OUTPUT_PREC))

    def __repr__(self):
        return 'Plane: {}x + {}y + {}z = {}'.format(*round(self.normalVec, OUTPUT_PREC).coords, round(self.k, OUTPUT_PREC))

    def isParallel(self, l):
        return self.normalVec.isParallel(l.normalVec)

    def isSame(self, l):
        if self.normalVec.isZero():
            return approx(self.k, l.k) if l.normalVec.isZero() else False
        connection = Vector([self.basepoint.coords[2] - l.basepoint.coords[2], self.basepoint.coords[1] - l.basepoint.coords[1], self.basepoint.coords[0] - l.basepoint.coords[0]])
        return connection.isOrthogonal(self.normalVec) and connection.isOrthogonal(l.normalVec)

    def multiply(self, coef):
        self.normalVec.multiply(coef)
        self.k *= coef

    def intersection(self, l):
        if self.isSame(l):
            return 'two planes are the same!'
        elif self.isParallel(l):
            return 'there are no intersections'
        # consider A = 0
        # A, B = self.normalVec.coords
        # C, D = l.normalVec.coords
        # return Point((D * self.k - B * l.k) / (A * D - B * C), (A * l.k - C * self.k) / (A * D - B * C))
        return 'intersect'

    def firstNonzero(self):
        if self.normalVec.isZero():
            return -1
        for i in range(self.normalVec.dimension):
            if not approx(self.normalVec[i], 0):
                return i

    @staticmethod
    def computeBase(normalVec, k):
        return Point(0, 0, k / normalVec.coords[2])

    @staticmethod
    def computeK(normalVec, basepoint):
        v = normalVec.orthogonalComponent().coords # WILL RAISE AN ERROR!!!
        return v[0] / v[1] * (-basepoint.coords[0] + v[1] / v[0] * basepoint.coords[1])

class LinearSystem:

    def __init__(self, *planes):
        planes = list(planes)
        if not all(map(lambda i: i.dimension == planes[0].dimension, planes[1:])):
            raise ValueError('all planes must have the same dimension')
        self.planes = planes
        self.dimension = self.planes[0].dimension

    def __str__(self):
        return 'Linear System:\n  {}'.format('\n  '.join('{}'.format(i) for i in self.planes))

    def __repr__(self):
        return 'Linear System:\n  {}'.format('\n  '.join('{}'.format(i) for i in self.planes))

    def __len__(self):
        return len(self.planes)

    def __getitem__(self, i):
        return self.planes[i]

    def __setitem__(self, i, plane):
        if plane.dimension == self.dimension:
            self.planes[i] = plane
        else:
            raise ValueError('plane must have dimension of {}'.format(self.dimension))

    def nonZeros(self):
        nonZeroIndices = []
        for plane in self.planes:
            for i, j in enumerate(plane.normalVec.coords):
                if j != 0:
                    nonZeroIndices.append(i)
                    break
            else:
                raise ValueError('cannot find nonzero term in the plane: {}'.format(plane))
        return nonZeroIndices

    def swap(self, row1, row2):
        self[row1], self[row2] = self[row2], self[row1]

    def multiply(self, coef, row):
        self[row].multiply(coef)

    def addMultipleRow(self, coef, row, target):
        newPlane = _copy.deepcopy(self[row])
        newPlane.multiply(coef)
        self[target].normalVec.add(newPlane.normalVec)
        self[target].k += newPlane.k

    def triangularForm(self):
        '''Compute the triangular form of a linear system.'''
        system = _copy.deepcopy(self)
        nonZeros = system.nonZeros()[:system.dimension]
        for i in range(system.dimension, len(system)):
            system[i] = Plane(Vector([0] * system.dimension), 0)
        for i, j in zip(range(len(nonZeros)), nonZeros):
            if i < j:
                system.swap(i, nonZeros.index(i))
                # print('Swap eq. {} and eq. {}'.format(i + 1, nonZeros.index(i) + 1))
            for k in range(i + 1, len(nonZeros)):
                if system[k].normalVec[i] != 0:
                    coef = -system[k].normalVec[i] / system[i].normalVec[i]
                    system.addMultipleRow(coef, i, k)
                    # print('Add {} times eq. {} to {}'.format(coef, i + 1, k + 1))
        return system

    def rref(self):
        '''Compute the reduced row-echelon form of a linear system.'''
        system = self.triangularForm()
        for i in range(len(system) - 1, -1, -1):
            currentVar = system[i].firstNonzero()
            if currentVar == -1:
                continue
            system.multiply(1 / system[i].normalVec[currentVar], i)
            currentVarVal = system[i].k
            for j in range(i - 1, -1, -1):
                coef = -system[j].normalVec[currentVar]
                system.addMultipleRow(coef, i, j)
        return system

    def solve(self):
        rref = self.rref()
        for plane in rref.planes:
            if plane.normalVec.isZero() and not approx(plane.k, 0):
                return 'inconsistent system (no solution)'
        solution = [0] * len(rref.planes)
        leadVars = []
        for plane in rref.planes:
            leadVars.append(plane.firstNonzero())
        for i in range(rref.dimension):
            if i not in leadVars:
                return rref.parametrize()
        solution = [i.k for i in rref if i.firstNonzero() != -1]
        return Point(*solution)

    def parametrize(self):
        rref = self #.rref() # should already be in rref form!
        pivotVars = [i.firstNonzero() for i in rref.planes if i.firstNonzero() != -1]
        freeVars = [i for i in range(rref.dimension) if i not in pivotVars]
        pivotNum = len(pivotVars)
        freeNum = rref.dimension - pivotNum
        basepoint = Vector([0] * rref.dimension)
        directionVectors = [_copy.deepcopy(Vector([0] * rref.dimension)) for i in range(freeNum)]
        for pivot, plane in zip(pivotVars, rref):
            if pivot == -1:
                # for i, plane in enumerate(rref[:rref.dimension]):
                #     directionVectors[param - freeNum - 1][i] = -plane.normalVec[param]
                continue
            basepoint[pivot] = plane.k
        for param in freeVars:
            for i, plane in enumerate(rref[:rref.dimension]):
                directionVectors[param - freeNum - 1][i] = -plane.normalVec[param]
        return Parametrization(basepoint, directionVectors)

class Parametrization:

    def __init__(self, basepoint, directionVectors):
        self.basepoint = basepoint
        self.directionVectors = directionVectors
        self.dimension = self.basepoint.dimension
        self.parameters = len(directionVectors)

    def __str__(self):
        return 'Parametrization: {} + {}{}p_2'.format(self.basepoint, '{}p_1 + '.format(round(self.directionVectors[0], OUTPUT_PREC).coords) if self.parameters == 2 else '', round(self.directionVectors[-1], OUTPUT_PREC).coords)

    def __repr__(self):
        return 'Parametrization: {} + {}{}p_2'.format(self.basepoint, '{}p_1 + '.format(round(self.directionVectors[0], OUTPUT_PREC).coords) if self.parameters == 2 else '', round(self.directionVectors[-1], OUTPUT_PREC).coords)

class HyperPlane:

    def __init__(self):
        pass

def approx(a, b, precision=1e-5):
    return abs(a - b) < precision

def add(u, v):
    return Vector([i + j for i, j in zip(u.coords, v.coords)])

def subtract(u, v):
    return Vector([i - j for i, j in zip(u.coords, v.coords)])

def multiply(v, c):
    return Vector([i * c for i in v.coords])

def dot(u, v):
    return sum(i * j for i, j in zip(u.coords, v.coords))

def angleBetween(u, v, mode='rad'):
    val = dot(u, v) / (u.magnitude() * v.magnitude())
    val = max(min(val, 1), -1) # decimal inaccuracies
    result = _math.acos(val)
    if mode.lower() in ('r', 'rad', 'radians'):
        return result
    elif mode.lower() in ('d', 'deg', 'degrees'):
        return result * (180 / _math.pi)

def isParallel(u, v):
    # return approx(abs(u.normalize()), abs(v.normalize())) # robust approach, optimization needed
    return approx(angleBetween(u, v), 0) or approx(angleBetween(u, v), _math.pi)

def isOrthogonal(u, v):
    return approx(dot(u, v), 0)

def projection(v, b):
    return multiply(b.normalize(), dot(v, b.normalize()))

def orthogonalComponent(v, b):
    return v - projection(v, b) # temporary implementation, inefficient?

def cross(u, v):
    return Vector([u.coords[1] * v.coords[2] - u.coords[2] * v.coords[1],
                   -(u.coords[0] * v.coords[2] - u.coords[2] * v.coords[0]),
                   u.coords[0] * v.coords[1] - u.coords[1] * v.coords[0]])
