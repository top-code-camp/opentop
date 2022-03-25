# -*- coding: utf-8 -*-
# This python script is intended for topology optimization of continuum structures based on SIMP model.
# This script allows users to define the design domain by hand drawing, the positions for boundary constraints and loading by hand selecting. 
# The design domain is discretized by triangular elements. 
# The design variables are updated by OC method.

# import necessary packages; If not installed, use for example "pip install taichi" to install
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import h5py
import pygmsh
import fem_utils
'''
from numpy.linalg import *
import matplotlib
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from time import process_time
'''

class topOpt:
    def __init__(self, points, frac, cpNode=None, fdof=None):
    
        # x: initial, current, and final design
        # I suggest we add design as feature of the topopt class (Suguang)
        self.x = []
        
        self.points = points
        self.h = 0.01     # Thickness of plate
        self.r = 0.006    # Filter radius
        self.p = 3        # Penalization factor for SIMP model
        self.xmin=0.001   # Lower bound for the density value (I added this, and used it in OC, Liwei)
        self.Emin=1e-9    # Avoid possible singularity issues in the stiffness matrix (Liwei)
        self.frac = frac  # Volume fraction
        self.cpNode = cpNode if cpNode is not None else None # displacement BC
        self.fdof = fdof if fdof is not None else None # force BC
        self.meshGenTri() 
        # Subscript vector in assembling the sparse stiffness matrix
        self.iK = np.kron(self.eleDof, np.ones((6, 1))).flatten()
        self.jK = np.kron(self.eleDof, np.ones((1, 6))).flatten()

        self.dof = 2 * self.nnode
        self.fac = None #element volume?  (Liwei: I think this is the numerator of the filtering operation)
        self.sum1 = None #total volume?  ( Liwei: This should be the denominator of the filtering operation)
        self.neiborEle() # Assemble the filter (Should we add other filtering techniques? Liwei)
        self.D = np.zeros((3, 3))
        self.stiffMatrix()
        self.KE = None
        self.eleMatrixTri()

    def meshGenTri(self):
        with pygmsh.geo.Geometry() as geom:
            geom.add_polygon(self.points, mesh_size=0.005)
            mesh = geom.generate_mesh()
        self.nodeList = mesh.points[:, 0:2]
        '''
        # self.eleNodeList = mesh.cells[1][1]  
        # Execution of this line of code results into the following error: "Error: *** TypeError: 'CellBlock' object is not subscriptable"
        # Is it related to the version of the packages or operating system? (Yongpeng, Suguang) 
        # To allow the program to proceed, this line of code is modified as: self.eleNodeList=mesh.get_cells_type("triangle")
        '''
        self.eleNodeList=mesh.get_cells_type("triangle")
        self.nele = len(self.eleNodeList)
        self.nnode = len(self.nodeList)
        self.eleCenter = np.zeros((self.nele, 2), dtype=np.float32)
        pos = self.nodeList[self.eleNodeList]
        self.eleCenter[:, 0] = np.sum(pos[:, :, 0], 1) / 3.0
        self.eleCenter[:, 1] = np.sum(pos[:, :, 1], 1) / 3.0
        self.eleDof = np.zeros((self.nele, 6), dtype=np.uint32)
        self.eleDof[:, 0], self.eleDof[:, 1] = 2 * self.eleNodeList[:, 0], 2 * self.eleNodeList[:, 0] + 1
        self.eleDof[:, 2], self.eleDof[:, 3] = 2 * self.eleNodeList[:, 1], 2 * self.eleNodeList[:, 1] + 1
        self.eleDof[:, 4], self.eleDof[:, 5] = 2 * self.eleNodeList[:, 2], 2 * self.eleNodeList[:, 2] + 1

    def neiborEle(self):
        ik = np.zeros(20 * self.nele)
        jk = np.zeros(20 * self.nele)
        sk = np.zeros(20 * self.nele)
        count = 0
        for i in range(self.nele):
            xd = self.eleCenter[:, 0] - self.eleCenter[i, 0]
            yd = self.eleCenter[:, 1] - self.eleCenter[i, 1]
            d = np.sqrt(xd ** 2 + yd ** 2)
            dif1 = self.r - d
            idx = np.argwhere(dif1 > 0).flatten()
            ik[count:count + len(idx)] = i
            jk[count:count + len(idx)] = idx
            sk[count:count + len(idx)] = dif1[idx]
            count += len(idx)
        ik = ik[0:count]
        jk = jk[0:count]
        sk = sk[0:count]
        self.fac = csc_matrix((sk, (ik, jk)), shape=(self.nele, self.nele))
        self.sum1 = self.fac.sum(1).flatten()

    def stiffMatrix(self):
        E = 200000.0
        v = 0.3
        self.D = fem_utils.constitutive_matrix(E,v)

    def eleMatrixTri(self):
        pos = self.nodeList[self.eleNodeList]
        pos1 = np.reshape(pos, (self.nele, 6))
        xi, yi, xj, yj, xm, ym = pos1[:, 0], pos1[:, 1], pos1[:, 2], pos1[:, 3], pos1[:, 4], pos1[:, 5]
        ai, aj, am, bi, bj, bm, gi, gj, gm = xj * ym - yj * xm, yi * xm - xi * ym, xi * yj - yi * xj, yj - ym, ym - yi, yi - yj, xm - xj, xi - xm, xj - xi
        A = xi * (yj - ym) + xj * (ym - yi) + xm * (yi - yj)
        self.area = A / 2
        ai, aj, am, bi, bj, bm, gi, gj, gm = ai / A, aj / A, am / A, bi / A, bj / A, bm / A, gi / A, gj / A, gm / A
        B = np.zeros((self.nele, 3, 6))
        B[:, 0, 0], B[:, 0, 2], B[:, 0, 4] = bi, bj, bm
        B[:, 1, 1], B[:, 1, 3], B[:, 1, 5] = gi, gj, gm
        B[:, 2, 0], B[:, 2, 1], B[:, 2, 2], B[:, 2, 3], B[:, 2, 4], B[:, 2, 5] = gi, bi, gj, bj, gm, bm
        self.KE = np.transpose(B, (0, 2, 1)) @ self.D @ B

    def FEM(self, x, fdof, cpNode):
        allDof = np.arange(0, self.dof)
        cpDof = np.sort(np.append(2 * cpNode, 2 * cpNode + 1))
        freeDof = np.setdiff1d(allDof, cpDof)
        KE1 = self.KE * np.transpose(x[np.newaxis][np.newaxis] ** self.p, (2, 1, 0))
        # Should we use the following line to avoid singularity issues? (Liwei)
        # KE1 = self.KE * np.transpose(self.Emin+(1.0-self.Emin)*x[np.newaxis][np.newaxis] ** self.p, (2, 1, 0)) 
        sK = KE1.flatten()
        K = csc_matrix((sK, (self.iK, self.jK)), shape=(self.dof, self.dof))
        f = np.zeros(self.dof)
        f[fdof] = 1
        u = np.zeros(self.dof)
        u[freeDof] = spsolve(K[freeDof, :][:, freeDof], f[freeDof])
        return u

    def check(self, x):
        xnew = x
        temp1 = self.fac @ np.transpose(x)
        xnew = temp1 / self.sum1
        return np.array(xnew).flatten()

    def OC(self, x, dc):
        move = 0.2
        l1 = 0
        l2 = 1e9
        xnew = np.zeros(self.nele)
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = (l1 + l2) / 2
            # xt = x * np.sqrt(-dc / lmid)
            # xnew[:] = np.maximum(0.001,np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / lmid)))))
            #xnew[:] = np.maximum(0.001, np.maximum(x * (1 - move), np.minimum(1.0, np.minimum(x * (1 + move),
            #                                                                                  x * np.sqrt(
            #                                                                                      -dc / lmid)))))
            xnew[:] = np.maximum(self.xmin, np.maximum(x * (1 - move), np.minimum(1.0, np.minimum(x * (1 + move),
                                                                                              x * np.sqrt(
                                                                                                  -dc / lmid)))))            
            if np.sum(xnew) - self.frac * self.nele > 0:
                l1 = lmid
            else:
                l2 = lmid
        return xnew

    def run(self, cpNode, fdof, call_back_func):
        self.fdof = np.unique(fdof)
        self.cpNode = np.unique(cpNode)
        x = self.frac * np.ones(self.nele)
        change = 1
        loop = 0
        maxloop=50
        while change > 0.001 :
            loop += 1
            xold = x
            # start1=process_time()
            u = self.FEM(x, self.fdof, self.cpNode)
            # end1=process_time()
            # print('FEM Time ', end1 - start1)
            c = np.sum(u[self.fdof])
            ce = np.transpose(u[self.eleDof][np.newaxis], (1, 0, 2)) @ self.KE @ np.transpose(u[self.eleDof][np.newaxis],
                                                                                        (1, 2, 0))
            ce = ce.flatten()
            dc = (-self.p * x ** (self.p - 1)) * ce
            dc = 5 * dc / np.abs(np.min(dc))
            # Suggest to add the following line to apply filter on the sensitivity value, since filter is used for x. (Liwei)
            #dc = self.check(dc)
            # Also, we might need to explicitly write out dv, so that we could apply filter on it and use it in OC.
            x = self.OC(x, dc) # In the future, we could include MMA to extend the code for more geenral design cases.
            x = self.check(x)
            call_back_func(1-x, self.cpNode) # call function to refresh screen
            print(np.sum(x) - self.frac *self.nele)
            print(loop)
            self.x.append(x)
            if abs(loop- maxloop)<1e-3:
                print ("The program is paused because loop reaches maxloop")
                self.save()
                return
            change = np.linalg.norm(x - xold, np.inf)
        self.save()
        return
    
    def save(self): # save the result
        with h5py.File("topopt_result.hdf5",'w') as f:
            f.create_dataset('points', data=self.points, compression='gzip', compression_opts=4)
            f.create_dataset('bc', data=self.cpNode, compression='gzip', compression_opts=4)
            f.create_dataset('force', data=self.fdof, compression='gzip', compression_opts=4)
            f.create_dataset('density', data=self.x, compression='gzip', compression_opts=4)
            f.attrs['volume_frac'] = self.frac
