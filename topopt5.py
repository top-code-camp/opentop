# This python script is intended for topology optimization of continuum structures based on SIMP model.
# This script allows users to define the design domain by hand drawing, the positions for boundary constraints and loading by hand selecting. 
# The design domain is discretized by triangular elements. 
# The design variables are updated by OC method.

# import necessary packages; If not installed, use for example "pip install taichi" to install
import taichi as ti
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
        self.frac = frac  # Volume fraction
        self.cpNode = cpNode if cpNode is not None else None
        self.fdof = fdof if fdof is not None else None
        self.meshGenTri() 
        # Stiffness matrix assembly parameters
        self.iK = np.kron(self.eleDof, np.ones((6, 1))).flatten()
        self.jK = np.kron(self.eleDof, np.ones((1, 6))).flatten()

        self.dof = 2 * self.nnode
        self.fac = None #element volume?
        self.sum1 = None #total volume? 
        self.neiborEle()
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
            xnew[:] = np.maximum(0.001, np.maximum(x * (1 - move), np.minimum(1.0, np.minimum(x * (1 + move),
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
            u = topIns.FEM(x, self.fdof, self.cpNode)
            # end1=process_time()
            # print('FEM Time ', end1 - start1)
            c = np.sum(u[self.fdof])
            ce = np.transpose(u[self.eleDof][np.newaxis], (1, 0, 2)) @ self.KE @ np.transpose(u[self.eleDof][np.newaxis],
                                                                                        (1, 2, 0))
            ce = ce.flatten()
            dc = (-self.p * x ** (self.p - 1)) * ce
            dc = 5 * dc / np.abs(np.min(dc))
            x = topIns.OC(x, dc)
            x = topIns.check(x)
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


class Gui(ti.GUI): # 
    def __init__(self):
        super().__init__('Topology optimization', (1600, 1600), background_color=0xFFFFFF)
        self.a1 = None
        self.b1 = None
        self.c1 = None
        self.nodeList1 = None
        self.cpNode = None
        self.status =None

    def set_disp_param(self, nodeList1, eleNodeList):
         # a1, b1, c1 are the vertices for triangular elements
        self.nodeList1 = nodeList1
        self.a1 = nodeList1[eleNodeList[:, 0]]
        self.b1 = nodeList1[eleNodeList[:, 1]]
        self.c1 = nodeList1[eleNodeList[:, 2]]

    def disp(self, dispx, cpNode): # refresh the screen
        self.triangles(self.a1, self.b1, self.c1, color=ti.rgb_to_hex([dispx, dispx, dispx]))
        for node_no in cpNode:
            X = ti.Vector([self.nodeList1[node_no, 0], self.nodeList1[node_no, 1]])
            self.circle(pos=X, color=0xFF0000, radius=2)
        self.show()
    
    def select_bc(self):
        pass
    
    def select_f(self):
        pass

if __name__ == '__main__':

    # Suguang:
    # GUI is fun when it works. It does not work for me so far.
    # It may be not so convenient when one has to repeat one design many times. 
    # I also suggest we support hdf5 file as input and output file for professional use.
    
    frac = 0.25

    ti.init(arch=ti.gpu)
    gui = Gui()

    points=[]
    draw = 1
    # Define the design domain by hand drawing. 
    # A polyline will be generated by connecting each two subsequently created points.
    # A closed polygon will be generated based on the polylines. 
    while draw == 1:
        gui.text(content=f'Draw anything you want, press SPACE to continue.',
                 pos=(0, 0.99),
                 font_size=80,
                 color=0x0)
        # Press 'Shift' to enforce the orthogonality between the newly generated line and the last generated line
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.LMB:
                if gui.is_pressed('Shift'):
                    if len(points)<1:
                        points.append(e.pos)
                    else:
                        num = len(points)
                        pre_point = points[num-1]
                        dx = np.abs(pre_point[0]-e.pos[0])
                        dy = np.abs(pre_point[1]-e.pos[1])
                        if dx<dy:
                            temp_point = [pre_point[0],e.pos[1]]
                        else:
                            temp_point = [e.pos[0],pre_point[1]]
                        points.append(temp_point)
                else:
                    points.append(e.pos)
            # Press 'SPACE' to complete the drawing of the design domain
            if e.key == ti.GUI.SPACE:
                draw = 0
        if len(points)>0:
            gui.circles(pos=np.array(points), color=0xFF0000, radius=10)
            for i in range(len(points)-1):
                gui.line(begin=points[i], end=points[i+1], radius=2, color=0xFF0000)
        gui.show()

    points = points[0:len(points)-1]
    # Initialize the optimization parameters
    topIns = topOpt(points, frac)
    # KE = topIns.KE
    # p = topIns.p
    eleNodeList = topIns.eleNodeList
    nodeList = topIns.nodeList
    # eleCenter = topIns.eleCenter
    # eleDof = topIns.eleDof
    # nd = topIns.nele
    nnode = topIns.nnode  
    cpNode = np.zeros(nnode,dtype=int)  # cpNode is the nodes for definition of boundary constraints
    ncp = 0
    nfdof = 0
    fdof = np.zeros(100,dtype=int)
    scale = 1
    nodeList1 = scale * nodeList
    gui.set_disp_param(nodeList1, eleNodeList)
   
    count = 0
    while True:
        dispx = 1 - topIns.x[-1] if len(topIns.x) != 0 else (1-topIns.frac)*np.ones(topIns.nele)
        gui.triangles(gui.a1, gui.b1, gui.c1, color=ti.rgb_to_hex([dispx, dispx, dispx]))
        for e in gui.get_events(ti.GUI.PRESS):
            # Press 'LMB' (Left Mouse Button) to proceed into the selection of points for boundary and force definition
            if e.key == ti.GUI.LMB:
                # Press 'b' to select points for definition of boundary constraints
                if gui.is_pressed('b'):
                    if count == 0:
                        print("Start point", e.pos)
                        start1 = np.array([e.pos[0], e.pos[1]])
                        distx0 = (start1[0] - nodeList1[:, 0])
                        disty0 = (start1[1] - nodeList1[:, 1])
                    if count == 1:
                        print("End point", e.pos)
                        end1 = np.array([e.pos[0], e.pos[1]])
                        distx1 = (end1[0] - nodeList1[:, 0])
                        disty1 = (end1[1] - nodeList1[:, 1])
                        idx1 = np.argwhere((distx0 * distx1) < 0).flatten()
                        idx2 = np.argwhere((disty0 * disty1) < 0).flatten()
                        idx = np.intersect1d(idx1,idx2)  # Only select the points locating within the rectangle defined by two diagonal vertices [start1, end1]
                        cpNode[ncp:ncp+len(idx)] = idx.astype(int)
                        ncp = ncp+len(idx)
                    count = (count+1) % 2
                # Press 'f' to select points for application of external force
                if gui.is_pressed('f'):
                    pos1 = np.array([e.pos[0], e.pos[1]])
                    dist0 = (pos1[0] - nodeList1[:, 0]) ** 2
                    dist1 = (pos1[1] - nodeList1[:, 1]) ** 2
                    dist = np.sqrt(dist0 + dist1)
                    idx = np.argmin(dist)
                    fdof[nfdof] = 2 * idx + 1
                    nfdof += 1
                    gui.circle(pos=ti.Vector([e.pos[0], e.pos[1]]), color=0xFF0000, radius=10)
            # Press 'r' to run the optimization program
            if e.key == 'r':
                fdof = fdof[:nfdof]
                cpNode = cpNode[:ncp]
                topIns.run(cpNode, fdof, gui.disp)
        gui.circles(pos=nodeList1[cpNode[0:ncp], :], color=0xFF0000, radius=2)
        gui.show()
