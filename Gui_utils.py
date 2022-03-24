# -*- coding: utf-8 -*-
import taichi as ti

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

    def key_press_monitor(self):
        pass
