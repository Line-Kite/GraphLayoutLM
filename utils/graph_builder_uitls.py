
import json

OPPOSITE={"left":"right",
          "right":"left",
          "up":"down",
          "down":"up",
          "parent":"child",
          "child":"parent"}

class TreeNode:
    def __init__(self,id,box):
        self.id=id
        self.box=box
        self.parent=None
        self.children=[]

def json_loader(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data

def json_saver(data,path):
    json_data = json.dumps(data, ensure_ascii=False)
    with open(path, "w", encoding='utf-8') as f:
        f.write(json_data)

def posotion_judge(box_a,box_b):
    center=[box_a[0]+box_a[2],box_a[1]+box_a[3]]
    box_b2=[2*i for i in box_b]
    if box_b2[2]<center[0]:
        if box_b2[3]<center[1]:
            return "up-left"
        elif box_b2[1]<=center[1]<=box_b2[3]:
            return "left"
        elif box_b2[1]>center[1]:
            return "down-left"
    elif box_b2[0]<=center[0]<=box_b2[2]:
        if box_b2[3]<center[1]:
            return "up"
        elif box_b2[1]<=center[1]<=box_b2[3]:
            return "center"
        elif box_b2[1]>center[1]:
            return "down"
    elif box_b2[0]>center[0]:
        if box_b2[3] < center[1]:
            return "up-right"
        elif box_b2[1] <= center[1] <= box_b2[3]:
            return "right"
        elif box_b2[1] > center[1]:
            return "down-right"
        
def node_box_update(node_p,node_c):
    if node_p.box[0]>node_c.box[0]:
        node_p.box[0]=node_c.box[0]
    if node_p.box[1]>node_c.box[1]:
        node_p.box[1]=node_c.box[1]
    if node_p.box[2]<node_c.box[2]:
        node_p.box[2]=node_c.box[2]
    if node_p.box[3]<node_c.box[3]:
        node_p.box[3]=node_c.box[3]