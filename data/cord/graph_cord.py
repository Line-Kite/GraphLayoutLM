import os
import sys

sys.path.append("..") 

from utils.graph_builder_uitls import OPPOSITE, TreeNode, json_loader, json_saver, node_box_update, posotion_judge


def get_item(id,list):
    for item in list:
        if id==item["id"]:
            return item


def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [x0, y0, x1, y1]
    return bbox


def quad_to_box(quad):
    box = (
        max(0, quad["x1"]),
        max(0, quad["y1"]),
        quad["x3"],
        quad["y3"]
    )
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box


def data_preprocess(data):
    if "id" not in data["valid_line"][0].keys():
        for i,item in enumerate(data["valid_line"]):
            item["id"]=i
    return data


def insertion_reorder(nodes):
    sorted_nodes = []
    for node in nodes:
        len_sorted=len(sorted_nodes)
        if len_sorted==0:
            sorted_nodes.append(node)
            continue
        for i in range(len_sorted):
            rel = posotion_judge(sorted_nodes[i].box,node.box)
            if rel=="up-left" or rel=="left" or rel=="up" or rel=="up-right":
                sorted_nodes.insert(i, node)
                break
            else:
                if i==len_sorted-1:
                    sorted_nodes.append(node)
    assert len(sorted_nodes)==len(nodes)
    return sorted_nodes


def get_relationship(node_a,node_list,edges):
    for node in node_list:
        if node.id==node_a.id:
            continue
        rel=posotion_judge(node_a.box,node.box)
        if rel=="left" or rel=="right" or rel=="up" or rel=="down":
            edges["edges"].append({"head": node_a.id, "tail": node.id, "rel": rel})


def tree_builder(data):
    img_width=data["meta"]["image_size"]["width"]
    img_height = data["meta"]["image_size"]["height"]
    root=TreeNode(-1,[0,0,img_width,img_height])
    groups={}
    for item in data["valid_line"]:
        words_box=[]
        for word in item["words"]:
            quad=word["quad"]
            words_box.append(quad_to_box(quad))
        if item["group_id"] not in groups.keys():
            groups[item["group_id"]]=[]
        groups[item["group_id"]].append(TreeNode(item["id"], get_line_bbox(words_box)))
    for key in groups.keys():
        sorted_nodes=insertion_reorder(groups[key])
        child_tree_root=None
        for i,node in enumerate(sorted_nodes):
            if i==0:
                child_tree_root=node
                node.parent=root
                root.children.append(node)
                node.children=sorted_nodes[1:]
                continue
            else:
                node.parent=child_tree_root
                node_box_update(child_tree_root,node)
    sorted_group_nodes=insertion_reorder(root.children)
    root.children=sorted_group_nodes
    return root


def get_graph_and_reordered_nodes(root,data):
    sorted_valid_line=[]
    edges={"edges":[]}
    valid_line=data["valid_line"]
    for child in root.children:
        sorted_valid_line.append(get_item(child.id,valid_line))
        get_relationship(child,root.children,edges)
        for node in child.children:
            edges["edges"].append({"head": child.id, "tail": node.id, "rel": "child"})
            edges["edges"].append({"head": node.id, "tail": child.id, "rel": "parent"})
            sorted_valid_line.append(get_item(node.id, valid_line))
            get_relationship(node, child.children, edges)
    assert len(valid_line)==len(sorted_valid_line)
    data["valid_line"]=sorted_valid_line
    com_edge=[]
    for edge in edges["edges"]:
        com_edge.append(edge)
        if {"head": edge["tail"], "tail": edge["head"], "rel": OPPOSITE[edge["rel"]]} not in edges["edges"]:
            com_edge.append({"head": edge["tail"], "tail": edge["head"], "rel": OPPOSITE[edge["rel"]]})
    edges["edges"]=com_edge
    return edges,data


def graph_builder(path):
    data_types=["train", "dev", "test"]
    for data_type in data_types:
        json_path=os.path.join(path,data_type,"json")
        graph_path=os.path.join(path,data_type,"graph")
        if not os.path.exists(graph_path):
            os.mkdir(graph_path)
        reordered_json_path=os.path.join(path,data_type,"reordered_json")
        if not os.path.exists(reordered_json_path):
            os.mkdir(reordered_json_path)
        for filename in os.listdir(json_path):
            file_path=os.path.join(json_path,filename)
            reordered_file_path = os.path.join(reordered_json_path, filename)
            graph_file_path = os.path.join(graph_path, filename)
            data=json_loader(file_path)
            data=data_preprocess(data)
            tree=tree_builder(data)
            edges,reordered_data=get_graph_and_reordered_nodes(tree,data)
            assert len(data["valid_line"])==len(reordered_data["valid_line"])
            json_saver(edges,graph_file_path)
            json_saver(reordered_data,reordered_file_path)