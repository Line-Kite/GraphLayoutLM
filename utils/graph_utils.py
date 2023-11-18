rel2ids = {
    "parent":1,
    'child':2,
    'up':3,
    'down':4,
    'left':5,
    'right':6,
}

def set_nodes(node_ids):
    nodes=[]
    new_node_ids=[]
    i=0
    while i < len(node_ids):
        node=[-1,-1]
        node[0]=i
        temp_id=node_ids[i]
        while temp_id==node_ids[i]:
            node[1]=i
            i+=1
            if i==len(node_ids):
                break
        nodes.append(node)
        new_node_ids.append(temp_id)
    return nodes,new_node_ids

def set_edges(edges_data,node_ids):
    edges=[]
    for i in range(len(edges_data["head"])):
        if (node_ids.count(edges_data["head"][i])!=0)and(node_ids.count(edges_data["tail"][i])!=0):
            new_edge=[node_ids.index(edges_data["head"][i]),node_ids.index(edges_data["tail"][i]),rel2ids[edges_data["rel"][i]]]
            edges.append(new_edge)
    return edges 