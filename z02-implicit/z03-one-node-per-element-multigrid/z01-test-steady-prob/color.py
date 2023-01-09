'''
distance-2 coloring
'''
def color2(fina, cola, nnode):
    import numpy as np 
    # input: nnode is number of nodes to color
    undone = True 
    ncolor = 0
    need_color = np.ones((nnode), dtype=bool)
    whichc = np.zeros((nnode), dtype=int)
    while undone:
        ncolor = ncolor + 1
        for inode in range(nnode):
            if (not need_color[inode]):
                continue # this node is already colored
            lcol = False # this is a flag to indicate if we've found a node of the same color within distance 2
            for count in range(fina[inode], fina[inode+1]):
                jnode = cola[count]
                if (whichc[jnode]==ncolor) : 
                    lcol=True # found a node of same color in distance 1
                    break 
                for count2 in range(fina[jnode], fina[jnode+1]):
                    if (whichc[cola[count2]]==ncolor) : 
                        lcol=True # found a node of smae color in distance 2
                        break 
                if (lcol): 
                    break 
            if ( not lcol):
                whichc[inode] = ncolor
            # print('===ncolor',ncolor,'===inode',inode,'====\n')
            # print(whichc)
            # print(need_color)
            # print(undone)
        need_color = (whichc==0)
        undone = np.any(need_color)
    return whichc, ncolor