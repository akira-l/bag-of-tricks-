import os
import pickle

import pdb

class Node:

    def __init__(self,freq):
        self.left = None
        self.right = None
        self.father = None
        self.freq = freq
    def isLeft(self):
        return self.father.left == self

def createNodes(freqs):
    return [Node(freq) for freq in freqs]

def createHuffmanTree(nodes):
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item:item.freq)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.freq + node_right.freq)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]

def huffmanEncoding(nodes,root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:
            if node_tmp.isLeft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes

if __name__ == '__main__':
    #chars = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
    #freqs = [10,4,2,5,3,4,2,6,4,4,3,7,9,6]
    count_filepath = '../ali_product/train_count.pkl'
    count_data = pickle.load(open(count_filepath, 'rb'))
    chars_freqs = []
    for cls_key in count_data.keys():
        chars_freqs.append((cls_key, 1))#count_data[cls_key]))
    nodes = createNodes([item[1] for item in chars_freqs])
    root = createHuffmanTree(nodes)
    codes = huffmanEncoding(nodes,root)

    gather = []
    save_dict = {}
    for item in zip(chars_freqs,codes):
        print('Character:%s freq:%-2d   encoding: %s' % (item[0][0],item[0][1],item[1]))
        save_dict[item[0][0]] = item[1] 
        gather.append(len(str(item[1])))
    print(max(gather))


    '''
    with open('huffman_save.pkl', 'wb') as handle:
        pickle.dump(save_dict, handle)
    '''
