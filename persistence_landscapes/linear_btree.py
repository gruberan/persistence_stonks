import itertools

class _LinearNode:
    def __init__(self,x,y,m=None):
        self.left, self.right = None, None
        self.x, self.y = x, y
        if not m == None:
            self.m, self.b = m, y - m*x
    def get_prev(self,root):
        """Returns in-order previous node."""
        if not self.left == None:
            return self.left.get_rightmost()
        prev = None
        while not root == None:
            if self.x > root.x:
                prev = root
                root = root.right
            elif self.x < root.x:
                root = root.left
            else:
                break
        return prev
    def get_next(self,root):
        """Returns in-order successor node."""
        if not self.right == None:
            return self.right.get_leftmost()
        succ = None
        while not root == None:
            if self.x < root.x:
                succ = root
                root = root.left
            elif self.x > root.x:
                root = root.right
            else:
                break
        return succ
    def get_leftmost(self):
        """Returns leftmost node of subtree with root self."""
        current = self
        while not current == None:
            if current.left == None:
                break
            current = current.left
        return current
    def get_rightmost(self):
        """Returns rightmost node of subtree with root self."""
        current = self
        while not current == None:
            if current.right == None:
                break
            current = current.right
        return current
    def __iter__(self):
        """Ordered traversal."""
        if self.left:
            for node in self.left:
                yield node
        yield self
        if self.right:
            for node in self.right:
                yield node
    def _pairwise_iterator(self):
        """Ordered traversal with consecutive pairs."""
        i, j = itertools.tee(self)
        next(j,None)
        return zip(i,j)
    
class Linear_BTree:
    """Binary search tree class stores linear parameters to successor node. Capable of linear interpolation between nodes."""
    
    def __init__(self):
        self.root = None
        
    def insert(self, x, y, m=None, delay_update=False):
        """Inserts a new node into the tree.
        
        x, y: Coordinates of node to insert
        m: slope to next node. Can be taken as an arg if precomputed (such as when converting a list)
        delay_update (bool): if True, will not update the linear parameters of adjacent node after insert
        """
        if self.root == None:
            self.root = _LinearNode(x,y,m)
        else:
            self._insert(self.root,x,y,m,delay_update)
            
    def _insert(self,node, x, y, m=None, delay_update=False): 
        if x < node.x:
            if node.left == None:
                node.left = _LinearNode(x,y,m)
                if not delay_update and m == None:
                    # Update linear parameters for new node
                    node.left.m = (node.y - y)/(node.x - x)
                    node.left.b = y - node.left.m * x
                if not delay_update:
                    # Update linear parameters for node previous to new node
                    prev = node.left.get_prev(self.root)
                    if not prev == None:
                        prev.m = (y - prev.y)/(x - prev.x)
                        prev.b = prev.y - prev.m * prev.x
            else:
                self._insert(node.left,x,y,m)
        elif x > node.x:
            if node.right == None:
                node.right = _LinearNode(x,y,m)
                if not delay_update and m == None:
                    # Update linear parameters for new node
                    succ = node.right.get_next(self.root)
                    if not succ == None:
                        node.right.m = (succ.y - y)/(succ.x - x)
                        node.right.b = y - node.right.m * x
                if not delay_update:
                    # Update linear parameters for current node
                    node.m = (y - node.y)/(x - node.x)
                    node.b = node.y - node.m * node.x
            else:
                self._insert(node.right,x,y,m)
        else:
            # Overwrites if node with same x value already exists
            if not (node.y == y) or not delay_update:
                node.y = y
                if m == None:
                    # Update linear parameters for successor node
                    succ = node.get_next(self.root)
                    if not succ == None:
                        node.m = (succ.y - y)/(succ.x - x)
                        node.b = y - node.m * x
                else:
                    node.m, node.b = m, y - m * x
                # Update linear parameters for previous node
                prev = node.get_prev(self.root)
                if not prev == None:
                    prev.m = (y - prev.y)/(x - prev.x)
                    prev.b = prev.y - prev.m * prev.x
                    
    @classmethod
    def from_list(cls, X, Y, already_sorted=False):
        """Returns a new linear binary tree.
        
        Arguments:
            X (list): X values
            Y (list): Y values
            already_sorted (list): indicates that X and Y are already in sorted order by X value
        """
        new_lbtree = cls()
        if already_sorted:
            M = [(y2 - y1)/(x2 - x1) if x1 != x2 else 0.0 for x1, x2, y1, y2 in zip(X,X[1:],Y,Y[1:])]
            new_lbtree._from_list(list(zip(X, Y, M+[0.0])), 0, len(X)-1)
        else:
            new_lbtree._from_list([(x,y,None) for x,y in zip(X,Y)], 0, len(X)-1)
            new_lbtree._update()
        return new_lbtree
        
    def _from_list(self, nodes, a, b):
        if a > b:
            return
        mid = int(a + (b - a) / 2)
        self.insert(nodes[mid][0], nodes[mid][1], nodes[mid][2], delay_update=True)
        self._from_list(nodes, a, mid-1)
        self._from_list(nodes, mid+1, b)
        
    def _update(self):
        """Updates the slope and intercept for all nodes."""
        for node1, node2 in self.root._pairwise_iterator():
            node1.m = (node2.y - node1.y)/(node2.x - node1.x)
            node1.b = node1.y - node1.m * node1.x
        self.root.get_rightmost().m, self.root.get_rightmost().b = 0.0, 0.0
    
    def evaluate(self, x):
        """ Find largest node.x below x and return linear interpolation. """
        return self._evaluate(x, self.root)
        
    def _evaluate(self, x, node):
        if node == None:
            return None
        if x == node.x:
            return node.y
        if x > node.x:
            y = self._evaluate(x,node.right)
            if y == None:
                y = (node.m)*x + node.b
            return y
        if x < node.x:
            return self._evaluate(x,node.left)
    
    def deepcopy(self):
        new_lbtree = Linear_BTree()
        for node in self:
            new_lbtree.insert(node.x, node.y, node.m, delay_update=True)
        return new_lbtree
    
    def __iter__(self):
        """ Yields sorted x values. """
        for node in self.root:
            yield node.x