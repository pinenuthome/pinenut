class BackwardEngine:
    def __init__(self):
        self.ops = []
        self.op_set = set()

    def add_op(self, op):
        '''
        This function is used to add an operator to the graph.
        insert op into self.ops list, and keep the order of the list.
        '''
        if op in self.op_set:
            return
        self.op_set.add(op)

        n = len(self.ops)
        for i in range(n):
            if op.rank < self.ops[i].rank:
                self.ops.insert(i, op)
                return
        self.ops.append(op)

    def run_backward(self, tensor, grad, retain_grad=False):
        '''
        This function is used to run backward propagation.
        '''
        self.add_op(tensor.creator)

        while len(self.ops) > 0:
            op = self.ops.pop()

            outputs = tuple(output() for output in op.outputs)  # access the weakref object
            out_grad = tuple(tensor.grad for tensor in outputs)

            grad_xs = op.backward(*out_grad)  # call the backward function of the operator
            if not isinstance(grad_xs, (tuple, list)):
                grad_xs = (grad_xs,)  # make sure grad_xs is a tuple

            for x, grad_x in zip(op.inputs, grad_xs):
                if x.grad is None:
                    x.grad = grad_x # set the gradient
                else:
                    x.grad = x.grad + grad_x # accumulate the gradient

                if x.creator is not None:
                    self.add_op(x.creator)  # add the creator of x to the graph

            if not retain_grad:
                for x in op.outputs:
                    x().grad = None  # access the weak ref object
