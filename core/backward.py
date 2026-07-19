class BackwardEngine:
    def run_backward(self, tensor, grad, retain_grad=False):
        '''
        This function is used to run backward propagation.
        retain_grad: if retain_grad is True, the gradient of each tensor will be retained.
        '''
        ops = []
        op_set = set()

        def add_op(op):
            if op is None or op in op_set:
                return
            op_set.add(op)

            for i, queued_op in enumerate(ops):
                if op.rank < queued_op.rank:
                    ops.insert(i, op)
                    return
            ops.append(op)

        add_op(tensor.creator)

        while ops:
            op = ops.pop()

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
                    add_op(x.creator)  # add the creator of x to the graph

            if not retain_grad:
                for x in op.outputs:
                    x().grad = None  # access the weak ref object
