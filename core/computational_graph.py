import subprocess
import webbrowser
from pathlib import Path

from pinenut.core import Tensor


class ComputationalGraph:
    def __init__(self, tensor: Tensor, comment=None, to_file=None, view=True, verbose=False):
        self.tensor = tensor
        self.visited = set()
        self.view = view
        self.verbose = verbose
        self.dot = Digraph(comment=comment, filename=to_file)

    def build(self):
        self.build_graph()
        self.dot.render(self.view)

    def build_graph(self):
        self._build_graph(self.tensor)

    def _build_graph(self, tensor=None):
        if tensor is None or id(tensor) in self.visited:
            return

        if self.verbose:
            self.dot.node(id(tensor), tensor.label, type='tensor')
        else:
            self.dot.node(id(tensor), tensor.name, type='tensor')

        self.visited.add(id(tensor))

        creator = tensor.creator
        if creator is None:
            return

        self.dot.node(id(creator), creator.name, type='operator')
        self.dot.edge(id(creator), id(tensor))

        inputs = creator.inputs
        for x in inputs:
            self.dot.edge(id(x), id(creator))
            self._build_graph(x)


class Digraph:
    def __init__(self, comment, filename=None):
        self.comment = comment
        self.filename = Path(filename or 'computational_graph.png')
        self.nodes = []
        self.edges = []

    def node(self, id, name, type):
        self.nodes.append((id, name, type))

    def edge(self, from_node, to_node):
        self.edges.append((from_node, to_node))

    def render(self, view):
        dot_path = Path(str(self.filename) + '.dot')
        with dot_path.open('w') as f:
            f.write('digraph g {\n')
            if self.comment is not None:
                f.write('label="%s"\n' % self.comment)

            for node in self.nodes:
                if node[2] == 'tensor':
                    f.write('%s [label="%s", shape=oval, color=pink, style=filled]\n' % (node[0], node[1]))
                else:
                    f.write('%s [label="%s", shape=box, color=lightblue, style=filled]\n' % (node[0], node[1]))

            for edge in self.edges:
                f.write('%s -> %s\n' % (edge[0], edge[1]))
            f.write('}\n')
        try:
            subprocess.run(
                ['dot', '-Tpng', str(dot_path), '-o', str(self.filename)],
                check=True)
        except FileNotFoundError as error:
            raise RuntimeError('Graphviz is required to render a graph') from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError('Graphviz failed to render the graph') from error

        if view:
            webbrowser.open(self.filename.resolve().as_uri())


def build_graph(tensor: Tensor, to_file=None, verbose=True, view=True, comment=None):
    graph = ComputationalGraph(tensor, to_file=to_file, verbose=verbose, view=view, comment=comment)
    graph.build()
