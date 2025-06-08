import networkx as nx
import numpy as np
from collections import defaultdict
from N2G import drawio_diagram

from ...probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit


class DrawIoExporter:
    """
    Export a probabilistic circuit to a draw.io diagram (.drawio XML).
    """

    def __init__(self, model: ProbabilisticCircuit):
        self.model = model

    # ------------------------------------------------------------------ #
    # Fallback: guaranteed positions for *all* nodes                     #
    # ------------------------------------------------------------------ #
    def _layer_layout(self):
        """
        Return {node: (x, y)} using the circuit's BFS layers starting
        from every in-degree-0 node.  x = layer index, y = row (-down).
        """
        roots = [n for n in self.model if self.model.in_degree(n) == 0]
        layer_map = defaultdict(list)

        for root in roots:
            for depth, layer in enumerate(self.model.bfs_layers(root)):
                layer_map[depth].extend(layer)

        positions = {}
        for x, nodes in sorted(layer_map.items()):
            for y, node in enumerate(nodes):
                positions[node] = (x, -y)          # y downward for readability
        return positions

    # ------------------------------------------------------------------ #
    # Public export method                                               #
    # ------------------------------------------------------------------ #
    def export(self) -> drawio_diagram:
        # ---------- 1. Choose a layout -------------------------------- #
        try:
            pos = nx.drawing.bfs_layout(self.model, self.model.root)
            if len(pos) != len(self.model):           # some nodes missing
                raise nx.NetworkXError
        except nx.NetworkXError:
            pos = self._layer_layout()

        # ---------- 2. Build the diagram ------------------------------ #
        unique_id = 1000
        diagram   = drawio_diagram()
        diagram.add_diagram("Structure", width=1360, height=1864)

        for unit, (x, y) in pos.items():
            diagram.add_node(id=str(hash(unit)),
                             x_pos=x * 100, y_pos=y * 100,
                             **unit.drawio_style)
            if unit.is_leaf:
                txt_style = ("text;html=1;align=left;verticalAlign=middle;"
                             "whiteSpace=wrap;rounded=0;fontFamily=Helvetica;"
                             "fontSize=12;fontColor=default;")
                diagram.add_node(id=str(unique_id),
                                 x_pos=x * 100 + 35, y_pos=y * 100,
                                 style=txt_style,
                                 label=unit.variables[0].name,
                                 width=100, height=30)
                unique_id += 1
            else:
                diagram.current_root[-1].attrib["label"] = ""

        # ---------- 3. Edges ------------------------------------------ #
        for u, v in self.model.unweighted_edges:
            diagram.add_link(str(hash(u)), str(hash(v)),
                             style='endArrow=classic;html=1;rounded=0;')

        for u, v, w in self.model.log_weighted_edges:
            diagram.add_link(str(hash(u)), str(hash(v)),
                             label=f"{round(w, 2)}",
                             style=(f'endArrow=classic;html=1;rounded=0;'
                                    f'opacity={np.exp(w) * 100};'))

        return diagram
