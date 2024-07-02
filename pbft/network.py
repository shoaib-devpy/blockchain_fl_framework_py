# pbft/network.py
class Network:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def broadcast(self, message):
        for node in self.nodes:
            node.receive_message(message)
