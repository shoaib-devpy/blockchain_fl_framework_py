# pbft/pbft_node.py
from pbft.pbft_message import PBFTMessage

class PBFTNode:
    def __init__(self, node_id, network):
        self.node_id = node_id
        self.network = network
        self.state = 'normal'
        self.log = []
        self.view = 0

    def send_message(self, message):
        self.network.broadcast(message)

    def receive_message(self, message):
        if message.message_type == 'pre-prepare':
            self.prepare(message)
        elif message.message_type == 'prepare':
            self.commit(message)
        elif message.message_type == 'commit':
            self.apply_block(message.block)

    def pre_prepare(self, block):
        message = PBFTMessage('pre-prepare', self.view, len(self.log), block, self.node_id)
        self.send_message(message)
        self.log.append(message)

    def prepare(self, message):
        prepare_message = PBFTMessage('prepare', message.view, message.sequence_number, message.block, self.node_id)
        self.send_message(prepare_message)
        self.log.append(prepare_message)

    def commit(self, message):
        commit_message = PBFTMessage('commit', message.view, message.sequence_number, message.block, self.node_id)
        self.send_message(commit_message)
        self.log.append(commit_message)

    def apply_block(self, block):
        # Implementation to apply the block to the blockchain
        pass
