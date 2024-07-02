# blockchain/blockchain.py
import hashlib
import time
from pbft.pbft_node import PBFTNode
from pbft.network import Network

class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash

class Blockchain:
    def __init__(self):
        self.chain = []
        self.network = Network()  # Initialize the network attribute
        self.create_genesis_block()
        self.initialize_pbft_nodes()

    def create_genesis_block(self):
        genesis_block = Block(0, "0", time.time(), "Genesis Block", self.hash_block(0, "0", time.time(), "Genesis Block"))
        self.chain.append(genesis_block)

    def initialize_pbft_nodes(self):
        if not hasattr(self, 'network'):
            self.network = Network()  # Initialize the network if it doesn't exist
        self.nodes = [PBFTNode(i, self.network) for i in range(4)]
        for node in self.nodes:
            self.network.add_node(node)

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, data):
        previous_block = self.get_latest_block()
        new_index = previous_block.index + 1
        new_timestamp = time.time()
        new_hash = self.hash_block(new_index, previous_block.hash, new_timestamp, data)
        new_block = Block(new_index, previous_block.hash, new_timestamp, data, new_hash)

        primary_node = self.nodes[0]  # Assuming the first node is primary
        primary_node.pre_prepare(new_block)
        
        # Assuming all nodes eventually reach consensus and add the block
        self.chain.append(new_block)
        return new_block

    def hash_block(self, index, previous_hash, timestamp, data):
        value = f"{index}{previous_hash}{timestamp}{data}"
        return hashlib.sha256(value.encode()).hexdigest()

    def print_chain(self):
        for block in self.chain:
            print(f"Block(index={block.index}, previous_hash={block.previous_hash}, timestamp={block.timestamp}, data={block.data}, hash={block.hash})")

    def get_block_by_hash(self, block_hash):
        for block in self.chain:
            if block.hash == block_hash:
                return block
        return None

    def get_block_by_index(self, block_index):
        for block in self.chain:
            if block.index == block_index:
                return block
        return None
