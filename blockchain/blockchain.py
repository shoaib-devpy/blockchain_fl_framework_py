# blockchain/blockchain.py
import hashlib
import time

class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash

    def __repr__(self):
        return f"Block(index={self.index}, previous_hash={self.previous_hash}, timestamp={self.timestamp}, data={self.data}, hash={self.hash})"

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        genesis_data = "Genesis Block"
        return Block(0, "0", time.time(), genesis_data, self.hash_block(genesis_data, "0", time.time()))

    def hash_block(self, data, previous_hash, timestamp):
        sha = hashlib.sha256()
        sha.update(f"{data}{previous_hash}{timestamp}".encode('utf-8'))
        return sha.hexdigest()

    def add_block(self, data):
        last_block = self.chain[-1]
        new_block = Block(len(self.chain), last_block.hash, time.time(), data, self.hash_block(data, last_block.hash, time.time()))
        self.chain.append(new_block)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            # Check if the current block's hash is valid
            if current_block.hash != self.hash_block(current_block.data, current_block.previous_hash, current_block.timestamp):
                return False

            # Check if the current block's previous_hash is correct
            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def print_chain(self):
        for block in self.chain:
            print(block)
