# pbft/pbft_message.py
class PBFTMessage:
    def __init__(self, message_type, view, sequence_number, block, sender):
        self.message_type = message_type
        self.view = view
        self.sequence_number = sequence_number
        self.block = block
        self.sender = sender
