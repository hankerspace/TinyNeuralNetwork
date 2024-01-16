import struct

def int_to_bits(integer, bits):
    return [integer >> i & 1 for i in range(bits - 1, -1, -1)]

def bits_to_int(bits):
    return sum(b << i for i, b in enumerate(reversed(bits)))


def float_to_bits(f):
    packed = struct.pack('!d', f)
    return [int(bit) for byte in packed for bit in f'{byte:08b}']


def bits_to_float(b):
    bytes_ = bytes(int(''.join(str(bit) for bit in b[i:i+8]), 2) for i in range(0, len(b), 8))
    return struct.unpack('!d', bytes_)[0]

def bool_to_bits(b):
    if b:
        return [1]
    else:
        return [0]

def bits_to_bool(b):
    return b[0] == 1

