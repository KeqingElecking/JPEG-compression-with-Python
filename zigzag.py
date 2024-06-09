import numpy as np
from collections import Counter, defaultdict
import heapq

# Helper function to perform zigzag scan
def zigzag_scan(block):
    zigzag_order = [
         0,  1,  5,  6, 14, 15, 27, 28,
         2,  4,  7, 13, 16, 26, 29, 42,
         3,  8, 12, 17, 25, 30, 41, 43,
         9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    ]
    return [block.flatten()[i] for i in zigzag_order]

# Helper function to perform run-length encoding
def run_length_encode(coeffs):
    rle = []
    zeros = 0
    for i in range(1, len(coeffs)):
        if coeffs[i] == 0:
            zeros += 1
        else:
            while zeros > 15:
                rle.append((15, 0))
                zeros -= 16
            rle.append((zeros, coeffs[i]))
            zeros = 0
    rle.append((0, 0))  # EOB
    return rle

# Helper function to build Huffman tree
def build_huffman_tree(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

# Helper function to generate Huffman codes from the Huffman tree
def generate_huffman_codes(huffman_tree):
    huffman_codes = {}
    for symbol, code in huffman_tree:
        huffman_codes[symbol] = code
    return huffman_codes

# Step 1: Prepare example data
block = np.array([
    [52, 55, 61, 66, 70, 61, 64, 73],
    [63, 59, 66, 90, 109, 85, 69, 72],
    [62, 59, 68, 113, 144, 104, 66, 73],
    [63, 58, 71, 122, 154, 106, 70, 69],
    [67, 61, 68, 104, 126, 88, 68, 70],
    [79, 65, 60, 70, 77, 68, 58, 75],
    [85, 71, 64, 59, 55, 61, 65, 83],
    [87, 79, 69, 68, 65, 76, 78, 94]
])

# Step 2: Perform zigzag scan and run-length encoding
zigzag_coeffs = zigzag_scan(block)
rle_coeffs = run_length_encode(zigzag_coeffs)

# Step 3: Calculate frequencies of DC differences and AC (RUNLENGTH, SIZE) pairs
previous_dc = 52
dc_diff = zigzag_coeffs[0] - previous_dc
dc_frequencies = Counter([dc_diff])

ac_frequencies = Counter(rle_coeffs)

# Step 4: Build Huffman trees
dc_huffman_tree = build_huffman_tree(dc_frequencies)
ac_huffman_tree = build_huffman_tree(ac_frequencies)

# Step 5: Generate Huffman codes
dc_huffman_codes = generate_huffman_codes(dc_huffman_tree)
ac_huffman_codes = generate_huffman_codes(ac_huffman_tree)

# Step 6: Encode the data using the generated Huffman codes
def huffman_encode(value, huffman_codes):
    return huffman_codes.get(value, '')

# Encoding the DC coefficient
dc_encoded = huffman_encode(dc_diff, dc_huffman_codes) + format(dc_diff, 'b')

# Encoding the AC coefficients
ac_encoded = ''.join([huffman_encode(pair, ac_huffman_codes) + format(pair[1], 'b') if pair[1] != 0 else huffman_encode(pair, ac_huffman_codes) for pair in rle_coeffs])

# Concatenate DC and AC encoded strings
encoded_block = dc_encoded + ac_encoded

print("Zigzag Scanned Coefficients:", zigzag_coeffs)
print("Run-Length Encoded Coefficients:", rle_coeffs)
print("DC Huffman Codes:", dc_huffman_codes)
print("AC Huffman Codes:", ac_huffman_codes)
print("DC Encode", ac_encoded)
print("Huffman Encoded Block:", encoded_block)