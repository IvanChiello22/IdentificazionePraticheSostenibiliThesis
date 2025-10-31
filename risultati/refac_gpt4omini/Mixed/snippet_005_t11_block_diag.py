import numpy as np
from scipy.sparse import block_diag, random as sprand

def create_block_diagonal_matrix(num_blocks, block_size, density):
    blocks = [sprand(block_size, block_size, density=density, format='csr') for _ in range(num_blocks)]
    return block_diag(blocks, format='csr')

def main():
    num_blocks = 8
    block_size = 200
    density = 0.01
    A = create_block_diagonal_matrix(num_blocks, block_size, density)
    x = np.random.randn(A.shape[1])
    y = A @ x
    print(A.shape, A.nnz, y.shape)

if __name__ == "__main__":
    main()