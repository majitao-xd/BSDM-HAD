import numpy as np


def run_rx(data):
    """
    :param data: HSI with M * N * B
    :return: RX result with M * N
    """
    M, N, B = data.shape
    data = np.reshape(data, [M*N, B]).T
    mu = np.mean(data, axis=1)
    sigma = np.cov(data.T, rowvar=False)
    z = data - mu[:, np.newaxis]
    sig_inv = np.linalg.pinv(sigma)
    dist_data = np.zeros(M*N)
    for i in range(M*N):
        dist_data[i] = z[:, i].T @ sig_inv @ z[:, i]
    dist_data = np.reshape(dist_data, [M, N])

    return dist_data