network_config = {
    'resnet34': {
        'net_dim': [64] * 4 + [128] * 4 + [256] * 6  + [512] * 3,
        1: [],  # End-to-end
        17: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3],
             [2, 0, 4], [2, 1, 5], [2, 2, 6], [2, 3, 7],
             [3, 0, 8], [3, 1, 9], [3, 2, 10], [3, 3, 11], [3, 4, 12], [3, 5, 13],
             [4, 0, 14], [4, 1, 15]],
    },
    'resnet101': {
        'net_dim': [64] + [64] * 3 + [128] * 4 + [256] * 23 + [512] * 3,
        1: [],  # End-to-end
        34: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3],
             [2, 0, 4], [2, 1, 5], [2, 2, 6], [2, 3, 7],
             [3, 0, 8], [3, 1, 9], [3, 2, 10], [3, 3, 11], [3, 4, 12], [3, 5, 13], [3, 6, 14], [3, 7, 15], [3, 8, 16], [3, 9, 17], [3, 10, 18], [3, 11, 19], [3, 12, 20], [3, 13, 21], [3, 14, 22], [3, 15, 23], [3, 16, 24], [3, 17, 25], [3, 18, 26], [3, 19, 27], [3, 20, 28], [3, 21, 29], [3, 22, 30],
             [4, 0, 31], [4, 1, 32],
             ],
    },
    'vgg13': {
        'net_dim': [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2 + [512] * 2,
        1: [],
        10: [0, 1,
             3, 4,
             6, 7,
             9, 10,
             12,]
    },
    'vgg19': {
        'net_dim': [64] * 2 + [128] * 2 + [256] * 4 + [512] * 4 + [512] * 4,
        1: [],
        16: [0, 1,
             3, 4,
             6, 7, 8, 9,
             11, 12, 13, 14,
             16, 17, 18]
    },
    'resnet50': {
        'net_dim': [64] + [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3,
        1: [],  # End-to-end
        17: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3],
             [2, 0, 4], [2, 1, 5], [2, 2, 6], [2, 3, 7],
             [3, 0, 8], [3, 1, 9], [3, 2, 10], [3, 3, 11], [3, 4, 12], [3, 5, 13],
             [4, 0, 14], [4, 1, 15]],

    }
}