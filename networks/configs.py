network_config = {
    'resnet32': {
        'net_dim': [16] * 6 + [32] * 5 + [64] * 5,
        'net_config': [[0, 0, 0],
                       [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5],
                       [2, 0, 6], [2, 1, 7], [2, 2, 8], [2, 3, 9], [2, 4, 10],
                       [3, 0, 11], [3, 1, 12], [3, 2, 13], [3, 3, 14], [3, 4, 15]],
        16: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5],
             [2, 0, 6], [2, 1, 7], [2, 2, 8], [2, 3, 9], [2, 4, 10],
             [3, 0, 11], [3, 1, 12], [3, 2, 13], [3, 3, 14]],
        2: [[0, 0, 0],
            ],
        3: [[0, 0, 0],
            [1, 0, 1]],
        4: [[0, 0, 0],
            [1, 0, 1], [1, 1, 2]],
        5: [[0, 0, 0],
            [1, 0, 1], [1, 1, 2], [1, 2, 3]],
        6: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4]],
        7: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5],
             ],
        8: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5],
             [2, 0, 6]],
        9: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5],
             [2, 0, 6], [2, 1, 7]],
        10: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5],
             [2, 0, 6], [2, 1, 7], [2, 2, 8]],
        11: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5],
             [2, 0, 6], [2, 1, 7], [2, 2, 8], [2, 3, 9]],
        12: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5],
             [2, 0, 6], [2, 1, 7], [2, 2, 8], [2, 3, 9], [2, 4, 10]],
        13: [[0, 0, 0],
            [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5],
            [2, 0, 6], [2, 1, 7], [2, 2, 8], [2, 3, 9], [2, 4, 10],
            [3, 0, 11]],

    },
    'resnet110': {
        'net_dim': [16] * 19 + [32] * 18 + [64] * 18,
        'net_config': [[0, 0, 0],
                       [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 7], [1, 7, 8],
                       [1, 8, 9],
                       [1, 9, 10], [1, 10, 11], [1, 11, 12], [1, 12, 13], [1, 13, 14], [1, 14, 15], [1, 15, 16],
                       [1, 16, 17],
                       [1, 17, 18],
                       [2, 0, 19], [2, 1, 20], [2, 2, 21], [2, 3, 22], [2, 4, 23], [2, 5, 24], [2, 6, 25], [2, 7, 26],
                       [2, 8, 27],
                       [2, 9, 28], [2, 10, 29], [2, 11, 30], [2, 12, 31], [2, 13, 32], [2, 14, 33], [2, 15, 34],
                       [2, 16, 35],
                       [2, 17, 36],
                       [3, 0, 37], [3, 1, 38], [3, 2, 39], [3, 3, 40], [3, 4, 41], [3, 5, 42], [3, 6, 43], [3, 7, 44],
                       [3, 8, 45],
                       [3, 9, 46], [3, 10, 47], [3, 11, 48], [3, 12, 49], [3, 13, 50], [3, 14, 51], [3, 15, 52],
                       [3, 16, 53], [3, 17, 54],
                       ],
        55: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 7], [1, 7, 8],
             [1, 8, 9],
             [1, 9, 10], [1, 10, 11], [1, 11, 12], [1, 12, 13], [1, 13, 14], [1, 14, 15], [1, 15, 16],
             [1, 16, 17],
             [1, 17, 18],
             [2, 0, 19], [2, 1, 20], [2, 2, 21], [2, 3, 22], [2, 4, 23], [2, 5, 24], [2, 6, 25], [2, 7, 26],
             [2, 8, 27],
             [2, 9, 28], [2, 10, 29], [2, 11, 30], [2, 12, 31], [2, 13, 32], [2, 14, 33], [2, 15, 34],
             [2, 16, 35],
             [2, 17, 36],
             [3, 0, 37], [3, 1, 38], [3, 2, 39], [3, 3, 40], [3, 4, 41], [3, 5, 42], [3, 6, 43], [3, 7, 44],
             [3, 8, 45],
             [3, 9, 46], [3, 10, 47], [3, 11, 48], [3, 12, 49], [3, 13, 50], [3, 14, 51], [3, 15, 52],
             [3, 16, 53],
             ],
        54: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 7], [1, 7, 8],
             [1, 8, 9],
             [1, 9, 10], [1, 10, 11], [1, 11, 12], [1, 12, 13], [1, 13, 14], [1, 14, 15], [1, 15, 16],
             [1, 16, 17],
             [1, 17, 18],
             [2, 0, 19], [2, 1, 20], [2, 2, 21], [2, 3, 22], [2, 4, 23], [2, 5, 24], [2, 6, 25], [2, 7, 26],
             [2, 8, 27],
             [2, 9, 28], [2, 10, 29], [2, 11, 30], [2, 12, 31], [2, 13, 32], [2, 14, 33], [2, 15, 34],
             [2, 16, 35],
             [2, 17, 36],
             [3, 0, 37], [3, 1, 38], [3, 2, 39], [3, 3, 40], [3, 4, 41], [3, 5, 42], [3, 6, 43], [3, 7, 44],
             [3, 8, 45],
             [3, 9, 46], [3, 10, 47], [3, 11, 48], [3, 12, 49], [3, 13, 50], [3, 14, 51], [3, 15, 52],
             ],
        53: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 7], [1, 7, 8],
             [1, 8, 9],
             [1, 9, 10], [1, 10, 11], [1, 11, 12], [1, 12, 13], [1, 13, 14], [1, 14, 15], [1, 15, 16],
             [1, 16, 17],
             [1, 17, 18],
             [2, 0, 19], [2, 1, 20], [2, 2, 21], [2, 3, 22], [2, 4, 23], [2, 5, 24], [2, 6, 25], [2, 7, 26],
             [2, 8, 27],
             [2, 9, 28], [2, 10, 29], [2, 11, 30], [2, 12, 31], [2, 13, 32], [2, 14, 33], [2, 15, 34],
             [2, 16, 35],
             [2, 17, 36],
             [3, 0, 37], [3, 1, 38], [3, 2, 39], [3, 3, 40], [3, 4, 41], [3, 5, 42], [3, 6, 43], [3, 7, 44],
             [3, 8, 45],
             [3, 9, 46], [3, 10, 47], [3, 11, 48], [3, 12, 49], [3, 13, 50], [3, 14, 51]
             ],
        52: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 7], [1, 7, 8],
             [1, 8, 9],
             [1, 9, 10], [1, 10, 11], [1, 11, 12], [1, 12, 13], [1, 13, 14], [1, 14, 15], [1, 15, 16],
             [1, 16, 17],
             [1, 17, 18],
             [2, 0, 19], [2, 1, 20], [2, 2, 21], [2, 3, 22], [2, 4, 23], [2, 5, 24], [2, 6, 25], [2, 7, 26],
             [2, 8, 27],
             [2, 9, 28], [2, 10, 29], [2, 11, 30], [2, 12, 31], [2, 13, 32], [2, 14, 33], [2, 15, 34],
             [2, 16, 35],
             [2, 17, 36],
             [3, 0, 37], [3, 1, 38], [3, 2, 39], [3, 3, 40], [3, 4, 41], [3, 5, 42], [3, 6, 43], [3, 7, 44],
             [3, 8, 45],
             [3, 9, 46], [3, 10, 47], [3, 11, 48], [3, 12, 49], [3, 13, 50]
             ],
        51: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 7], [1, 7, 8],
             [1, 8, 9],
             [1, 9, 10], [1, 10, 11], [1, 11, 12], [1, 12, 13], [1, 13, 14], [1, 14, 15], [1, 15, 16],
             [1, 16, 17],
             [1, 17, 18],
             [2, 0, 19], [2, 1, 20], [2, 2, 21], [2, 3, 22], [2, 4, 23], [2, 5, 24], [2, 6, 25], [2, 7, 26],
             [2, 8, 27],
             [2, 9, 28], [2, 10, 29], [2, 11, 30], [2, 12, 31], [2, 13, 32], [2, 14, 33], [2, 15, 34],
             [2, 16, 35],
             [2, 17, 36],
             [3, 0, 37], [3, 1, 38], [3, 2, 39], [3, 3, 40], [3, 4, 41], [3, 5, 42], [3, 6, 43], [3, 7, 44],
             [3, 8, 45],
             [3, 9, 46], [3, 10, 47], [3, 11, 48], [3, 12, 49]
             ],
        50: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 7], [1, 7, 8],
             [1, 8, 9],
             [1, 9, 10], [1, 10, 11], [1, 11, 12], [1, 12, 13], [1, 13, 14], [1, 14, 15], [1, 15, 16],
             [1, 16, 17],
             [1, 17, 18],
             [2, 0, 19], [2, 1, 20], [2, 2, 21], [2, 3, 22], [2, 4, 23], [2, 5, 24], [2, 6, 25], [2, 7, 26],
             [2, 8, 27],
             [2, 9, 28], [2, 10, 29], [2, 11, 30], [2, 12, 31], [2, 13, 32], [2, 14, 33], [2, 15, 34],
             [2, 16, 35],
             [2, 17, 36],
             [3, 0, 37], [3, 1, 38], [3, 2, 39], [3, 3, 40], [3, 4, 41], [3, 5, 42], [3, 6, 43], [3, 7, 44],
             [3, 8, 45],
             [3, 9, 46], [3, 10, 47], [3, 11, 48]
             ],
        49: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 7], [1, 7, 8],
             [1, 8, 9],
             [1, 9, 10], [1, 10, 11], [1, 11, 12], [1, 12, 13], [1, 13, 14], [1, 14, 15], [1, 15, 16],
             [1, 16, 17],
             [1, 17, 18],
             [2, 0, 19], [2, 1, 20], [2, 2, 21], [2, 3, 22], [2, 4, 23], [2, 5, 24], [2, 6, 25], [2, 7, 26],
             [2, 8, 27],
             [2, 9, 28], [2, 10, 29], [2, 11, 30], [2, 12, 31], [2, 13, 32], [2, 14, 33], [2, 15, 34],
             [2, 16, 35],
             [2, 17, 36],
             [3, 0, 37], [3, 1, 38], [3, 2, 39], [3, 3, 40], [3, 4, 41], [3, 5, 42], [3, 6, 43], [3, 7, 44],
             [3, 8, 45],
             [3, 9, 46], [3, 10, 47]
             ],
        48: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 7], [1, 7, 8],
             [1, 8, 9],
             [1, 9, 10], [1, 10, 11], [1, 11, 12], [1, 12, 13], [1, 13, 14], [1, 14, 15], [1, 15, 16],
             [1, 16, 17],
             [1, 17, 18],
             [2, 0, 19], [2, 1, 20], [2, 2, 21], [2, 3, 22], [2, 4, 23], [2, 5, 24], [2, 6, 25], [2, 7, 26],
             [2, 8, 27],
             [2, 9, 28], [2, 10, 29], [2, 11, 30], [2, 12, 31], [2, 13, 32], [2, 14, 33], [2, 15, 34],
             [2, 16, 35],
             [2, 17, 36],
             [3, 0, 37], [3, 1, 38], [3, 2, 39], [3, 3, 40], [3, 4, 41], [3, 5, 42], [3, 6, 43], [3, 7, 44],
             [3, 8, 45],
             [3, 9, 46]
             ],
        47: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], [1, 6, 7], [1, 7, 8],
             [1, 8, 9],
             [1, 9, 10], [1, 10, 11], [1, 11, 12], [1, 12, 13], [1, 13, 14], [1, 14, 15], [1, 15, 16],
             [1, 16, 17],
             [1, 17, 18],
             [2, 0, 19], [2, 1, 20], [2, 2, 21], [2, 3, 22], [2, 4, 23], [2, 5, 24], [2, 6, 25], [2, 7, 26],
             [2, 8, 27],
             [2, 9, 28], [2, 10, 29], [2, 11, 30], [2, 12, 31], [2, 13, 32], [2, 14, 33], [2, 15, 34],
             [2, 16, 35],
             [2, 17, 36],
             [3, 0, 37], [3, 1, 38], [3, 2, 39], [3, 3, 40], [3, 4, 41], [3, 5, 42], [3, 6, 43], [3, 7, 44],
             [3, 8, 45],
             ],
    },
    'vgg19': {
        'cifar10size': [32] * 2 + [16] * 2 + [8] * 4 + [4] * 4 + [2] * 4,
        'svhnsize': [32] * 2 + [16] * 2 + [8] * 4 + [4] * 4 + [2] * 4,
        'stl10size': [96] * 2 + [48] * 2 + [24] * 4 + [12] * 4 + [6] * 4,
        'net_dim': [64] * 2 + [128] * 2 + [256] * 4 + [512] * 4 + [512] * 4,
        'net_config': [0, 1,
                       3, 4,
                       6, 7, 8, 9,
                       11, 12, 13, 14,
                       16, 17, 18],
        1: [],
        16: [0, 1,
             3, 4,
             6, 7, 8, 9,
             11, 12, 13, 14,
             16, 17, 18],
        15: [0, 1,  # d=1
             3, 4,
             6, 7, 8, 9,
             11, 12, 13, 14,
             16, 17],
        14: [0, 1,  # d=2
             3, 4,
             6, 7, 8, 9,
             11, 12, 13, 14,
             16, ],
        13: [0, 1,  # d=3
             3, 4,
             6, 7, 8, 9,
             11, 12, 13, 14,
             ],
        12: [0, 1,  # d=4
             3, 4,
             6, 7, 8, 9,
             11, 12, 13,
             ],
        11: [0, 1,  # d=5
             3, 4,
             6, 7, 8, 9,
             11, 12,
             ]
    },
    'resnet50': {
        'net_dim': [16] + [16] * 3 + [32] * 4 + [64] * 6 + [128] * 3,
        1: [],  # End-to-end
        2: [[4, 2, 16]],
        17: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3],
             [2, 0, 4], [2, 1, 5], [2, 2, 6], [2, 3, 7],
             [3, 0, 8], [3, 1, 9], [3, 2, 10], [3, 3, 11], [3, 4, 12], [3, 5, 13],
             [4, 0, 14], [4, 1, 15]],
        18: [[0, 0, 0],
             [1, 0, 1], [1, 1, 2], [1, 2, 3],
             [2, 0, 4], [2, 1, 5], [2, 2, 6], [2, 3, 7],
             [3, 0, 8], [3, 1, 9], [3, 2, 10], [3, 3, 11], [3, 4, 12], [3, 5, 13],
             [4, 0, 14], [4, 1, 15], [4, 2, 16]],

    },
    'mobilenet': {
        19: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    },
    'efficientnet': {
        17: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    },
    'regnet': {
        23: []
    }
}