filenames = [
    'F1.csv',
    'F2.csv',
    'F3.csv',
    'M1.csv',
    'T1.csv',
    'Th1.csv',
    'Th2.csv',
    'W1.csv'
]

attacks = {
    'BENIGN': 0, 
    'Bot': 1, 
    'DDoS': 2, 
    'DoS GoldenEye': 3, 
    'DoS Hulk': 4, 
    'DoS Slowhttptest': 5, 
    'DoS slowloris': 6, 
    'FTP-Patator': 7, 
    'Heartbleed': 8, 
    'Infiltration': 9,
    'PortScan': 10, 
    'SSH-Patator': 11, 
    'Web Attack-Brute Force': 12, 
    'Web Attack-Sql Injection': 13, 
    'Web Attack-XSS': 14
}

underSample = {
    0 : 230000, 
    1 : 1966, 
    2 : 128027, 
    3 : 10293, 
    4 : 231073, 
    5 : 5499, 
    6 : 5796, 
    7 : 7938, 
    8 : 11, 
    9 : 36,
    10 : 158390, 
    11 : 5897, 
    12 : 1507, 
    13 : 21, 
    14 : 652
}

overSample = {
    0 : 230000, 
    1 : 5000, 
    2 : 128027, 
    3 : 10293, 
    4 : 231073, 
    5 : 5499, 
    6 : 5796, 
    7 : 7938, 
    8 : 5000, 
    9 : 5000,
    10 : 158390, 
    11 : 5897, 
    12 : 5000, 
    13 : 5000, 
    14 : 5000
}