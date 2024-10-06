tidrec_conf = {
    'model_list': ['co', 'write', 'toge'],
    'distill_dict': {
        ('co', 'write'): 1.0,
        ('write', 'co'): 1.0,
        ('co', 'toge'): 1.0,
        ('toge', 'co'): 1.0,
        ('write', 'toge'): 1.0,
        ('toge', 'write'): 1.0,
    },
    'l_author': [1, 2],
    'l_paper': [0],
}