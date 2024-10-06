tidrec_no_conf = {
    'model_list': ['co', 'write', 'toge'],
    'distill_dict': {
        ('co', 'write'): 0.0,
        ('write', 'co'): 0.0,
        ('co', 'toge'): 0.0,
        ('toge', 'co'): 0.0,
        ('write', 'toge'): 0.0,
        ('toge', 'write'): 0.0,
    },
    'l_author': [1, 2],
    'l_paper': [0],
}