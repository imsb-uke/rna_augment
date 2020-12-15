import pandas as pd

def load_data():
    source_1 = pd.read_csv('data/example_gtex_train_data_1.zip', compression='zip', index_col='sample_id')
    source_2 = pd.read_csv('data/example_gtex_train_data_2.zip', compression='zip', index_col='sample_id')
    source = pd.concat([source_1, source_2], axis=0).values.astype('float32')
    target = pd.read_csv('data/example_tcga_test_data.zip', compression='zip', index_col='sample_id').values.astype('float32')
    bias = pd.read_csv('data/example_sra_train_data.zip', compression='zip', index_col='sample_id').values.astype('float32')
    
    return source, target, bias