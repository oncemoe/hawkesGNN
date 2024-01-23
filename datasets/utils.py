import numpy as np

def shrink_edge_index(df):
        _, new_edge_index = np.unique(df[['x', 'y']].values, return_inverse=True)
        new_edge_index = new_edge_index.reshape(len(df), 2)
        df['x'] = new_edge_index[:, 0]
        df['y'] = new_edge_index[:, 1]