import json
import pandas as pd
import numpy as np

def run_audit():
    print("RUNNING COMPREHENSIVE MILESTONE 2 AUDIT")
    report = {}

    for t in ['t1', 't2', 't3']:
        train_path = f'data/processed/{t}_train.csv'
        test_path = f'data/processed/{t}_test.csv'
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        train_df['dt'] = pd.to_datetime(train_df['datetime'])
        test_df['dt'] = pd.to_datetime(test_df['datetime'])
        
        # Target timestamp is feature dt + 1 hour
        train_df['target_dt'] = train_df['dt'] + pd.Timedelta(hours=1)
        test_df['target_dt'] = test_df['dt'] + pd.Timedelta(hours=1)
        
        # Audit 1 & 6: Check target timestamps crossing into test
        train_target_in_2017 = train_df['target_dt'] >= '2017-01-01'
        print(f"\n[{t.upper()}] Audit 1 & 6: Train target timestamp in 2017 count: {train_target_in_2017.sum()}")
        if train_target_in_2017.sum() > 0:
            offending_rows = train_df[train_target_in_2017]
            print(f"  Offending row datetime: {offending_rows['datetime'].values}, target_dt: {offending_rows['target_dt'].values}")
            
        test_feature_in_train = test_df['dt'] < '2017-01-01'
        print(f"[{t.upper()}] Test feature timestamp before 2017 count: {test_feature_in_train.sum()}")
        
        test_target_in_train = test_df['target_dt'] < '2017-01-01'
        print(f"[{t.upper()}] Test target timestamp before 2017 count: {test_target_in_train.sum()}")

    print("\nAudit completed.")

if __name__ == '__main__':
    run_audit()

