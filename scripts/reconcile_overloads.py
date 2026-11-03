import pandas as pd

df = pd.read_csv('data/raw/transformer_hourly_festival_with_temperature.csv')
df['datetime'] = pd.to_datetime(df['datetime_IST'])

for t in ['t1', 't2', 't3']:
    print(f"\n{'='*20} {t.upper()} OVERLOAD RECONCILIATION {'='*20}")
    train_df = pd.read_csv(f'data/processed/{t}_train.csv')
    test_df = pd.read_csv(f'data/processed/{t}_test.csv')
    train_df['dt'] = pd.to_datetime(train_df['datetime'])
    test_df['dt'] = pd.to_datetime(test_df['datetime'])
    
    raw_overload = df[df[f'{t}_load_pct'] > 100.0]
    raw_ov_train = raw_overload[raw_overload['datetime'] < '2017-01-01']
    raw_ov_test = raw_overload[raw_overload['datetime'] >= '2017-01-01']
    
    print(f"Raw Overloads (>100%): Total={len(raw_overload)} | 2013-2016={len(raw_ov_train)} | 2017={len(raw_ov_test)}")
    print(f"Train Processed (rows={len(train_df)}):")
    print(f"  Rows where current load_pct > 100%: {(train_df['load_pct'] > 100.0).sum()}")
    print(f"  Rows where target_next_hour_overload == 1: {(train_df['target_next_hour_overload'] == 1.0).sum()}")
    
    print(f"Test Processed (rows={len(test_df)}):")
    print(f"  Rows where current load_pct > 100%: {(test_df['load_pct'] > 100.0).sum()}")
    print(f"  Rows where target_next_hour_overload == 1: {(test_df['target_next_hour_overload'] == 1.0).sum()}")
    
    # Check dropped overloads in train
    if len(raw_ov_train) > 0:
        dropped_train_ov = raw_ov_train[~raw_ov_train['datetime'].isin(train_df['dt'])]
        print(f"  Train overloads dropped due to outage/lag NaNs: {len(dropped_train_ov)}")
        if len(dropped_train_ov) > 0:
            print(f"  Dates: {dropped_train_ov['datetime_IST'].values}")
            
    # Check dropped overloads in test
    if len(raw_ov_test) > 0:
        dropped_test_ov = raw_ov_test[~raw_ov_test['datetime'].isin(test_df['dt'])]
        print(f"  Test overloads dropped due to outage/lag NaNs: {len(dropped_test_ov)}")
        if len(dropped_test_ov) > 0:
            print(f"  Dates: {dropped_test_ov['datetime_IST'].values}")

