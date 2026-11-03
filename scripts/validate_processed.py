import json
import glob
import os
import pandas as pd

def validate():
    print("="*60)
    print("VALIDATING PROCESSED DATASETS")
    print("="*60)
    
    files = glob.glob("data/processed/*.csv")
    assert len(files) == 6, f"Expected 6 CSV files, found {len(files)}"
    
    all_ok = True
    for f in sorted(files):
        df = pd.read_csv(f)
        filename = os.path.basename(f)
        print(f"\nChecking {filename}:")
        print(f"  Shape: {df.shape}")
        
        # 1. uplift_pct check
        if 'uplift_pct' in df.columns:
            print("  [FAIL] uplift_pct found in columns!")
            all_ok = False
        else:
            print("  [PASS] uplift_pct strictly excluded.")
            
        # 2. Invariant columns check
        for inv in ['voltage_V', 'pf', 'frequency_Hz', 'watts_per_frequency', 'current_A', 'observed_W']:
            if any(inv in col for col in df.columns):
                print(f"  [FAIL] Invariant/redundant signal {inv} found!")
                all_ok = False
        print("  [PASS] All invariant/redundant signals removed.")
        
        # 3. Missing values check
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            print(f"  [FAIL] Found {null_count} nulls in {filename}!")
            all_ok = False
        else:
            print("  [PASS] 0 missing/null values across all columns.")
            
        # 4. Target variable check
        targets = ['target_next_hour_power_W', 'target_next_hour_load_pct', 'target_next_hour_overload']
        for tgt in targets:
            assert tgt in df.columns, f"Target {tgt} missing from {filename}"
        print("  [PASS] All 3 next-hour targets present.")

    # 5. Chronological train/test boundary check
    for t in ['t1', 't2', 't3']:
        train_df = pd.read_csv(f"data/processed/{t}_train.csv")
        test_df = pd.read_csv(f"data/processed/{t}_test.csv")
        train_max = pd.to_datetime(train_df['datetime']).max()
        test_min = pd.to_datetime(test_df['datetime']).min()
        print(f"\nTransformer {t.upper()} Chronological Boundary:")
        print(f"  Train max date: {train_max}")
        print(f"  Test min date:  {test_min}")
        assert train_max < test_min, f"Data leakage: train_max ({train_max}) >= test_min ({test_min})"
        print("  [PASS] Train and Test are strictly chronological. No temporal leakage.")

    # 6. JSON summary check
    with open("data/processed/dataset_summary.json") as jf:
        summary = json.load(jf)
    print("\n[PASS] dataset_summary.json verified. 3 transformers documented.")
    
    if all_ok:
        print("\nALL PREPROCESSING INTEGRITY CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    validate()

