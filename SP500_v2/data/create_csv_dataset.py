"""
Example script to create S&P 500 dataset and save as CSV files
"""

import os
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.dataset import create_sp500_training_dataset, SP500Dataset


def main():
    """
    Create S&P 500 dataset and demonstrate CSV functionality
    """
    print("📊 S&P 500 Dataset CSV Creation Example")
    print("=" * 50)
    
    # Create the complete dataset with CSV saving enabled
    dataset = create_sp500_training_dataset(
        symbol='^GSPC',
        target_type='return_1d',
        save_csv=True
    )
    
    # Show CSV file information
    print("\n📁 CSV Files Information:")
    print("-" * 30)
    csv_files = dataset.get_csv_files()
    
    for name, info in csv_files.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Path: {info['path']}")
        print(f"  Exists: {'✅ Yes' if info['exists'] else '❌ No'}")
        
        if info['exists']:
            # Load and show basic info about each CSV
            try:
                df = SP500Dataset.load_csv_data(info['path'])
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {df.columns.tolist()[:5]}{'...' if len(df.columns) > 5 else ''}")
                print(f"  Date range: {df.index.min()} to {df.index.max()}")
            except Exception as e:
                print(f"  Error loading: {e}")
    
    # Example of loading CSV data independently
    print(f"\n🔄 Example: Loading training data from CSV")
    print("-" * 40)
    
    try:
        train_df = SP500Dataset.load_csv_data(dataset.train_csv_path)
        print(f"✅ Successfully loaded training data from CSV")
        print(f"   Shape: {train_df.shape}")
        print(f"   Features: {train_df.shape[1] - 1} (+ 1 target)")
        print(f"   Sample data:")
        print(train_df.head(3).to_string())
        
        # Show target column
        print(f"\n🎯 Target column statistics:")
        target_col = train_df['target']
        print(f"   Mean: {target_col.mean():.6f}")
        print(f"   Std: {target_col.std():.6f}")
        print(f"   Min: {target_col.min():.6f}")
        print(f"   Max: {target_col.max():.6f}")
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
    
    print(f"\n🎉 CSV creation complete!")
    print(f"📂 All files saved in: {dataset.csv_dir}")
    print(f"💡 You can now use these CSV files in any ML framework or tool")


if __name__ == "__main__":
    main()
