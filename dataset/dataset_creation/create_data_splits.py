import pandas as pd
import numpy as np

def create_data_splits(csv_path, save_dir, num_splits=None, seed=42):
    # 1. Load your dataset
    df = pd.read_csv(csv_path)

    # 2. Get unique group names (one per session)
    unique_groups = df[['group_name']].drop_duplicates().reset_index(drop=True)

    # 3. Extract ID, Year, Month, Day using Regex
    # This looks for: Numbers_YYYY_MM_DD at the start of the string
    extracted = unique_groups['group_name'].str.extract(
        r'^(?P<ID>\d+)_(?P<Year_Before>\d{4})_(?P<Month_Before>\d{2})_(?P<Day_Before>\d{2})'
    )

    # 4. Create a temporary 'Date' column to ensure correct chronological sorting
    extracted['Date'] = pd.to_datetime(
        extracted[['Year_Before', 'Month_Before', 'Day_Before']].rename(
            columns={'Year_Before': 'year', 'Month_Before': 'month', 'Day_Before': 'day'}
        )
    )

    # Sort by Patient ID and Date (to ensure A -> B -> C order)
    extracted = extracted.sort_values(by=['ID', 'Date']).reset_index(drop=True)

    # 5. Group by Patient ID and shift the dates up by 1 to create the "After" columns
    extracted['Year_After'] = extracted.groupby('ID')['Year_Before'].shift(-1)
    extracted['Month_After'] = extracted.groupby('ID')['Month_Before'].shift(-1)
    extracted['Day_After'] = extracted.groupby('ID')['Day_Before'].shift(-1)

    # 6. Drop the rows that do not have an "After" pair (the last timepoint for each patient)
    pairs = extracted.dropna(subset=['Year_After', 'Month_After', 'Day_After']).copy()

    # Ensure "After" columns keep their string/zero-padded format (shift converts them to float)
    for col in ['Year_After', 'Month_After', 'Day_After']:
        pairs[col] = pairs[col].astype(str).str.split('.').str[0]
        if 'Month' in col or 'Day' in col:
            pairs[col] = pairs[col].str.zfill(2)

    # 7. Select final columns
    final_cols = [
        'ID', 
        'Year_Before', 'Month_Before', 'Day_Before', 
        'Year_After', 'Month_After', 'Day_After'
    ]
    final_df = pairs[final_cols]

    if num_splits is None:
        # 8. Save to CSV
        final_df.to_csv(save_dir + '/longitudinal_pairs.csv', index=False)
    else:
        # 1. Isolate unique Patient IDs
        unique_pids = final_df['ID'].unique()

        # 2. Shuffle the PIDs randomly (set seed for reproducible splits)
        np.random.seed(seed)
        np.random.shuffle(unique_pids)

        # 3. Split the shuffled PIDs into 5 distinct arrays
        pid_splits = np.array_split(unique_pids, num_splits)

        # 4. Map the grouped PIDs back to the final dataframe
        data_splits = {}

        for i, pids_in_fold in enumerate(pid_splits):
            # Filter the main dataframe to only keep rows where the ID is in this specific split
            split_df = final_df[final_df['ID'].isin(pids_in_fold)].copy()
            
            # Store in dictionary
            data_splits[f'split_{i+1}'] = split_df

            split_df.to_csv(save_dir + f'/longitudinal_pairs_fold_{i+1}.csv', index=False)

if __name__ == "__main__":
    create_data_splits("/home/yhchoi/EEG_Data/260403_Leqembi/Embeddings_LUNA/metadata.csv", "/home/yhchoi/EEG_Data/260403_Leqembi/Embeddings_LUNA/splits", 5)