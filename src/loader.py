import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt

def clean_bluetooth_data(filepath, config):
    """
    Ingests the raw Bluetooth CSV, kicks out the sentinels, and maps time to 5-minute slots.
    """
    # Bypass the broken header
    columns = ['timestamp', 'user_a', 'user_b', 'rssi']
    df = pd.read_csv(filepath, skiprows=1, names=columns, header=None)
    # Isolate sentinels for the coverage map
    sentinels_df = df[df['user_b'] < 0].copy()
    # Kick out the sentinels
    valid_contacts = df[df['user_b'] >= 0].copy()
    # Enforce RSSI threshold
    base_threshold = min(config['bluetooth_parameters']['rssi_thresholds'])
    valid_contacts = valid_contacts[valid_contacts['rssi'] >= base_threshold].copy()
    # Create the 5-minute grid
    slot_size = config['time_parameters']['slot_duration_sec']
    valid_contacts['slot_id'] = valid_contacts['timestamp'] // slot_size

    # Export the baseline files (contacts.parquet and coverage.parquet)
    output_dir = config['paths']['processed_data_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    contacts_path = os.path.join(output_dir, 'contacts.parquet')
    valid_contacts.to_parquet(contacts_path, engine='pyarrow')
    
    coverage_path = os.path.join(output_dir, 'coverage.parquet')
    sentinels_df.to_parquet(coverage_path, engine='pyarrow')
    """
    print(f"Original row count: {len(df)}")
    print(f"Cleaned row count: {len(valid_contacts)}")
    print("\nFirst 5 rows of cleaned data:")
    print(valid_contacts.head()) 
    """
    return valid_contacts

def extract_gatherings(clean_contacts, config):
    """
    Phase 1: Instantiates the network graphs from discretized time slots.
    """
    # Parameter extraction
    min_k = config['clustering_parameters']['min_gathering_size']   # minimum number of nodes that form a gathering
    gatherings_list = []
    gathering_id_counter = 0
    
    # Temporal partitioning (Split-Apply-Combine)
    for slot_id, slot_data in clean_contacts.groupby('slot_id'):
        # Graph instantiation
        G = nx.Graph()
        # Edge population
        for _, row in slot_data.iterrows():
            G.add_edge(row['user_a'], row['user_b'], rssi=row['rssi'])
        """
        # Graph visualization
        print(f"Drawing network for Time Slot {slot_id}...")        
        # Make the plot look nice
        plt.figure(figsize=(10, 8))
        nx.draw(G, 
                with_labels=True, 
                node_color='lightblue', 
                edge_color='gray', 
                node_size=500, 
                font_weight='bold')
        plt.title(f"Campus Gatherings - Slot {slot_id}")
        
        # Open the window!
        plt.show()
        break
        """
        # Network mining: extract sub-graphs (connected components)
        for component in nx.connected_components(G):
            # Enforce the minimum gathering size (min_k)
            if len(component) >= min_k:
                # Isolate the original raw data for just these specific users
                component_data = slot_data[
                    (slot_data['user_a'].isin(component)) & 
                    (slot_data['user_b'].isin(component))
                ]
                # Add these data into the gatherings list
                gatherings_list.append({
                    'slot_id': slot_id,
                    'gathering_id': gathering_id_counter,
                    'participant_set': list(component), 
                    'start_time': component_data['timestamp'].min(),
                    'end_time': component_data['timestamp'].max(),
                    'mean_rssi': component_data['rssi'].median() 
                })
                gathering_id_counter += 1

    gatherings_df = pd.DataFrame(gatherings_list)   # dataframe to parquet
    output_dir = config['paths']['processed_data_dir']
    os.makedirs(output_dir, exist_ok=True)  # checks if the processed_data_dir exist or not and it creates it accordingly
    output_path = os.path.join(output_dir, 'gatherings_v1.parquet')
    gatherings_df.to_parquet(output_path, engine='pyarrow') # saves the parquet file
    print(f"Successfully exported {len(gatherings_df)} gatherings to {output_path}")
    return gatherings_df

def load_facebook_data(filepath):
    """
    Ingests the static social network, bypassing the broken '# user_a' header 
    to ensure user IDs remain integers.
    """
    columns = ['user_a', 'user_b']
    return pd.read_csv(filepath, skiprows=1, names=columns, header=None)
    
def load_gender_data(filepath):
    """
    Ingests demographic data, bypassing the broken '# user' header.
    """
    columns = ['user_id', 'is_female']
    return pd.read_csv(filepath, skiprows=1, names=columns, header=None)

def load_telecom_data(filepath, log_type='sms'):
    """
    Ingests telecom logs. 
    If loading calls, it isolates the '-1' missed call sentinel into a boolean feature.
    """
    df = pd.read_csv(filepath)
    
    if log_type == 'calls':
        # Safely flag missed calls so downstream algorithms don't compute negative durations
        df['is_missed_call'] = df['duration'] == -1
        
    return df

# Loads secondary CSVs, applies shallow cleaning, and exports them as Parquet 
# to establish a unified, high-performance processed data layer.
def export_secondary_datasets(config, raw_data_path):

    out_dir = config['paths']['processed_data_dir']
    
    # Facebook Network
    fb_df = load_facebook_data(f"{raw_data_path}/fb_friends.csv")
    fb_df.to_parquet(os.path.join(out_dir, 'fb_friends.parquet'), engine='pyarrow')
    
    # Gender Demographics
    gender_df = load_gender_data(f"{raw_data_path}/genders.csv")
    gender_df.to_parquet(os.path.join(out_dir, 'genders.parquet'), engine='pyarrow')
    
    # Call Logs
    calls_df = load_telecom_data(f"{raw_data_path}/calls.csv", log_type='calls')
    calls_df.to_parquet(os.path.join(out_dir, 'calls.parquet'), engine='pyarrow')
    
    # SMS Logs
    sms_df = load_telecom_data(f"{raw_data_path}/sms.csv", log_type='sms')
    sms_df.to_parquet(os.path.join(out_dir, 'sms.parquet'), engine='pyarrow')
    
    print("Successfully exported all secondary datasets to Parquet format.")

# Temporary test block
if __name__ == "__main__":
    from config import load_config
    real_config = load_config("/Users/secon/DocumentsLocal/Data Mining I/UU-DataMining-2026/config.yaml")  # load the config file

    # Bluetooth Pipeline
    test_filepath = "data/raw/7267433/bt_symmetric.csv"
    cleaned_df = clean_bluetooth_data(test_filepath, config=real_config)  # pass the config file
    final_df = extract_gatherings(cleaned_df, config=real_config)
    print("\nFirst 3 Extracted Gatherings:")
    print(final_df.head(3))

    # Secondary Datasets Pipeline
    raw_dir = "/Users/secon/DocumentsLocal/Data Mining I/UU-DataMining-2026/data/raw/7267433"
    export_secondary_datasets(real_config, raw_data_path=raw_dir)