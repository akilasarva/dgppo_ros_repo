import json
import re
from collections import Counter
from typing import Dict, List, Tuple

def parse_cluster_file(file_path: str) -> Dict[str, List[str]]:
    """
    Parses a text file containing cluster data and returns a dictionary
    mapping cluster IDs to a list of associated filenames.

    Args:
        file_path (str): The path to the input text file.

    Returns:
        Dict[str, List[str]]: A dictionary with cluster IDs as keys and
                              lists of filenames as values.
    """
    clusters = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # The pattern "Cluster X: file1, file2, ..."
                match = re.match(r"Cluster (-?\d+): (.*)", line)
                if match:
                    cluster_id = match.group(1)
                    # Split the filenames by ", " and clean up whitespace
                    filenames = [
                        name.strip() for name in match.group(2).split(',') if name.strip()
                    ]
                    clusters[cluster_id] = filenames
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred while parsing the file: {e}")
    
    return clusters

def parse_ground_truth_json(file_path: str) -> List[Dict]:
    """
    Parses a ground truth JSON file and returns the data.

    The JSON is expected to be a list of objects, where each object has
    'start_time', 'end_time', and 'label' keys.

    Args:
        file_path (str): The path to the ground truth JSON file.

    Returns:
        List[Dict]: The parsed JSON data.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}.")
        return []
    except Exception as e:
        print(f"An error occurred while parsing the JSON file: {e}")
        return []

def get_label_from_timestamp(timestamp: int, ground_truth: List[Dict]) -> str:
    """
    Finds the label for a given timestamp from the ground truth data.

    Args:
        timestamp (int): The timestamp to look up.
        ground_truth (List[Dict]): The parsed ground truth data.

    Returns:
        str: The label associated with the timestamp, or "unknown" if not found.
    """
    for item in ground_truth:
        if item['start_time'] <= timestamp <= item['end_time']:
            return item['label']
    return "unknown"

def assign_labels_to_clusters(
    clusters: Dict[str, List[str]], ground_truth: List[Dict]
) -> Dict[str, str]:
    """
    Assigns a single, most frequent label to each cluster.

    Args:
        clusters (Dict[str, List[str]]): Dictionary of cluster IDs and filenames.
        ground_truth (List[Dict]): Parsed ground truth data.

    Returns:
        Dict[str, str]: A dictionary mapping cluster IDs to their assigned label.
    """
    cluster_labels = {}
    
    # Updated regex to extract the timestamp from the new filename format
    # e.g., '00240' from '00240.pcd'
    timestamp_pattern = re.compile(r"(\d+)\.pcd")

    for cluster_id, filenames in clusters.items():
        # Use a Counter to find the most frequent label
        labels_for_cluster = Counter()
        for filename in filenames:
            match = timestamp_pattern.match(filename)
            if match:
                # The numeric part of the filename is the timestamp
                timestamp = int(match.group(1))
                label = get_label_from_timestamp(timestamp, ground_truth)
                if label != "unknown":
                    labels_for_cluster[label] += 1
        
        # Assign the most common label, or 'unlabeled' if none are found
        if labels_for_cluster:
            most_common_label = labels_for_cluster.most_common(1)[0][0]
            cluster_labels[cluster_id] = most_common_label
        else:
            cluster_labels[cluster_id] = "unlabeled"
            
    return cluster_labels

def main():
    """Main function to run the labeling process and print JSON output."""
    # Define file paths
    cluster_file_path = "encoder_weights/full_bag/train_cluster_to_filepaths_full_bag.txt"
    ground_truth_file_path = "encoder_weights/full_bag/ground_truth.json"

    ground_truth_data = parse_ground_truth_json(ground_truth_file_path)

    clusters = parse_cluster_file(cluster_file_path)
    if not clusters or not ground_truth_data:
        print("Could not process files. Please check paths and content.")
        return

    labeled_clusters = assign_labels_to_clusters(clusters, ground_truth_data)
    
    # Print the output as a formatted JSON string
    print(json.dumps(labeled_clusters, indent=4))

if __name__ == "__main__":
    main()
