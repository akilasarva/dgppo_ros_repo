# import json
# import re
# from collections import Counter
# from typing import Dict, List, Tuple
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# def parse_cluster_file(file_path: str) -> Dict[str, List[str]]:
#     """
#     Parses a text file containing cluster data and returns a dictionary
#     mapping cluster IDs to a list of associated filenames.

#     Args:
#         file_path (str): The path to the input text file.

#     Returns:
#         Dict[str, List[str]]: A dictionary with cluster IDs as keys and
#                               lists of filenames as values.
#     """
#     clusters = {}
#     try:
#         with open(file_path, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue

#                 # The pattern "Cluster X: file1, file2, ..."
#                 match = re.match(r"Cluster (-?\d+): (.*)", line)
#                 if match:
#                     cluster_id = match.group(1)
#                     # Split the filenames by ", " and clean up whitespace
#                     filenames = [
#                         name.strip() for name in match.group(2).split(',') if name.strip()
#                     ]
#                     clusters[cluster_id] = filenames
#     except FileNotFoundError:
#         print(f"Error: The file at {file_path} was not found.")
#     except Exception as e:
#         print(f"An error occurred while parsing the file: {e}")
    
#     return clusters

# def parse_json_file(file_path: str) -> Dict[str, str] | List[Dict]:
#     """
#     Parses a JSON file from the given path.

#     Args:
#         file_path (str): The path to the JSON file.

#     Returns:
#         Dict[str, str] | List[Dict]: The parsed JSON data.
#     """
#     try:
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#             return data
#     except FileNotFoundError:
#         print(f"Error: The file at {file_path} was not found.")
#         return {}
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON from {file_path}.")
#         return {}
#     except Exception as e:
#         print(f"An error occurred while parsing the file: {e}")
#         return {}
# def calculate_metrics(file_to_cluster: Dict[str, str], cluster_to_label: Dict[str, str], ground_truth: List[Dict]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
#     """
#     Calculates the confusion matrix and category-wise accuracy,
#     including an "unlabeled" category.

#     Args:
#         file_to_cluster (Dict[str, str]): A map from filename to its cluster ID.
#         cluster_to_label (Dict[str, str]): A map from cluster ID to its assigned label.
#         ground_truth (List[Dict]): Parsed ground truth data.

#     Returns:
#         Tuple[Dict, Dict]: A tuple containing the confusion matrix and
#                            a dictionary of category-wise accuracies,
#                            and a dictionary of false positives and negatives.
#     """
    
#     # Get all unique labels from the ground truth data
#     unique_labels = sorted(list(set(item['label'] for item in ground_truth)))

#     # --- CHANGE START ---
#     # Explicitly add the "unlabeled" category to our list of labels
#     all_labels = unique_labels + ["unlabeled"]
    
#     # Initialize the confusion matrix with all labels
#     confusion_matrix = {label: {pred_label: 0 for pred_label in all_labels} for label in all_labels}
#     # --- CHANGE END ---
    
#     # Initialize dictionaries for false positives and negatives
#     false_positives_and_negatives = {label: {'False Positives': 0, 'False Negatives': 0, 'True Positives': 0} for label in all_labels}
    
#     # Track totals for accuracy calculation
#     total_per_category = Counter()
#     correct_per_category = Counter()

#     # Iterate through all files based on the ground truth timestamps
#     for item in ground_truth:
#         actual_label = item['label']
#         for timestamp in range(item['start_time'], item['end_time'] + 1):
#             filename = f"{timestamp:05d}.pcd"
            
#             # Find the cluster ID for this file
#             predicted_cluster_id = file_to_cluster.get(filename)
            
#             # If the file is not in a cluster, the predicted label is "unlabeled"
#             if predicted_cluster_id is None:
#                 predicted_label = "unlabeled"
#             else:
#                 # Use the cluster's assigned label as the prediction
#                 predicted_label = cluster_to_label.get(predicted_cluster_id, "unlabeled")

#             # --- CHANGE START ---
#             # Update the confusion matrix for all labels, including "unlabeled"
#             if actual_label in confusion_matrix and predicted_label in confusion_matrix[actual_label]:
#                 confusion_matrix[actual_label][predicted_label] += 1
#             # --- CHANGE END ---
            
#             # Update counts for accuracy calculation
#             total_per_category[actual_label] += 1
#             if actual_label == predicted_label:
#                 correct_per_category[actual_label] += 1
#                 false_positives_and_negatives[actual_label]['True Positives'] += 1
#             else:
#                 # This is a misclassification
#                 # It's a False Negative for the actual_label
#                 if actual_label in false_positives_and_negatives:
#                     false_positives_and_negatives[actual_label]['False Negatives'] += 1
#                 # It's a False Positive for the predicted_label
#                 if predicted_label in false_positives_and_negatives:
#                     false_positives_and_negatives[predicted_label]['False Positives'] += 1


#     # Calculate category-wise accuracy
#     accuracy_per_category = {}
#     for label in unique_labels:
#         if total_per_category[label] > 0:
#             accuracy_per_category[label] = correct_per_category[label] / total_per_category[label]
#         else:
#             accuracy_per_category[label] = 0.0
            
#     # Return all counts for overall accuracy calculation
#     return confusion_matrix, accuracy_per_category, total_per_category, correct_per_category, false_positives_and_negatives

# def save_json_to_file(data: Dict, filename: str):
#     """
#     Saves a dictionary as a formatted JSON file.

#     Args:
#         data (Dict): The dictionary to save.
#         filename (str): The path and name of the file to save to.
#     """
#     try:
#         with open(filename, 'w') as f:
#             json.dump(data, f, indent=4)
#         print(f"Successfully saved data to {filename}")
#     except Exception as e:
#         print(f"Error saving file {filename}: {e}")

# def save_confusion_matrix_as_image(confusion_matrix: Dict, filename: str, is_normalized=False):
#     """
#     Generates and saves a heatmap image of the confusion matrix.

#     Args:
#         confusion_matrix (Dict): The confusion matrix as a dictionary.
#         filename (str): The path and name of the image file to save to.
#         is_normalized (bool): Whether to save a normalized matrix.
#     """
#     try:
#         # Convert the dictionary to a pandas DataFrame
#         df_cm = pd.DataFrame.from_dict(confusion_matrix)

#         # Define the custom order of the labels
#         label_order = [
#             "Along Wall",
#             "Around Corner",
#             "Past Building",
#             "On Bridge",
#             "In Intersection",
#             "Exit Intersection/Enter Bridge",
#             "Enter Intersection/Exit Bridge",
#             "Open Space",
#             "Road",
#             "unlabeled"
#         ]

#         # Reindex the DataFrame to match the custom order
#         df_cm = df_cm.reindex(index=label_order, columns=label_order)

#         # Normalize if requested
#         if is_normalized:
#             df_cm = df_cm.div(df_cm.sum(axis=1), axis=0).fillna(0)
#             fmt = ".2f"
#             title = "Normalized Confusion Matrix - Training Data"
#         else:
#             fmt = "d"
#             title = "Confusion Matrix"

#         # Create the plot
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(df_cm, annot=True, fmt=fmt, cmap="Blues")
#         plt.xlabel("Predicted Label")
#         plt.ylabel("True Label")
#         plt.title(title)
#         plt.tight_layout()
        
#         # Save the figure
#         plt.savefig(filename, dpi=300)
#         print(f"Successfully saved confusion matrix image to {filename}")
#         plt.close()

#     except Exception as e:
#         print(f"Error saving confusion matrix image: {e}")

# def create_error_bar_chart(false_positives_and_negatives: Dict, filename: str):
#     """
#     Generates a grouped bar chart to visualize false positives and false negatives.

#     Args:
#         false_positives_and_negatives (Dict): A dictionary with FP/FN counts for each label.
#         filename (str): The path and name of the image file to save to.
#     """
#     try:
#         labels = list(false_positives_and_negatives.keys())
#         false_positives = [data['False Positives'] for data in false_positives_and_negatives.values()]
#         false_negatives = [data['False Negatives'] for data in false_positives_and_negatives.values()]
        
#         x = np.arange(len(labels))
#         width = 0.35

#         fig, ax = plt.subplots(figsize=(12, 8))
#         rects1 = ax.bar(x - width/2, false_positives, width, label='False Positives', color='darkred')
#         rects2 = ax.bar(x + width/2, false_negatives, width, label='False Negatives', color='sienna')

#         # Add some text for labels, title and custom x-axis tick labels
#         ax.set_ylabel('Number of Samples')
#         ax.set_title('False Positives and False Negatives per Label')
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels, rotation=45, ha="right")
#         ax.legend()
#         ax.set_ylim(0, max(max(false_positives), max(false_negatives)) * 1.2)
        
#         plt.tight_layout()
#         plt.savefig(filename, dpi=300)
#         print(f"Successfully saved error bar chart to {filename}")
#         plt.close()

#     except Exception as e:
#         print(f"Error creating error bar chart: {e}")

# def create_stacked_bar_chart(false_positives_and_negatives: Dict, filename: str):
#     """
#     Generates a stacked bar chart to visualize True Positives and False Negatives.

#     Args:
#         false_positives_and_negatives (Dict): A dictionary with TP/FN counts for each label.
#         filename (str): The path and name of the image file to save to.
#     """
#     try:
#         labels = list(false_positives_and_negatives.keys())
#         true_positives = [data['True Positives'] for data in false_positives_and_negatives.values()]
#         false_negatives = [data['False Negatives'] for data in false_positives_and_negatives.values()]
        
#         # Define colors: Gray for correct predictions, Red for errors
#         tp_color = 'lightgray'
#         fn_color = 'darkred'

#         fig, ax = plt.subplots(figsize=(12, 8))
#         ax.bar(labels, true_positives, label='True Positives', color=tp_color)
#         ax.bar(labels, false_negatives, bottom=true_positives, label='False Negatives', color=fn_color)
        
#         # Add labels and title
#         ax.set_ylabel('Number of Samples')
#         ax.set_title('True Positives and False Negatives per Label')
#         ax.set_xticks(range(len(labels)))
#         ax.set_xticklabels(labels, rotation=45, ha="right")
#         ax.legend()
        
#         plt.tight_layout()
#         plt.savefig(filename, dpi=300)
#         print(f"Successfully saved stacked bar chart to {filename}")
#         plt.close()

#     except Exception as e:
#         print(f"Error creating stacked bar chart: {e}")

# def create_precision_stacked_bar_chart(false_positives_and_negatives: Dict, filename: str):
#     """
#     Generates a stacked bar chart to visualize True Positives and False Positives.

#     Args:
#         false_positives_and_negatives (Dict): A dictionary with TP/FP counts for each label.
#         filename (str): The path and name of the image file to save to.
#     """
#     try:
#         labels = list(false_positives_and_negatives.keys())
#         true_positives = [data['True Positives'] for data in false_positives_and_negatives.values()]
#         false_positives = [data['False Positives'] for data in false_positives_and_negatives.values()]
        
#         # Define colors: Gray for correct predictions, Red for errors
#         tp_color = 'lightgray'
#         fp_color = 'darkred'

#         fig, ax = plt.subplots(figsize=(12, 8))
#         ax.bar(labels, true_positives, label='True Positives', color=tp_color)
#         ax.bar(labels, false_positives, bottom=true_positives, label='False Positives', color=fp_color)
        
#         # Add labels and title
#         ax.set_ylabel('Number of Samples')
#         ax.set_title('True Positives and False Positives per Predicted Label - Training Data')
#         ax.set_xticks(range(len(labels)))
#         ax.set_xticklabels(labels, rotation=45, ha="right")
#         ax.legend()
        
#         plt.tight_layout()
#         plt.savefig(filename, dpi=300)
#         print(f"Successfully saved precision stacked bar chart to {filename}")
#         plt.close()

#     except Exception as e:
#         print(f"Error creating precision stacked bar chart: {e}")

# def create_grouped_stacked_bar_chart(false_positives_and_negatives: Dict, filename: str):
#     """
#     Generates a chart with stacked TP/FN and grouped FP bars.

#     Args:
#         false_positives_and_negatives (Dict): A dictionary with TP/FN/FP counts.
#         filename (str): The path and name of the image file to save to.
#     """
#     try:
#         labels = list(false_positives_and_negatives.keys())
#         true_positives = np.array([data['True Positives'] for data in false_positives_and_negatives.values()])
#         false_negatives = np.array([data['False Negatives'] for data in false_positives_and_negatives.values()])
#         false_positives = np.array([data['False Positives'] for data in false_positives_and_negatives.values()])

#         x = np.arange(len(labels))
#         width = 0.35

#         fig, ax = plt.subplots(figsize=(14, 8))

#         # First set of bars: Stacked TP and FN
#         rects1_stack = ax.bar(x - width/2, true_positives, width, label='True Positives', color='lightgray')
#         rects2_stack = ax.bar(x - width/2, false_negatives, width, bottom=true_positives, label='False Negatives', color='darkred')
        
#         # Second bar grouped next to it: FP
#         rects3_grouped = ax.bar(x + width/2, false_positives, width, label='False Positives', color='sienna')

#         # Add some text for labels, title and custom x-axis tick labels
#         ax.set_ylabel('Number of Samples')
#         ax.set_title('Recall and False Positives per Label - Training Data')
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels, rotation=45, ha="right")
#         ax.legend()
        
#         plt.tight_layout()
#         plt.savefig(filename, dpi=300)
#         print(f"Successfully saved grouped stacked bar chart to {filename}")
#         plt.close()

#     except Exception as e:
#         print(f"Error creating grouped stacked bar chart: {e}")

# def main():
#     """Main function to run the labeling and analysis process."""

#     # Define file paths
#     cluster_file_path = "encoder_weights/full_bag/train_cluster_to_filepaths_full_bag.txt"
#     ground_truth_file_path = "encoder_weights/full_bag/ground_truth.json"
#     cluster_to_label_file_path = "encoder_weights/full_bag/cluster_id_to_label_full_bag.json"
    
#     # Load data from files
#     ground_truth_data = parse_json_file(ground_truth_file_path)
#     clusters = parse_cluster_file(cluster_file_path)
#     cluster_to_label = parse_json_file(cluster_to_label_file_path)

#     if not clusters or not ground_truth_data or not cluster_to_label:
#         print("Could not process files. Please check paths and content.")
#         return

#     # Create a reverse map from filename to cluster ID
#     file_to_cluster = {}
#     for cluster_id, filenames in clusters.items():
#         for filename in filenames:
#             file_to_cluster[filename] = cluster_id

#     # Calculate metrics using the loaded data
#     confusion_matrix, accuracy_per_category, total_per_category, correct_per_category, false_positives_and_negatives = calculate_metrics(file_to_cluster, cluster_to_label, ground_truth_data)
    
#     print("--- Confusion Matrix ---")
#     print(json.dumps(confusion_matrix, indent=4))
#     save_json_to_file(confusion_matrix, "confusion_matrix.json")
#     save_confusion_matrix_as_image(confusion_matrix, "confusion_matrix.png")
    
#     # Save the normalized confusion matrix image
#     save_confusion_matrix_as_image(confusion_matrix, "normalized_confusion_matrix.png", is_normalized=True)

#     print("\n--- Category-wise Accuracy ---")
#     print(json.dumps(accuracy_per_category, indent=4))
#     save_json_to_file(accuracy_per_category, "category_accuracy.json")
    
#     print("\n--- False Positives and False Negatives ---")
#     print(json.dumps(false_positives_and_negatives, indent=4))
#     save_json_to_file(false_positives_and_negatives, "false_positives_and_negatives.json")

#     # Create and save all three charts
#     # create_stacked_bar_chart(false_positives_and_negatives, "tp_fn_stacked_bar_chart.png")
#     # create_precision_stacked_bar_chart(false_positives_and_negatives, "tp_fp_stacked_bar_chart.png")
#     # create_error_bar_chart(false_positives_and_negatives, "fp_fn_grouped_bar_chart.png")
#     create_grouped_stacked_bar_chart(false_positives_and_negatives, "tp_fn_fp_bar_chart.png")

#     # Calculate and print total accuracy
#     total_correct = sum(correct_per_category.values())
#     total_files = sum(total_per_category.values())
#     if total_files > 0:
#         total_accuracy = total_correct / total_files
#         print(f"\n--- Total Accuracy ---\n{total_accuracy:.2f}")
#     else:
#         print("\n--- Total Accuracy ---\nCannot calculate total accuracy (no files found).")

# if __name__ == "__main__":
#     main()

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict

def save_confusion_matrix_as_image(confusion_matrix: Dict, filename: str, is_normalized=False):
    """
    Generates and saves a heatmap image of the confusion matrix from a dictionary.

    Args:
        confusion_matrix (Dict): The confusion matrix as a dictionary.
        filename (str): The path and name of the image file to save to.
        is_normalized (bool): Whether to save a normalized matrix.
    """
    try:
        # Convert the dictionary to a pandas DataFrame
        df_cm = pd.DataFrame.from_dict(confusion_matrix)

        # Define the custom order of the labels to ensure consistency
        label_order = [
            "Along Wall",
            "Around Corner",
            "Past Building",
            "On Bridge",
            "In Intersection",
            "Exit Intersection/Enter Bridge",
            "Enter Intersection/Exit Bridge",
            "Open Space",
            "Road",
            "unlabeled"
        ]

        # Reindex the DataFrame to match the custom order
        df_cm = df_cm.reindex(index=label_order, columns=label_order)

        # Normalize if requested
        if is_normalized:
            # Drop the 'unlabeled' row for normalization, as it's not a true label
            # that is predicted by the model.
            df_for_norm = df_cm.drop(index=["unlabeled"])
            df_cm = df_for_norm.div(df_for_norm.sum(axis=1), axis=0).fillna(0)
            fmt = ".2f"
            title = "Normalized Confusion Matrix - Test Data"
        else:
            fmt = "d"
            title = "Confusion Matrix"

        # Create the plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(df_cm, annot=True, fmt=fmt, cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)
        plt.tight_layout()

        # Save the figure
        plt.savefig(filename, dpi=300)
        print(f"Successfully saved flipped confusion matrix image to {filename}")
        plt.close()

    except Exception as e:
        print(f"Error saving confusion matrix image: {e}")

def main():
    """Main function to run the confusion matrix generation process."""
    
    # Load the JSON data from the provided artifact
    confusion_matrix_json = json.dumps({
    "Along Wall": {
        "Along Wall": 230,
        "Around Corner": 10,
        "Enter Intersection/Exit Bridge": 0,
        "Exit Intersection/Enter Bridge": 17,
        "In Intersection": 0,
        "On Bridge": 0,
        "Open Space": 6,
        "Past Building": 10,
        "Road": 0,
        "unlabeled": 0
    },
    "Around Corner": {
        "Along Wall": 2,
        "Around Corner": 200,
        "Enter Intersection/Exit Bridge": 0,
        "Exit Intersection/Enter Bridge": 0,
        "In Intersection": 0,
        "On Bridge": 0,
        "Open Space": 0,
        "Past Building": 10,
        "Road": 0,
        "unlabeled": 4
    },
    "Enter Intersection/Exit Bridge": {
        "Along Wall": 0,
        "Around Corner": 0,
        "Enter Intersection/Exit Bridge": 230,
        "Exit Intersection/Enter Bridge": 0,
        "In Intersection": 31,
        "On Bridge": 6,
        "Open Space": 0,
        "Past Building": 0,
        "Road": 68,
        "unlabeled": 15
    },
    "Exit Intersection/Enter Bridge": {
        "Along Wall": 4,
        "Around Corner": 0,
        "Enter Intersection/Exit Bridge": 0,
        "Exit Intersection/Enter Bridge": 200,
        "In Intersection": 12,
        "On Bridge": 8,
        "Open Space": 0,
        "Past Building": 0,
        "Road": 30,
        "unlabeled": 10
    },
    "In Intersection": {
        "Along Wall": 0,
        "Around Corner": 0,
        "Enter Intersection/Exit Bridge": 12,
        "Exit Intersection/Enter Bridge": 10,
        "In Intersection": 250,
        "On Bridge": 0,
        "Open Space": 0,
        "Past Building": 0,
        "Road": 61,
        "unlabeled": 20
    },
    "On Bridge": {
        "Along Wall": 0,
        "Around Corner": 0,
        "Enter Intersection/Exit Bridge": 0,
        "Exit Intersection/Enter Bridge": 0,
        "In Intersection": 0,
        "On Bridge": 260,
        "Open Space": 0,
        "Past Building": 0,
        "Road": 0,
        "unlabeled": 2
    },
    "Open Space": {
        "Along Wall": 0,
        "Around Corner": 0,
        "Enter Intersection/Exit Bridge": 0,
        "Exit Intersection/Enter Bridge": 0,
        "In Intersection": 0,
        "On Bridge": 0,
        "Open Space": 250,
        "Past Building": 5,
        "Road": 0,
        "unlabeled": 0
    },
    "Past Building": {
        "Along Wall": 4,
        "Around Corner": 0,
        "Enter Intersection/Exit Bridge": 0,
        "Exit Intersection/Enter Bridge": 0,
        "In Intersection": 0,
        "On Bridge": 0,
        "Open Space": 10,
        "Past Building": 200,
        "Road": 0,
        "unlabeled": 20
    },
    "Road": {
        "Along Wall": 0,
        "Around Corner": 0,
        "Enter Intersection/Exit Bridge": 7,
        "Exit Intersection/Enter Bridge": 0,
        "In Intersection": 3,
        "On Bridge": 0,
        "Open Space": 0,
        "Past Building": 15,
        "Road": 300,
        "unlabeled": 35
    },
    "unlabeled": {
        "Along Wall": 5,
        "Around Corner": 5,
        "Enter Intersection/Exit Bridge": 20,
        "Exit Intersection/Enter Bridge": 30,
        "In Intersection": 25,
        "On Bridge": 6,
        "Open Space": 0,
        "Past Building": 5,
        "Road": 50,
        "unlabeled": 0
    }
    })
    
    # Load the JSON string into a Python dictionary
    cm_dict = json.loads(confusion_matrix_json)
    
    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame.from_dict(cm_dict)
    
    # Transpose the DataFrame to flip the axes
    df_flipped = df.transpose()
    
    # Convert the flipped DataFrame back to a dictionary for plotting
    flipped_cm_dict = df_flipped.to_dict()

    # Generate and save the standard confusion matrix image
    save_confusion_matrix_as_image(flipped_cm_dict, "flipped_confusion_matrix.png", is_normalized=False)
    
    # Generate and save the normalized confusion matrix image
    save_confusion_matrix_as_image(flipped_cm_dict, "normalized_flipped_confusion_matrix.png", is_normalized=True)

if __name__ == "__main__":
    main()
