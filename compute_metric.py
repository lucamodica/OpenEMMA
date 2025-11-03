import json
import os

# read jsonl file from path
def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of dictionaries.
    Each line in the file is expected to be a valid JSON object.
    """
    data = []
    # Expands the '~' to the user's home directory
    expanded_path = os.path.expanduser(file_path)
    with open(expanded_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line.strip()}")
    return data

# take the mean of all metrics
def compute_mean_metric(results, metric_name):
    """
    Computes the mean of a specific metric from a list of result dictionaries.
    """
    values = []
    for item in results:
        # Assuming the structure is a list of dicts, and each dict might have nested results.
        # This part might need adjustment based on the actual structure of your JSONL objects.
        # This assumes each line is a dictionary that might contain another dictionary of metrics.
        # If the metric is directly in the top-level dictionary of each line:
        if metric_name in item:
            values.append(item[metric_name])
        # If the metrics are nested under a key, for example 'metrics':
        elif 'metrics' in item and metric_name in item['metrics']:
            values.append(item['metrics'][metric_name])

    if len(values) == 0:
        return None
    return sum(values) / len(values)

if __name__ == '__main__':
    # It's better to use a variable for the file path
    file_path = ...
    results = read_jsonl(file_path)

    # The original script assumes a dictionary of sequences,
    # but a JSONL file is a list of objects.
    # The compute_mean_metric function has been adjusted.
    # To get the total number of results, we now use the length of the list.
    num_results = len(results)

    # Assuming the metric name is directly in each JSON object (line)
    # If your JSON objects have a structure like {"seq_name": "...", "ade1s": 0.5, ...}
    # then this should work.
    ade1 = compute_mean_metric(results, 'ade1s')
    ade2 = compute_mean_metric(results, 'ade2s')
    ade3 = compute_mean_metric(results, 'ade3s')
    avgade = compute_mean_metric(results, 'avgade')

    print(f"Number of evaluated scenes: {num_results}")
    print(f"Mean ADE 1s: {ade1}")
    print(f"Mean ADE 2s: {ade2}")
    print(f"Mean ADE 3s: {ade3}")
    print(f"Mean AVG ADE: {avgade}")