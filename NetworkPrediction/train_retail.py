def create_2d_list_from_csv(filename):
    data_2d = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Skip header and process each line
        for line in lines[1:]:  # Skip first line (header)
            values = line.strip().split(',')
            # Take the first 6 values from each row
            first_six = values[:6]
            data_2d.append(first_six)
    
    return data_2d

# Usage
filename = r"D:\NetworkPrediction\misceleaneous\inspected_data.csv"
result = create_2d_list_from_csv(filename)

# Print first few rows to verify
for i in range(min(60, len(result))):
    print(result[i])