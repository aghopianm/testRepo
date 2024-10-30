import pandas as pd

# Load the data from your original CSV file
df = pd.read_csv("PeopleTrainngDate.csv")

# Load new records from the new CSV file
new_records = pd.read_csv("PeopleTrainingDateUpdate.csv", header=None)

# Specify the correct columns for new_records. Assuming the format is: Updated, Email and ID, Title, Name, Company
new_records.columns = ['Updated', 'Email_ID', 'Title', 'Name', 'Company']

# Extract the email and ID from the combined 'Email_ID' column
# Assuming the format is "email id", split by space
new_records[['Email', 'ID']] = new_records['Email_ID'].str.split(' ', n=1, expand=True)

# Convert 'Updated' to datetime format
new_records['Updated'] = pd.to_datetime(new_records['Updated'], dayfirst=True)

# Reorganize new_records DataFrame to have 'Updated' first
new_records = new_records[['Updated', 'Title', 'Name', 'Company', 'Email', 'ID']]

# Ensure the original DataFrame's 'Updated' is also first
original_columns = ['Updated'] + [col for col in df.columns if col != 'Updated']
df = df[original_columns]

# Convert the 'Updated' column to datetime format for the original DataFrame
df['Updated'] = pd.to_datetime(df['Updated'], dayfirst=True)

# Combine the original DataFrame with new records
combined_df = pd.concat([df, new_records], ignore_index=True)

# Sort the combined DataFrame by the 'Updated' column in ascending order
combined_df.sort_values(by='Updated', inplace=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("CombinedRecords.csv", index=False)

# Display the sorted DataFrame
print(combined_df)
