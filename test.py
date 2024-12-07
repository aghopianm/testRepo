import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

#setting style for sns plotting later to come in the programme, I liked whitegrid the most

sns.set(style="whitegrid")

#Main variables that are used through the programme are initialised here at the top. 
#I have had to do this due to scope issues. 

activity_logs_dataframe = None
component_codes_dataframe = None
user_log_dataframe = None
fully_merged_dataset = None
reshaped_data = None
#this varialbe is to ensure that you HAVE to rename&remove (task1&2, before you are allowed to merge)

is_data_removed_and_renamed = False

# Function to load and clean data from CSV files

def data_load_from_csv_and_clean():
    global activity_logs_dataframe, component_codes_dataframe, user_log_dataframe
    file_paths = filedialog.askopenfilenames(title="Select CSV files, activity logs first, component codes second, user logs last", filetypes=[("CSV files", "*.csv")])
    if len(file_paths) != 3:
        messagebox.showerror("Error", "Please select exactly three CSV files, activity logs first, component codes second, user logs last.")
        return

    #Load all CSVs to a pandas dataframe, pandas > np in this case as we are going to have to
    #heavily manipulate the dataset and reshape it.
    activity_logs_dataframe = pd.read_csv(file_paths[0])
    component_codes_dataframe = pd.read_csv(file_paths[1])
    user_log_dataframe = pd.read_csv(file_paths[2])

    #Strip 
    activity_logs_dataframe.columns = activity_logs_dataframe.columns.str.strip()
    component_codes_dataframe.columns = component_codes_dataframe.columns.str.strip()
    user_log_dataframe.columns = user_log_dataframe.columns.str.strip()

    #Clean activity logs first
    activity_logs_dataframe.fillna({'Component': 'Unknown', 'Action': 'Unknown', 'Target': 'Unknown', 'User Full Name *Anonymized': 0}, inplace=True)
    activity_logs_dataframe['Target'] = activity_logs_dataframe['Target'].str.replace('_', '', regex=False)
    activity_logs_dataframe['Component'] = activity_logs_dataframe['Component'].str.replace('_', '', regex=False)
    
    #Clean user logs next
    #ensuring that the data column is in a format I can manipulate in the future 
    user_log_dataframe['Date'] = pd.to_datetime(user_log_dataframe['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
    user_log_dataframe['Date'] = user_log_dataframe['Date'].dt.strftime('%d/%m/%Y')
    #flling in missing dates with the date for Christmas.
    user_log_dataframe['Date'].fillna('25/11/2023', inplace=True)
    user_log_dataframe['Time'] = user_log_dataframe['Time'].str.strip()
    #filling in a blanket time of 00 for time that is missing.
    user_log_dataframe['Time'].fillna('00:00:00', inplace=True)

    #Clean component codes last
    component_codes_dataframe['Component'] = component_codes_dataframe['Component'].str.replace('_', '', regex=False)
    component_codes_dataframe['Code'] = component_codes_dataframe['Code'].str.replace('_', '', regex=False)

    update_status_bar("CSV files loaded and cleaned successfully.")
    messagebox.showinfo("Info", "Data loaded and cleaned successfully.")

#save data from the csv to a json format.

#This function saves the three cleaned dataframes to three seperate JSON files.
#I use JSON because of its lightweight and readable format, it is also great in this instance
#where the nature of the relationship between the 'variable' and teh data is in a traditional
#key value pair type. 

def save_cleaned_data():
    global activity_logs_dataframe, component_codes_dataframe, user_log_dataframe
    if activity_logs_dataframe is None or component_codes_dataframe is None or user_log_dataframe is None:
        messagebox.showerror("Error", "Data not loaded. Please load and clean data first.")
        update_status_bar("Error with data saving")
        return

    activity_logs_dataframe.to_json("activity_logs_in_json_format.json", orient="records", lines=False, indent=4)
    component_codes_dataframe.to_json("component_codes_in_json_format.json", orient="records", lines=False, indent=4)
    user_log_dataframe.to_json("user_logs_in_json_format.json", orient="records", lines=False, indent=4)

    update_status_bar("files saved to JSON.")
    messagebox.showinfo("Info", "CSV files are now saved as JSON files.")

#function to load data from json to dataframes
def data_load_from_json():
    global activity_logs_dataframe, component_codes_dataframe, user_log_dataframe, is_data_removed_and_renamed
    file_paths = filedialog.askopenfilenames(title="Select JSON files, activity logs first, component codes second, user logs last", filetypes=[("JSON files", "*.json")])
    
    if len(file_paths) != 3:
        messagebox.showerror("Error", "Please select exactly three JSON files, first one activity logs, second one component codes, last one user logs.")
        return

    # Load the JSON files into DataFrames
    activity_logs_dataframe = pd.read_json(file_paths[0], orient='records')
    component_codes_dataframe = pd.read_json(file_paths[1], orient='records')
    user_log_dataframe = pd.read_json(file_paths[2], orient='records')

    #RECLEAN to ensure consistency, this is JUST IN CASE the user has not cleaned properly 
    #in the first step or there is some missmatch when reading from json/to avoid errors.
    #whilst this does increase the length of the code and I am repeating myself, this is just to handle
    #errors.
    activity_logs_dataframe.columns = activity_logs_dataframe.columns.str.strip()
    component_codes_dataframe.columns = component_codes_dataframe.columns.str.strip()
    user_log_dataframe.columns = user_log_dataframe.columns.str.strip()

    activity_logs_dataframe.fillna({'Component': 'Unknown', 'Action': 'Unknown', 'Target': 'Unknown', 'User Full Name *Anonymized': 0}, inplace=True)
    activity_logs_dataframe['Target'] = activity_logs_dataframe['Target'].str.replace('_', '', regex=False)
    activity_logs_dataframe['Component'] = activity_logs_dataframe['Component'].str.replace('_', '', regex=False)

    #this is NECESSARY, as further in the programme if you try to perform the reshaping once you load
    #FROM the JSON and not from the CSV, this will throw an error as date is NOT being treated
    #as a string. 
    if 'Date' in user_log_dataframe.columns:
        user_log_dataframe['Date'] = pd.to_datetime(user_log_dataframe['Date'], format='%d/%m/%Y', errors='coerce')
        user_log_dataframe['Date'] = user_log_dataframe['Date'].dt.strftime('%d/%m/%Y')
    
    #further cleaning that follows the same code as the code to clean the csv files.
    user_log_dataframe['Date'].fillna('25/11/2023', inplace=True)
    user_log_dataframe['Time'] = user_log_dataframe['Time'].str.strip()
    user_log_dataframe['Time'].fillna('00:00:00', inplace=True)
    
    component_codes_dataframe['Component'] = component_codes_dataframe['Component'].str.replace('_', '', regex=False)
    component_codes_dataframe['Code'] = component_codes_dataframe['Code'].str.replace('_', '', regex=False)
    
    update_status_bar("loaded files from JSON.")
    # Inform the user of success
    messagebox.showinfo("Info", "Loaded all three JSON files successfully.")

#Data manipulation and outputs tasks:
#Task 1 and 2
def remove_and_rename():
    global activity_logs_dataframe, user_log_dataframe, component_codes_dataframe, is_data_removed_and_renamed
    if activity_logs_dataframe is None or component_codes_dataframe is None or user_log_dataframe is None:
        messagebox.showerror("Error", "Please load and clean the data first.")
        return
    #Remove task:
    #Remove system and folder 
    activity_logs_dataframe = activity_logs_dataframe[~activity_logs_dataframe['Component'].isin(['System', 'Folder'])]
    #Rename task:
    activity_logs_dataframe.rename(columns={'User Full Name *Anonymized': 'User_ID'}, inplace=True)
    user_log_dataframe.rename(columns={'User Full Name *Anonymized': 'User_ID'}, inplace=True)

    is_data_removed_and_renamed = True

    update_status_bar("removed&renamed .")
    messagebox.showinfo("Info", "Thank you, you have removed and renamed successsfully")

#Task 3 
# Function to merge data
def merge_data():
    global fully_merged_dataset
    if not is_data_removed_and_renamed:  #check if use has performed remove&rename step first.
        messagebox.showerror("Error", "Please complete the Remove & Rename step first.")
        return
    #ensure that there is data there to be merged.
    if activity_logs_dataframe is None or component_codes_dataframe is None or user_log_dataframe is None:
        messagebox.showerror("Error", "Please load and clean the data first, and remove&rename.")
        return
    messagebox.showinfo("Info", "This process may take up to 5-10 minutes. Please wait.")
    #merge only takes two arguments, so I need to merge two dataframes first, then take the merged
    #dataframe and merge that into one final dataframe to have a fully merged dataset
    first_merge = pd.merge(activity_logs_dataframe, component_codes_dataframe, on='Component', how='left')
    fully_merged_dataset = pd.merge(first_merge, user_log_dataframe, on='User_ID', how='left')
    
    update_status_bar("data merged.")
    messagebox.showinfo("Info", "Data merged into one dataframe successfully.")

#Task 4 (Task 5 is also nested here)
# Function to reshape data
def reshape_data():
    global reshaped_data
    #handle errors where I cannot reshape the data as it has not already been merged.
    if fully_merged_dataset is None:
        messagebox.showerror("Error", "Please merge the data first.")
        return
    messagebox.showinfo("Info", "This process may take up to 5-10 minutes. Please wait.")
    
    #I am string slicing here to get the month.
    fully_merged_dataset['Month'] = fully_merged_dataset['Date'].str[3:5]
    #Task 5 of counting, interaction count = task 5
    interaction_count = fully_merged_dataset.groupby(['User_ID', 'Component', 'Month']).size().reset_index(name='Interaction_Count')
    reshaped_data = pd.pivot_table(
        interaction_count,
        index=['User_ID'],
        columns=['Component', 'Month'],
        values='Interaction_Count',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    update_status_bar("Data reshaped")
    messagebox.showinfo("Info", "Data reshaped successfully.")

#The following function is for converting the reshaped and merged datframes into two seperate
#json files. The reason why I do this is to preserve the state of the application, so that if a
#user has already gone through the process of cleaning the data, merging and reshaping, they can
#then save this data to two JSON files, so they can load from it in the future, avoiding the need
#for them to have to go back and go through the process of loading, cleaning etc again.
#I have chosen to save and load in chunks, as on my computer (m1 macbook air), I could not save/load
#all of the data in one process. I suspest this is due to the fact that JSON is uncompressed and large.
#Furthermore, I dont think I explicitly need to save the merged data as well as the reshaped data, as
#all functions moving forward are only requiring the reshaped data, which is a much shorter process.
#I have, however, chosen to include saving/loading the merged data also, in order to preserve the
#integrity of the data, in future iterations of this programme, the reshaping may be altered, and 
#therefore it is important to still have access to the three files in a merged format to be able to reshape.

def save_large_json(dataframe, filename, chunk_size=1000):
    #json files opened for writing.
    with open(filename, 'w') as f:
        #initiate array.
        f.write("[\n")
        
        #dataframe looped through in chunks ( size can be given by me here)
        for i in range(0, len(dataframe), chunk_size):
            #get chunk
            chunk = dataframe.iloc[i:i + chunk_size]
            #pass to a json
            chunk_json = chunk.to_json(orient="records", lines=False)
            #then take the json to a file.
            f.write(chunk_json[1:-1])  # Exclude the outer brackets for each chunk
            
            # addcomma between chunks. except after the last chunk to avoid erros.
            if i + chunk_size < len(dataframe):
                f.write(",\n")
            #progress printed to console so I dont think its not working.
            print(f"Printed chunk {i // chunk_size + 1} for {filename}")

        #close array. 
        f.write("\n]")

#purpose of this function is a helper function specifically for the reshaped data as it was throwing
#errors when I used savelargejson() on the reshaped data. This function has a much samaller chunksize
def save_reshaped_data():
    global reshaped_data
    if reshaped_data is None:
        messagebox.showerror("Error", "Data not yet reshaped.")
        return
    
    #reshaped data was throwing erros when using larger chunk sizes so I have put a smaller size. 
    save_large_json(reshaped_data, "reshaped_data.json", chunk_size=10)
    print("Reshaped data saved successfully.")

#Function to save the fully merged and reshaped data to seperate json files for future use and to
#maintain the programme moving forward.
def save_prepared_data():
    global fully_merged_dataset, reshaped_data
    if fully_merged_dataset is None or reshaped_data is None:
        messagebox.showerror("Error", "Data not yet merged and reshaped.")
        return
    messagebox.showinfo("Info", "This process will take a VERY LONG TIME - Please wait up to 20 mins depending on your system")
    save_large_json(fully_merged_dataset, "fully_merged_dataset.json")
    print("Merged data saved, now I will do the reshaped data.")
    save_reshaped_data()

    update_status_bar("All files saved as Merged data and Reshaped data")
    messagebox.showinfo("Info", "Prepared data saved as two seperate JSON files.")

#Function to load the fully merged and reshaped data from seperate files so you don't have to 
#go through the whole process of cleaning, merging, loading etc from the csv every time if you have
#a prepared dataset.

def load_large_json(filename, chunk_size=1000):

    #generator function within the load large json. this handles chunk generation 
    def json_chunk_generator(file_obj, chunk_size):
        buffer = []
        in_array = False
        for line in file_obj:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Handle start of array
            if line == '[':
                in_array = True
                continue
            # Handle end of array
            elif line == ']':
                if buffer:
                    yield '[' + ','.join(buffer) + ']'
                break
                
            # Handle normal/standard lines.
            if in_array:
                #remove commas if present
                if line.endswith(','):
                    line = line[:-1]
                buffer.append(line)
                
                if len(buffer) >= chunk_size:
                    yield '[' + ','.join(buffer) + ']'
                    buffer = []
    
    data_chunks = []
    try:
        with open(filename, 'r') as f:
            for chunk_json in json_chunk_generator(f, chunk_size):
                try:
                    chunk_df = pd.read_json(chunk_json, orient='records')
                    data_chunks.append(chunk_df)
                    print(f"Loaded chunk with {len(chunk_df)} records from {filename}")
                except ValueError as e:
                    print(f"Error processing chunk: {e}")
                    continue
                    
        if not data_chunks:
            raise ValueError("No valid data chunks were loaded")
            
        result = pd.concat(data_chunks, ignore_index=True)
        #I will get 50,000 at a time from my pre existing JSON, this takes a very long time on my
        #m1 macbook air and I am not sure if this is the best way to programme this.
        #perhaps a solution moving forward could be using threads for IO operaitons, there is
        #lots of issues in this programme with IO that could be optimised through threading.
        print(f"Successfully loaded total of {len(result)} records from {filename}")
        return result
        
    except Exception as e:
        print(f"Error loading file {filename}: {str(e)}")
        raise

def load_prepared_data():
    global fully_merged_dataset, reshaped_data
    try:
        messagebox.showinfo("Info", "This process will take a VERY LONG TIME - Please wait up to 60 mins depending on your hardware")
        
        #Load the fully merged dataset with a larger chunk size
        print("Loading fully merged dataset...")
        fully_merged_dataset = load_large_json("fully_merged_dataset.json", chunk_size=500)
        
        #Load the reshaped data with a smaller chunk size since it is much, much wider.
        print("Loading reshaped data...")
        reshaped_data = load_large_json("reshaped_data.json", chunk_size=100)
        
        update_status_bar("Data loaded.")
        messagebox.showinfo("Info", "Prepared data loaded successfully.")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error loading prepared data: {str(e)}")
        print(f"Detailed error: {str(e)}")

#Task 6
def output_statistics():
    global reshaped_data
    
    if reshaped_data is None:
        messagebox.showerror("Error", "Please reshape the data first.")
        return

    #throughout my programme I will keep making copyies to avoid messing with the global reshaped data
    #variable as I had issues when flattening columns from one function to another.
    manipulated_data = reshaped_data.copy()

    # Components of interest that I need to extract.
    components_of_interest = ['Quiz', 'Lecture', 'Assignment', 'Attendence', 'Survey']
    
    # Flatten columns to access them easily in the copied data.
    manipulated_data.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in manipulated_data.columns]

    # Get columns that contain component-month pairs
    relevant_columns = [col for col in manipulated_data.columns if any(comp in col for comp in components_of_interest)]

    #Task 6a
    # Monthly statistics (September, October, November, December)
    monthly_statistics = {}
    for month in ['09', '10', '11', '12']:
        month_columns = [col for col in relevant_columns if f'_{month}' in col]
        month_data = manipulated_data[month_columns]

        monthly_statistics[month] = {}
        
        #mean for each month
        monthly_statistics[month]['Mean'] = month_data.mean().round(2)
        
        #median for each month
        monthly_statistics[month]['Median'] = month_data.median().to_dict()
        
        #first mode returned for each month with [0]
        mode_values = month_data.mode()
        if not mode_values.empty:
            monthly_statistics[month]['Mode'] = mode_values.iloc[0].to_dict()  #each mode for component.
        else:
            #Handle the case where there's no mode as i think thats more useful than returning
            #nan or 0 to the console.
            monthly_statistics[month]['Mode'] = 'No mode'  
    #Task 6b
    #Semester statistics 09-12, september to december.
    semester_statistics = {}
    for component in components_of_interest:
        # get components across every single month
        component_columns = [col for col in relevant_columns if col.startswith(component)]
        
        component_data = manipulated_data[component_columns]
        
        #all month means calculated.
        semester_statistics[component] = {}
        semester_statistics[component]['Mean'] = component_data.mean().mean()  # Mean of all months combined
        
        #medians calculated across all months. (i.e: i am getting the median of medians, mean of means etc)
        semester_statistics[component]['Median'] = component_data.median().median()  #Median of all months combined
        
        #all  month modes.
        # Combine data from all months for the component and calculate mode
        # Stack all the month data into a single column, I did it this way as I did not want to have many
        #modes simply for the semester, it didnt really make sense having individual month modes for the
        #semester as this was what I was getting when I coded initially.
        combined_data = component_data.stack() 
        mode_values = combined_data.mode()
        if not mode_values.empty:
            semester_statistics[component]['Mode'] = mode_values.iloc[0] 
        else:
            semester_statistics[component]['Mode'] = 'No mode'  # Handle the case where there's no mode
    
    # Prepare output message for monthly statistics
    stats_message = "Monthly statistics:\n"
    for month, stats in monthly_statistics.items():
        stats_message += f"\nMonth: {month}\nMean:\n{stats['Mean']}\nMedian:\n{stats['Median']}\nMode:\n{stats['Mode']}\n"
    
    # Prepare output message for semester statistics
    stats_message += "\nSemester statistics:\n"
    for component, stats in semester_statistics.items():
        stats_message += f"\nComponent: {component}\nMean: {stats['Mean']}\nMedian: {stats['Median']}\nMode: {stats['Mode']}\n"
    
    # Print statistics to the console
    print(stats_message)

    #keep me updated.
    update_status_bar("Statistics shown.")
    messagebox.showinfo("Info", "Statistics have been printed to the console, please check.")

#Task 7 - the first part of the task, plotting the graphs
def plot_bar_graphs():
    global reshaped_data
    
    if reshaped_data is None:
        messagebox.showerror("Error", "Please reshape the data first.")
        return

    components_of_interest = ['Assignment', 'Quiz', 'Lecture', 'Book', 'Project', 'Course']

    months_of_interest = ['09', '10', '11', '12']

    # Make a copy of reshaped_data to avoid modifying the original
    plot_data = reshaped_data.copy()
    
    #i flatten here to make it easier to plot.
    plot_data.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in plot_data.columns]
    
    for component in components_of_interest:
        #Filter columns for this componentwithin the months I want to plot.
        component_columns = [col for col in plot_data.columns 
                             if col.startswith(component) and any(f"_{month}" in col for month in months_of_interest)]
        
        if component_columns:  
            # Suminteractions across all users for each month
            component_data = plot_data[component_columns].sum()
            
            #ensure the data is ordered by month. 
            component_data = component_data.sort_index(key=lambda x: [int(m.split('_')[-1]) for m in x])
            
            # Create bar plot
            plt.figure(figsize=(8, 6))
            plt.bar(component_data.index, component_data.values)
            plt.title(f'User Interactions with {component} Across September through to December')
            plt.xlabel('Month & Component')
            plt.ylabel('Total Interactions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        update_status_bar("Bar graphs showing.")

#These two functions below help with task 7 correlation calculations.
#pearson correlation calculation
def pearson_corr(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2)) * np.sqrt(np.sum((y - y_mean)**2))
    return numerator / denominator

    #spearman correlation calculation
def spearman_corr(x, y):
    # Rank data
    x_rank = pd.Series(x).rank().to_numpy()
    y_rank = pd.Series(y).rank().to_numpy()
    # Use Pearson correlation on ranks
    return pearson_corr(x_rank, y_rank)
    
#Task 7 seond half, calculating and plotting correlation
def plot_user_component_correlation():
    global reshaped_data

    if reshaped_data is None:
        messagebox.showerror("Error", "Please reshape the data first.")
        return

    #again, making a copy of reshaped data to avoid errors with modifying the original.
    correlation_data = reshaped_data.copy()

    # Set index with userID, empty string as a tuple as well due to the multi-index nature of the
    #dataframe.
    correlation_data = correlation_data.set_index(('User_ID', ''))

    #columns flattened so I can access each component/month combo like quiz_09 in one line
    correlation_data.columns = ['_'.join(map(str, col)) for col in correlation_data.columns]

    #get USERID
    user_id_data = correlation_data.index.get_level_values(0) 

    #get columns for the components except for USERID
    component_columns = [col for col in correlation_data.columns if 'User_ID' not in col]

    #results storage, best in a dictionary.
    correlation_results = {}

    #calculate correlations with a for loop
    for component in component_columns:
        component_data = correlation_data[component].to_numpy()

        pearson_correlation = pearson_corr(component_data, user_id_data)

        spearman_correlation = spearman_corr(component_data, user_id_data)

        correlation_results[component] = {
            'Pearson Correlation': pearson_correlation,
            'Spearman Correlation': spearman_correlation,
        }
        
    #print results to console
    print("Correlation Results with User_ID:")
    for component, result in correlation_results.items():
        print(f"\nComponent: {component}")
        print(f"Pearson Correlation: {result['Pearson Correlation']}")
        print(f"Spearman Correlation: {result['Spearman Correlation']}")

    #i will plot using a dataframe so results must be converted.
    correlation_df = pd.DataFrame(correlation_results).T

    #heatmap for both pearson and spearman correlation, splitting them
    #pearson first
    plt.figure(figsize=(10, 6))
    #I have set annot to false as it was cluttering my output, annot just shows the numbers on top of
    #the visualisation. 
    sns.heatmap(correlation_df[['Pearson Correlation']], annot=False, cmap='coolwarm', center=0)
    plt.title("Pearson Correlation of Components with User_ID")
    plt.show()

    #spearman correlation next
    plt.figure(figsize=(10, 6))
    #I have set annot to false as it was cluterring my output - see above comment.
    sns.heatmap(correlation_df[['Spearman Correlation']], annot=False, cmap='coolwarm', center=0)
    plt.title("Spearman Correlation of Components with User_ID")
    plt.show()

    update_status_bar("Correlation calculated and shown.")
    messagebox.showinfo("Info", "Correlation results with User_ID have been printed to the console, please check.")


def update_status_bar(message):
    status_bar.config(text=message)
    root.update_idletasks()

#POTENTIAL FINAL SOLUTION FOR GUI. 
# Setup Tkinter GUI
root = tk.Tk()
root.title("GUI for my prototype application")

#Main menu bar
menubar = tk.Menu(root)
root.config(menu=menubar)

loading_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="1. Load your files", menu=loading_menu)
loading_menu.add_command(label="Load and Clean Data FROM CSV", command=data_load_from_csv_and_clean)
loading_menu.add_command(label="Load the three JSON files and clean (individual files)", command=data_load_from_json)
loading_menu.add_command(label="Load Prepared Data from JSON (merged and reshaped)\n BEWARE, VERY LONG PROCESS", 
                              command=load_prepared_data)

#data storage menu
saving_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="2. Save your data", menu=saving_menu)
saving_menu.add_command(label="Save Cleaned Data to JSON (individual files)", command=save_cleaned_data)
saving_menu.add_command(label="Save Prepared Data to JSON (one big JSON file)\nBEWARE, VERY LARGE PROCESS", 
                              command=save_prepared_data)

#All things data processing in one menu
data_processing_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="3. Data processing", menu=data_processing_menu)
data_processing_menu.add_command(label="Remove and Rename", command=remove_and_rename)
data_processing_menu.add_command(label="Merge Data", command=merge_data)
data_processing_menu.add_command(label="Reshape Data", command=reshape_data)

#statistics and visualiation menu
visualization_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="4. Statistics & Visualization", menu=visualization_menu)
visualization_menu.add_command(label="Output Statistics", command=output_statistics)
visualization_menu.add_command(label="Component Interactions Bar Graphs", command=plot_bar_graphs)
visualization_menu.add_command(label="User-Component Correlation Heatmap/Correlation", command=plot_user_component_correlation)

#status bar at the bottom, not sure how much value this adds but it looks good.
status_bar = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

#instruction and welcome menu
welcome_label = ttk.Label(root, text="Prototype application, welcome!\n\nPlease use the menus above to:\n1. Load your data\n2. Save your data, don't lose progress!\n3. Process your data (either CSV or JSON)\n 4. Visualize results and output statistics", justify=tk.CENTER, padding=20)
welcome_label.pack(expand=True)

root.mainloop()