import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os

def extract_learning_rate(filename):
    """Extracts learning rate from filename."""
    match = re.search(r"lr_([\d.eE-]+)\.csv", filename)
    if match:
        return match.group(1)
    return None

def extract_task_number(row):
    """Extracts the 'trained_upto_task' number from ModelLoaded or EvaluationName."""
    model_loaded = row['ModelLoaded']
    eval_name = row['EvaluationName']

    match_model = re.search(r'task(\d+)\.pth', model_loaded)
    if match_model:
        return int(match_model.group(1))

    if 'eval_task1_after_task1' == eval_name: # Exact match for baseline
        return 1
    if 'eval_all_after_task' in eval_name: # e.g., eval_all_after_task6
        match_eval = re.search(r'task(\d+)', eval_name)
        if match_eval:
            return int(match_eval.group(1))
    if 'eval_cumul_after_task' in eval_name: # e.g., eval_cumul_after_task2
        match_eval = re.search(r'task(\d+)', eval_name)
        if match_eval:
            return int(match_eval.group(1))
    if 'eval_task1_after_task' in eval_name: # e.g., eval_task1_after_task2
        match_eval = re.search(r'task(\d+)', eval_name)
        if match_eval:
            return int(match_eval.group(1))
    if '_new' in eval_name and 'task' in model_loaded : # e.g. eval_task2_new, model is incremental_add_one_task2.pth
         match_model_fallback = re.search(r'task(\d+)\.pth', model_loaded)
         if match_model_fallback:
            return int(match_model_fallback.group(1))

    print(f"Warning: Could not determine task number for row: {eval_name}, {model_loaded}")
    return None


def define_eval_type(evaluation_name):
    """Defines the evaluation type based on the EvaluationName."""
    if 'eval_task1_after_task' in evaluation_name:
        return 'Task 1 Performance (Forgetting)'
    elif '_new' in evaluation_name: # e.g. eval_task2_new, eval_task3_new
        return 'New Task Performance'
    elif 'cumul_after_task' in evaluation_name or 'all_after_task' in evaluation_name :
        return 'Cumulative Performance'
    return 'Other'

def main():
    results_path_pattern = "results/results_incremental_add_one_epochs_1_lr_*.csv"
    output_dir = "generated_plots"
    os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist

    all_dfs = []
    csv_files = glob.glob(results_path_pattern)

    if not csv_files:
        print(f"No CSV files found matching the pattern: {results_path_pattern}")
        print("Please ensure your CSV files are in a 'results' subdirectory and match the naming convention.")
        return

    print(f"Found {len(csv_files)} CSV files:")
    for f_path in csv_files:
        print(f"  - {f_path}")
        lr = extract_learning_rate(os.path.basename(f_path))
        if lr:
            try:
                df = pd.read_csv(f_path)
                df['lr'] = lr
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading or processing {f_path}: {e}")
        else:
            print(f"Could not extract learning rate from: {f_path}")

    if not all_dfs:
        print("No data loaded. Exiting.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\nSuccessfully loaded and combined data from {len(all_dfs)} files.")
    print("Combined DataFrame head:")
    print(df_all.head())

    df_all['TrainedUptoTask'] = df_all.apply(extract_task_number, axis=1)
    df_all['EvaluationType'] = df_all['EvaluationName'].apply(define_eval_type)
    df_all['CurrentTask'] = df_all['TrainedUptoTask'] # For these plots, CurrentTask evaluated is same as tasks trained

    df_all.dropna(subset=['TrainedUptoTask'], inplace=True)
    df_all['TrainedUptoTask'] = df_all['TrainedUptoTask'].astype(int)
    df_all['CurrentTask'] = df_all['CurrentTask'].astype(int)


    print("\nProcessed DataFrame info:")
    df_all.info()
    print("\nValue counts for EvaluationType:")
    print(df_all['EvaluationType'].value_counts())
    print("\nValue counts for TrainedUptoTask:")
    print(df_all['TrainedUptoTask'].value_counts())

    if not df_all.empty and 'lr' in df_all.columns:
        unique_lrs_str = df_all['lr'].unique()
        try:
            lr_float_map = {lr_str: float(lr_str) for lr_str in unique_lrs_str}
            sorted_unique_lrs_str = sorted(unique_lrs_str, key=lambda x: lr_float_map[x], reverse=True)
        except ValueError:
            print("Warning: Could not convert all learning rates to float for sorting. Using lexicographical sort.")
            sorted_unique_lrs_str = sorted(unique_lrs_str, reverse=True) # Fallback to string sort
    else:
        sorted_unique_lrs_str = []
    
    print(f"Determined hue order for learning rates: {sorted_unique_lrs_str}")


    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Task 1 Accuracy (Forgetting)
    df_forget = df_all[df_all['EvaluationType'] == 'Task 1 Performance (Forgetting)']
    if not df_forget.empty:
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_forget, x='CurrentTask', y='Accuracy', hue='lr', 
                     hue_order=sorted_unique_lrs_str, # Apply consistent order
                     marker='o', palette='viridis', errorbar='sd')
        plt.title('Task 1 Accuracy vs. Number of Tasks Trained', fontsize=16)
        plt.xlabel('Tasks Trained', fontsize=14)
        plt.ylabel('Accuracy on Task 1 (%)', fontsize=14)
        if not df_forget['CurrentTask'].empty:
             plt.xticks(sorted(df_forget['CurrentTask'].unique()), fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Learning Rate', title_fontsize='13', fontsize='11', frameon=True, shadow=True, fancybox=True)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plot1_path = os.path.join(output_dir, "task1_forgetting_accuracy_multicsv.png")
        plt.savefig(plot1_path)
        # plt.show()
        print(f"\nPlot 1: Task 1 Forgetting Accuracy saved to {plot1_path}")
    else:
        print("\nNo data found for 'Task 1 Performance (Forgetting)' plot.")

    # Plot 2: Cumulative Accuracy
    df_cumulative_raw = df_all[df_all['EvaluationType'] == 'Cumulative Performance'].copy()

    df_task1_initial = df_all[
        (df_all['EvaluationType'] == 'Task 1 Performance (Forgetting)') &
        (df_all['CurrentTask'] == 1)
    ].copy()

    if not df_task1_initial.empty:
        df_task1_initial['EvaluationType'] = 'Cumulative Performance' # Match type for concat
        df_cumulative = pd.concat([df_task1_initial, df_cumulative_raw], ignore_index=True)
        df_cumulative = df_cumulative.sort_values(by=['lr', 'CurrentTask']).drop_duplicates(subset=['lr', 'CurrentTask'], keep='first')
    else:
        print("Warning: Could not find initial Task 1 performance data (eval_task1_after_task1) to include in cumulative plot.")
        df_cumulative = df_cumulative_raw


    if not df_cumulative.empty:
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=df_cumulative, x='CurrentTask', y='Accuracy', hue='lr',
                     hue_order=sorted_unique_lrs_str, # Apply consistent order
                     marker='o', palette='viridis', errorbar='sd')
        plt.title('Overall Cumulative Accuracy vs. Number of Tasks Trained', fontsize=16)
        plt.xlabel('Tasks Trained', fontsize=14)
        plt.ylabel('Cumulative Accuracy on All Tasks Seen (%)', fontsize=14)
        if not df_cumulative['CurrentTask'].empty:
            plt.xticks(sorted(df_cumulative['CurrentTask'].unique()), fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Learning Rate', title_fontsize='13', fontsize='11', frameon=True, shadow=True, fancybox=True)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plot2_path = os.path.join(output_dir, "cumulative_accuracy_multicsv.png")
        plt.savefig(plot2_path)
        # plt.show()
        print(f"Plot 2: Cumulative Accuracy saved to {plot2_path}")
    else:
        print("\nNo data found for 'Cumulative Performance' plot.")

    print("\nScript finished.")

if __name__ == '__main__':
    main()