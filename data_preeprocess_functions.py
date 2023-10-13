from external_libs_imports import *


def make_time_serialized_categories(df, prefix, time_lines):
    cat_cols = []
    # Extract unique categories within the specified prefix
    convert_columns = df.filter(like=prefix)
    categories = convert_columns.stack().dropna().unique().tolist()
    print(f'converting {len(convert_columns.columns)} time periods within {len(time_lines)}  {prefix} columns to {len(categories)} categorical binary time seriesed columns and dropping the originals')

    for category in categories:
        for timeline in time_lines:
            for time_point in ['1hodesh', '3hodesh', '6hodesh', '9hodesh', '12hodesh']:
                # Create binary columns for time line for each category and time point
                df[f'{prefix}_{category.replace(" ", "_")}_{time_point}_{timeline}'] = np.where(df[f'{prefix}_{time_point}_{timeline}'] == category, 1, 0)
                cat_cols.append(f'{prefix}_{category.replace(" ", "_")}_{time_point}_{timeline}')

    # Drop the original columns with prefixes and time points
    df.drop(convert_columns.columns, axis=1, inplace=True)
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('/', '_or_')
    return df, cat_cols


def sort_key(column_name):
    numeric_part = re.search(r'\d+', column_name)
    if numeric_part:
        return int(numeric_part.group())
    else:
        return -1


def sort_columns(df, reverse=False):
    sorted_column_names = sorted(df.columns, key=lambda x: (re.split('_|\d+', x)[0], int(re.search(r'\d+', x).group(0)) if re.search(r'\d+', x) else 0), reverse=reverse)
    return df[sorted_column_names]


def get_columns_by_sub_name(df, sub_name):
    df_sub_name = df[[col for col in df.columns if sub_name in col]]
    return df_sub_name



def filter_columns_by_prefixes(df, prefixes, time_line=None):
    filtered_columns = [col for col in df.columns if any(col.startswith(prefix) and (time_line in col if time_line else True) for prefix in prefixes)]
    return df[filtered_columns]



def apply_decay(df):
    bef_before_cols = df.columns[df.columns.str.contains('befor')]
    sorted_columns = sorted(bef_before_cols, key=sort_key)
    decay_factor = 1.0
    last_group_index = None

    for column_name in sorted_columns:
        group_index = (re.search(r'\d+', column_name)).group()
        if last_group_index is None:
            last_group_index = group_index
        else:
            if last_group_index == group_index:
                df[column_name] = df[column_name].astype('float').multiply(decay_factor)
            else:
                decay_factor -= 0.1
                df[column_name] = df[column_name].astype('float').multiply(decay_factor)
        last_group_index = group_index


def process_time_columns(df, periods, time_line, reverse):
    if reverse == True:
        peri = periods[::-1]
    else:
        peri = periods

    time_line_cols = df.columns[df.columns.str.contains(time_line)]
    df[time_line_cols].fillna(0)
    unique_features = list(set([name[:re.search("\d+", name).start()].strip(' _') for name in time_line_cols]))

    print(f'creating columns with potential time flow information over {time_line} timeline in period order {peri} for features: {unique_features}')
    for feature in unique_features:
        for i in range(len(peri) - 1):
            current_period = df.filter(regex=re.compile(f'{feature}.*{peri[i]}')).columns[0]
            next_period = df.filter(regex=re.compile(f'{feature}.*{peri[i + 1]}')).columns[0]
            # Compute rolling mean across two peri
            df[f"{feature}_RollingMean_{peri[i]}_{peri[i + 1]}"] = df[[current_period, next_period]].mean(axis=1)
            # # Shift the rolling mean values down by 1 row
            # df[f'{current_period}_Shift1'] = df[current_period].shift(1, axis=1)

    df_with_original_cols = df
    df_with_no_original_cols = df.drop(time_line_cols, axis=1)
    return df_with_original_cols, df_with_no_original_cols


def one_hot_encode_with_bins(df, col_bins_tuples, drop_last=True, debug=False):
    df_bin = df.copy()
    for col, bins in col_bins_tuples:
        # Check if column exists
        if col not in df_bin.columns:
            print(f"Warning: Column {col} not found in DataFrame.")
            continue

        # Create bin labels
        bin_labels = [f"{b}" for b in range(len(bins) - 1)]

        # Cut the column into bins
        df_bin[col + '_binned'] = np.digitize(df[col], bins=bins)

        # Perform one-hot encoding
        dummies = pd.get_dummies(df_bin[col + '_binned'], prefix=col, drop_first=False)

        # Optionally drop the last column
        if drop_last == True:
            dummies = dummies.iloc[:, :-1]

        # Drop original and temporary columns
        df_bin.drop([col, col + '_binned'], axis=1, inplace=True)

        # Concatenate the one-hot encoded columns to original DataFrame
        df_bin = pd.concat([df_bin, dummies], axis=1)

        if debug:
            print(f"dummies column names are {dummies.columns}")
            print(f"dummies value counts: {dummies.sum()}")

    return df_bin


def process_categorical(df_in, categorical_cols):
    df_in = df_in.copy()
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first')
    df_encoded = pd.DataFrame(encoder.fit_transform(df_in[categorical_cols]))
    df_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    df_in.drop(categorical_cols, axis=1, inplace=True)
    df_rv = pd.concat([df_in, df_encoded], axis=1)
    print(df_rv.columns)
    return df_rv, df_encoded




def handle_outliers(df_in, z_thresh=3.0, ignore_zero_inflated=False, debug=False):
    df_out = df_in.copy()

    for col in df_in.columns:
        if df_in[col].dtype == np.dtype('object'):  # Skip non-numeric columns
            continue

        col_data = df_in[col]

        # Detect zero-inflated columns
        zero_count = (col_data == 0).sum()
        if ignore_zero_inflated and zero_count / len(col_data) > 0.5:
            if debug:
                print(f"Ignoring zero-inflated column: {col}")
            continue

        # Remove zeros if column is zero-inflated
        if zero_count / len(col_data) > 0.5:
            col_data = col_data[col_data != 0]
            if debug:
                print(f"Ignoring zeros in column: {col}")

        z_scores = (col_data - col_data.mean()) / col_data.std()
        abs_z_scores = np.abs(z_scores)

        # Identify outliers
        outliers = (abs_z_scores > z_thresh)

        if debug:
            print(f"Outliers detected in {col}: {outliers.sum()}")

        # Cap outliers
        df_out.loc[outliers, col] = col_data.mean() + z_thresh * col_data.std()

    return df_out


def sum_periods_appointments(df_in, prf_lst, debug=False):
    df_sum_periods = pd.DataFrame()
    df_in = df_in.fillna(0)

    for prefix in prf_lst:
        # Debugging information
        if debug:
            print(f"Processing prefix: {prefix}")

        # Find columns that start with the current prefix
        relevant_cols = [col for col in df_in.columns if col.startswith(prefix)]

        if not relevant_cols:
            if debug:
                print(f"No columns found for prefix: {prefix}")
            continue

        # Sum along the rows (axis=1) for those columns
        sum_series = df_in[relevant_cols].sum(axis=1)

        # Add this sum as a new column to df_sum_periods
        df_sum_periods[prefix + '_sum'] = sum_series

        if debug:
            print(f"Added summed column for prefix: {prefix}")

    return df_sum_periods


def sum_boolean_categories(df_before, df_after, prf_lst):
    df_sum_periods = pd.DataFrame()
    for prefix in prf_lst:
        df_temp = df_before[[col for col in df_before.columns if col.startswith(tuple(prf_lst))]]
        df_sum_periods = pd.concat([df_sum_periods,
                                   df_temp[df_temp > 0].all(axis=1)
                                   ],axis=1)
        df_temp = df_after[[col for col in df_after.columns if col.startswith(tuple(prf_lst))]]
        df_sum_periods = pd.concat([df_sum_periods,
                                   df_temp[df_temp > 0].all(axis=1)
                                   ],axis=1)
    return df_sum_periods


def scale_columns(df, only_min_max=True, handle_nulls='zero', null_handler_func=None, debug=False):
    scaled_df = df.copy()

    # Initialize scalers
    scaler_minmax = MinMaxScaler()
    final_scaler = MinMaxScaler()

    for col in df.columns:

        # Debugging information
        if debug:
            print(f"Processing column: {col}")

        # Skip all-zero columns and object-type columns
        if (df[col] == 0).all() or df[col].dtype == 'object':
            if debug:
                print(f"Skipped {col} due to all zeros or non-numeric type.")
            continue

        # Skip binary columns
        if set(df[col].dropna().unique()) == {0, 1}:
            if debug:
                print(f"Skipped {col} due to binary values.")
            continue

        # Handle null values
        if df[col].isnull().any():
            if handle_nulls == 'zero':
                df[col].fillna(0, inplace=True)
            elif handle_nulls == 'custom' and null_handler_func:
                df[col] = null_handler_func(df[col])

        # Check if column is zero-inflated
        if np.sum(df[col] == 0) > len(df[col]) * 0.5:
            non_zero_idx = df[col] != 0
            scaled_df.loc[non_zero_idx, col] = scaler_minmax.fit_transform(df.loc[non_zero_idx, [col]])

        else:
            # Scale based on distribution
            if only_min_max == True:
                scaled_df[col] = scaler_minmax.fit_transform(df[[col]])
            else:
                scaled_df[col] = QuantileTransformer().fit_transform(df[[col]])

        if debug:
            print(f"Scaled {col}.")

    # Final Min-Max scaling to bring all features into a common range (0, 1)
    scaled_df = pd.DataFrame(final_scaler.fit_transform(scaled_df), columns=scaled_df.columns)

    return scaled_df


def verify_preprocessing(df, target_name):
    # Ensure no data leakage and correct train-test split
    X = df.drop(target_name, axis=1)
    y = df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test




def generate_synthetic_data(df, target, num_synthetic_samples):
    positive_samples = df[df[target] == 1]
    synthetic_samples = positive_samples.copy()
    for _ in range((num_synthetic_samples // len(positive_samples)) - 1):
        noise = np.random.normal(0, 0.01, size=positive_samples.shape)

        new_samples = pd.DataFrame(positive_samples.values + noise, columns=positive_samples.columns.to_list())
        synthetic_samples = pd.concat([synthetic_samples, new_samples],
                                      axis=0).reset_index(drop=True)
    df_rv = pd.concat([df,synthetic_samples],axis=0)
    return df_rv


def balance_and_split(df, target, real_duplicate_percentage=0, synthetic_percentage=0,
                      test_size=0.15, train_size=0.70, val_size=0.15):
    # Ensure that the sum of test_size, train_size, and val_size is 1
    assert sum([test_size, train_size, val_size]) == 1, "The sum of test_size, train_size, and val_size must be 1"
    print(df.shape)
    # Split the original data into train_val_data and test_data
    train_val_data, test_data = train_test_split(df, test_size=test_size,
                                                 stratify=df[target], random_state=42)
    print(train_val_data.shape)
    print(test_data.shape)

    # Separate features and labels
    X_test, y_test = test_data.drop(target, axis=1), test_data[target].astype(int)

    # Split the train_val_data into train_data and val_data
    train_data, val_data = train_test_split(train_val_data, test_size=val_size,
                                                 stratify=train_val_data[target], random_state=42)
    print(train_data.shape)
    print(val_data.shape)

    # Calculate the number of real duplicate and synthetic samples to generate based on train_data
    num_real_duplicates_train = int((real_duplicate_percentage / 100) * len(train_data))
    num_synthetic_samples_train = int((synthetic_percentage / 100) * len(train_data))
    # Generate real duplicate and synthetic samples
    df_minority_train = train_data[train_data[target] == 1]
    train_all_data = train_data
    if real_duplicate_percentage != 0:
        df_minority_upsampled_train = df_minority_train.sample(n=num_real_duplicates_train, replace=True, random_state=42)
        train_all_data = shuffle(pd.concat([train_all_data, df_minority_upsampled_train], axis=0),
                                 random_state=42)
    if synthetic_percentage != 0:
        df_synthetic_train = generate_synthetic_data(train_data, target, num_synthetic_samples_train)
        train_all_data = shuffle(pd.concat([train_all_data, df_synthetic_train], axis=0), random_state=42)
    X_train, y_train = train_all_data.drop(target, axis=1), train_all_data[target].astype(int)


    # Calculate the number of real duplicate and synthetic samples to generate based on val_data
    num_real_duplicates_val = int((real_duplicate_percentage / 100) * len(val_data))
    num_synthetic_samples_val = int((synthetic_percentage / 100) * len(val_data))
    # Generate real duplicate and synthetic samples
    df_minority_val = val_data[val_data[target] == 1]
    val_all_data = val_data
    if real_duplicate_percentage != 0:
        df_minority_upsampled_val = df_minority_val.sample(n=num_real_duplicates_val, replace=True, random_state=42)
        val_all_data = shuffle(pd.concat([val_all_data, df_minority_upsampled_val], axis=0),
                                 random_state=42)
    if synthetic_percentage != 0:
        df_synthetic_val = generate_synthetic_data(val_data, target, num_synthetic_samples_val)
        val_all_data = shuffle(pd.concat([val_all_data, df_synthetic_val], axis=0), random_state=42)
    X_val, y_val = val_all_data.drop(target, axis=1), val_all_data[target].astype(int)


    # Concatenate train_val_data, real duplicate, and synthetic samples
    train_val_data = shuffle(pd.concat([train_all_data, val_all_data], axis=0), random_state=42)
    X_train_val, y_train_val = train_val_data.drop(target, axis=1), train_val_data[target].astype(int)

    print(X_train.shape)
    print(X_val.shape)

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'train_val': (X_train_val, y_train_val)
    }



def get_digits_from_string(s):
    return ''.join(re.findall(r'\d', s))


def create_shifted_period_datasets(df, df_before_sorted, df_after_sorted, df_not_periods, target_keyword="Suicide_attempt", debug=False):
    period_datasets = {}
    periods = sorted(set(int(get_digits_from_string(s)) for s in df_after_sorted.columns if 'hodesh' in s), key=int)

    # Special case for 0 to 1 period
    if debug:
        print("Processing special case: 0 to 1")
    # Features for 0_to_1 will be just the 'before' and 'not_periods' data
    df_0_to_1 = pd.concat([df_before_sorted, df_not_periods], axis=1)
    # Targets will be from the first period ("1")
    target_0_to_1_cols = [col for col in df.columns if (get_digits_from_string(col)).isdigit() and int(
        get_digits_from_string(col)) == 1 and 'after' in col and target_keyword in col]
    target_0_to_1 = df[target_0_to_1_cols]
    # Fill NaN values with 0
    df_0_to_1 = df_0_to_1.fillna(0)
    # Store feature dataframe and target vector in dictionary
    period_datasets["0_to_1"] = {"X": df_0_to_1, "y":target_0_to_1}

    for i in range(len(periods) - 1):
        current_period = periods[i]
        next_period = periods[i + 1]
        if debug:
            print(f"Processing current period: {current_period}, next period: {next_period}")
        # Define columns for all periods up to the current period
        current_and_before_cols = [col for col in df_after_sorted.columns if int(get_digits_from_string(col)) <= current_period]
        # Create feature dataframe using all data up to the current period
        df_current = pd.concat([df_before_sorted, df_after_sorted[current_and_before_cols], df_not_periods], axis=1)
        # Create target vector for the next period
        target_cols = [col for col in df.columns if get_digits_from_string(col).isdigit() and
                       int(get_digits_from_string(col)) == next_period and 'after' in col and target_keyword in col]
        target_next = df[target_cols]
        # Fill NaN values with 0
        df_current = df_current.fillna(0)
        if debug:
            print(f"Feature shape: {df_current.shape}, Target shape: {target_next.shape}")
        # Store feature dataframe and target vector in dictionary
        period_datasets[f"{current_period}_to_{next_period}"] = {"X": df_current, "y": target_next}
    return period_datasets
