
from classifires_data import *
from data_preeprocess_classes import *
from data_preeprocess_functions import *
from input_data_attributes import *
from model_selection_classes import *
from model_selection_functions import *
from plots import *
from plots_testing import *
from external_libs_imports import *
from pipeline_functions import *




if __name__ == '__main__':

    df = pd.read_csv(f'{input_lib}/{input_file}')


    # remove columns not relevant to training model
    teudat_zehut_src = df.teudat_zehut_src
    df.drop('teudat_zehut_src', axis=1, inplace=True)
    # avh_more_12hodesh_befor = df.avh_more_12hodesh_befor
    # df.drop('avh_more_12hodesh_befor', axis=1, inplace=True)

    # define list for data manipulations
    time_lines = ['befor','after']
    time_series_categories_pref = ['mrs', 'avh']
    bikurim_pref = ['bkrcnt','bkrcnt_mish','bkrcnt_mik','bkrcnt_emg','bkrcnt_emgbrn','missbkr_cnt']
    periods = [1, 3, 6, 9, 12]
    categorical_cols = ['kvutza_demografit', 'kod_min']
    fill_zero = ['zman_bein_ishpuz',]
    fill_mean = ['zion_sotzio', 'mekadem']
    hospitalashhion_cols =['mesheh_ishpuz', 'ishpuzim_num']
    cols_bikurim = [col for col in df.columns if col.startswith(tuple(bikurim_pref))]
    numeric_cols = ['godel_mishpacha', 'mispar_yeladim', 'zion_sotzio', 'mekadem', ]
    time_bins = [('gil_shihrur', [0, 20, 35, 50, 65, 80, 200]), ('zman_bein_ishpuz',[0, 7, 30, 90, 180, 360, float('inf')])]
    time_bin_cols = ['gil_shihrur', 'zman_bein_ishpuz']

    # make 'gil_shihrur' categorical column
    df_time_bins = one_hot_encode_with_bins(df[time_bin_cols], time_bins)

    # make binary categories
    all_cat_cols = []
    for prefix in time_series_categories_pref:
        df, cat_cols = make_time_serialized_categories(df, prefix, time_lines)
        all_cat_cols += cat_cols



    # Process categorical columns
    df , df_encoded = process_categorical(df, categorical_cols)

    # get after and before data frames
    df_before = get_columns_by_sub_name(df, 'befor')
    df_after = get_columns_by_sub_name(df, 'after')

    df_b= sum_periods_appointments(df_before, bikurim_pref, debug=True)
    df_before[df_b.columns] = df_b

    prefixes = [re.search(r'^[^\d]*', string).group() for string in df_before[[col for col in df_before.columns if col in all_cat_cols]].columns]
    df_before = sum_periods_appointments(df_before, prefixes, debug=True)


    # get dictionary of outliers
    # dfClean, dfOutliers, outlierCountsDict = handle_outliers(df[cols_bikurim])
    df[cols_bikurim] = handle_outliers(df[cols_bikurim], z_thresh=2, ignore_zero_inflated=True, debug=False)
    df[numeric_cols] = handle_outliers(df[numeric_cols], z_thresh=2, ignore_zero_inflated=True, debug=False)


    # sort data frames
    df_before_sorted = sort_columns(df_before, reverse=True)
    df_after_sorted = sort_columns(df_after, reverse=False)

    df_numericm = df[numeric_cols]
    df_hospitalashhion = df[hospitalashhion_cols]
    df_not_periods = pd.concat([df_encoded, df_time_bins, df_hospitalashhion], axis=1)


    period_data = create_shifted_period_datasets(df, df_before_sorted, df_after_sorted, df_not_periods,
                                       target_keyword="Suicide_attempt", debug=True)

    classeifire_metric_dict = {}
    custom_scorer = make_scorer(custom_scorer, greater_is_better=True, needs_proba=True, debug=True)
    for period, data in period_data.items():
        X = data['X']
        y = data['y']
        data_splits = balance_and_split(pd.concat([X,y],axis=1), y.columns[0], real_duplicate_percentage=10, synthetic_percentage=0,
                              test_size=0.15, train_size=0.65, val_size=0.20)

        X_train = data_splits['train'][0]
        y_train= data_splits['train'][1]
        X_val = data_splits['val'][0]
        y_val= data_splits['val'][1]
        X_test = data_splits['test'][0]
        y_test= data_splits['test'][1]

        best_params = {}
        best_folds = {}

        all_data = np.vstack((X_train, X_val))
        all_labels = np.hstack((y_train, y_val))

        best_custom_cv =  CustomCV(n_splits=5, len_train=len(X_train))

        # Loop over each classifier in the dictionary
        for name, model_info in model_params.items():

            # Skip classifiers that don't support either p_value or hypothesis_testing
            if not model_info.get('p_value_support', False) and not model_info.get('hypothesis_testing', False):
                print(f'{name} is not calculated')
                continue

            pipeline, pipe_params = create_pipeline(name, model_info, debug=False)

            # Create RandomizedSearchCV object with multiple scorers
            random_search = RandomizedSearchCV(pipeline, param_distributions=pipe_params,
                                               scoring=custom_scorer, refit=custom_scorer, cv=5)

            random_search.fit(all_data, all_labels)
            print(random_search.get_params())
            # get all evaluation results
            classeifire_metric_dict[f'{name}_test'] = model_checks(name, random_search, X_train, y_train, X_test, y_test)

    plotting_results(classeifire_metric_dict)


