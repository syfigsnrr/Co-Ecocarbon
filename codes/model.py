from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


'''
The main functions of this script include:
1. Performing carbon balance correction for site-level data.
2. Converting PFTs from character labels to corresponding numeric codes.
3. Training the model and searching for optimal hyperparameters.
4. Building the final model based on the best parameter set.

'''


igbp_codes = {
    'ENF': 1,  # Evergreen Needleleaf Forest
    'DBF': 4,  # Deciduous Broadleaf Forest
    'EBF': 2,  # Evergreen Broadleaf Forest
    'MF': 5,  # Mixed Forest
    'CRO': 12,  # Cropland
    'GRA': 10,  # Grassland
    'WSA': 8,  # Wetland
    'SAV': 9,  # Savanna
    'OSH': 7,  # Open Shrubland
    'CSH': 6,  # Closed Shrubland
    'WET': 11  # Wetland
}
cols = ['dem', 'temperature_2m', 'soil_temperature_level_1', 'volumetric_soil_water_layer_1',
        'solar_rad_dw', 'temperature_2m_min', 'temperature_2m_max',
        'vpd', 'nirv', 'lai', 'sif', 'CO2', 'igbp_index']


#carbon balance correction
def carbon_balance_correction(path=r'FLUX_dataset.csv', outpath=r'FLUX_dataset_corr.csv'):

    df = pd.read_csv(path)

    # in-situ variables :'NEE_VUT_REF', 'GPP_NT_VUT_REF', 'RECO_NT_VUT_REF'
    df['abs_diff'] = abs((df['GPP_NT_VUT_REF'] + df['NEE_VUT_REF'] - df['RECO_NT_VUT_REF']))
    df['abs_per_diff'] = abs(df['abs_diff'] / df['NEE_VUT_REF'])
    # The threshold of imbalance carbon
    df = df[df['abs_diff'] < 0.1]
    # The threshold of carbon imbalance ratio
    df = df[df['abs_per_diff'] < 0.05]

    # Correction
    df['diff'] = df['GPP_NT_VUT_REF'] - df['RECO_NT_VUT_REF'] + df['NEE_VUT_REF']
    df['GPP_NT_VUT_REF_RAT'] = df['GPP_NT_VUT_REF'] - df['diff'] * df['GPP_NT_VUT_REF'] / (
                df['GPP_NT_VUT_REF'] + df['RECO_NT_VUT_REF'])
    df['RECO_NT_VUT_REF_RAT'] = df['RECO_NT_VUT_REF'] + df['diff'] * df['RECO_NT_VUT_REF'] / (
                df['GPP_NT_VUT_REF'] + df['RECO_NT_VUT_REF'])

    # Exclude the values that do not conform to the actual physical meaning after correction
    df = df[df['GPP_NT_VUT_REF_RAT'] >= 0]
    df = df[df['RECO_NT_VUT_REF_RAT'] >= 0]
    df.to_csv(outpath)
    return df


def pft_convert(df,outpath=r'FLUX_dataset_corr.csv'):
    forest_types = ['ENF', 'DBF', 'EBF', 'MF', 'CRO', 'GRA', 'WSA', 'SAV', 'OSH', 'CSH', 'WET']
    for igbp_type in forest_types:
        df.loc[df['IGBP'] == igbp_type, 'igbp_index'] = igbp_codes[igbp_type]
    df.to_csv(outpath)
    return df


def para_optimize(df):
    X = df[cols]
    y = df[['GPP_NT_VUT_REF_RAT', 'NEE_VUT_REF', 'RECO_NT_VUT_REF_RAT']]

    # Note that the parameter values here are the initial Settings.
    # The further adjustment process is specifically as follows:
    # Based on the optimal parameters obtained each time, gradually narrow down the range of parameter Settings until the optimal combination is found.
    # Avoiding setting too many parameter combinations at once during the initial setup is to improve efficiency
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10,20,30],
        'min_samples_split': [4, 8, 16, 32],
        'min_samples_leaf': [2, 4, 8, 16],
        'bootstrap': [True]
    }

    rf = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    return best_params

def model_construct(df,best_params,modelpath=r'F:\CarbonSynEst\06_models\Co-EcocarbonV01.joblib'):
    X = df[cols]
    y = df[['GPP_NT_VUT_REF_RAT', 'NEE_VUT_REF', 'RECO_NT_VUT_REF_RAT']]

    # 按照 8:2 的比例划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
                               min_samples_split=best_params['min_samples_split'],
                               min_samples_leaf=best_params['min_samples_leaf'],
                               random_state=42)

    rf.fit(X_train, y_train)

    dump(rf, modelpath)
    print("The model has been saved to 'Co-EcocarbonV01.joblib'")

def main():
    path=r'F:\CarbonSynEst\01_sitesvalues\FLUX_dataset.csv'
    outpath = r'F:\CarbonSynEst\01_sitesvalues\FLUX_dataset_corr.csv'
    model_path = r'F:\CarbonSynEst\06_models\Co-EcocarbonV01.joblib'
    data=carbon_balance_correction(path, outpath)
    data=pft_convert(data,outpath)
    params=para_optimize(data) #Please note that this step needs to be executed repeatedly. Based on the best parameter Settings obtained each time, narrow down the range of parameter search defined in the function until the best parameter combination is found
    model_construct(data,params,model_path)

if __name__ == "__main__":
    main()









