'''Script to generate a synthetic dataset for resale price prediction'''

import argparse

import numpy as np
import pandas as pd


def round_price(number: float, hard: bool = True) -> int:
    '''
    Rounds up a given price to end in 9, based on the value of the price and
    the rounding strategy.

    Args:
        number (float): The price to be rounded.
        hard (bool, optional): The flag that determines the rounding strategy.
            If set to True (default), prices greater than 1000 will be rounded
            up to end in 99. Otherwise, all prices will be rounded up to end in 9.

    Returns:
        int: The rounded price.

    Examples:
        round_price(1055.25) -> 1099
        round_price(445.65) -> 449
        round_price(1055.25, hard=False) -> 1059
    '''
    if number > 1000 and hard == True:
        base = 100 # Round to 99
    else:
        base = 10 # Round to 9
        
    next_multiple = int(number / base) + 1
    price = next_multiple * base - 1
    
    return price


def generate_notebook_data() -> pd.DataFrame:
    '''
    Generates a DataFrame containing details of different notebook models from various brands.
    
    This function simulates various details like brand, model name, release year, screen size,
    storage size, RAM size, weight, and retail price for a list of notebook brands and their 
    associated models. The retail price is a calculated value, influenced by various parameters
    like screen size, storage size, RAM, release year, and a brand-specific markup.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - 'Brand': Brand name of the notebook.
            - 'Model': Model name of the notebook.
            - 'Release year': Year when the model was released.
            - 'Screen size (inches)': Size of the screen in inches.
            - 'Hard drive size (GB)': Storage capacity of the hard drive in GB.
            - 'RAM size (GB)': RAM size in GB.
            - 'Weight (grams)': Weight of the notebook in grams.
            - 'Retail price': Calculated retail price of the notebook.

    Notes:
        - Weight is simulated to generally get lighter over time (i.e., with newer release years).
        - The function assumes certain patterns in model naming for determining release years.
    '''
    
    # Dictionary of notebook brands and models
    names = {
        'Lemon': ['AeroBook Pro', 'AeroBook Elite', 'Skyline Series', 'Horizon Hues', 'Nest Navigator', 'Ultra L5'],
        'Pine': ['TimberTouch 6', 'TimberTouch 7', 'EcoWave 2', 'EcoWave 3'],
        'Orion': ['Celestial C3', 'Galaxy Guardian', 'AstroArtisan Pro', 'Oasis Pro', 'Celestial C4'],
        'Nexa': ['WaveBook', 'Node', 'WaveBook Plus', 'Node Plus', 'Navigator'],
        'Mystic': ['Arcane Artisan', 'Enigma Elite', 'WizardWave W5', 'Spectral X'],
        'Nebula': ['StarSlate S2', 'StarSlate S3'],
        'Solar': ['SunSeeker S8', 'Photon Pro', 'Eclipse Elite', 'LunarLite', 'Helio H6'],
        'Ocean': ['AquaArtisan Plus', 'Tidal Touch', 'Mariner Max', 'DeepBlue D3', 'CoralCraft'],
        'Terra': ['EarthBook Elite', 'GeoFlex G8', 'Plateau Pro', 'Highland H3'],
        'Vivid': ['Color Canvas C7', 'Color Canvas C8', 'Spectrum S6', 'ChromaCraft Pro'],
        'Echo': ['Eon E5', 'Reverb R3', 'Reverb R4'],
        'Pixel': ['PixelBook PX4', 'PixelBook PX3', 'Graphite Pro', 'PixelNet Prime', 'Focus Pro'],
        'Crest': ['PeakPro P9', 'Summit S7', 'Elevation Elite', 'Pinnacle Prime', 'Ridge Raider R6'],
        'Zephyr': ['Breeze Pro', 'WindRider W2', 'Atmos A4', 'CloudCraft C3', 'Nimbus N5'],
        'Iris': ['SightSync S3', 'Optic Oasis', 'Visionary V5', 'Insight Inception', 'Pupil Pro'],
        'Quantum': ['Nanobook Air', 'Nanobook Pro', 'Fusion Flex'],
    }
    screen_size_options = [
        11, 12, 13, 14, 15, 17, 19,
    ]
    storage_size_options = [
        128, 256, 512, 1028,
    ]
    ram_size_options = [
        4, 8, 16,
    ]
    year_options = [
        2015, 2016, 2017, 2018,
    ]
    year_distribution = [
        0.1, 0.2, 0.3, 0.4,
    ]
        
    # Notebook data generation
    data = []

    for brand, models in names.items():
        
        # Set a value that reflects how expensive a brand is in general (e.g. due to their software)
        brand_markup = 1 + np.random.exponential(0.05) * 1.5
        last_model = []
        last_release_year = None
        
        for model in models:
            
            # A newer iteration of the same model should be released later
            try:
                if int(model[-1:]) == int(last_model[-1:]) + 1:
                    release_year = last_release_year + 1
                else:
                    raise BaseException()
            except:
                release_year = np.random.choice(year_options, p=year_distribution)
                
            # Notebooks get lighter over time
            weight = np.random.normal(1400, 100) - release_year + 2000
            
            # Random number of model versions (different screen sizes, etc.)
            n_versions = range(np.random.randint(1, 6))
            
            for _ in n_versions:
                
                # Versions of each model vary by screen size, storage and weight
                screen_size = np.random.choice(screen_size_options)
                storage_size = np.random.choice(storage_size_options)
                ram_size = np.random.choice(ram_size_options)
                
                # The weight depends on the model and screen size
                version_weight = int(weight * screen_size / 19)
                
                # The retail price is a function of screen size, storage, release year, and brand markup
                retail_price = 300 + 5e-4*(screen_size**5) + 2*storage_size + 2*(ram_size**2.2) + 5e-4*((release_year-2010)**5)
                retail_price = round_price(retail_price * brand_markup)
                
                data.append([
                    brand,
                    model,
                    brand_markup,
                    release_year,
                    screen_size,
                    storage_size,
                    ram_size,
                    version_weight,
                    retail_price,
                ])

    # Return pandas dataframe
    return pd.DataFrame(data, columns=[
        'Brand',
        'Model',
        'Brand markup',
        'Release year',
        'Screen size (inches)',
        'Hard drive size (GB)',
        'RAM size (GB)',
        'Weight (grams)',
        'Retail price',
    ])
    

def expand_dataset(df: pd.DataFrame, N_SAMPLES: int) -> pd.DataFrame:
    '''
    Expands or reduces the dataframe size to the desired number of samples.
    
    If the input dataframe has more than or equal to `N_SAMPLES` rows, it slices 
    the dataframe to retain only the first `N_SAMPLES` rows. If the input dataframe 
    has fewer rows, it samples additional rows with replacement until reaching the 
    desired `N_SAMPLES` and adds a variance to the retail price of the new rows.
    
    Args:
        df (pd.DataFrame): The input dataframe to expand or reduce.
        N_SAMPLES (int): The desired number of samples in the output dataframe.
        
    Returns:
        pd.DataFrame: The expanded or reduced dataframe with size `N_SAMPLES`.
    
    Notes:
        - The function assumes the retail price column is the ninth column
            (0-indexed) in the dataframe.
        - The variance added to the retail price is sampled from a normal
            distribution with mean 0 and standard deviation 50.
    '''
    
    # Expand or reduce the dataframe size to the desired number of samples
    if len(df) >= N_SAMPLES:
        return df.iloc[:N_SAMPLES]
    else:
        new_rows = df.sample(n=(N_SAMPLES - len(df)), replace=True)
        # Add variance to the retail price, since it can change over time or due to discounts
        for idx in range(len(new_rows)):
            new_price = new_rows.iloc[idx, 8] + np.random.normal(0, 50)
            new_rows.iloc[idx, 8] = round_price(new_price, hard=False)
        return pd.concat([df, new_rows], ignore_index=True)


def append_industry_and_lease_duration(df: pd.DataFrame) -> None:
    '''
    Appends random industry, corresponding depreciation value, and lease
    duration to a given dataframe.

    This function takes in a dataframe and appends three new columns to it: 
    1) 'Industry': Randomly chosen from predefined industries.
    2) 'Depreciation': Depreciation value corresponding to the chosen industry.
    3) 'Lease Duration (months)': Randomly selected lease duration based on
        predefined probabilities.

    Args:
        df (pd.DataFrame): The input dataframe to which the new columns will be
            appended.
        
    Notes:
        - The industry types and their corresponding yearly depreciation rates
            are based on a predefined dictionary.
        - This function modifies the input dataframe in-place and does not
            return anything.
    '''
    
    # Industries of companies that may lease notebooks and corresponding yearly
    # depreciation rates
    industries = {
        'Information Technology': 0.33,
        'Financial Services and Banking': 0.26,
        'Education': 0.27,
        'Healthcare': 0.24,
        'Consulting': 0.28,
        'Media and Entertainment': 0.32,
        'Telecommunications': 0.26,
        'Real Estate and Property Management': 0.21,
        'Travel and Hospitality': 0.24,
        'Retail and E-Commerce': 0.22,
        'Advertising and Marketing': 0.30,
        'Research and Development': 0.33, 
        'Energy and Utilities': 0.23,
        'Government and Public Sector': 0.20,
        'Automotive and Transportation': 0.26,
        'Aerospace and Defense': 0.27,
        'Non-Profit Organizations': 0.20,
        'Law and Legal Services': 0.23,
        'Construction and Engineering': 0.27,
        'Manufacturing and Production': 0.27,
        'Logistics and Supply Chain Management': 0.24,
        'Event Management': 0.25,
        'Sports and Recreation': 0.23,
        'Agriculture and Farming': 0.22,
    }

    # Duration of the notebook leases and their frequencies
    lease_duration = [
        6, 12, 24, 36, 48, 60,
    ]
    lease_distribution = [
        0.1, 0.22, 0.27, 0.2, 0.12, 0.09
    ]
    
    # Randomly select industries and their corresponding depreciation values
    random_industries = np.random.choice(list(industries.keys()), size=len(df))
    depreciation_values = [industries[industry] for industry in random_industries]
    
    # Randomly select lease durations
    random_lease_durations =  np.random.choice(lease_duration,
                                               size=len(df),
                                               p=lease_distribution)
    
    # Append these values to the dataframe
    df['Industry'] = random_industries
    df['Depreciation'] = depreciation_values
    df['Contract Lease Duration (months)'] = random_lease_durations

    # Actual lease duration can deviate from contract due to early termination or reposession issues
    df['Actual Lease Duration (months)'] = (
        random_lease_durations * np.random.normal(1, 0.1, len(df))
    ).astype(int)
    
    df['Battery capacity (%)'] = abs(100 - np.random.gamma(3, 3.5, len(df))).round(2)

def append_resale_price(df: pd.DataFrame) -> None:
    '''
    Appends an `Observed resale price` column to the given dataframe.

    This function calculates the resale price for notebook models based on
    multiple factors: retail price, depreciation, release year, screen size,
    hard drive size, RAM size, weight, lease duration, and battery capacity.

    Args:
        df (pd.DataFrame): A dataframe containing the relevant notebook model
            and lease information

    Notes:
        - The calculated resale price includes a random variance (normal distribution) 
          to introduce unpredictability, simulating real-world price variations.
        - This function modifies the input dataframe in-place and does not return anything.
    '''
    
    # Compute observed resale price
    resale_prices = []

    for idx in df.index:
        
        # Functions
        price_function = 0.15 * np.exp(-df.iloc[idx]['Retail price'] / 5e4)
        depr_function = 1.1 * df.iloc[idx]['Depreciation']
        year_function = 0.004 * (df.iloc[idx]['Release year'] - 2010)
        screen_function = 0.0002 * (df.iloc[idx]['Screen size (inches)'] - 20)**2 + 0.01
        drive_function = 5e-7 * (df.iloc[idx]['Hard drive size (GB)'] - 400)**2 + 0.01
        ram_function = 0.04 / np.exp(df.iloc[idx]['RAM size (GB)'] / 10)
        weight_function = df.iloc[idx]['Weight (grams)'] / 20000
        durations_function = 1.3 * ((np.log(1 + df.iloc[idx]['Actual Lease Duration (months)']/15)**1e-3**1e-3) - 0.09*(df.iloc[idx]['Actual Lease Duration (months)']/30)**2) - 0.95 * (0.9**df.iloc[idx]['Actual Lease Duration (months)'])
        
        depreciation = (
            price_function +
            depr_function +
            year_function +
            screen_function +
            drive_function +
            ram_function +
            weight_function +
            durations_function
        )
        depreciation = ((depreciation * (1 - (df.iloc[idx]['Brand markup'] - 1))) / 3) / (df.iloc[idx]['Battery capacity (%)'] / 85)
        
        resale_price = df.iloc[idx]['Retail price'] * (1 - depreciation)
        if resale_price < 0:
            resale_price = 0
            
        resale_price = int(resale_price * (1 + np.random.normal(0, 0.05)))
            
        resale_prices.append(resale_price)
        
    df['Observed resale price'] = resale_prices
    df.drop(columns=['Brand markup', 'Depreciation'], inplace=True)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Script to generate a synthetic dataset for resale price prediction. Saves resulting dataframe as `output.csv` in the current working directory.')

    parser.add_argument(
        '-n',
        type=int,
        default=5000,
        help='(int) number of samples to generate',
    )
    args = parser.parse_args()

    # Create dataset of notebooks
    df = expand_dataset(generate_notebook_data(), args.n)

    # Add leasing and resale data
    append_industry_and_lease_duration(df)
    append_resale_price(df)
    
    # Save final dataframe as a .csv file
    df.sort_values(by=['Brand', 'Model']).to_csv('output.csv', index=False)
