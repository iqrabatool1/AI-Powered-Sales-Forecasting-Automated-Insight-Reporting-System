import pandas as pd
import numpy as np


def load_data(data_dir='../data'):
    """Load raw train and store CSVs."""
    train = pd.read_csv(f'{data_dir}/train.csv', low_memory=False)
    store = pd.read_csv(f'{data_dir}/store.csv')
    return train, store


def clean_and_merge(train, store):
    """Merge train with store info and fix missing values."""
    train['Date'] = pd.to_datetime(train['Date'])
    df = train.merge(store, on='Store', how='left')

    # Fix missing values logically
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
    df['PromoInterval'] = df['PromoInterval'].fillna('None')

    return df


def engineer_features(df):
    """Add calendar, lag, rolling, and promo/competition features."""
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    # Calendar features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week

    # Lag features
    df['Sales_Lag_7'] = df.groupby('Store')['Sales'].shift(7)
    df['Sales_Lag_14'] = df.groupby('Store')['Sales'].shift(14)

    # Rolling averages (shifted to avoid leakage)
    df['Sales_RollingMean_7'] = df.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(7).mean())
    df['Sales_RollingMean_30'] = df.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(30).mean())

    # Drop rows with insufficient history
    df = df.dropna(subset=['Sales_Lag_7', 'Sales_Lag_14',
                            'Sales_RollingMean_7', 'Sales_RollingMean_30'])

    # Keep only open days
    df = df[df['Open'] == 1]

    # Holiday / promo / competition features
    df['IsHoliday'] = ((df['StateHoliday'] != '0') | (df['SchoolHoliday'] == 1)).astype(int)
    df['IsPromo2Active'] = df['Promo2']
    df['CompetitionOpenMonths'] = (df['Year'] - df['CompetitionOpenSinceYear']) * 12 + \
                                   (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpenMonths'] = df['CompetitionOpenMonths'].apply(lambda x: x if x > 0 else 0)

    return df


def get_feature_columns():
    """The list of columns used to train the model."""
    return [
        'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'IsHoliday',
        'Year', 'Month', 'Day', 'WeekOfYear',
        'Sales_Lag_7', 'Sales_Lag_14', 'Sales_RollingMean_7', 'Sales_RollingMean_30',
        'CompetitionDistance', 'CompetitionOpenMonths', 'Promo2', 'IsPromo2Active'
    ]


def prepare_full_dataset(data_dir='../data'):
    """Full pipeline: load -> clean -> engineer features. Returns final df."""
    train, store = load_data(data_dir)
    df = clean_and_merge(train, store)
    df = engineer_features(df)
    return df