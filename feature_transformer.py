import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth"""
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def assign_city(lat, lon):
    # Manhattan (famous for business, tourism)
    if 40.7 <= lat <= 40.88 and -74.02 <= lon <= -73.91:
        return 'Manhattan'
    # Long Island City (subset of Queens, famous for arts and proximity to Manhattan)
    elif 40.73 <= lat <= 40.76 and -73.96 <= lon <= -73.91:
        return 'Long Island City'
    # Brooklyn (famous for culture, nightlife)
    elif 40.57 <= lat <= 40.74 and -74.04 <= lon <= -73.83:
        return 'Brooklyn'
    # Queens (includes JFK, LaGuardia)
    elif 40.54 <= lat <= 40.8 and -73.96 <= lon <= -73.7:
        return 'Queens'
    # Bronx
    elif 40.79 <= lat <= 40.92 and -73.93 <= lon <= -73.76:
        return 'Bronx'
    # Staten Island
    elif 40.49 <= lat <= 40.65 and -74.26 <= lon <= -74.05:
        return 'Staten Island'
    # Jersey City (famous for commuters, skyline views)
    elif 40.69 <= lat <= 40.75 and -74.12 <= lon <= -74.03:
        return 'Jersey City'
    # Hoboken (famous for nightlife, proximity to Manhattan)
    elif 40.73 <= lat <= 40.76 and -74.04 <= lon <= -73.99:
        return 'Hoboken'
    # Newark (includes Newark Airport)
    elif 40.65 <= lat <= 40.78 and -74.25 <= lon <= -74.11:
        return 'Newark'
    # Yonkers
    elif 40.91 <= lat <= 40.98 and -73.91 <= lon <= -73.83:
        return 'Yonkers'
    else:
        return 'Other'

def transform_features(input_data):
    """Transform raw input data into features used by the model"""
    # Create a DataFrame from the input data
    df = pd.DataFrame([input_data])

    # Convert pickup_datetime to datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # Time features
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['day_of_week'] = df['pickup_datetime'].dt.weekday

    # Cyclical encoding of hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Business hours
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)

    # Distance calculation
    df['distance_km'] = haversine(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    # Manhattan boundaries
    manhattan_bounds = {
        'north': 40.879, 'south': 40.703,
        'east': -73.907, 'west': -74.030
    }

    # Manhattan features
    df['dropoff_in_manhattan'] = (
        (df['dropoff_latitude'] >= manhattan_bounds['south']) &
        (df['dropoff_latitude'] <= manhattan_bounds['north']) &
        (df['dropoff_longitude'] >= manhattan_bounds['west']) &
        (df['dropoff_longitude'] <= manhattan_bounds['east'])
    ).astype(int)

    df['manhattan_to_manhattan'] = (
        ((df['pickup_latitude'] >= manhattan_bounds['south']) &
         (df['pickup_latitude'] <= manhattan_bounds['north']) &
         (df['pickup_longitude'] >= manhattan_bounds['west']) &
         (df['pickup_longitude'] <= manhattan_bounds['east'])) &
        df['dropoff_in_manhattan']
    ).astype(int)

    # Trip characteristics
    df['is_long_trip'] = (df['distance_km'] > 10).astype(int)
    df['is_short_trip'] = (df['distance_km'] < 2).astype(int)

    # Location features
    # Apply to pickup and dropoff locations
    df['pickup_city'] = df.apply(lambda row: assign_city(row['pickup_latitude'], row['pickup_longitude']), axis=1)
    df['dropoff_city'] = df.apply(lambda row: assign_city(row['dropoff_latitude'], row['dropoff_longitude']), axis=1)
    df['is_city_to_city'] = (df.apply(lambda row: assign_city(row['pickup_latitude'], row['pickup_longitude']), axis=1) != df['dropoff_city']).astype(int)

    # Popular Locations (Hotspots) - Top 3 Pickup & Dropoff
    top_pickups = df['pickup_city'].value_counts().nlargest(3).index
    top_dropoffs = df['dropoff_city'].value_counts().nlargest(3).index
    df['pickup_hotspot'] = df['pickup_city'].isin(top_pickups).astype(int)
    df['dropoff_hotspot'] = df['dropoff_city'].isin(top_dropoffs).astype(int)

    # Select only the features the model expects
    final_features = [
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'distance_km', 'day', 'month', 'year',
        'day_of_week', 'hour_sin', 'hour_cos',
        'is_business_hours', 'manhattan_to_manhattan',
        'is_long_trip', 'is_short_trip', 'dropoff_city',
        'is_city_to_city', 'pickup_hotspot', 'dropoff_hotspot'
    ]

    return df[final_features]
