from distance_functions import *

distance_df = pd.read_csv('distance_xy.csv')
eye_screen_distance = DistanceCalculator()
eye_screen_distance.calculate_distance(
    distance_df['distance_pixel'], distance_df['distance_cm'])