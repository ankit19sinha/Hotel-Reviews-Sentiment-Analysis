import re
import numpy as np
import pandas as pd

# Import dataset
df = pd.read_csv("Hotel_Reviews.csv")

# Get negative reviews
df_negative = df[["Hotel_Name", "Negative_Review"]]
df_negative = df_negative.loc[df_negative["Negative_Review"] != "No Negative"]
df_negative["Sentiment"] = 0
df_negative.columns = ["Hotel_Name", "Review", "Sentiment"]

# Get positive reviews
df_positive = df[["Hotel_Name", "Positive_Review"]]
df_positive = df_positive.loc[df_positive["Positive_Review"] != "No Positive"]
df_positive["Sentiment"] = 1
df_positive.columns = ["Hotel_Name", "Review", "Sentiment"]

# Merge the two dfs
df_merged = pd.concat([df_negative, df_positive])

def preprocessText(review):
  # Convert to lowercase
  review = review.lower()

  # Remove punctuations
  review = re.sub('[^A-Za-z0-9]+', ' ', review)

  # Remove whitespaces at both ends
  review = review.strip()

  return review

# Apply preprocessing on all reviews
df_merged["Review"] = df_merged["Review"].apply(preprocessText)

# Replace blank rows with NaN
df_merged.replace("", np.nan, inplace=True)

# Drop empty rows
df_merged.dropna(inplace=True)

# Double check for missing values
df_merged.isnull().any()

# Randomly pick 100,000 rows
hotelreviewsdf = df_merged.sample(n = 100000)

# Save the file as csv
hotelreviewsdf.to_csv("hotel_reviews_processed.csv", sep = ",", index = False)