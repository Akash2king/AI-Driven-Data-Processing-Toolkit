import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define data lists for realistic Indian context
states = ['Maharashtra', 'Gujarat', 'Rajasthan', 'Punjab', 'Haryana', 'Uttar Pradesh', 
          'Bihar', 'West Bengal', 'Odisha', 'Tamil Nadu', 'Karnataka', 'Kerala',
          'Andhra Pradesh', 'Telangana', 'Madhya Pradesh', 'Chhattisgarh', 'Jharkhand',
          'Assam', 'Tripura', 'Meghalaya', 'Manipur', 'Mizoram', 'Nagaland', 'Arunachal Pradesh']

cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Chennai', 'Kolkata',
          'Surat', 'Pune', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane',
          'Bhopal', 'Visakhapatnam', 'Pimpri-Chinchwad', 'Patna', 'Vadodara', 'Ghaziabad',
          'Ludhiana', 'Agra', 'Nashik', 'Faridabad', 'Meerut', 'Rajkot', 'Kalyan-Dombivli']

# Add some variations and typos for cities
cities_with_errors = cities + ['Mumabi', 'Deli', 'Bangalor', 'Hydrabad', 'Ahemdabad', 
                               'Chenai', 'Kolkatta', 'Puna', 'Jaipr', 'Luckno', '']

first_names = ['Rahul', 'Priya', 'Amit', 'Sneha', 'Rajesh', 'Anita', 'Suresh', 'Meera',
               'Vikash', 'Sunita', 'Ajay', 'Kavita', 'Ravi', 'Pooja', 'Manoj', 'Rekha',
               'Sanjay', 'Neha', 'Deepak', 'Shweta', 'Arun', 'Geeta', 'Vinod', 'Seema',
               'Karan', 'Ritu', 'Mohit', 'Jyoti', 'Sachin', 'Asha']

last_names = ['Sharma', 'Patel', 'Singh', 'Kumar', 'Gupta', 'Agarwal', 'Joshi', 'Shah',
              'Verma', 'Yadav', 'Mishra', 'Tiwari', 'Pandey', 'Sinha', 'Jain', 'Chopra',
              'Kapoor', 'Malhotra', 'Bansal', 'Saxena', 'Arora', 'Goyal', 'Mittal', 'Agrawal']

occupations = ['Software Engineer', 'Teacher', 'Doctor', 'Businessman', 'Farmer', 'Student',
               'Homemaker', 'Government Employee', 'Private Employee', 'Self-employed',
               'Retired', 'Unemployed', 'Engineer', 'Nurse', 'Lawyer', 'Accountant']

# Add variations and errors for occupations
occupations_with_errors = occupations + ['Sofware Engineer', 'Techer', 'Docter', 'Bussinessman',
                                         'Govt Employee', 'Pvt Employee', 'Self Employed', '', 'NA', 'N/A']

education_levels = ['Primary', 'Secondary', 'Higher Secondary', 'Graduate', 'Post Graduate', 
                   'Diploma', 'Professional Course', 'Doctorate', 'Illiterate']

education_with_errors = education_levels + ['primary', 'graduate', 'post graduate', 'PhD', 'Masters',
                                           'Bachelor', '10th', '12th', 'College', '', 'Not Applicable']

income_ranges = ['Below 2 Lakh', '2-5 Lakh', '5-10 Lakh', '10-20 Lakh', 'Above 20 Lakh', 'Prefer not to say']
income_with_errors = income_ranges + ['<2L', '2-5L', '5-10L', '10-20L', '>20L', 'Less than 2 lakhs',
                                     'Between 5-10 lakhs', 'More than 20 lakhs', '', 'NA']

def generate_phone():
    """Generate realistic Indian phone numbers with some errors"""
    if random.random() < 0.1:  # 10% chance of error
        return random.choice(['', '123456789', '98765432', '9876543210123', 'Not Available', 'NA'])
    
    # Valid Indian mobile number
    return f"+91-{random.choice(['9', '8', '7'])}{random.randint(100000000, 999999999)}"

def generate_email(first_name, last_name):
    """Generate email addresses with some errors"""
    if random.random() < 0.15:  # 15% chance of error or missing
        return random.choice(['', 'invalid-email', 'test@', '@gmail.com', 'NA', 'Not provided'])
    
    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'rediffmail.com']
    separators = ['.', '_', '']
    
    email = f"{first_name.lower()}{random.choice(separators)}{last_name.lower()}{random.randint(1, 999)}@{random.choice(domains)}"
    return email

def generate_age():
    """Generate age with some unrealistic values"""
    if random.random() < 0.05:  # 5% chance of error
        return random.choice([0, -5, 150, 200, '', 'Twenty Five', 'NA'])
    return random.randint(18, 80)

def generate_survey_response():
    """Generate survey responses with errors"""
    responses = ['Strongly Agree', 'Agree', 'Neutral', 'Disagree', 'Strongly Disagree']
    responses_with_errors = responses + ['SA', 'A', 'N', 'D', 'SD', 'Agree Strongly', 
                                        'Disagree Strongly', '5', '4', '3', '2', '1', 
                                        '', 'No Response', 'Skip']
    return random.choice(responses_with_errors)

def generate_date():
    """Generate survey dates with some format inconsistencies"""
    base_date = datetime(2024, 1, 1)
    random_days = random.randint(0, 365)
    survey_date = base_date + timedelta(days=random_days)
    
    if random.random() < 0.2:  # 20% chance of format error
        formats = ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y']
        return survey_date.strftime(random.choice(formats))
    else:
        return survey_date.strftime('%Y-%m-%d')

# Generate the dataset
data = []
used_ids = set()

print("Generating 20,000 survey records with quality issues...")

for i in range(20000):
    # Generate unique ID with some duplicates (data quality issue)
    if random.random() < 0.02:  # 2% chance of duplicate ID
        survey_id = random.choice(list(used_ids)) if used_ids else f"SUR_{i+1:05d}"
    else:
        survey_id = f"SUR_{i+1:05d}"
        used_ids.add(survey_id)
    
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    
    # Introduce missing values randomly
    record = {
        'Survey_ID': survey_id,
        'First_Name': first_name if random.random() > 0.03 else '',
        'Last_Name': last_name if random.random() > 0.02 else '',
        'Age': generate_age(),
        'Gender': random.choice(['Male', 'Female', 'male', 'female', 'M', 'F', '', 'Other', 'Prefer not to say']) if random.random() > 0.01 else '',
        'Phone': generate_phone(),
        'Email': generate_email(first_name, last_name),
        'City': random.choice(cities_with_errors),
        'State': random.choice(states) if random.random() > 0.05 else '',
        'Education': random.choice(education_with_errors),
        'Occupation': random.choice(occupations_with_errors),
        'Income_Range': random.choice(income_with_errors),
        'Survey_Date': generate_date(),
        'Q1_Product_Satisfaction': generate_survey_response(),
        'Q2_Service_Quality': generate_survey_response(),
        'Q3_Price_Fairness': generate_survey_response(),
        'Q4_Recommendation': random.choice(['Yes', 'No', 'Maybe', 'yes', 'no', 'YES', 'NO', '1', '0', '', 'Unsure']),
        'Q5_Overall_Rating': random.choice([1, 2, 3, 4, 5, '1', '2', '3', '4', '5', 'One', 'Five', '', 'NA', 6, 0, -1]) if random.random() > 0.08 else '',
        'Comments': random.choice(['Good service', 'Needs improvement', 'Excellent experience', 
                                  'Poor quality', 'Average', '', 'NA', 'No comments', 'Very satisfied',
                                  'Could be better', 'Will recommend']) if random.random() > 0.3 else '',
        'Source': random.choice(['Online', 'Phone', 'In-person', 'Email', 'SMS', 'online', 'phone', '', 'Other'])
    }
    
    data.append(record)
    
    if (i + 1) % 5000 == 0:
        print(f"Generated {i + 1} records...")

# Create DataFrame
df = pd.DataFrame(data)

# Introduce some additional inconsistencies
print("Adding additional data quality issues...")

# Add some completely duplicate rows
num_duplicates = 100
duplicate_indices = random.sample(range(len(df)), num_duplicates)
duplicate_rows = df.iloc[duplicate_indices].copy()
df = pd.concat([df, duplicate_rows], ignore_index=True)

# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
filename = 'indian_survey_data_unprocessed.csv'
df.to_csv(filename, index=False)

print(f"\nDataset generated successfully!")
print(f"File saved as: {filename}")
print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# Print data quality summary
print("\n" + "="*50)
print("DATA QUALITY ISSUES SUMMARY:")
print("="*50)

print(f"1. Missing values per column:")
for col in df.columns:
    missing_count = df[col].isnull().sum() + (df[col] == '').sum()
    missing_pct = (missing_count / len(df)) * 100
    if missing_count > 0:
        print(f"   {col}: {missing_count} ({missing_pct:.1f}%)")

print(f"\n2. Duplicate Survey_IDs: {df['Survey_ID'].duplicated().sum()}")
print(f"3. Completely duplicate rows: {df.duplicated().sum()}")

print(f"\n4. Data type inconsistencies:")
print(f"   Age column has mixed types: {df['Age'].dtype}")
print(f"   Q5_Overall_Rating has mixed types: {df['Q5_Overall_Rating'].dtype}")

print(f"\n5. Format inconsistencies:")
print(f"   Survey_Date has multiple formats")
print(f"   Gender has multiple representations")
print(f"   City names have typos and variations")

print(f"\n6. Invalid/Unrealistic values:")
print(f"   Phone numbers with invalid formats")
print(f"   Email addresses with errors")
print(f"   Age values outside reasonable range")

print("\n" + "="*50)
print("CLEANING TASKS TO PRACTICE:")
print("="*50)
print("1. Handle missing values")
print("2. Standardize gender values")
print("3. Clean and validate phone numbers")
print("4. Fix email format issues")
print("5. Correct city name typos")
print("6. Standardize date formats")
print("7. Remove/merge duplicate records")
print("8. Standardize survey responses")
print("9. Clean occupation and education categories")
print("10. Validate and clean rating scales")
print("11. Handle income range inconsistencies")
print("12. Clean text fields and comments")

print(f"\nHappy data cleaning practice! 🧹📊")