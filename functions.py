import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re
import warnings

def add_space():
	print(" " * 100)
	print(" " * 100)
	print(" " * 100)


# encoding 
def encode_category_data(train_set, test_set):
	train_set = train_set.copy()
	test_set = test_set.copy()

	# Binary categorical features — use LabelEncoder
	binary_cols = ['Married/Single', 'Car_Ownership']
	for col in binary_cols:
		le = LabelEncoder()
		train_set[col] = le.fit_transform(train_set[col])
		test_set[col] = le.transform(test_set[col])

	# Multi-class categorical features — use OrdinalEncoder
	multi_class_cols = ['House_Ownership', 'Profession', 'CITY', 'STATE', 'Marital_Home_Status']
	oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

	train_set[multi_class_cols] = oe.fit_transform(train_set[multi_class_cols])
	test_set[multi_class_cols] = oe.transform(test_set[multi_class_cols])

	return train_set, test_set


# Print number of unique items in dataset
def print_unique_items(df):
	if not isinstance(df, pd.DataFrame):
		warnings.warn("⚠️ The provided input is not a pandas DataFrame. Please pass a valid DataFrame.")
		return

	unique_values = df.nunique()
	print("🔍 Number of unique values per column:\n")
	print(unique_values)
	print()

	for col in df.columns:
		print(f"🧾 Unique values in column '{col}':")
		print(df[col].unique())
		print()


# Print Min and Max Values
def print_min_and_max_value(data, train_df):
	if data not in train_df.columns:
		warnings.warn(f"⚠️ The column '{data}' does not exist in the DataFrame.")
		return

	print(f"Min {data}: {train_df[data].min()}")
	print(f"Max {data}: {train_df[data].max()}")


# Data Visualisation Functions
def income(df):
	print(df.groupby('Risk_Flag')['Income'].describe())

	add_space()

	plt.figure(figsize=(8, 6))
	sns.boxplot(x='Risk_Flag', y='Income', data=df)
	plt.title('Income Distribution by Risk Flag')
	plt.xlabel('Risk Flag (0 = Low Risk, 1 = High Risk)')
	plt.ylabel('Income')
	plt.show()

	add_space()

	plt.figure(figsize=(10, 6))
	sns.kdeplot(data=df[df['Risk_Flag'] == 0], x='Income', label='Low Risk (0)', fill=True)
	sns.kdeplot(data=df[df['Risk_Flag'] == 1], x='Income', label='High Risk (1)', fill=True)
	plt.title('Income Distribution by Risk Category')
	plt.xlabel('Income')
	plt.ylabel('Density')
	plt.legend()
	plt.show()

	mean_income_by_risk = df.groupby('Risk_Flag')['Income'].mean()
	print("Mean Income by Risk Flag:\n", mean_income_by_risk)

	print_min_and_max_value("Income", df)

	bins = [0, 10000, 20000, 30000, 40000, 50000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, float('inf')]
	labels = [
		'0-10k', '10k-20k', '20k-30k', '30k-40k', '40k-50k',
		'50k-100k', '100k-200k', '200k-300k', '300k-400k', '400k-500k',
		'500k-600k', '600k-700k', '700k-800k', '800k-900k', '900k+'
	]

	income_range_series = pd.cut(df['Income'], bins=bins, labels=labels, right=False)
	risk_counts = df.groupby(income_range_series, observed=True)['Risk_Flag'].agg(['sum', 'count'])
	risk_counts['High_Risk_Percent'] = (risk_counts['sum'] / risk_counts['count']) * 100

	print(risk_counts)

	add_space()

	plt.figure(figsize=(8, 8))
	plt.pie(risk_counts['High_Risk_Percent'], labels=risk_counts.index, autopct='%1.1f%%', startangle=140)
	plt.title('High Risk Percentage by Income Range')
	plt.axis('equal')
	plt.show()


def age(train_df):
	print_min_and_max_value("Age", train_df)

	temp_df = train_df.copy()

	age_bins = [21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, float('inf')]
	age_labels = ['21-25', '26-30', '31-35', '36-40', '41-45',
				  '46-50', '51-55', '56-60', '61-65', '66-70',
				  '71-75', '76+']

	temp_df['Age_Group'] = pd.cut(temp_df['Age'], bins=age_bins, labels=age_labels, right=False)

	age_risk_viz = df.copy()
	age_risk_viz['Age_Group'] = pd.cut(age_risk_viz['Age'], bins=age_bins, labels=age_labels, right=False)
	age_risk_grouped = age_risk_viz.groupby(['Age_Group', 'Risk_Flag']).size().unstack(fill_value=0)
	age_risk_grouped['Total'] = age_risk_grouped.sum(axis=1)
	age_risk_grouped['High_Risk_Percent'] = (age_risk_grouped[1] / age_risk_grouped['Total']) * 100
	age_risk_grouped['Low_Risk_Percent'] = (age_risk_grouped[0] / age_risk_grouped['Total']) * 100

	print(age_risk_grouped[['Total', 1, 0, 'High_Risk_Percent', 'Low_Risk_Percent']])

	add_space()

	print("This shows distribution among the High Risk Flag")
	age_risk_clean = age_risk_grouped.dropna()
	labels = age_risk_clean.index.astype(str)
	sizes = age_risk_clean[1]
	plt.figure(figsize=(8, 8))
	plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
	plt.title('High Risk Percentage by Age Group')
	plt.axis('equal')
	plt.show()

	add_space()

	plt.figure(figsize=(10, 6))
	sns.kdeplot(data=train_df[train_df['Risk_Flag'] == 1], x='Age', label='High Risk (1)', fill=True, color='red')
	sns.kdeplot(data=train_df[train_df['Risk_Flag'] == 0], x='Age', label='Low Risk (0)', fill=True, color='blue')
	plt.title('Age Distribution by Risk Flag')
	plt.xlabel('Age')
	plt.ylabel('Density')
	plt.legend()
	plt.show()


def experience(df):
    print_min_and_max_value("Experience", df)

    temp_df = df.copy()

    experience_bins = [0, 5, 10, 15, 20]
    experience_labels = ['0-5 years', '5-10 years', '10-15 years', '15-20 years']

    temp_df['Experience_Group'] = pd.cut(temp_df['Experience'], bins=experience_bins, labels=experience_labels, right=False)

    experience_risk = temp_df.groupby(['Experience_Group', 'Risk_Flag'], observed=True).size().unstack(fill_value=0)

    experience_risk['Total'] = experience_risk.sum(axis=1)

    experience_risk['High_Risk_Percent'] = (experience_risk[1] / experience_risk['Total']) * 100
    experience_risk['Low_Risk_Percent'] = (experience_risk[0] / experience_risk['Total']) * 100

    print(experience_risk[['Total', 1, 0, 'High_Risk_Percent', 'Low_Risk_Percent']])

    add_space()

    experience_percent = experience_risk[['High_Risk_Percent', 'Low_Risk_Percent']]

    experience_percent.plot(
      kind='bar',
      stacked=True,
      figsize=(10, 6),
      color=['#ff4d4d', '#4CAF50']
    )
    plt.title('Risk Distribution by Experience Group (in %)')
    plt.xlabel('Experience Group')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.legend(['High Risk (1)', 'Low Risk (0)'], loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    add_space()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=experience_risk, x=experience_risk.index.astype(str), y='High_Risk_Percent', marker='o')
    plt.title('Distribution of High-Risk Percentage by Experience Group')
    plt.xlabel('Experience Group')
    plt.ylabel('High Risk (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    add_space()

### Married/Single
def relationship_status(df):
    # Work on a copy to avoid modifying the original DataFrame
    temp_df = df.copy()

    # Group by 'Married/Single' and 'Risk_Flag'
    marital_risk = temp_df.groupby(['Married/Single', 'Risk_Flag']).size().unstack(fill_value=0)

    # Calculate totals and risk percentages
    marital_risk['Total'] = marital_risk.sum(axis=1)
    marital_risk['High_Risk_Percent'] = (marital_risk.get(1, 0) / marital_risk['Total']) * 100
    marital_risk['Low_Risk_Percent'] = (marital_risk.get(0, 0) / marital_risk['Total']) * 100

    # Print risk distribution table
    print("📊 Marital Status Risk Distribution:")
    print(marital_risk[['Total', 1, 0, 'High_Risk_Percent', 'Low_Risk_Percent']])

    add_space()

    # Visualization
    marital_percent = marital_risk[['High_Risk_Percent', 'Low_Risk_Percent']]

    marital_percent.plot(
        kind='bar',
        stacked=True,
        figsize=(8, 6),
        color=['#ff4d4d', '#4CAF50']
    )

    plt.title('Risk Distribution by Marital Status (in %)')
    plt.xlabel('Marital Status')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.legend(['High Risk (1)', 'Low Risk (0)'], loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



### House Ownership
def house_ownership(df):
    # Work on a copy of the dataframe to avoid modifying the original
    temp_df = df.copy()

    # Group by 'House_Ownership' and 'Risk_Flag'
    house_ownership_risk = temp_df.groupby(['House_Ownership', 'Risk_Flag']).size().unstack(fill_value=0)

    # Calculate totals and percentages
    house_ownership_risk['Total'] = house_ownership_risk.sum(axis=1)
    house_ownership_risk['High_Risk_Percent'] = (house_ownership_risk.get(1, 0) / house_ownership_risk['Total']) * 100
    house_ownership_risk['Low_Risk_Percent'] = (house_ownership_risk.get(0, 0) / house_ownership_risk['Total']) * 100
    # Display results
    print("📊 House Ownership Risk Distribution:")
    print(house_ownership_risk[['Total', 1, 0, 'High_Risk_Percent', 'Low_Risk_Percent']])

    add_space()

    # Stacked bar chart
    house_percent = house_ownership_risk[['High_Risk_Percent', 'Low_Risk_Percent']]
    house_percent.plot(
        kind='bar',
        stacked=True,
        figsize=(8, 6),
        color=['#ff4d4d', '#4CAF50']
    )

    plt.title('Risk Distribution by House Ownership (in %)')
    plt.xlabel('House Ownership')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.legend(['High Risk (1)', 'Low Risk (0)'], loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    add_space()

    # Count plot (bar plot) for House Ownership vs Risk Flag
    plt.figure(figsize=(10, 6))
    sns.countplot(data=temp_df, x='House_Ownership', hue='Risk_Flag', palette='coolwarm')

    plt.title('Risk Distribution by House Ownership')
    plt.xlabel('House Ownership')
    plt.ylabel('Count')
    plt.legend(title='Risk Flag', labels=['Low Risk (0)', 'High Risk (1)'])
    plt.grid(True)
    plt.tight_layout()
    plt.show()


### CITY
def analyze_city_clusters(df):
    # Work on a copy of the dataframe
    temp_df = df.copy()

    # Group by 'CITY' and 'Risk_Flag'
    city_risk = temp_df.groupby(['CITY', 'Risk_Flag']).size().unstack(fill_value=0)

    # Add Total and Risk Percentages
    city_risk['Total'] = city_risk.sum(axis=1)
    city_risk['High_Risk_Percent'] = (city_risk.get(1, 0) / city_risk['Total']) * 100
    city_risk['Low_Risk_Percent'] = (city_risk.get(0, 0) / city_risk['Total']) * 100

    # Sort and display
    city_risk_sorted = city_risk.sort_values(by='High_Risk_Percent', ascending=False)
    print("📊 City-wise Risk Distribution:")
    print(city_risk_sorted[['Total', 1, 0, 'High_Risk_Percent', 'Low_Risk_Percent']])

    add_space()

    # Prepare for clustering
    city_features = city_risk[['High_Risk_Percent', 'Total']].copy()
    scaler = StandardScaler()
    city_scaled = scaler.fit_transform(city_features)

    # Apply KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    city_risk['Cluster'] = kmeans.fit_predict(city_scaled)

    # Scatterplot of clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=city_risk,
        x='High_Risk_Percent',
        y='Total',
        hue='Cluster',
        palette='tab10'
    )
    plt.title('City Clusters: High-Risk % vs Total Applicants')
    plt.xlabel('High Risk Percentage')
    plt.ylabel('Total Applicants')
    plt.grid(True)
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    add_space()

    # Cluster summary
    cluster_summary = city_risk.groupby('Cluster')[['High_Risk_Percent', 'Total']].agg(['mean', 'count'])
    print("\n📌 Cluster Summary:")
    print(cluster_summary)

    add_space()

    # Cluster count bar chart
    cluster_counts = city_risk['Cluster'].value_counts().sort_index()
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Cities in Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Cities')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    add_space()

    # Save mapping of city to cluster
    city_clusters = city_risk.reset_index()[['CITY', 'Cluster']]
    city_clusters.to_csv('city_clusters.csv', index=False)
    print("✅ City-cluster mapping saved to 'city_clusters.csv'")



### Car Ownership
def car_ownership(df):
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Group by Car_Ownership and Risk_Flag
    car_risk = df_copy.groupby(['Car_Ownership', 'Risk_Flag']).size().unstack(fill_value=0)

    # Total applicants per car ownership category
    car_risk['Total'] = car_risk.sum(axis=1)

    # High and low risk percentages
    car_risk['High_Risk_Percent'] = (car_risk[1] / car_risk['Total']) * 100
    car_risk['Low_Risk_Percent'] = (car_risk[0] / car_risk['Total']) * 100

    # Display
    print(car_risk[['Total', 1, 0, 'High_Risk_Percent', 'Low_Risk_Percent']])

    add_space()
    
    # Bar Chart
    car_risk[['High_Risk_Percent', 'Low_Risk_Percent']].plot(
    kind='bar',
    figsize=(8, 5),
    stacked=True,
    color=['red', 'green']
    )

    plt.title('Loan Risk Percentage by Car Ownership')
    plt.ylabel('Percentage')
    plt.xlabel('Car Ownership')
    plt.xticks(rotation=0)
    plt.legend(['High Risk', 'Low Risk'])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    add_space()

    # Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
    data=df_copy[df_copy['Car_Ownership'] == 'yes'],
    x='Risk_Flag',
    fill=True,
    label='Owns Car',
    color='blue'
    )
    sns.kdeplot(
    data=df_copy[df_copy['Car_Ownership'] == 'no'],
    x='Risk_Flag',
    fill=True,
    label='No Car',
    color='orange'
    )

    plt.title('Risk Distribution by Car Ownership')
    plt.xlabel('Risk_Flag (0 = Low Risk, 1 = High Risk)')
    plt.ylabel('Density')
    plt.legend(title='Car Ownership')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


### Profession
def profession_vs_risk_flag(df, n_clusters=5):
    # Work on a copy of the dataframe
    temp_df = df.copy()

    # Group by Profession and Risk_Flag
    grouped = temp_df.groupby(['Profession', 'Risk_Flag']).size().unstack(fill_value=0)
    grouped.columns = ['Low_Risk_Count', 'High_Risk_Count']

    # Calculate percentages
    grouped['Total'] = grouped.sum(axis=1)
    grouped['High_Risk_Percent'] = (grouped['High_Risk_Count'] / grouped['Total']) * 100
    grouped['Low_Risk_Percent'] = (grouped['Low_Risk_Count'] / grouped['Total']) * 100

    # Reset index to make 'Profession' a column again
    grouped = grouped.reset_index()

    # Sort by High Risk Percentage
    grouped_sorted = grouped.sort_values(by='High_Risk_Percent', ascending=False)
    print("📊 Profession-wise Risk Distribution:")
    print(grouped_sorted[['Profession', 'Total', 'High_Risk_Count', 'Low_Risk_Count', 'High_Risk_Percent', 'Low_Risk_Percent']])

    # Prepare for clustering
    profession_features = grouped_sorted[['High_Risk_Percent', 'Total']].copy()
    scaler = StandardScaler()
    profession_scaled = scaler.fit_transform(profession_features)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    grouped_sorted['Cluster'] = kmeans.fit_predict(profession_scaled)

    # Scatterplot of clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=grouped_sorted,
        x='High_Risk_Percent',
        y='Total',
        hue='Cluster',
        palette='tab10',
        s=100,
        edgecolors='k'
    )
    plt.title('Profession Clusters: High-Risk % vs Total Applicants')
    plt.xlabel('High Risk Percentage')
    plt.ylabel('Total Applicants')
    plt.grid(True)
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    # Cluster summary
    cluster_summary = grouped_sorted.groupby('Cluster')[['High_Risk_Percent', 'Total']].agg(['mean', 'count'])
    print("\n📌 Cluster Summary:")
    print(cluster_summary)

    # Cluster count bar chart
    cluster_counts = grouped_sorted['Cluster'].value_counts().sort_index()
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Professions in Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Professions')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return grouped_sorted[['Profession', 'Cluster']]



### State
def state_vs_risk_flag(df):
    # Create a percentage table
    state_risk_counts = df.groupby(['STATE', 'Risk_Flag']).size().unstack(fill_value=0)
    state_risk_percentage = state_risk_counts.div(state_risk_counts.sum(axis=1), axis=0) * 100

    # Print percentage table
    print("📊 STATE vs Risk_Flag Percentage Table:")
    print(state_risk_percentage.round(2))

    add_space()

    # Bar chart of percentage distribution
    state_risk_percentage.plot(kind='bar', stacked=True, figsize=(14, 6), colormap='Set2')
    plt.title("Risk_Flag Percentage Distribution by STATE")
    plt.ylabel("Percentage")
    plt.xlabel("State")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Risk_Flag", loc='upper right')
    plt.tight_layout()
    plt.show()

    add_space()

    # Distribution plot (count) with color adjustment
    plt.figure(figsize=(14, 6))
    sns.countplot(data=df, x='STATE', hue='Risk_Flag', palette={0: 'green', 1: 'red'})
    plt.title("Risk_Flag Count Distribution by STATE")
    plt.xlabel("State")
    plt.ylabel("Number of Applicants")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Risk_Flag")
    plt.tight_layout()
    plt.show()

### CURRENT JOB YEARS
def analyze_job_years_vs_risk(df):
    # Group and calculate counts
    job_risk = df.groupby(['CURRENT_JOB_YRS', 'Risk_Flag']).size().unstack(fill_value=0)

    # Calculate percentage distribution
    job_risk_percent = job_risk.div(job_risk.sum(axis=1), axis=0) * 100
    print("📊 CURRENT_JOB_YRS vs Risk_Flag Percentage Table:")
    print(job_risk_percent.round(2))

    # Bar chart
    job_risk.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')
    plt.title("CURRENT_JOB_YRS vs Risk_Flag - Stacked Bar Chart")
    plt.xlabel("Current Job Years")
    plt.ylabel("Number of Applicants")
    plt.legend(title='Risk_Flag')
    plt.tight_layout()
    plt.show()

    # Bar chart with percentage
    job_risk_percent.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='coolwarm')
    plt.title("CURRENT_JOB_YRS vs Risk_Flag - Percentage Bar Chart")
    plt.xlabel("Current Job Years")
    plt.ylabel("Percentage")
    plt.legend(title='Risk_Flag')
    plt.tight_layout()
    plt.show()

    # Distribution plot (histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='CURRENT_JOB_YRS', hue='Risk_Flag', multiple='stack', bins=11, palette='Set1')
    plt.title("Distribution of Risk_Flag across CURRENT_JOB_YRS")
    plt.xlabel("Current Job Years")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


### CURRENT_HOUSE_YRS
def analyze_house_years_vs_risk(df_train):
    # Group and calculate counts for CURRENT_HOUSE_YRS vs Risk_Flag
    house_risk = df_train.groupby(['CURRENT_HOUSE_YRS', 'Risk_Flag']).size().unstack(fill_value=0)

    # Calculate percentage distribution
    house_risk_percent = house_risk.div(house_risk.sum(axis=1), axis=0) * 100
    print("📊 CURRENT_HOUSE_YRS vs Risk_Flag Percentage Table:")
    print(house_risk_percent.round(2))

    # Bar chart for count distribution
    house_risk.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')
    plt.title("CURRENT_HOUSE_YRS vs Risk_Flag - Stacked Bar Chart")
    plt.xlabel("Current House Years")
    plt.ylabel("Number of Applicants")
    plt.legend(title='Risk_Flag')
    plt.tight_layout()
    plt.show()

    # Bar chart for percentage distribution
    house_risk_percent.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='coolwarm')
    plt.title("CURRENT_HOUSE_YRS vs Risk_Flag - Percentage Bar Chart")
    plt.xlabel("Current House Years")
    plt.ylabel("Percentage")
    plt.legend(title='Risk_Flag')
    plt.tight_layout()
    plt.show()

    # Distribution plot (histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_train, x='CURRENT_HOUSE_YRS', hue='Risk_Flag', multiple='stack', bins=11, palette='Set1')
    plt.title("Distribution of Risk_Flag across CURRENT_HOUSE_YRS")
    plt.xlabel("Current House Years")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
