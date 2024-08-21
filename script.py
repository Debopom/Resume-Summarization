import sys
import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the LabelEncoder and model
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
loaded_model = pickle.load(open("voting_classifier_model.pkl", 'rb'))

# Get the input CSV file path from the command line arguments
input_csv_path = sys.argv[1]
data = pd.read_csv(input_csv_path)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 2]
    return ' '.join(words)

data_cleaned = data.dropna(subset=['Resume_str'])
data_cleaned['Resume_str_cleaned'] = data_cleaned['Resume_str'].apply(preprocess_text)

ngrams_to_remove = ['city state', 'company name', 'name city', 'name city state',
                    'current company name', 'high school', 'microsoft office', 'team member']
def remove_ngrams(text, ngrams):
    for ngram in ngrams:
        text = text.replace(ngram, '')
    return text

data_cleaned['Resume_str_cleaned'] = data_cleaned['Resume_str_cleaned'].apply(lambda x: remove_ngrams(x, ngrams_to_remove))

# Transform the cleaned text using TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = tfidf.fit_transform(data_cleaned['Resume_str_cleaned'])

# Predict 
predictions = loaded_model.predict(X)

# Convert predictions back to original category names
predicted_categories = label_encoder.inverse_transform(predictions)

# Save the output CSV file
output_df = pd.DataFrame({
    'Filename': data['ID'], 
    'Category': predicted_categories
})
output_csv_path = "categorized_resumes.csv"
output_df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")
# Move or save HTML files to respective category folders
for index, row in data.iterrows():
    resume_html_content = row['Resume_html']  
    predicted_category = predicted_categories[index]
    
    category_folder = os.path.join('categorized_resumes', predicted_category)
    os.makedirs(category_folder, exist_ok=True)

    resume_id = row['ID']  
    filename = f"{resume_id}.html"
    file_path = os.path.join(category_folder, filename)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(resume_html_content)

print("HTML files have been saved to their respective category folders.")
