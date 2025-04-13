import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

# Change dataset paths 
training_dataset_path = 'split/training.csv'
validation_dataset_path = 'split/validation.csv'
testing_dataset_path = 'split/testing.csv'

train_df = pd.read_csv(training_dataset_path, header=None)
validation_df = pd.read_csv(validation_dataset_path, header=None)
test_df = pd.read_csv(testing_dataset_path, header=None)
train_df.columns = validation_df.columns = test_df.columns = ['sender_email', 'email_body', 'label']

x_train = train_df['email_body'].values
y_train = train_df['label'].values
x_validation = validation_df['email_body'].values
y_validation = validation_df['label'].values
x_test = test_df['email_body'].values
y_test = test_df['label'].values

# TF-IDF vectorize email text
vectorizer = TfidfVectorizer(max_features=5000)
x_train = vectorizer.fit_transform(x_train).toarray()
x_validation = vectorizer.transform(x_validation).toarray()
x_test = vectorizer.transform(x_test).toarray()

model = Sequential([
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy()])
model.fit(x_train, y_train, epochs=5, validation_data=(x_validation, y_validation))

test_predictions = (model.predict(x_test) >= 0.5).astype('int')
with open('trained_baseline.txt', 'w') as file:
    for label in test_predictions:
        file.write(str(label[0]) + '\n')