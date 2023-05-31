import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page title
st.title("CSV File Explorer")

# File selection
file = st.file_uploader("Upload a CSV file", type=["csv"])

if file is not None:
    # Read CSV file
    df = pd.read_csv(file)

    # Display column names
    st.write("Column Names:")
    selected_columns = st.multiselect("Select columns", df.columns.tolist())

    # Filter dataframe based on selected columns and language
    if selected_columns:
        filtered_df = df[selected_columns]
        filtered_df_en = filtered_df[filtered_df["language"] == "en"]

        # Number of tweets selection
        st.write("Number of Tweets:")
        option = st.radio("Choose the number of tweets", ("All", "Specify"))

        if option == "All":
            st.write(filtered_df_en)
            num_tweets = len(filtered_df_en)
        else:
            num_tweets = st.number_input("Enter the number of tweets", min_value=1, max_value=len(filtered_df_en),
                                         value=len(filtered_df_en), step=1)
            st.write(filtered_df_en.head(num_tweets))

        # Classification using your model
        st.write("Classification Results:")
        model_path = "D:/UNI/fyp/New folder/lstm_depressed.h5"  # Replace with the path to your saved model
        model = load_model(model_path)

        # Tokenizer
        num_words = 10000  # Replace with your desired number of words
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(filtered_df_en["tweet"])

        # Set the maximum sequence length
        max_sequence_length = 30 # Replace with your desired maximum sequence length

        depressed_count = 0
        non_depressed_count = 0
        depressed_tweets=[]
        non_depressed_tweets=[]
        for i in range(num_tweets):
            tweet = filtered_df_en.iloc[i]["tweet"]
            # Tokenize the tweet
            tweet_sequence = tokenizer.texts_to_sequences([tweet])
            tweet_sequence = pad_sequences(tweet_sequence, maxlen=max_sequence_length)

            # Perform classification using your LSTM model on the tokenized tweet
            # Replace the placeholder code below with your actual classification logic
            predicted_label = model.predict(tweet_sequence)

            if predicted_label > 0.5:
                depressed_count += 1
                depressed_tweets.append(tweet)
            else:
                non_depressed_count += 1
                non_depressed_tweets.append(tweet)

        # Calculate percentage of depressed tweets
        depressed_percentage = (depressed_count / num_tweets) * 100
        non_depressed_percentage = (non_depressed_count / num_tweets) * 100

        # Create a pie chart
        labels = ['Depressed', 'Non-Depressed']
        sizes = [depressed_count, non_depressed_count]
        explode = (0.1, 0)  # Explode the depressed slice
        colors = ['#ff9999', '#66b3ff']
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Tweets Classification')
        st.pyplot(plt)

        # Display tables for depressed and non-depressed tweets
        st.write("Depressed Tweets:")
        depressed_df = pd.DataFrame({"Tweet": depressed_tweets})
        st.table(depressed_df)

        st.write("Non-Depressed Tweets:")
        non_depressed_df = pd.DataFrame({"Tweet": non_depressed_tweets})
        st.table(non_depressed_df)

    else:
        st.write("Please select at least one column.")
