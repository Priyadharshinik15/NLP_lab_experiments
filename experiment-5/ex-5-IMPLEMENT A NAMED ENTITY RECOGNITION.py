import nltk
from nltk import word_tokenize, pos_tag

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# User input
text = input("Enter legal text: ")

# Tokenization
tokens = word_tokenize(text)

# POS tagging
tags = pos_tag(tokens)

print("\nDetected Named Entities:")

count = 0

for word, tag in tags:
    if tag == "NNP":
        print(word, "-> ENTITY")
        count += 1

# Actual number of entities
actual = int(input("\nEnter actual number of entities: "))

# Accuracy calculation
accuracy = (min(count, actual) / max(count, actual)) * 100

print("\nPredicted Entities:", count)
print("NER Accuracy:", round(accuracy, 2), "%")