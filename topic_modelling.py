from preprocessing import preprocess_text
import nltk, re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
with open("lyrics.txt", "r", encoding="utf-8") as file:
    lyrics = file.read()

corpus = lyrics.split("\n")

preprocessed_corpus = [preprocess_text(song) for song in corpus]
#print(preprocessed_corpus[5:45])
stop_list = ["yeah", "oh", "ooh", "ha", "no", "La-la-la-la-la-la-la", "ya", "'em", "na-na-na-na", "motherfucker", "please",
             "think", "cause", "feel", "best", "never", "like", "wan", "gon", "good", "begin", "forever", "come", "make", "baby"]

def filter_out_stop_words(corpus):
    no_stops_corpus = []
    for song in corpus:
        no_stops_songs = " ".join([word for word in song.split() if word not in stop_list])
        no_stops_corpus.append(no_stops_songs)
    return no_stops_corpus

filtered_for_stops = filter_out_stop_words(preprocessed_corpus)

#print(filtered_for_stops[5:30])

bag_of_words_creator = CountVectorizer()
bag_of_words = bag_of_words_creator.fit_transform(filtered_for_stops)


tfidf_creator = TfidfVectorizer()
tfidf = tfidf_creator.fit_transform(filtered_for_stops)

lda_bag_of_words_creator = LatentDirichletAllocation(learning_method='online', n_components=10)
lda_bag_of_words = lda_bag_of_words_creator.fit_transform(bag_of_words)

# creating the tf-idf LDA model
lda_tfidf_creator = LatentDirichletAllocation(learning_method='online', n_components=10)
lda_tfidf = lda_tfidf_creator.fit_transform(tfidf)

print("~~~ Topics found by bag of words LDA ~~~")
for topic_id, topic in enumerate(lda_bag_of_words_creator.components_):
  message = "Topic #{}: ".format(topic_id + 1)
  message += " ".join([bag_of_words_creator.get_feature_names_out()[i] for i in topic.argsort()[:-5 :-1]])
  print(message)

print("\n\n~~~ Topics found by tf-idf LDA ~~~")
for topic_id, topic in enumerate(lda_tfidf_creator.components_):
  message = "Topic #{}: ".format(topic_id + 1)
  message += " ".join([tfidf_creator.get_feature_names_out()[i] for i in topic.argsort()[:-5 :-1]])
  print(message)