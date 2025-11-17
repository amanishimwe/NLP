Ideally, Humans communicate using language as a form of exchanging ideas, and words form the integral part of the language. Human are so good at procesing information that words convey. Forexample a single word "City" will convey information about a Country's or Nation's large town as shown below.
![Tokyo](https://upload.wikimedia.org/wikipedia/commons/b/b2/Skyscrapers_of_Shinjuku_2009_January.jpg)
*Figure 2: Tokyo, the capital city of Japan.*
Now, Computers do not understand ideas that come from words,therefore, we need to find a way of converting words into a format that computers will understand--this is where embeddings come into action. The representation of words into vector spaces is called word embeddings. They are so essential that they help us to solve complex Natural Language Processings Tasks like document classification, and clustering e.t.c.

The two most popular word embeddings are Word to Vec and Glove. 
This tutorial will concentrate on Word to Vec, which was invented by researchers at google to map words to higher dimensional vector Spaces.

## So What is Word to Vec?
Word to vec works on a fundemental principle that words with the same meanings should have the same Vector Spaces. Word to Vec Presents 02 Models -- Continous Bag of Words (CBOW) and Skip Gram.

## Choice for CBOW and Skip Gram
In the CBOW, the model uses surrounding context words to predict the center (target) word, whereas, skip-gram does the reverse, predicting context words from a given target word.

Empirically, CBOW typically trains faster, but skip-gram produces higher-quality embeddings for infrequent word. The choice between them depends on the application: CBOW is often chosen for efficiency when frequent words are most important, while skip-gram is chosen when the corpus is smaller or when capturing rare-word semantics is crucial.

