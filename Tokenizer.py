"""
visit tiktokenizer.vercel.app to understand tokenization process.
Let's start from the basic how python understand text sequence,
Text - Strings are immutable sequences of Unicode code points ie. every single character is represented with unique number according to the unicode standards.
Eg : 'h' - 104
In python we use ord() function to get the unicode (It works only with char not string)
Unicode standards keep on changing so we cant use it for encoding texts instead we can use encodings like UTF-8, UTF-16, UTF-32
Encoding stores the text data as binary data.
UTF-8 is commonly used for encoding characters.
Since each CHAR have unique numbers, one simple word will have long list of numbers.
Tokenization process turns WORDS into unique numbers ie. tokens
"""
