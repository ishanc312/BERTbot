# BERTbot
# This profanity filtering bot inputs user messages sent in the server to a Neural Network, which forms a BERT Embedding of the message before feeding it into an LSTM for evaluation.
# This model is based on binary classification, classifying user messages as either vulgar or not. 
# The original dataset was originally imbalanced; we utilize oversampling on the minority class to balance the training dataset.
# For future work on the bot, I hope to first further optimize the model, alongside strengthen the filter with additional data + traditional filtering methods to supplement the bot to provide a second set of checks on a message.
# Additionally, I hope to expand on its moderation capabilities from merely deleting messages to keeping track of users, kicks/bans for certain infractions, etc.
