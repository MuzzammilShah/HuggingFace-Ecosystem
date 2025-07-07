import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class NLPEngine:
    def __init__(self, device=-1):
        device_name = 'cuda' if device == 0 else 'mps' if device == 'mps' else 'cpu'
        print(f"Initializing NLPEngine on device: {device_name}")

        self.sentiment = pipeline(
            'sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english',
            device=device
        )
        self.summarizer = pipeline(
            'summarization',
            model='facebook/bart-large-cnn',
            device=device
        )
        self.ner = pipeline(
            'ner',
            model='dslim/bert-base-NER',
            aggregation_strategy='simple',
            device=device
        )
        self.qa = pipeline(
            'question-answering',
            model='deepset/roberta-base-squad2',
            device=device
        )
        self.generator = pipeline(
            'text-generation',
            model='gpt2-medium',
            device=device
        )
        ## For initial tests of semantic search
        # self.retriever = pipeline(
        #     'feature-extraction',
        #     model='sentence-transformers/all-MiniLM-L6-v2',
        #     device=device
        # )
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        print("NLPEngine initialized successfully.")

    def analyze_sentiment(self, text):
        return self.sentiment(text)

    def summarize_text(self, text, max_length=150, min_length=30):
        return self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

    def extract_entities(self, text):
        return self.ner(text)

    def answer_question(self, question, context):
        return self.qa(question=question, context=context)

    def generate_text(self, prompt, max_length=50, num_return_sequences=1):
        return self.generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)

    def get_embeddings(self, text_or_texts):

        ## For initial tests of semantic search
        ## I tried to manually calculate and adjust the embedding sizes, until i directly used the sentence transformer from HF
        # embeddings = self.retriever(text_or_texts)
        # if isinstance(text_or_texts, str):
        #     return torch.mean(torch.tensor(embeddings[0]), dim=0)
        # else:
        #     return torch.stack([torch.mean(torch.tensor(emb), dim=0) for emb in embeddings])
        return torch.tensor(self.sentence_model.encode(text_or_texts))

if __name__ == "__main__":
    pass

    # Uncomment all the codes from here if you would like to run this as a stand alone script and test on your terminal how each pipeline works
    # =================================

    # # Check for available hardware acceleration
    # if torch.cuda.is_available():
    #     selected_device = 0  # CUDA
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     selected_device = 'mps'  # MPS for Apple Silicon
    # else:
    #     selected_device = -1  # CPU

    # print("Starting NLPEngine tests...")
    # engine = NLPEngine(device=selected_device)

    # sample_text_sentiment = "Hugging Face is a great platform for NLP."
    # print(f"\nSentiment for '{sample_text_sentiment}': {engine.analyze_sentiment(sample_text_sentiment)}")

    # sample_text_summarize = """
    # The Hugging Face ecosystem provides a wide array of tools and models for natural language processing.
    # It includes transformers for state-of-the-art models, datasets for accessing and sharing data,
    # and a model hub for discovering and using pre-trained models. Developers can leverage these
    # resources to build powerful NLP applications with relative ease. The platform also supports
    # various tasks such as text classification, summarization, translation, and question answering.
    # The quick brown fox jumps over the lazy dog. This sentence is repeated multiple times to ensure
    # the text is long enough for summarization to be meaningful. The quick brown fox jumps over the lazy dog.
    # The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.
    # """
    # print(f"\nSummary: {engine.summarize_text(sample_text_summarize, min_length=20, max_length=50)}")

    # sample_text_ner = "Apple Inc. is looking at buying U.K. startup for $1 billion. Tim Cook is the CEO. The meeting is in New York."
    # print(f"\nNER for '{sample_text_ner}': {engine.extract_entities(sample_text_ner)}")

    # sample_context_qa = "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre Museum."
    # sample_question_qa = "What is Paris known for?"
    # print(f"\nQA for context '{sample_context_qa}' and question '{sample_question_qa}': {engine.answer_question(question=sample_question_qa, context=sample_context_qa)}")

    # sample_prompt_generate = "In a world powered by AI,"
    # print(f"\nGenerated Text from prompt '{sample_prompt_generate}': {engine.generate_text(sample_prompt_generate, max_length=30)}")

    # # sample_text_retriever1 = "This is a test sentence for semantic search."
    # # sample_text_retriever2 = "Another sentence to compare for similarity."
    # # embedding1 = engine.get_embeddings(sample_text_retriever1)
    # # embedding2 = engine.get_embeddings(sample_text_retriever2)
    # # print(f"\nEmbedding shape for a single sentence: {embedding1.shape}")

    # corpus = ["The weather is sunny today.", "I enjoy walking in the park on a beautiful day.", "AI is transforming many industries."]
    # query = "What is the forecast for today?"

    # query_embedding = engine.get_embeddings(query)
    # corpus_embeddings = engine.get_embeddings(corpus)

    # print(f"Query embedding shape: {query_embedding.shape}")
    # print(f"Corpus embeddings shape: {corpus_embeddings.shape}")

    # if query_embedding.ndim == 1:
    #     query_embedding = query_embedding.unsqueeze(0)

    # similarities = torch.nn.functional.cosine_similarity(query_embedding, corpus_embeddings, dim=1)
    # print(f"\nSimilarities between '{query}' and corpus sentences: {similarities.tolist()}")

    # print("\nNLPEngine tests completed.")