# Covenant University Student Book Chatbot

This repository contains a Streamlit-based chatbot designed to interact with the Covenant University Student Handbook. The chatbot uses advanced natural language processing (NLP) techniques to answer questions based on the content of the handbook. It leverages the LangChain framework, Hugging Face embeddings, and the Groq API for efficient and accurate responses.

## Features

- **PDF Processing**: Automatically loads and processes the Covenant University Student Handbook from a provided URL.
- **Text Splitting**: Splits the document into manageable chunks for efficient processing.
- **Embedding Generation**: Uses Hugging Face's `sentence-transformers/all-mpnet-base-v2` model to generate embeddings for the document.
- **Vector Storage**: Stores the embeddings in a FAISS vector store for quick retrieval.
- **Question Answering**: Allows users to ask questions and retrieves answers along with the sources from the handbook.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- Streamlit
- LangChain
- Hugging Face Transformers
- Groq API key (stored in `.env` file)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/covenant-university-chatbot.git
   cd covenant-university-chatbot
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment variables**:
   Create a `.env` file in the root directory and add your Groq API key:
   ```plaintext
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

1. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **Process the Handbook**:
   - Open the sidebar and click the "Process Handbook" button.
   - The application will download the handbook, split it into chunks, generate embeddings, and store them in a vector index.

3. **Ask Questions**:
   - Once the processing is complete, you can enter your question in the text input box.
   - The chatbot will retrieve the answer and display it along with the relevant sources from the handbook.

## Code Overview


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://langchain.com/) for the powerful NLP framework.
- [Hugging Face](https://huggingface.co/) for the embeddings model.
- [Groq](https://groq.com/) for the LLM API.
- [Streamlit](https://streamlit.io/) for the easy-to-use web app framework.

---

Feel free to explore the code and customize it according to your needs. Happy coding! 🚀
