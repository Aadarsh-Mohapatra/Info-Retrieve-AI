# Info-Retrieve-AI

## Introduction

Welcome to Info-Retrieve-AI!

By integrating machine learning (ML) and natural language processing (NLP) techniques based on RAG approach, the initiative aims to revolutionize the processing and analysis of data from diverse sources, with a primary focus on any PDF documents and web-based newsletters/blogs. This approach is expected to bolster secondary research capabilities across the companys various operational departments.

## Getting Started

These instructions will get your copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or newer
- pip (Python package installer)
- API Keys for various models like Langchain, DeepLake, OpenAI, Gemini, Hugging Face, Ngrok, Pinecone, etc

## Setup Instructions

Follow these steps to get your development environment running:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Aadarsh-Mohapatra/Info-Retrieve-AI.git
   cd Info-Retrieve-AI
   ```

2. **Configure Settings**

   - Create your `config.py` based on the `dummy_config.py` template to store API keys and other configurations so that the IRS is up and running.

3. **Change directory**

   - Change the path into your local directory so that the repository can be up and running smoothly.

4. **Setup PowerShell Execution Policy (Windows PowerShell only)**

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   ```

   - **This is optional as you can directly run this in command prompt (cmd)**

5. **Create and Activate Virtual Environment**

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

6. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

7. **Running the IRS Application**

   - **Launch the IRS Model Application**

   ```bash
   streamlit run main\app.py
   ```

   - **Launch the Performance Metrics UI**

   ```bash
   streamlit run main\app_pm.py
   ```

   - **_"Please note that unless the entire model is executed as well as RAGA code files, the application for Performance Metrics UI will not display any statistics or plots"_**

## Additional Information

- **Folder Structure**: Ensure your project folder is set up according to the guidelines provided in the repository structure.
- **Dependencies**: All dependencies can be found in the `requirements.txt` file. Make sure to update this file as you add or update libraries.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Collaborators

- The main collaborators behind the project are -

Aadarsh Mohapatra - www.github.com/Aadarsh-Mohapatra
Aayush Oberoi - www.github.com/aayushoberoi

Using their combined knowledge of Web Scraping, Modeling, knowledge of various performance metrics and fine tuning LLM models, they have completed this entire project within a span of 3 months. Given above are their GitHub profile links that can be accessed to see and run the project and have a chatbot ready to serve your purpose.
