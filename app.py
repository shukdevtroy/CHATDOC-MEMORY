import gradio as gr
from llama_index.core import VectorStoreIndex, Document  # Imports VectorStoreIndex for semantic search and Document class for text storage
from llama_index.llms.openai import OpenAI  # Imports OpenAI language model integration from LlamaIndex
from llama_index.core import Settings  # Imports Settings to configure global LlamaIndex parameters
import os  # Imports OS module for interacting with the operating system (file paths, environment variables)
import pdfplumber  # Imports pdfplumber library for extracting text from PDF files
from docx import Document as DocxDocument  # Imports Document class from python-docx, renamed to avoid conflict with LlamaIndex's Document
import json  # Imports JSON module for parsing and creating JSON data
from datetime import datetime  # Imports datetime class for working with dates and times
import hashlib  # Imports hashlib for creating hash functions (MD5, SHA, etc.) for data integrity/unique identifiers

# Global variables
chat_engine = None  # Stores the LlamaIndex chat engine instance; None until initialized with documents
conversation_history = []  # Empty list to store all chat messages (user questions and AI responses)
current_user_id = None  # Stores hashed identifier for current user based on their API key

# Function to generate user ID from API key
def get_user_id(api_key):  # Defines function that takes API key string as input
    if not api_key:  # Checks if api_key is None, empty string, or falsy value
        return None  # Returns None if no API key provided
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]  # Encodes key to bytes, creates SHA-256 hash, converts to hex string, returns first 16 characters as unique user ID

# Function to get user-specific filename
def get_user_file(api_key):  # Defines function that generates unique filename for each user
    user_id = get_user_id(api_key)  # Calls get_user_id to generate unique identifier from API key
    if not user_id:  # Checks if user_id is None (happens when api_key is invalid/empty)
        return None  # Returns None if no valid user ID could be generated
    return f"conversations_{user_id}.json"  # Returns formatted string with user-specific filename for storing conversation history

# Function to read PDF files
def read_pdf(file_path):  # Defines function that takes a file path string as parameter
    with pdfplumber.open(file_path) as pdf:  # Opens PDF file using context manager (auto-closes after use)
        text = ''  # Initializes empty string to accumulate extracted text
        for page in pdf.pages:  # Loops through each page object in the PDF
            text += page.extract_text() + '\n'  # Extracts text from current page and appends it with newline character
    return text  # Returns the complete concatenated text from all pages

# Function to read DOCX files
def read_docx(file_path):  # Defines function that takes a file path string as parameter
    doc = DocxDocument(file_path)  # Creates a Document object by loading the .docx file
    text = ''  # Initializes empty string to store extracted text
    for paragraph in doc.paragraphs:  # Iterates through each paragraph object in the document
        text += paragraph.text + '\n'  # Extracts text from current paragraph and appends with newline
    return text  # Returns the complete text from all paragraphs

# Function to load and index documents
def load_data(files, api_key):  # Defines function that accepts uploaded files list and API key string
    global chat_engine, current_user_id  # Declares these as global so changes persist outside function scope
    
    if not api_key:  # Checks if API key is missing, empty, or None
        return "Please provide your OpenAI API key first."  # Returns error message prompting for API key
    
    if not files:  # Checks if files list is empty, None, or falsy
        return "Please upload files to proceed."  # Returns error message prompting for file upload
    
    try:  # Begins try block to catch any errors during document processing
        # Set current user
        current_user_id = get_user_id(api_key)  # Generates and stores unique user ID from API key in global variable
        
        docs = []  # Initializes empty list to store Document objects
        for file in files:  # Loops through each uploaded file object in the files list
            if file.name.endswith('.pdf'):  # Checks if filename ends with .pdf extension
                text = read_pdf(file.name)  # Extracts all text from PDF using previously defined function
                docs.append(Document(text=text))  # Creates LlamaIndex Document object from text and adds to list
            elif file.name.endswith('.docx'):  # Checks if filename ends with .docx extension
                text = read_docx(file.name)  # Extracts all text from Word document using previously defined function
                docs.append(Document(text=text))  # Creates Document object from extracted text and appends to list
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = api_key  # Sets environment variable so OpenAI library can automatically access the API key
        
        Settings.llm = OpenAI(  # Configures the global LLM (Large Language Model) settings for LlamaIndex
            model="gpt-5-nano",  # Specifies which OpenAI model to use (GPT-4 optimized mini version)
            temperature=0.5,  # Sets randomness level (0=deterministic/focused, 1=creative/random); 0.5 is balanced
            api_key=api_key,  # Passes API key directly to OpenAI client for authentication
            system_prompt="You are a helpful AI assistant that answers questions based on the provided documents. Always base your answers on the content of the uploaded documents. If the answer cannot be found in the documents, clearly state that. Be accurate, concise, and cite specific information from the documents when possible."  # Instructions that guide the AI's behavior, response style, and constrain it to document content
        )
        
        index = VectorStoreIndex.from_documents(docs)  # Creates vector embeddings of all documents for semantic similarity search
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)  # Converts index to conversational chat interface; condense_question mode reformulates follow-up questions using conversation context; verbose=True prints debug info
        
        return "Documents loaded and indexed successfully! You can now start chatting."  # Returns success message to display to user
    except Exception as e:  # Catches any error that occurred anywhere in the try block
        return f"Error loading documents: {str(e)}"  # Returns formatted error message with details of what went wrong

# Function to handle chat
def chat_with_docs(message, history, api_key):  # Defines function that takes user's message, chat history list, and API key as parameters
    global chat_engine, conversation_history, current_user_id  # Declares global variables so function can read/modify them
    
    if not api_key:  # Checks if API key is missing, empty, or None
        return history + [{"role": "assistant", "content": "Please enter your OpenAI API key first."}]  # Returns existing history plus error message as assistant response
    
    # Update current user
    current_user_id = get_user_id(api_key)  # Generates unique user ID from API key and stores in global variable
    
    if chat_engine is None:  # Checks if chat_engine hasn't been initialized (no documents loaded yet)
        return history + [  # Returns history with two new messages added to the list
            {"role": "user", "content": message},  # Adds user's question to history as dictionary
            {"role": "assistant", "content": "Please upload and load documents first before asking questions."}  # Adds assistant's error response
        ]
    
    try:  # Begins try block to catch errors during chat interaction
        response = chat_engine.chat(message)  # Sends user message to chat engine, which searches documents and generates response
        conversation_history.append({"role": "user", "content": message})  # Adds user message to global conversation history list
        conversation_history.append({"role": "assistant", "content": response.response})  # Adds AI response to global conversation history (response.response extracts text from response object)
        
        return history + [  # Returns updated history by concatenating existing history with new messages
            {"role": "user", "content": message},  # Adds current user message
            {"role": "assistant", "content": response.response}  # Adds AI's response text
        ]
    except Exception as e:  # Catches any error that occurred during chat processing
        return history + [  # Returns history with error message instead of crashing
            {"role": "user", "content": message},  # Still adds user's message to show what they asked
            {"role": "assistant", "content": f"Error: {str(e)}"}  # Adds error details as assistant response for debugging
        ]

# Function to save conversation (user-specific)
def save_conversation(api_key):  # Defines function that saves conversation to user-specific file
    global conversation_history  # Accesses global conversation_history variable
    
    if not api_key:  # Checks if API key is missing or empty
        return "Please enter your OpenAI API key first."  # Returns error message and exits function
    
    if not conversation_history:  # Checks if conversation_history list is empty (no messages to save)
        return "No conversation to save."  # Returns message indicating nothing to save
    
    try:  # Begins try block to handle file writing errors
        user_file = get_user_file(api_key)  # Generates unique filename based on user's API key (e.g., "conversations_abc123.json")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Gets current date/time and formats as string (e.g., "2026-01-14_15-30-45")
        
        with open(user_file, "a") as f:  # Opens user's file in append mode ("a" means add to end without overwriting); auto-closes when done
            conv_data = {  # Creates dictionary to structure the conversation data
                "timestamp": timestamp,  # Stores when conversation was saved
                "messages": conversation_history  # Stores all messages from current conversation
            }
            json.dump(conv_data, f)  # Converts dictionary to JSON format and writes to file
            f.write("\n")  # Adds newline character so each saved conversation is on separate line in file
        return "Conversation saved successfully!"  # Returns success message to display to user
    except Exception as e:  # Catches any errors during file operations (permission issues, disk full, etc.)
        return f"Error saving conversation: {str(e)}"  # Returns formatted error message with details

# Function to delete all conversations (user-specific)
def delete_all_conversations(api_key):  # Defines function to permanently delete user's conversation file
    if not api_key:  # Checks if API key is missing or empty
        return "Please enter your OpenAI API key first."  # Returns error message requiring API key
    
    try:  # Begins try block to handle file deletion errors
        user_file = get_user_file(api_key)  # Generates filename for this user's conversations
        if os.path.exists(user_file):  # Checks if file actually exists before attempting deletion
            os.remove(user_file)  # Deletes the file from disk permanently
            return "All your conversations deleted successfully!"  # Returns success confirmation message
        return "No conversations to delete."  # Returns message if file doesn't exist (nothing to delete)
    except Exception as e:  # Catches errors like permission denied, file in use, etc.
        return f"Error deleting conversations: {str(e)}"  # Returns error message with details for debugging

# Function to load previous conversations (user-specific)
def load_conversations(api_key):  # Defines function that retrieves and displays user's saved conversations
    if not api_key:  # Checks if API key is missing, empty, or None
        return "Please enter your OpenAI API key first to view your conversations."  # Returns error message prompting for API key
    
    user_file = get_user_file(api_key)  # Generates unique filename for this user based on their API key (e.g., "conversations_abc123.json")
    
    if os.path.exists(user_file):  # Checks if the user's conversation file actually exists on disk
        try:  # Begins try block to handle file reading and parsing errors
            with open(user_file, "r") as f:  # Opens user's file in read mode; auto-closes when done
                conversations = [json.loads(line) for line in f]  # List comprehension: reads each line, parses JSON, creates list of conversation dictionaries
            
            conv_text = ""  # Initializes empty string to build formatted conversation display
            for i, conv in enumerate(conversations):  # Loops through conversations with index (i) and conversation data (conv)
                conv_text += f"\n{'='*50}\nConversation {i + 1}\n{'='*50}\n"  # Adds separator line (50 equals signs), conversation number header, and another separator
                timestamp = conv.get("timestamp", "Unknown time")  # Retrieves timestamp from conversation dict; defaults to "Unknown time" if key doesn't exist
                conv_text += f"Timestamp: {timestamp}\n\n"  # Adds timestamp to output with two newlines for spacing
                
                messages = conv.get("messages", conv)  # Gets messages list from conversation; if "messages" key doesn't exist, uses entire conv dict as fallback
                for message in messages:  # Loops through each message dictionary in the messages list
                    role = message.get('role', 'unknown')  # Extracts role (user/assistant); defaults to 'unknown' if not found
                    content = message.get('content', '')  # Extracts message content; defaults to empty string if not found
                    conv_text += f"{role.upper()}: {content}\n\n"  # Adds formatted message with role in uppercase, content, and spacing
            
            return conv_text if conv_text else "No previous conversations found."  # Returns formatted text if any exists; otherwise returns "not found" message (ternary operator)
        except Exception as e:  # Catches any errors during file reading or JSON parsing
            return f"Error loading conversations: {str(e)}"  # Returns error message with exception details
    return "No previous conversations found for your account."  # Returns message if file doesn't exist (user has no saved conversations)

# Function to clear current conversation
def clear_conversation():  # Defines function to reset the current chat session
    global conversation_history  # Accesses global conversation_history variable to modify it
    conversation_history = []  # Resets conversation_history to empty list, clearing all messages
    return []  # Returns empty list to clear the Gradio chat interface display

# Create Gradio interface
with gr.Blocks(title="Chat with Documents 💬 📚", theme=gr.themes.Ocean()) as demo:  # Creates Gradio app using Blocks API (custom layout); sets browser tab title and applies Ocean color theme; assigns to 'demo' variable
    gr.Markdown("# Chat with Documents 💬 📚")  # Displays large heading text using Markdown syntax (# = h1)
    gr.Markdown("Upload PDF or DOCX files and chat with them using AI!")  # Displays instruction text as second line
    gr.Markdown("**Privacy Notice:** Your conversations are private and tied to your API key. Only you can see your saved conversations.")  # Displays privacy notice in bold (**text** = bold in Markdown)
    
    with gr.Row():  # Creates horizontal row container to arrange elements side-by-side
        with gr.Column(scale=2):  # Creates column inside row with scale=2 (takes 2/3 of width when combined with scale=1 column later)
            api_key_input = gr.Textbox(  # Creates text input box for API key
                label="OpenAI API Key",  # Sets label displayed above the textbox
                type="password",  # Masks input characters with dots/asterisks for security
                placeholder="Enter your OpenAI API key here..."  # Shows gray hint text when box is empty
            )
            
            file_upload = gr.File(  # Creates file upload widget
                label="Upload PDF or DOCX files",  # Sets label above file upload area
                file_count="multiple",  # Allows user to select multiple files at once
                file_types=[".pdf", ".docx"]  # Restricts file picker to only show PDF and DOCX files
            )
            
            load_btn = gr.Button("Load Documents", variant="primary")  # Creates button with text "Load Documents"; variant="primary" makes it blue/highlighted
            load_status = gr.Textbox(label="Status", interactive=False)  # Creates read-only textbox to display status messages; interactive=False prevents user editing
            
            load_btn.click(  # Defines what happens when load_btn is clicked
                fn=load_data,  # Calls load_data function when button clicked
                inputs=[file_upload, api_key_input],  # Passes file_upload and api_key_input values as arguments to load_data
                outputs=load_status  # Displays return value from load_data in load_status textbox
            )
    
    with gr.Row():  # Creates another horizontal row below the first one
        with gr.Column(scale=3):  # Creates column with scale=3 (takes 3/4 width; main chat area)
            chatbot = gr.Chatbot(  # Creates chatbot interface component for displaying conversation
                label="Chat",  # Sets label above chat window
                height=400  # Sets chat window height to 400 pixels
            )
            msg = gr.Textbox(  # Creates text input for user to type questions
                label="Your Question",  # Label displayed above input box
                placeholder="Ask a question about your documents..."  # Hint text shown when empty
            )
            
            with gr.Row():  # Creates row inside column for button group
                submit_btn = gr.Button("Send", variant="primary")  # Creates primary (highlighted) Send button
                clear_btn = gr.Button("Clear Chat")  # Creates Clear Chat button with default styling
            
            with gr.Row():  # Creates another row for save functionality
                save_btn = gr.Button("Save Conversation")  # Creates button to save chat history
                save_status = gr.Textbox(label="Save Status", interactive=False)  # Read-only textbox for save confirmation messages
        
        with gr.Column(scale=1):  # Creates sidebar column with scale=1 (takes 1/4 width; for conversation history)
            gr.Markdown("### Your Previous Conversations")  # Displays h3 heading (### = h3 in Markdown)
            load_convs_btn = gr.Button("Load Your Conversations")  # Creates button to retrieve saved conversations
            convs_display = gr.Textbox(  # Creates large textbox for displaying conversation history
                label="Conversation History",  # Label above the textbox
                lines=20,  # Sets textbox height to 20 lines of text
                interactive=False  # Makes textbox read-only (user cannot edit)
            )
            delete_all_btn = gr.Button("Delete All Your Conversations", variant="stop")  # Creates red warning-style button for deletion; variant="stop" makes it red
            delete_status = gr.Textbox(label="Delete Status", interactive=False)  # Read-only textbox for deletion confirmation messages
    
    # Event handlers
    submit_btn.click(  # Defines behavior when Send button is clicked
        fn=chat_with_docs,  # Calls chat_with_docs function
        inputs=[msg, chatbot, api_key_input],  # Passes message text, current chat history, and API key as arguments
        outputs=chatbot  # Updates chatbot display with return value (new conversation history)
    ).then(  # Chains another action after the first completes
        lambda: "",  # Anonymous function that returns empty string
        outputs=msg  # Clears the message input box after sending
    )
    
    msg.submit(  # Defines behavior when user presses Enter key in message textbox
        fn=chat_with_docs,  # Calls same chat function as submit button
        inputs=[msg, chatbot, api_key_input],  # Same inputs as submit button
        outputs=chatbot  # Updates chat display
    ).then(  # Chains follow-up action
        lambda: "",  # Returns empty string
        outputs=msg  # Clears message box after Enter is pressed
    )
    
    clear_btn.click(  # Defines behavior when Clear Chat button clicked
        fn=clear_conversation,  # Calls clear_conversation function (resets conversation_history to [])
        outputs=chatbot  # Updates chatbot display with empty list (clears visible chat)
    )
    
    save_btn.click(  # Defines behavior when Save Conversation button clicked
        fn=save_conversation,  # Calls save_conversation function to write to JSON file
        inputs=[api_key_input],  # Passes API key to identify which user's file to save to
        outputs=save_status  # Displays success/error message in save_status textbox
    )
    
    load_convs_btn.click(  # Defines behavior when Load Your Conversations button clicked
        fn=load_conversations,  # Calls load_conversations function to read from user's JSON file
        inputs=[api_key_input],  # Passes API key to identify which user's file to load
        outputs=convs_display  # Displays formatted conversation history in convs_display textbox
    )
    
    delete_all_btn.click(  # Defines behavior when Delete All button clicked
        fn=delete_all_conversations,  # Calls delete_all_conversations function to remove user's file
        inputs=[api_key_input],  # Passes API key to identify which file to delete
        outputs=delete_status  # Displays confirmation/error message in delete_status textbox
    )

if __name__ == "__main__":  # Python idiom: only runs code below if script is executed directly (not imported as module)
    ### demo.launch()  # Starts Gradio web server and opens app in browser; makes app accessible at local URL (e.g., http://127.0.0.1:7860)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860))) ### render purpose
