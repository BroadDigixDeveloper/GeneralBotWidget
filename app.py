# updated on 9th April
# from flask import Flask, Response, render_template, request, session
# import requests
# import json
# import logging
# import os
# import re
# from logging.handlers import RotatingFileHandler
# from typing import List, Dict
# from datetime import datetime

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = os.urandom(24)

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# if not os.path.exists('logs'):
#     os.makedirs('logs')

# file_handler = RotatingFileHandler(
#     'logs/medical_bot.log',
#     maxBytes=10485760,
#     backupCount=10
# )
# file_handler.setFormatter(logging.Formatter(
#     '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
# ))
# logger.addHandler(file_handler)

# class MedicalBot:
#     def __init__(self):
#         self.endpoint ="https://capps-backend-br4xnxo2jang4.wonderfulfield-f0927b5c.westus.azurecontainerapps.io/chat/stream"
#         # self.endpoint = "https://capps-backend-52y7qfxjzrxiq.calmsky-7b1632e3.westus.azurecontainerapps.io/chat/stream"
#         self.messages: List[Dict] = []
#         self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
#     def remove_citations(self, text: str) -> str:
#         """Comprehensive citation removal function."""
#         if not text:
#             return text
        
#         # First pass: Remove all content within square brackets
#         text = re.sub(r'\[[^\]]*\]', '', text)
        
#         # Second pass: Remove specific patterns that might remain
#         patterns = [
#             # PDF files with page numbers
#             r'\[\s*[^\]]*\.pdf\s*#\s*page\s*=\s*\d+\s*[^\]]*\]',
#             # File references
#             r'\[\s*[^\]]*\.(docx?|pdf|txt|xlsx?|csv)[^\]]*\]',
#             # Any remaining citations
#             r'\[\s*[^\]]*\bpage\s*\d+\s*[^\]]*\]',
#             r'\[\s*[^\]]*\bsource\s*:?[^\]]*\]',
#             r'\[\s*[^\]]*\b(file|document|reference)\s*[^\]]*\]',
#             # Empty brackets
#             r'\[\s*\]'
#         ]
        
#         for pattern in patterns:
#             text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
#         # Clean up extra whitespace
#         text = re.sub(r'\s+', ' ', text)
#         return text.strip()
        
#     def format_messages(self) -> List[Dict]:
#         return [{"content": msg["content"], "role": msg["role"]} for msg in self.messages]
    
#     def add_message(self, content: str, role: str = "user"):
#         if role == "assistant":
#             content = self.remove_citations(content)
#         self.messages.append({"content": content, "role": role})
#         logger.info(f"Added message - Role: {role}, Content: {content[:50]}...")
    
#     def clear_history(self):
#         self.messages = []
#         self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
#         logger.info("Chat history cleared")
    
#     def process_stream_chunk(self, chunk: dict) -> dict:
#         """Clean citations from streaming chunks."""
#         if 'choices' in chunk and chunk['choices']:
#             choice = chunk['choices'][0]
#             if 'delta' in choice and 'content' in choice['delta']:
#                 cleaned_content = self.remove_citations(choice['delta']['content'])
#                 choice['delta']['content'] = cleaned_content
#         return chunk
        

#     def get_stream(self, query: str, email: str = None):
#         try:
#             self.add_message(query)
#             headers = {"accept": "*/*", "Content-Type": "application/json"}
#             context = {
#                 "overrides": {
#                     "temperature": 0.3,
#                     "top": 3,
#                     "retrieval_mode": "hybrid",
#                     "semantic_ranker": True,
#                     "suggest_followup_questions": True
#                 }
#             }
#             if email:
#                 context["auth_claims"] = {"email": email}

#             payload = {
#                 "messages": self.format_messages(),
#                 "context": context,
#                 "session_state": None
#             }

#             logger.info(f"Sending request with {len(self.messages)} messages (Email: {email})")
#             response = requests.post(self.endpoint, json=payload, headers=headers, stream=True)
#             response.raise_for_status()

#             current_response = ""
#             for line in response.iter_lines():
#                 if line:
#                     try:
#                         chunk = json.loads(line)
#                         chunk = self.process_stream_chunk(chunk)
#                         if 'choices' in chunk and chunk['choices']:
#                             content = chunk['choices'][0].get('delta', {}).get('content', '')
#                             current_response += content
#                         yield f"data: {json.dumps(chunk)}\n\n"
#                         if chunk.get('end_turn') and current_response:
#                             cleaned = self.remove_citations(current_response)
#                             self.add_message(cleaned, "assistant")
#                     except json.JSONDecodeError:
#                         continue
#         except Exception as e:
#             logger.error(f"Stream error: {str(e)}")
#             yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"



#     # Dictionary to store bot instances
# session_bots = {}


# @app.route('/')
# def home():
#     if 'session_id' not in session:
#         session['session_id'] = datetime.now().strftime("%Y%m%d%H%M%S")
    
#     session_id = session['session_id']
#     session_bots[session_id] = MedicalBot()

#     email = request.args.get('email')
#     if email:
#         session['email'] = email
#         logger.info(f"Stored email in session: {email}")
#     else:
#         logger.info("No email parameter received")
    
#     return render_template('index.html')


# @app.route('/chat', methods=['POST'])
# def chat():
#     session_id = session.get('session_id')
#     if not session_id or session_id not in session_bots:
#         session['session_id'] = datetime.now().strftime("%Y%m%d%H%M%S")
#         session_id = session['session_id']
#         session_bots[session_id] = MedicalBot()
    
#     medical_bot = session_bots[session_id]
#     data = request.json
#     query = data.get('message')
    
#     if not query:
#         return {"error": "No message provided"}, 400

#     # ✅ Fetch email from session
#     email = session.get('email')

#     return Response(
#         medical_bot.get_stream(query, email=email),
#         mimetype='text/event-stream'
#     )


# @app.route('/history', methods=['GET'])
# def get_history():
#     session_id = session.get('session_id')
#     if not session_id or session_id not in session_bots:
#         session['session_id'] = datetime.now().strftime("%Y%m%d%H%M%S")
#         session_id = session['session_id']
#         session_bots[session_id] = MedicalBot()
    
#     medical_bot = session_bots[session_id]
#     return {"messages": medical_bot.format_messages()}

# @app.route('/clear', methods=['POST'])
# def clear_history():
#     session_id = session.get('session_id')
#     if session_id and session_id in session_bots:
#         session_bots[session_id].clear_history()
#     return {"status": "success"}

# if __name__ == "__main__":
#     app.run(debug=True)





from flask import Flask, Response, render_template, request, session, jsonify
import requests
import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from typing import List, Dict
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if not os.path.exists('logs'):
    os.makedirs('logs')
file_handler = RotatingFileHandler('logs/medical_bot.log', maxBytes=10485760, backupCount=10)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
logger.addHandler(file_handler)

class MedicalBot:
    def __init__(self):
        self.endpoint = "https://capps-backend-br4xnxo2jang4.wonderfulfield-f0927b5c.westus.azurecontainerapps.io/chat/stream"
        self.messages: List[Dict] = []
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")

    def remove_citations(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r'\[[^\]]*\]', '', text)
        patterns = [
            r'\[\s*[^\]]*\.pdf\s*#\s*page\s*=\s*\d+\s*[^\]]*\]',
            r'\[\s*[^\]]*\.(docx?|pdf|txt|xlsx?|csv)[^\]]*\]',
            r'\[\s*[^\]]*\bpage\s*\d+\s*[^\]]*\]',
            r'\[\s*[^\]]*\bsource\s*:?[^\]]*\]',
            r'\[\s*[^\]]*\b(file|document|reference)\s*[^\]]*\]',
            r'\[\s*\]'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', text).strip()

    def format_messages(self) -> List[Dict]:
        return [{"content": msg["content"], "role": msg["role"]} for msg in self.messages]

    def add_message(self, content: str, role: str = "user"):
        if role == "assistant":
            content = self.remove_citations(content)
        self.messages.append({"content": content, "role": role})
        logger.info(f"Added message - Role: {role}, Content: {content[:50]}...")

    def clear_history(self):
        self.messages = []
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        logger.info("Chat history cleared")

    def process_stream_chunk(self, chunk: dict) -> dict:
        if 'choices' in chunk and chunk['choices']:
            choice = chunk['choices'][0]
            if 'delta' in choice and 'content' in choice['delta']:
                choice['delta']['content'] = self.remove_citations(choice['delta']['content'])
        return chunk

    def get_stream(self, query: str, email: str = None, name: str = "Unknown"):
        try:
            self.add_message(query)
            headers = {"accept": "*/*", "Content-Type": "application/json"}
            context = {
                "overrides": {
                    "temperature": 0.3,
                    "top": 3,
                    "retrieval_mode": "hybrid",
                    "semantic_ranker": True,
                    "suggest_followup_questions": True
                }
            }
            if email:
                context["auth_claims"] = {"email": email}

            payload = {
                "messages": self.format_messages(),
                "context": context,
                "session_state": None
            }

            logger.info(f"Sending request with {len(self.messages)} messages (Email: {email})")
            response = requests.post(self.endpoint, json=payload, headers=headers, stream=True)
            response.raise_for_status()

            current_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        chunk = self.process_stream_chunk(chunk)
                        if 'choices' in chunk and chunk['choices']:
                            content = chunk['choices'][0].get('delta', {}).get('content', '')
                            current_response += content
                        yield f"data: {json.dumps(chunk)}\n\n"
                        if chunk.get('end_turn') and current_response:
                            cleaned = self.remove_citations(current_response)
                            self.add_message(cleaned, "assistant")
                            self.store_chat_log(email, name)
                            self.create_zendesk_ticket(email, name, query, cleaned)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    def store_chat_log(self, email, name):
        try:
            payload = {
                "user": {"email": email, "name": name},
                "messages": self.format_messages()
            }
            logger.info("Logging chat to MongoDB: %s", payload)
            response = requests.post("https://capps-backend-br4xnxo2jang4.wonderfulfield-f0927b5c.westus.azurecontainerapps.io/store_chat", json=payload)
            if response.status_code != 200:
                logger.warning(f"MongoDB logging failed: {response.text}")
        except Exception as e:
            logger.error(f"MongoDB logging error: {str(e)}")

    def create_zendesk_ticket(self, email, name, question, answer):
        try:
            payload = {
                "title": f"Widget Chat Query by {name}",
                "description": f"Question: {question}\n\nAnswer: {answer}",
                "username": email,
                "name": name
            }
            logger.info("Sending Zendesk ticket: %s", payload)
            response = requests.post("https://capps-backend-br4xnxo2jang4.wonderfulfield-f0927b5c.westus.azurecontainerapps.io/api/create", json=payload)
            if response.status_code != 200:
                logger.warning(f"Zendesk ticket creation failed: {response.text}")
        except Exception as e:
            logger.error(f"Zendesk ticket error: {str(e)}")


# Dictionary to store bot instances
session_bots = {}

@app.route('/')
def home():
    session['session_id'] = datetime.now().strftime("%Y%m%d%H%M%S")
    session_id = session['session_id']
    session_bots[session_id] = MedicalBot()

    session['email'] = request.args.get('email')
    session['name'] = request.args.get('name')
    logger.info(f"Session email: {session.get('email')}, name: {session.get('name')}")
    
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    session_id = session.get('session_id')
    if not session_id or session_id not in session_bots:
        session['session_id'] = datetime.now().strftime("%Y%m%d%H%M%S")
        session_id = session['session_id']
        session_bots[session_id] = MedicalBot()

    medical_bot = session_bots[session_id]
    data = request.json
    query = data.get('message')
    if not query:
        return {"error": "No message provided"}, 400

    email = session.get('email')
    name = session.get('name')

    return Response(
        medical_bot.get_stream(query, email=email, name=name),
        mimetype='text/event-stream'
    )

@app.route('/history', methods=['GET'])
def get_history():
    session_id = session.get('session_id')
    if not session_id or session_id not in session_bots:
        session['session_id'] = datetime.now().strftime("%Y%m%d%H%M%S")
        session_id = session['session_id']
        session_bots[session_id] = MedicalBot()

    return {"messages": session_bots[session_id].format_messages()}


@app.route('/clear', methods=['POST'])
def clear_history():
    session_id = session.get('session_id')
    if session_id and session_id in session_bots:
        session_bots[session_id].clear_history()
    return {"status": "success"}


if __name__ == "__main__":
    app.run(debug=True)
