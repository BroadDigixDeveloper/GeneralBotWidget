document.addEventListener('DOMContentLoaded', () => {
    const chatToggle = document.querySelector('.chat-toggle');
    const chatBox = document.querySelector('.chat-box');
    const input = document.querySelector('.chat-input input');
    const sendButton = document.querySelector('.send-button');
    const messagesContainer = document.querySelector('.chat-messages');
    const optionButtons = document.querySelectorAll('.option-button');

    chatToggle.addEventListener('click', () => {
        chatBox.classList.toggle('active');
    });

    function addMessage(text, type) {
        const messageContainer = document.createElement('div');
        messageContainer.className = type === 'bot' ? 'bot-message-container' : '';
    
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
    
        if (type === 'bot') {
            const aiIndicator = document.createElement('div');
            aiIndicator.className = 'ai-indicator';
            aiIndicator.innerHTML = '<i class="fas fa-robot"></i>';
            messageContainer.appendChild(aiIndicator);
    
            // Process the text with proper formatting
            let processedText = text;
    
            // Format numbered items with titles and descriptions
            processedText = processedText.replace(
                /(\d+\.)\s*\*\*(.*?)\*\*\s*:\s*([\s\S]*?)(?=(?:\d+\.)|$)/g,
                (match, number, title, description) => `
                    <div class="numbered-item">
                        <div class="item-header">
                            <span class="item-number">${number}</span>
                            <span class="item-title">${title}</span>
                        </div>
                        <div class="item-description">${description.trim()}</div>
                    </div>
                `
            );
    
            messageDiv.innerHTML = processedText;
        } else {
            messageDiv.textContent = text;
        }
    
        messageContainer.appendChild(messageDiv);
        messagesContainer.appendChild(messageContainer);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function addFollowUpQuestions(questions) {
        const optionsDiv = document.createElement('div');
        optionsDiv.className = 'options';
        
        questions.forEach(question => {
            const button = document.createElement('button');
            button.className = 'option-button';
            button.textContent = question;
            button.onclick = () => {
                addMessage(question, 'user');
                sendMessage(question);
            };
            optionsDiv.appendChild(button);
        });
        
        messagesContainer.appendChild(optionsDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function sendMessage(message) {
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        })
        .then(response => response.json())
        .then(data => {
            addMessage(data.response, 'bot');
            if (data.follow_up_questions && data.follow_up_questions.length > 0) {
                addFollowUpQuestions(data.follow_up_questions);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage('Sorry, there was an error processing your request.', 'bot');
        });
    }

    optionButtons.forEach(button => {
        button.addEventListener('click', () => {
            const message = button.textContent;
            addMessage(message, 'user');
            sendMessage(message);
        });
    });

    sendButton.addEventListener('click', () => {
        const message = input.value.trim();
        if (message) {
            addMessage(message, 'user');
            sendMessage(message);
            input.value = '';
        }
    });

    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const message = input.value.trim();
            if (message) {
                addMessage(message, 'user');
                sendMessage(message);
                input.value = '';
            }
        }
    });
});