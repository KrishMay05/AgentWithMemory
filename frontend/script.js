
/* script.js adjustments */
const form = document.getElementById('chat-form');
const input = document.getElementById('chat-input');
const chatBody = document.getElementById('chat-body');
const sendButton = document.querySelector('.send-button');
const themeToggle = document.getElementById('theme-toggle');
const toolsButton = document.getElementById('tools-button');
const toolsMenu = document.getElementById('tools-menu');
const searchToggle = document.getElementById('search-toggle');
const searchButton = document.getElementById('search-button');

// Theme toggle
themeToggle.addEventListener('click', () => {
  document.body.classList.toggle('dark-theme');
  themeToggle.textContent = document.body.classList.contains('dark-theme') ? '🌑' : '☀️';
});

// Tools menu toggle
toolsButton.addEventListener('click', () => {
  toolsMenu.classList.toggle('hidden');
});
// Close tools menu on outside click
window.addEventListener('click', (e) => {
  if (!toolsButton.contains(e.target) && !toolsMenu.contains(e.target)) {
    toolsMenu.classList.add('hidden');
  }
});

// Search toggle
searchToggle.addEventListener('change', () => {
  searchButton.classList.toggle('hidden', !searchToggle.checked);
});

function appendMessage(sender, text) {
  const msg = document.createElement('div');
  msg.className = `message ${sender}`;
  msg.textContent = text;
  chatBody.appendChild(msg);
  chatBody.scrollTop = chatBody.scrollHeight;
}

form.addEventListener('submit', async e => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  appendMessage('user', text);
  input.value = '';
  sendButton.disabled = true;

  try {
    console.log(searchToggle.checked);
    const res = await fetch('http://127.0.0.1:5050/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: text, search: searchToggle.checked })
    });
    const { response } = await res.json();
    appendMessage('bot', response);
  } catch (err) {
    appendMessage('bot', 'Error connecting to server.');
    console.error(err);
  } finally {
    sendButton.disabled = false;
  }
});
