const chat = document.querySelector('#chat');
const form = document.querySelector('#composer');
const prompt = document.querySelector('#prompt');
const messages = [];
const conversationId = crypto.randomUUID();

function addMessage(role, text) {
  const message = document.createElement('article');
  message.className = `message ${role}`;
  setMessageContent(message, role, text);
  chat.append(message);
  message.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return message;
}

function setMessageContent(element, role, text) {
  if (role === 'assistant' && window.marked && window.DOMPurify) {
    element.innerHTML = DOMPurify.sanitize(marked.parse(text, { gfm: true, breaks: true }));
  } else {
    element.textContent = text;
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = prompt.value.trim();
  if (!text) return;
  messages.push({ role: 'user', content: text });
  addMessage('user', text);
  prompt.value = '';
  const pending = addMessage('assistant', 'Searching PubMed, ClinicalTrials.gov and the web… (about 20 seconds)');
  form.querySelector('button').disabled = true;
  try {
    const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ messages, conversation_id: conversationId }) });
    const data = await response.json();
    const answer = data.answer || 'Sorry, I could not answer that.';
    setMessageContent(pending, 'assistant', answer);
    messages.push({ role: 'assistant', content: answer });
  } catch (error) {
    pending.textContent = 'Connection error. Is the Python server running?';
  } finally { form.querySelector('button').disabled = false; prompt.focus(); }
});
