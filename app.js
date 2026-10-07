const chat = document.querySelector('#chat');
const form = document.querySelector('#composer');
const prompt = document.querySelector('#prompt');
const messages = [];
const conversationId = crypto.randomUUID();
const STAGES = ['INITIAL', 'EVIDENCE_ENRICHING', 'DRUGS_ENRICHING', 'TRIALS_ENRICHING', 'VERIFICATION', 'FINAL'];
const LABELS = { INITIAL: 'Fast answer', EVIDENCE_ENRICHING: 'Evidence', DRUGS_ENRICHING: 'Drugs', TRIALS_ENRICHING: 'Trials', VERIFICATION: 'Verification', FINAL: 'Final' };

function render(element, text) {
  if (window.marked && window.DOMPurify) {
    element.innerHTML = DOMPurify.sanitize(marked.parse(text, { gfm: true, breaks: true }));
  } else {
    element.textContent = text;
  }
}

function addMessage(role, text) {
  const message = document.createElement('article');
  message.className = `message ${role}`;
  if (role === 'assistant') render(message, text); else message.textContent = text;
  chat.append(message);
  message.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return message;
}

// An assistant message with a stage bar, a timing line and the streamed answer.
function addStreamingMessage() {
  const message = document.createElement('article');
  message.className = 'message assistant streaming';
  message.innerHTML = '<div class="stagebar"></div><div class="timing"></div><div class="answer"></div>';
  const bar = message.querySelector('.stagebar');
  STAGES.forEach((stage) => { const chip = document.createElement('span'); chip.dataset.stage = stage; chip.textContent = LABELS[stage]; bar.append(chip); });
  chat.append(message);
  message.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return message;
}

function setStage(message, stage) {
  const index = STAGES.indexOf(stage);
  message.querySelectorAll('.stagebar span').forEach((chip, i) => {
    chip.className = i < index ? 'done' : i === index ? 'active' : '';
  });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = prompt.value.trim();
  if (!text) return;
  messages.push({ role: 'user', content: text });
  addMessage('user', text);
  prompt.value = '';
  const message = addStreamingMessage();
  const answerBox = message.querySelector('.answer');
  const timing = message.querySelector('.timing');
  timing.textContent = 'Searching PubMed, ClinicalTrials.gov and the web…';
  form.querySelector('button').disabled = true;

  let markdown = '';
  let pending = false;
  const schedule = () => {  // re-render at most once per animation frame while tokens stream in
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; render(answerBox, markdown); });
  };
  const t = {};

  const handle = (ev) => {
    if (ev.type === 'stage') setStage(message, ev.stage);
    else if (ev.type === 'token') { markdown += ev.text; schedule(); }
    else if (ev.type === 'warning') {  // a live notice: failed sources, or identifiers removed from the question
      const box = document.createElement('div');
      box.className = `warning warning-${ev.category}`; box.setAttribute('role', 'alert'); box.textContent = ev.message;
      message.insertBefore(box, answerBox);
    }
    else if (ev.type === 'milestone') {
      if (ev.name === 'first_useful_answer') { t.first = ev.first_token; t.useful = ev.seconds; }
      timing.textContent = `PRELIMINARY answer — First token ${t.first ?? '…'} s · first useful answer ${t.useful ?? '…'} s · the verified report will replace it…`;
    } else if (ev.type === 'verification') {
      timing.textContent = `Verification: ${ev.status} (${ev.checked.pmids} PMIDs, ${ev.checked.trials} trials checked)`;
    } else if (ev.type === 'final') {
      markdown = ev.markdown; render(answerBox, markdown); setStage(message, 'FINAL');
      message.classList.remove('streaming');
      const tm = ev.timings;
      timing.textContent = `✓ Final verified answer · first token ${tm.first_token_s} s · first useful answer ${tm.first_useful_s} s · retrieval ${tm.retrieval_deep_s} s · verified ${tm.verification_done_s} s · final ${tm.final_s} s`;
      messages.push({ role: 'assistant', content: markdown });
    } else if (ev.type === 'pdf' && ev.name) {
      const link = document.createElement('a');
      link.href = `demo_output/${ev.name}`; link.textContent = 'Download PDF'; link.target = '_blank'; link.className = 'pdf-link';
      message.append(link);
    } else if (ev.type === 'error') {
      answerBox.textContent = markdown ? markdown + `\n\n[Stopped: ${ev.message}]` : `Error: ${ev.message}`;
    }
  };

  try {
    const response = await fetch('/api/chat-stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ messages, conversation_id: conversationId }) });
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line) handle(JSON.parse(line));
      }
    }
  } catch (error) {
    answerBox.textContent = markdown || 'Connection error. Is the Python server running?';
  } finally {
    form.querySelector('button').disabled = false;
    prompt.focus();
    message.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }
});
