// Dependency-free state/request checks. This DOM stub does not verify browser layout.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page() {
  let document;
  class Node {
    constructor(tag = 'div', text = '') {
      this.tagName = tag; this.children = []; this._text = text; this.events = {};
      this.value = ''; this.hidden = false; this.disabled = false; this.open = false;
      this.dataset = {}; this.attributes = {}; this.style = { setProperty() {} };
      this.classes = new Set(); this.classList = { toggle: (name, on) => on ? this.classes.add(name) : this.classes.delete(name) };
      this.isConnected = true;
    }
    get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
    set textContent(text) { this._text = String(text); this.children = []; }
    append(...children) { for (const child of children) { child.parent = this; this.children.push(child); } }
    replaceChildren(...children) { this._text = ''; this.children = []; this.append(...children); }
    setAttribute(key, value) { this.attributes[key] = value; }
    addEventListener(event, handler) { (this.events[event] ||= []).push(handler); }
    async emit(event, extra = {}) { for (const handler of this.events[event] || []) await handler({ target: this, preventDefault() {}, ...extra }); }
    click() { return this.disabled ? Promise.resolve() : this.emit('click'); }
    focus() { if (!this.disabled) document.activeElement = this; }
    showModal() { this.open = true; }
    close() { this.open = false; return this.emit('close'); }
    closest(selector) { return selector === '[data-question]' && this.dataset.question ? this : this.parent?.closest(selector) || null; }
    requestSubmit() { return this.emit('submit'); }
  }
  const nodes = {};
  for (const match of html.matchAll(/<([\w-]+)\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
    assert(!nodes[match[2]], `Duplicate id ${match[2]}`);
    nodes[match[2]] = new Node(match[1]);
    nodes[match[2]].hidden = /\bhidden\b/.test(match[0]);
  }
  const walk = node => [node, ...node.children.flatMap(walk)];
  document = {
    activeElement: null,
    getElementById: id => { assert(nodes[id], `Missing ${id}`); return nodes[id]; },
    createElement: tag => new Node(tag), createTextNode: text => new Node('#text', text),
    querySelectorAll: selector => Object.values(nodes).flatMap(walk).filter(n =>
      selector === '[data-question]' ? n.dataset.question : selector === '[data-conversation-id]' ? n.dataset.conversationId : false),
    addEventListener() {},
  };
  const queue = [], calls = [];
  const context = vm.createContext({ document, URL, window: { location: { origin: 'http://localhost:8001' } }, fetch: async (url, options) => {
    calls.push({ url, options });
    const next = queue.shift();
    assert(next, `Unexpected ${url}`);
    if (next instanceof Error) throw next;
    if (typeof next === 'function') return next();
    return { status: next.status, ok: next.status >= 200 && next.status < 300, json: async () => next.body };
  } });
  vm.runInContext(script, context);
  return { nodes, queue, calls, read: expression => vm.runInContext(expression, context), walk };
}

const listing = (conversations = [], offset = 0) => ({ status: 200, body: { conversations, offset, limit: 20 } });
const item = (id, created_at = 1700000000) => ({ conversation_id: id, created_at });
const chatCalls = p => p.calls.filter(call => call.url === '/chat');
const historyButton = (p, id) => p.nodes['history-list'].children.find(node => node.dataset.conversationId === id);
async function login(p, status = 200, listResponse = listing(), username = 'alice') {
  p.nodes.username.value = ` ${username} `;
  p.nodes.password.value = 'offline-only-password';
  p.queue.push({ status, body: { access_token: username === 'alice' ? 'offline-only-token' : 'bob-only-token', username } });
  if (status === 200) p.queue.push(listResponse);
  await p.nodes['login-form'].emit('submit');
}
async function send(p, response, question = '查询 A1001') {
  p.nodes.question.value = question;
  p.queue.push(response);
  await p.nodes['chat-form'].emit('submit');
}

(async () => {
  let p = page();
  p.nodes.question.value = '尚未登录的问题';
  await p.nodes['chat-form'].emit('submit');
  assert.equal(p.calls.length, 0);
  assert.equal(p.nodes['login-dialog'].open, true);
  await login(p, 401);
  assert.match(p.nodes['login-feedback'].textContent, /用户名或密码错误/);
  assert.equal(p.read('accessToken'), '');
  assert.equal(p.nodes.password.value, '');
  await login(p);
  assert.deepEqual(JSON.parse(p.calls[1].options.body), { username: 'alice', password: 'offline-only-password' });
  assert.equal(p.nodes['login-dialog'].open, false);
  assert.equal(p.nodes['chat-panel'].hidden, false);
  assert.equal(p.nodes.question.value, '尚未登录的问题');
  await send(p, { status: 200, body: { conversation_id: 'thread-one', answer: 'A1001 的电脑已发货。' } });
  assert.deepEqual(JSON.parse(chatCalls(p)[0].options.body), { question: '查询 A1001', conversation_id: null });
  assert.equal(chatCalls(p)[0].options.headers.Authorization, 'Bearer offline-only-token');
  await send(p, { status: 200, body: { conversation_id: 'thread-one', answer: '刚才查询的是 A1001。' } }, '刚才查了什么？');
  assert.equal(JSON.parse(chatCalls(p)[1].options.body).conversation_id, 'thread-one');
  await p.nodes['new-chat'].click();
  assert.equal(p.read('accessToken'), 'offline-only-token');
  assert.equal(p.read('conversationId'), null);
  assert(!p.nodes['chat-messages'].textContent.includes('A1001'));
  await send(p, { status: 200, body: { conversation_id: 'thread-two', answer: '新会话。' } });
  assert.equal(JSON.parse(chatCalls(p)[2].options.body).conversation_id, null);
  p.queue.push({ status: 204 });
  await p.nodes['account-button'].click();
  assert.equal(p.calls.at(-1).url, '/logout');
  assert.equal(p.read('accessToken'), '');
  assert(!p.nodes['chat-messages'].textContent.includes('新会话。'));
  console.log('PASS: login failure/recovery, request headers, history id, new conversation and logout');

  for (const status of [401, 404, 422, 429, 503]) {
    p = page(); await login(p);
    await send(p, { status });
    assert.equal(p.nodes['send-button'].disabled, false);
    assert.equal(p.nodes['new-chat'].disabled, false);
    if (status === 401) { assert.equal(p.read('accessToken'), ''); assert.match(p.nodes['chat-messages'].textContent, /登录已失效/); }
    if (status === 404) { assert.equal(p.read('conversationId'), null); assert.match(p.nodes['chat-messages'].textContent, /不可访问/); }
    if (status === 422 || status === 429) assert.equal(p.nodes.question.value, '查询 A1001');
    if (status === 503) assert.match(p.nodes['chat-messages'].textContent, /未自动重试/);
    assert.equal(p.calls.length, 3);
  }
  p = page(); await login(p); await send(p, new Error('offline'));
  assert.match(p.nodes['chat-messages'].textContent, /先核对是否已保存/);
  assert.equal(p.calls.length, 3);
  console.log('PASS: 401/404/422/429/503/network handling; no automatic retry');

  p = page(); await login(p);
  let finish;
  const response = new Promise(resolve => { finish = resolve; });
  p.queue.push(() => response);
  p.nodes.question.value = '稍慢的请求';
  const pending = p.nodes['chat-form'].emit('submit');
  assert.equal(p.nodes['new-chat'].disabled, true);
  assert.equal(p.nodes['account-button'].disabled, true);
  assert.equal(p.nodes['chat-account'].disabled, true);
  await p.nodes['chat-form'].emit('submit');
  assert.equal(p.calls.length, 3);
  finish({ status: 200, ok: true, json: async () => ({ conversation_id: 'slow-thread', answer: '完成。' }) });
  await pending;
  assert.equal(p.nodes['new-chat'].disabled, false);
  console.log('PASS: in-flight requests disable identity/conversation changes and duplicate submission');

  p = page(); await login(p);
  await send(p, { status: 200, body: { conversation_id: 'safe-thread', answer: '<img src=x onerror=alert(1)> **已发货** [办理入口](/demo/return-application?order_id=A1001) [危险](javascript:evil) https://example.test/help。' } });
  const rendered = p.walk(p.nodes['chat-messages']);
  assert.equal(rendered.filter(n => n.tagName === 'img').length, 0);
  const links = rendered.filter(n => n.tagName === 'a');
  assert.equal(links.length, 2);
  assert.equal(links[0].href, 'http://localhost:8001/demo/return-application?order_id=A1001');
  assert.equal(links[1].href, 'https://example.test/help');
  assert(links.every(n => n.rel === 'noopener noreferrer'));
  assert(!p.nodes['chat-messages'].textContent.includes('offline-only-token'));
  assert(!/\b(?:localStorage|sessionStorage)\s*[.\[]|innerHTML\s*=|document\.cookie\s*=/.test(script));
  console.log('PASS: answer HTML stays text, unsafe links rejected, business links preserved, token not persisted');

  // A new page has no credential or selected id; login fetches owned metadata,
  // and only an explicit successful selection restores the previous thread.
  p = page();
  assert.equal(p.read('accessToken'), '');
  assert.equal(p.read('conversationId'), null);
  await login(p, 200, listing([item('saved-thread')]));
  assert.equal(p.calls[1].url, '/conversations?offset=0&limit=20');
  assert.equal(p.calls[1].options.headers.Authorization, 'Bearer offline-only-token');
  assert.equal(p.calls[1].options.cache, 'no-store');
  assert.equal(p.read('conversationId'), null);
  p.queue.push(listing([item('saved-thread')]));
  await p.nodes['history-toggle'].click();
  assert.equal(p.nodes['history-panel'].hidden, false);
  assert.equal(p.nodes['composer-wrap'].hidden, true);
  p.queue.push({ status: 200, body: { conversation_id: 'saved-thread', messages: [
    { role: 'user', content: '售后备注：青蛙跳过三片荷叶。<img src=x>' },
    { role: 'assistant', content: '**已记下** [帮助](https://example.test/help)' },
  ] } });
  await historyButton(p, 'saved-thread').click();
  assert.equal(p.calls.at(-1).url, '/conversations/saved-thread/messages');
  assert.equal(p.calls.at(-1).options.headers.Authorization, 'Bearer offline-only-token');
  assert.equal(p.read('conversationId'), 'saved-thread');
  assert.equal(p.nodes['history-panel'].hidden, true);
  assert.equal(p.nodes['composer-wrap'].hidden, false);
  assert.match(p.nodes['chat-messages'].textContent, /青蛙跳过三片荷叶/);
  assert(!p.nodes['chat-messages'].textContent.includes('嗨，我是小蛙'));
  assert.equal(p.walk(p.nodes['chat-messages']).filter(n => n.tagName === 'img').length, 0);
  await send(p, { status: 200, body: { conversation_id: 'saved-thread', answer: '继续原会话。' } }, '继续刚才的售后问题');
  assert.equal(JSON.parse(chatCalls(p).at(-1).options.body).conversation_id, 'saved-thread');
  p.queue.push({ status: 204 }); await p.nodes['account-button'].click();
  assert.equal(p.read('historyItems.length'), 0);
  assert(!p.nodes['chat-messages'].textContent.includes('青蛙跳过三片荷叶'));
  await login(p, 200, listing([item('bob-thread')]), 'bob');
  assert.equal(p.read('conversationId'), null);
  assert(!historyButton(p, 'saved-thread'));
  assert(historyButton(p, 'bob-thread'));
  assert.equal(p.calls.at(-1).options.headers.Authorization, 'Bearer bob-only-token');
  console.log('PASS: login loads owned sessions, explicit restore renders safely, continuation reuses id, logout/account switch clears old state');

  p = page();
  const firstPage = Array.from({ length: 20 }, (_, index) => item(`thread-${index}`));
  await login(p, 200, listing(firstPage));
  p.queue.push(listing(firstPage)); await p.nodes['history-toggle'].click();
  assert.equal(p.nodes['history-prev'].disabled, true);
  assert.equal(p.nodes['history-next'].disabled, false);
  p.queue.push(listing([], 20)); await p.nodes['history-next'].click();
  assert.equal(p.calls.at(-1).url, '/conversations?offset=20&limit=20');
  assert.equal(p.nodes['history-next'].disabled, true);
  assert.equal(p.nodes['history-prev'].disabled, false);
  assert.match(p.nodes['history-list'].textContent, /没有更多会话/);
  p.queue.push(listing(firstPage)); await p.nodes['history-prev'].click();
  p.queue.push(new Error('offline')); await p.nodes['history-refresh'].click();
  assert.equal(p.read('historyOffset'), 0);
  assert.equal(p.read('historyItems.length'), 20);
  assert.match(p.nodes['history-feedback'].textContent, /刷新列表/);
  assert.equal(p.nodes['history-refresh'].disabled, false);
  await p.nodes['history-toggle'].click();
  assert.equal(p.nodes['history-panel'].hidden, true);
  console.log('PASS: history pagination, empty last page, refresh failure retains list and unlocks controls');

  for (const response of [
    { status: 404 }, { status: 429 }, { status: 503 }, new Error('offline'),
    { status: 200, body: { conversation_id: 'wrong-id', messages: [] } },
    { status: 200, body: { conversation_id: 'candidate', messages: [{ role: 'tool', content: 'private' }] } },
  ]) {
    p = page(); await login(p);
    await send(p, { status: 200, body: { conversation_id: 'current-thread', answer: '保留的原对话。' } });
    p.queue.push(listing([item('candidate')])); await p.nodes['history-toggle'].click();
    const previous = p.nodes['chat-messages'].textContent, count = p.calls.length;
    p.queue.push(response); await historyButton(p, 'candidate').click();
    assert.equal(p.read('conversationId'), 'current-thread');
    assert.equal(p.nodes['chat-messages'].textContent, previous);
    assert.equal(p.nodes['history-panel'].hidden, false);
    assert.equal(p.calls.length, count + 1);
    assert.equal(p.nodes['new-chat'].disabled, false);
    assert.equal(p.nodes['history-toggle'].disabled, false);
  }
  for (const stage of ['list', 'messages']) {
    p = page(); await login(p, 200, listing([item('saved')]));
    p.queue.push(stage === 'list' ? { status: 401 } : listing([item('saved')]));
    await p.nodes['history-toggle'].click();
    if (stage === 'messages') { p.queue.push({ status: 401 }); await historyButton(p, 'saved').click(); }
    assert.equal(p.read('accessToken'), '');
    assert.equal(p.read('conversationId'), null);
    assert.equal(p.read('historyItems.length'), 0);
    assert.equal(p.nodes['history-panel'].hidden, true);
    assert.equal(p.nodes['new-chat'].disabled, false);
    assert.match(p.nodes['chat-status'].textContent, /登录已失效/);
  }
  p = page(); await login(p, 200, { status: 503 });
  assert.equal(p.read('accessToken'), 'offline-only-token');
  assert.equal(p.nodes['send-button'].disabled, false);
  console.log('PASS: failed/invalid restore preserves current chat, no automatic retries, list/history 401 clears all identity data');

  p = page(); await login(p, 200, listing([item('slow-restore')]));
  p.queue.push(listing([item('slow-restore')])); await p.nodes['history-toggle'].click();
  let completeHistory;
  p.queue.push(() => new Promise(resolve => { completeHistory = resolve; }));
  const restorePending = historyButton(p, 'slow-restore').click();
  const beforeDuplicate = p.calls.length;
  for (const id of ['new-chat', 'account-button', 'history-toggle', 'history-refresh', 'send-button']) assert.equal(p.nodes[id].disabled, true);
  await historyButton(p, 'slow-restore').emit('click');
  await p.nodes['new-chat'].emit('click');
  p.nodes.question.value = '不该在恢复中发送'; await p.nodes['chat-form'].emit('submit');
  assert.equal(p.calls.length, beforeDuplicate);
  completeHistory({ status: 200, ok: true, json: async () => ({ conversation_id: 'slow-restore', messages: [] }) });
  await restorePending;
  assert.equal(p.read('conversationId'), 'slow-restore');
  assert.match(p.nodes['chat-messages'].textContent, /没有可显示的消息/);
  assert.equal(p.nodes['account-button'].disabled, false);
  assert.equal(p.nodes['history-toggle'].disabled, false);
  await p.nodes['new-chat'].click();
  assert.equal(p.read('conversationId'), null);
  assert.equal(p.read('accessToken'), 'offline-only-token');
  console.log('PASS: pending restore locks identity, thread selection and sends; empty history and new session remain usable');
})().catch(error => { console.error(error); process.exitCode = 1; });
