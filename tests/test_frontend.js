// 运行: node tests/test_frontend.js（无需安装测试依赖）
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const row = name => ({name, start_str: '00:00:01', end_str: '00:00:05'});
const response = (data, ok = true) => ({ok, status: ok ? 200 : 400, json: async () => data});
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return {promise, resolve};
};
const flush = () => new Promise(setImmediate);

function harness() {
  const elements = new Map(), requests = [], songs = new Map();
  const hooks = [];
  class Element {
    constructor(tag = 'div') {
      this.tag = tag;
      this.children = [];
      this.inputs = {};
      this.style = {};
      this.dataset = {};
      this.value = '';
      this.valueAsNumber = NaN;
      this.textContent = '';
      this.classList = {add() {}, remove() {}, toggle() {}};
    }
    set innerHTML(html) {
      this.html = html;
      this.children = [];
      for (const name of ['name', 'start', 'end']) {
        const match = html.match(new RegExp(`class="in-${name}"[^>]*value="([^"]*)"`));
        if (match) this.inputs[`.in-${name}`] = {value: match[1], focus() {}};
      }
    }
    get innerHTML() { return this.html || ''; }
    appendChild(child) { child.parent = this; this.children.push(child); }
    get lastElementChild() { return this.children.at(-1); }
    remove() { this.parent.children.splice(this.parent.children.indexOf(this), 1); }
    querySelector(selector) {
      if (selector === 'tbody') return this.body || (this.body = new Element('tbody'));
      if (selector === 'td[colspan]') return this.innerHTML.includes('colspan=') ? {} : null;
      if (selector in this.inputs) return this.inputs[selector];
      return this.inputs[selector] || (this.inputs[selector] = new Element());
    }
    querySelectorAll() { return []; }
    addEventListener() {}
    setAttribute() {}
    removeAttribute() {}
    focus() {}
  }
  const get = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const context = vm.createContext({
    document: {
      getElementById: get, createElement: tag => new Element(tag), addEventListener() {},
      querySelectorAll: selector => selector === '#tbl-songs tbody tr' ? get('tbl-songs').querySelector('tbody').children : [],
      body: {style: {}},
    },
    window: {addEventListener() {}},
    localStorage: {getItem: () => '', setItem() {}},
    setTimeout() {}, alert() {}, confirm: () => true,
    fetch: async (url, options = {}) => {
      const body = options.body ? JSON.parse(options.body) : undefined;
      requests.push({url, body});
      const hook = hooks.findIndex(item => item.matches(url, body));
      if (hook !== -1) return hooks.splice(hook, 1)[0].handle(url, body);
      if (url === '/api/select-video') return response({video: body.video, result: `${body.video}/result`});
      if (url.startsWith('/api/candidates?')) return response({exists: false, candidates: []});
      if (url.startsWith('/api/songs?')) return response({songs: songs.get(new URL(url, 'http://localhost').searchParams.get('video')) || []});
      if (url === '/api/songs') { songs.set(body.video, body.songs); return response({count: body.songs.length}); }
      if (url.startsWith('/api/task/')) return response({status: 'done', messages: [], result: {items: [], out_dir: 'A/result'}});
      return response({task_id: 'test-task'});
    },
  });
  vm.runInContext(script, context);
  return {
    context, requests, songs, get, hooks,
    state: name => vm.runInContext(name, context),
    names: () => get('tbl-songs').querySelector('tbody').children.map(tr => tr.querySelector('.in-name').value),
    edit(name) {
      get('tbl-songs').querySelector('tbody').children[0].querySelector('.in-name').value = name;
      context.markSongsDirty();
    },
  };
}

const tests = [];
const test = (name, run) => tests.push({name, run});

test('switch loads the requested session and saves old edits to the old video', async () => {
  const h = harness();
  h.songs.set('A', [row('Song A')]);
  h.songs.set('B', [row('Song B')]);
  await h.context.setVideo('A');
  h.edit('Edited A');
  await h.context.setVideo('B');
  assert.deepEqual(h.names(), ['Song B']);
  assert.equal(h.songs.get('A')[0].name, 'Edited A');
  assert.equal(h.state('outDir'), 'B/result');
  assert(h.requests.some(r => r.url === '/api/songs?video=B'));
  assert(h.requests.some(r => r.url === '/api/candidates?video=B'));
});

test('failed save blocks switching and retains the editable old list', async () => {
  const h = harness();
  h.songs.set('A', [row('Song A')]);
  await h.context.setVideo('A');
  h.edit('Unsaved A');
  h.hooks.push({matches: url => url === '/api/songs', handle: () => response({detail: 'Invalid time'}, false)});
  await assert.rejects(h.context.setVideo('B'), /保存失败/);
  assert.equal(h.state('currentVideo'), 'A');
  assert.deepEqual(h.names(), ['Unsaved A']);
  assert(!h.requests.some(r => r.url === '/api/select-video' && r.body.video === 'B'));
});

test('late songs response cannot replace the newer session', async () => {
  const h = harness(), pending = deferred();
  h.songs.set('A', [row('Song A')]);
  h.songs.set('B', [row('Song B')]);
  await h.context.setVideo('A');
  h.hooks.push({matches: url => url === '/api/songs?video=A', handle: () => pending.promise});
  const oldLoad = h.context.loadSongs();
  await h.context.setVideo('B');
  pending.resolve(response({songs: [row('Old A')]}));
  await oldLoad;
  assert.deepEqual(h.names(), ['Song B']);
});

test('saving during session load cannot clear the stored list', async () => {
  const h = harness(), pending = deferred();
  h.songs.set('B', [row('Song B')]);
  h.hooks.push({matches: url => url === '/api/songs?video=B', handle: () => pending.promise});
  const switching = h.context.setVideo('B');
  await flush();
  assert.equal(await h.context.saveSongs(), false);
  await assert.rejects(h.context.appendSongRow('Wrong row', '00:00:01', '00:00:05'), /尚未加载/);
  assert.equal(h.songs.get('B')[0].name, 'Song B');
  assert.equal(h.get('btn-add-row').disabled, true);
  pending.resolve(response({songs: [row('Song B')]}));
  await switching;
  assert.deepEqual(h.names(), ['Song B']);
  assert.equal(h.get('btn-add-row').disabled, false);
});

test('late reload cannot discard edits made while loading', async () => {
  const h = harness(), pending = deferred();
  h.songs.set('A', [row('Song A')]);
  await h.context.setVideo('A');
  h.hooks.push({matches: url => url === '/api/songs?video=A', handle: () => pending.promise});
  const loading = h.context.loadSongs();
  h.edit('New edit');
  pending.resolve(response({songs: [row('Old saved row')]}));
  await loading;
  assert.deepEqual(h.names(), ['New edit']);
  assert.equal(h.state('songsDirty'), true);
});

test('rapid selections keep the latest selection despite response order', async () => {
  const h = harness(), pending = deferred();
  h.hooks.push({matches: (url, body) => url === '/api/select-video' && body.video === 'B', handle: () => pending.promise});
  const selectingB = h.context.setVideo('B');
  await h.context.setVideo('C');
  pending.resolve(response({video: 'B', result: 'B/result'}));
  assert.equal(await selectingB, false);
  assert.equal(h.state('currentVideo'), 'C');
  assert.equal(h.state('outDir'), 'C/result');
});

test('save requests are ordered and retain their captured video and rows', async () => {
  const h = harness(), pending = deferred();
  h.songs.set('A', [row('Song A')]);
  await h.context.setVideo('A');
  h.edit('First edit');
  h.hooks.push({matches: url => url === '/api/songs', handle: () => pending.promise});
  const first = h.context.saveSongs();
  await flush();
  h.edit('Second edit');
  const second = h.context.saveSongs();
  await flush();
  assert.equal(h.requests.filter(r => r.url === '/api/songs').length, 1);
  pending.resolve(response({count: 1}));
  assert.equal(await first, true);
  assert.equal(await second, true);
  assert.equal(h.songs.get('A')[0].name, 'Second edit');
  assert.equal(h.state('songsDirty'), false);
});

test('editing during auto-save blocks switching instead of discarding edits', async () => {
  const h = harness(), pending = deferred();
  h.songs.set('A', [row('Song A')]);
  await h.context.setVideo('A');
  h.edit('First edit');
  h.hooks.push({matches: url => url === '/api/songs', handle: () => pending.promise});
  const switching = h.context.setVideo('B');
  const rejected = assert.rejects(switching, /发生修改/);
  await flush();
  h.edit('Newer edit');
  pending.resolve(response({count: 1}));
  await rejected;
  assert.equal(h.state('currentVideo'), 'A');
  assert.deepEqual(h.names(), ['Newer edit']);
});

test('analysis completion keeps unsaved manual edits', async () => {
  const h = harness();
  h.songs.set('A', [row('Song A')]);
  await h.context.setVideo('A');
  h.edit('Unsaved title');
  const count = h.requests.filter(r => r.url.startsWith('/api/songs?')).length;
  await h.get('btn-analyze').onclick();
  assert.deepEqual(h.names(), ['Unsaved title']);
  assert.equal(h.state('songsDirty'), true);
  assert.equal(h.requests.filter(r => r.url.startsWith('/api/songs?')).length, count);
});

test('old recognition results cannot populate the new video', async () => {
  const h = harness(), pending = deferred();
  await h.context.setVideo('A');
  vm.runInContext('var rendered = 0; renderRecognizeResult = () => { rendered++; };', h.context);
  h.hooks.push({matches: url => url.startsWith('/api/task/'), handle: () => pending.promise});
  const recognizing = h.context.recognizeSegment(1, 5, '00:00:01', '00:00:05');
  await flush();
  await h.context.setVideo('B');
  pending.resolve(response({status: 'done', messages: [], result: {candidates: []}}));
  await recognizing;
  assert.equal(h.state('rendered'), 0);
});

test('old cut completion does not change the new output directory', async () => {
  const h = harness(), pending = deferred();
  h.songs.set('A', [row('Song A')]);
  await h.context.setVideo('A');
  h.hooks.push({matches: url => url.startsWith('/api/task/'), handle: () => pending.promise});
  const cutting = h.get('btn-cut').onclick();
  await flush();
  await h.context.setVideo('B');
  pending.resolve(response({status: 'done', messages: [], result: {out_dir: 'A/result', items: []}}));
  await cutting;
  assert.equal(h.state('outDir'), 'B/result');
});

(async () => {
  for (const {name, run} of tests) {
    await run();
    console.log(`PASS ${name}`);
  }
  console.log(`${tests.length} frontend regression tests passed`);
})().catch(error => { console.error(error); process.exitCode = 1; });
