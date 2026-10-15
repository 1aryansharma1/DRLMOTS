// Offline JavaScript control test using a small DOM stub; no browser/layout claim.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[2] || 'outputs/demos/drlmots-demo.html', 'utf8');
const embedded = html.match(/<script id="demo-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const isTemplate = /<body\b[^>]*data-demo="template"/.test(html);
const data = isTemplate ? null : JSON.parse(embedded);
let download;
let exported;
class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.listeners = {}; this.value = ''; }
  append(child) { this.children.push(child); if (this.tagName === 'select' && !this.value) this.value = child.value; }
  replaceChildren(...children) { this.children = children; }
  addEventListener(event, handler) { this.listeners[event] = handler; }
  click() { if (this.tagName === 'a') download = this; else this.listeners.click(); }
}
const elements = new Map();
for (const match of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"[^>]*>/g)) elements.set(match[2], new Element(match[1]));
elements.get('demo-data').textContent = embedded;
const document = { getElementById: id => elements.get(id), createElement: tag => new Element(tag) };
vm.runInNewContext(script, { document, Blob, URL: {
  createObjectURL: blob => { exported = blob; return 'blob:offline-test'; }, revokeObjectURL() {},
}, setTimeout: callback => callback() });

(async () => {
  if (isTemplate) {
    assert.equal(elements.get('scenario').children.length, 0);
    assert.match(html, /Open the generated professor demo/);
    assert.match(html, /href="\.\.\/\.\.\/\.\.\/outputs\/demos\/drlmots-demo\.html"/);
    console.log('Verified source-template guidance and safe startup without generated data.');
    return;
  }
  assert.equal(elements.get('scenario').value, 'seeded');
  const initial = data.scenarios.find(s => s.id === 'seeded').comparison.reports.find(r => r.policy === elements.get('policy').value);
  assert.equal(elements.get('completed').textContent, initial.completed_count);
  assert.equal(elements.get('failed').textContent, initial.failed_count);
  assert.equal(elements.get('policy').children.length, initial.placement ? 3 : 2);
  for (const scenario of data.scenarios) {
    elements.get('scenario').value = scenario.id;
    elements.get('scenario').listeners.change();
    for (const report of scenario.comparison.reports) {
      elements.get('policy').value = report.policy;
      elements.get('policy').listeners.change();
      assert.equal(elements.get('completed').textContent, report.completed_count);
      assert.equal(elements.get('failed').textContent, report.failed_count);
      assert.equal(elements.get('task-rows').children.length, report.outcomes.length);
      assert.equal(elements.get('placement-details').hidden, !report.placement);
      assert.equal(elements.get('placement-rows').children.length, report.placement ? report.outcomes.length : 0);
      assert.equal(elements.get('comparison-rows').children.length, scenario.comparison.reports.length);
      assert.match(elements.get('network-note').textContent, report.placement ? /uploads and propagation/ : /network delay is zero/);
      assert.equal(elements.get('pool').children.length, report.topology.vms.length);
      assert.equal(JSON.parse(elements.get('provenance').textContent).seed, 42);
      assert.match(elements.get('timeline').innerHTML, /^<svg\s/);
      assert.equal(elements.get('makespan').textContent, report.makespan_ms.toFixed(2) + ' ms');
    }
    elements.get('export').click();
    assert.equal(download.download, `drlmots-${scenario.id}-comparison.json`);
    assert.deepEqual(JSON.parse(await exported.text()), scenario.comparison);
  }
  console.log(`Verified all ${data.scenarios.reduce((sum,s)=>sum+s.comparison.reports.length,0)} scenario/policy selections, tables, SVG selection, provenance and 3 JSON exports (DOM stub; layout untested).`);
})().catch(error => { console.error(error); process.exitCode = 1; });
