importScripts('engine.js');
let X = null, y = null;
onmessage = e => {
  const m = e.data;
  if (m.type === "ping") { postMessage({ type: "pong" }); return; }
  if (m.type === "data") { X = m.X; y = m.y; return; }
  const s = Engine.split(y, m.testFrac, m.splitSeed);
  const md = Engine.fit(m.model, X, y, s.tr, m.modelSeed, m.nTrees);
  postMessage({ type: "res", id: m.id, acc: Engine.accuracy(md, X, y, s.te), nTest: s.te.length });
};
