const fs=require('fs'),path=require('path'),vm=require('vm');
const html=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8');
const escapeSource=html.slice(html.indexOf('function escapeHtml('),html.indexOf('async function apiError('));
const renderSource=html.slice(html.indexOf('function refreshChatModels('),html.indexOf('function updateMetrics('));
const malicious='Моя "модель" <img src=x onerror=alert(1)> & \'test\'';
const select={value:'safe',innerHTML:''};
const sandbox={INSTANCES:[{id:'safe',name:malicious,allowed:true}],ACTIVE:{safe:true},BUTLER_MODE:'ON',
  STATE_UI:{on:{lbl:'ON'}},advanceState:()=> 'on',updateChatColor:()=>{},
  document:{activeElement:null,getElementById:()=>select}};
vm.createContext(sandbox);vm.runInContext(escapeSource+renderSource,sandbox);
const escaped=sandbox.escapeHtml(malicious);
if(escaped.includes('<img')||!escaped.includes('&quot;')||!escaped.includes('&#39;')||!escaped.includes('&amp;'))throw Error('escaping incomplete');
sandbox.refreshChatModels(true);
if(select.innerHTML.includes('<img')||!select.innerHTML.includes('&lt;img'))throw Error('unsafe chat option');
if(sandbox.INSTANCES[0].name!==malicious)throw Error('display name was mutated');
if(!html.includes('value="${escapeHtml(inst.name)}"'))throw Error('unsafe input attribute');
for(const script of html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g))new vm.Script(script[1]);
console.log('PASS: Unicode/quoted/HTML names escaped in attributes and chat options; source names preserved; all inline JS parses');
