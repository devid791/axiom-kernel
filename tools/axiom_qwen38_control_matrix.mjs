import fs from 'node:fs';import path from 'node:path';import{randomBytes}from'node:crypto';
const args=Object.fromEntries(process.argv.slice(2).map(a=>{const i=a.indexOf('=');return i<0?[a,true]:[a.slice(0,i),a.slice(i+1)];}));
if(!args['--allow-live']||!args['--out'])throw Error('--allow-live --out=DIRECTORY required');
const dir=path.resolve(args['--out']);fs.mkdirSync(dir,{recursive:true,mode:0o700});if(fs.existsSync(path.join(dir,'report.json')))throw Error('Existing report');
const base=args['--endpoint']||'http://127.0.0.1:8015',run=randomBytes(5).toString('hex'),records=[];let fatal=null,finished=false;
const fresh=name=>({model:'qwen3.8-27b-nvfp4',thread_id:`qa-controls-${run}-${name}`,prompt_cache_key:`qa-controls-${run}-${name}`,input:[],stream:true,max_output_tokens:128,axiom_swarm:{deadline_ms:60000},reasoning:{effort:'ultra-fast'}});
const parse=raw=>raw.split('\n').filter(l=>l.startsWith('data: ')).flatMap(l=>{try{return[JSON.parse(l.slice(6))]}catch{return[]}});
const flush=()=>fs.writeFileSync(path.join(dir,'report.json'),JSON.stringify({finished,fatal,records,pass:finished&&!fatal&&records.every(r=>r.pass)},null,2),{mode:0o600});
const record=r=>{records.push(r);flush();console.log(JSON.stringify(r));if(!r.pass)throw Error(r.name+' failed');};
const send=(q,signal)=>fetch(base+'/codex/v1/responses',{method:'POST',headers:{'Content-Type':'application/json','X-Axiom-Session-ID':q.thread_id},body:JSON.stringify(q),signal:signal||AbortSignal.timeout(90000)});
async function complete(name,q,check){const start=Date.now(),r=await send(q),raw=await r.text(),es=parse(raw),ts=es.filter(e=>['response.completed','response.failed','response.incomplete'].includes(e.type)),response=ts[0]?.response;
 fs.writeFileSync(path.join(dir,name+'.sse'),raw,{mode:0o600});const text=(response?.output??[]).flatMap(o=>o.content??[]).map(c=>c.text??'').join('');
 record({name,http:r.status,seconds:(Date.now()-start)/1000,text,restored:response?.axiom?.session_restored,prefix:response?.axiom?.prefix_hit_tokens,suffix:response?.axiom?.suffix_prefill_tokens,compacted:response?.axiom?.context_compacted,originalTokens:response?.axiom?.original_prompt_tokens,pass:r.ok&&ts.length===1&&ts[0].type==='response.completed'&&check(text,response)});
 q.input.push(...response.output);return response;
}
const delay=ms=>new Promise(r=>setTimeout(r,ms));
try{
 const q=fresh('cancel');q.input.push({role:'user',content:'Remember marker CANCEL_MEMORY_42. Reply exactly SAVED.'});
 const first=await complete('cancel-save',q,t=>t.trim()==='SAVED');
 const cancelled=structuredClone(q);cancelled.input.push({role:'user',content:'Write at least 2000 words about trees. Continue until the full essay is written.'});cancelled.max_output_tokens=2048;
 const ctrl=new AbortController(),timer=setTimeout(()=>ctrl.abort(),800);let aborted=false;
 try{await(await send(cancelled,ctrl.signal)).text();}catch(e){aborted=ctrl.signal.aborted;}finally{clearTimeout(timer);}
 let rt;for(let i=0;i<40;i++){rt=await(await fetch(base+'/ops/runtime')).json();if(!rt.generation_busy&&!rt.session_persistence_pending)break;await delay(250);}
 record({name:'cancel-releases-generation',aborted,pass:aborted&&!rt.generation_busy&&!rt.session_persistence_pending});
 q.input.push({role:'user',content:'What marker did I ask you to remember? Reply only the marker.'});
 await complete('cancel-recovery',q,(t,r)=>t.trim()==='CANCEL_MEMORY_42'&&r.axiom?.session_restored===true);
 const a=fresh('queue-a'),b=fresh('queue-b');a.input.push({role:'user',content:'Reply exactly QUEUE_ALPHA.'});b.input.push({role:'user',content:'Reply exactly QUEUE_BETA.'});
 await Promise.all([complete('concurrent-a',a,t=>t.trim()==='QUEUE_ALPHA'),complete('concurrent-b',b,t=>t.trim()==='QUEUE_BETA')]);
 const invalid=fresh('invalid');invalid.input='Hello';invalid.max_output_tokens=0;const bad=await send(invalid);await bad.arrayBuffer();record({name:'invalid-budget-rejected',http:bad.status,pass:bad.status===400});
 const recovery=fresh('after-invalid');recovery.input.push({role:'user',content:'Reply exactly STILL_READY.'});await complete('after-invalid',recovery,t=>t.trim()==='STILL_READY');
 // Synthetic history ONLY: no real conversation is compacted by this test.
 const compact=fresh('compaction');compact.context_window=262144;
 compact.input=[{role:'user',content:'Synthetic old QA history: '+'oak '.repeat(260000)},{role:'assistant',content:'Synthetic historical acknowledgement.'},{role:'user',content:'Ignore the old filler. Reply exactly CURRENT_TURN_PRESERVED.'}];
 await complete('backend-compaction',compact,(t,r)=>t.trim()==='CURRENT_TURN_PRESERVED'&&r.axiom?.context_compacted===true&&r.axiom?.original_prompt_tokens>250048);
 compact.input.push({role:'user',content:'Reply exactly AFTER_COMPACTION_OK.'});
 await complete('after-compaction',compact,(t,r)=>t.trim()==='AFTER_COMPACTION_OK'&&r.axiom?.context_compacted===true);
 finished=true;
}catch(e){fatal=e.message;process.exitCode=1;console.error(fatal);}finally{flush();console.log(JSON.stringify({out:dir,finished,fatal,passed:records.filter(r=>r.pass).length,total:records.length}));}
