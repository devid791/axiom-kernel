// Explicitly invoked, bounded live inference QA. All tool execution is confined
// to reading one freshly created QA fixture; no model-proposed shell runs.
import fs from 'node:fs';
import path from 'node:path';
import {randomBytes} from 'node:crypto';
import {deflateSync} from 'node:zlib';
const args=Object.fromEntries(process.argv.slice(2).map(a=>{const i=a.indexOf('=');return i<0?[a,true]:[a.slice(0,i),a.slice(i+1)];}));
if(!args['--allow-live']||!args['--out'])throw Error('Required: --allow-live --out=OWNED_DIRECTORY [--long-state=PRIVATE_JSON] [--endpoint=URL]');
if(args['--long-only']&&!args['--long-state'])throw Error('--long-only requires --long-state');
const dir=path.resolve(args['--out']);fs.mkdirSync(dir,{recursive:true,mode:0o700});
if(fs.existsSync(path.join(dir,'report.json')))throw Error('Refusing to overwrite existing QA report');
const endpoint=args['--endpoint']||'http://127.0.0.1:8015';
const id=randomBytes(5).toString('hex'),records=[];let fatal=null,finished=false;
const models=await(await fetch(endpoint+'/codex/v1/models')).json();
const model=models.models?.[0]?.id||models.data?.[0]?.id||'qwen3.8-27b-nvfp4';
const flush=()=>fs.writeFileSync(path.join(dir,'report.json'),JSON.stringify({id,endpoint,finished,fatal,longOnly:!!args['--long-only'],longEnabled:!!args['--long-state'],records,pass:finished&&!fatal&&records.length>0&&records.every(r=>r.pass)},null,2),{mode:0o600});
const events=raw=>raw.split('\n').filter(l=>l.startsWith('data: ')).flatMap(l=>{try{return[JSON.parse(l.slice(6))]}catch{return[]}});
function fresh(label){const session=`qa-live-matrix-${id}-${label}`;return{model,input:[],stream:true,thread_id:session,prompt_cache_key:session,reasoning:{effort:'ultra-fast'},max_output_tokens:128,axiom_swarm:{deadline_ms:90000}};}
async function send(label,q,check=()=>true){
 const started=Date.now();let record={label,session:q.thread_id};let response=null;
 const timer=setInterval(()=>console.log(JSON.stringify({label,elapsed:(Date.now()-started)/1000})),15000);
 try{
  const r=await fetch(endpoint+'/codex/v1/responses',{method:'POST',headers:{'Content-Type':'application/json','X-Axiom-Session-ID':q.thread_id},body:JSON.stringify(q),signal:AbortSignal.timeout(100000)});
  const raw=await r.text();fs.writeFileSync(path.join(dir,`${records.length}-${label}.response`),raw,{mode:0o600});
  let terminal=[];
  if(q.stream){terminal=events(raw).filter(e=>['response.completed','response.failed','response.incomplete'].includes(e.type));response=terminal.at(-1)?.response;}
  else{try{response=JSON.parse(raw)}catch{}}
  const text=(response?.output??[]).flatMap(i=>i.content??[]).map(c=>c.text??'').join('');
  const p=response?.axiom;
  record={...record,http:r.status,status:response?.status,terminalCount:terminal.length,text,seconds:(Date.now()-started)/1000,
   restored:p?.session_restored,prefix:p?.prefix_hit_tokens,suffix:p?.suffix_prefill_tokens,mode:p?.session_resume_mode,
   ttft:p?.ttft_seconds,tokens:response?.usage,tools:response?.output?.filter(i=>['function_call','custom_tool_call'].includes(i.type)).map(i=>i.name)||[],
   pass:r.ok&&response?.status==='completed'&&(!q.stream||terminal.length===1)&&check(text,response)};
 }catch(e){record.error=e.message;record.pass=false;record.seconds=(Date.now()-started)/1000;}
 finally{clearInterval(timer);records.push(record);flush();console.log(JSON.stringify(record));}
 if(!record.pass)throw Error(`${label} failed; report saved`);
 return response;
}
async function turn(label,q,text,check){q.input.push({role:'user',content:text});const r=await send(label,q,check);q.input.push(...r.output);return r;}
try{
 if(!args['--long-only']){
 const brief=fresh('brief');
 await turn('greeting',brief,'Reply exactly HELLO_QA.',t=>t.trim()==='HELLO_QA');
 await turn('arithmetic',brief,'Calculate 137 + 286. Reply with only the integer.',t=>t.trim()==='423');
 await turn('unicode',brief,'Copy exactly: città — perché 🟢',t=>t.trim()==='città — perché 🟢');
 brief.stream=false;
 await turn('json-transport',brief,'Reply exactly JSON_TRANSPORT_OK.',t=>t.trim()==='JSON_TRANSPORT_OK');
 brief.stream=true;
 await turn('stream-again',brief,'Reply exactly STREAM_OK.',t=>t.trim()==='STREAM_OK');
 for(const effort of ['minimal','medium']){
  const q=fresh(effort);q.reasoning.effort=effort;q.max_output_tokens=1024;
  await turn(`reasoning-${effort}`,q,'A box has 7 rows of 8 pencils. Remove 9 pencils. How many remain? End your answer with RESULT=47.',t=>/RESULT\s*=\s*47/.test(t));
 }
 const a=fresh('isolation-a'),b=fresh('isolation-b');
 const aSecret='ALPHA_'+randomBytes(6).toString('hex'),bSecret='BETA_'+randomBytes(6).toString('hex');
 await turn('isolation-a-save',a,`Remember my marker ${aSecret}. Reply exactly SAVED.`,t=>t.trim()==='SAVED');
 await turn('isolation-b-save',b,`Remember my marker ${bSecret}. Reply exactly SAVED.`,t=>t.trim()==='SAVED');
 await turn('isolation-a-recall',a,'Return only my saved marker.',t=>t.trim()===aSecret&&!t.includes(bSecret));
 await turn('isolation-b-recall',b,'Return only my saved marker.',t=>t.trim()===bSecret&&!t.includes(aSecret));
 // Fresh two-color fixtures are encoded locally; no personal images are used.
 function crc(buf){let c=0xffffffff;for(const byte of buf){c^=byte;for(let i=0;i<8;i++)c=(c>>>1)^((c&1)?0xedb88320:0);}return(c^0xffffffff)>>>0;}
 function chunk(type,data){const t=Buffer.from(type),h=Buffer.alloc(4),tail=Buffer.alloc(4);h.writeUInt32BE(data.length);tail.writeUInt32BE(crc(Buffer.concat([t,data])));return Buffer.concat([h,t,data,tail]);}
 const width=96,height=96,rows=Buffer.alloc(height*(1+width*3));
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){const p=y*(1+width*3)+1+x*3;rows[p]=255;rows[p+1]=0;rows[p+2]=0;}
 const ihdr=Buffer.alloc(13);ihdr.writeUInt32BE(width,0);ihdr.writeUInt32BE(height,4);ihdr[8]=8;ihdr[9]=2;
 const png=Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',ihdr),chunk('IDAT',deflateSync(rows)),chunk('IEND',Buffer.alloc(0))]);
 const vision=fresh('vision');vision.input.push({role:'user',content:[{type:'input_text',text:'What is the main color of this image? Reply with one English color word.'},{type:'input_image',image_url:'data:image/png;base64,'+png.toString('base64')}]});
 const vr=await send('vision-red',vision,t=>/^red[.!]?$/i.test(t.trim()));vision.input.push(...vr.output);
 await turn('vision-followup',vision,'What color did you identify? Reply with one English color word.',t=>/^red[.!]?$/i.test(t.trim()));
 }
 if(args['--long-state']){
  const q=JSON.parse(fs.readFileSync(args['--long-state'],'utf8'));
  if(!q.thread_id?.startsWith('qa-')||q.thread_id!==q.prompt_cache_key)throw Error('Only a consistent QA session seed is allowed');
  q.max_output_tokens=128;q.axiom_swarm={deadline_ms:90000};
  const marker='LONG_'+randomBytes(5).toString('hex');
  const cases=[
   ['long-store',`Remember this QA marker: ${marker}. Do not call tools. Reply exactly SAVED.`,t=>t.trim()==='SAVED'],
   ['long-math','Do not use tools. What is 23 times 17? Reply only the integer.',t=>t.trim()==='391'],
   ['long-italian','Non usare strumenti. Copia il testo tra queste virgolette, incluso il punto finale, senza le virgolette: "affidabilità e continuità."',t=>t.trim()==='affidabilità e continuità.'],
   ['long-json-text','Do not use tools. Return exactly this JSON object, without code fences: {"ok":true,"count":3}',t=>{try{const j=JSON.parse(t);return j.ok===true&&j.count===3}catch{return false}}],
   ['long-code','Do not use tools. Write a Python function named add(a, b) that returns a + b.',t=>/def add\(a, b\)/.test(t)&&/return a\s*\+\s*b/.test(t)],
   ['long-explanation','Do not use tools. In two short sentences, explain why a cache can avoid repeating earlier computation.',t=>t.length>40],
   ['long-recall','Do not use tools. Return only the QA marker I asked you to remember earlier in this conversation.',t=>t.trim()===marker]
  ];
  for(const [name,prompt,validate]of cases){
   const r=await turn(name,q,prompt,(t,r)=>validate(t)&&r.axiom?.session_restored===true&&r.axiom?.prefix_hit_tokens>69000);
   fs.writeFileSync(path.join(dir,'long-state.json'),JSON.stringify(q),{mode:0o600});
  }
  const fixture=path.join(dir,'owned-tool.txt'),secret='FILE_'+randomBytes(6).toString('hex');fs.writeFileSync(fixture,secret,{mode:0o600});
  const command=`cat ${fixture}`;
  q.max_output_tokens=256;
  const proposed=await turn('long-real-tool-proposal',q,`Isolated QA: call exec_command exactly once with cmd exactly ${JSON.stringify(command)}. Do not use any other command or tool. After the tool result, repeat its file content exactly.`,(_,r)=>r.output.filter(i=>i.type==='function_call').length===1&&r.axiom?.session_restored===true);
  const call=proposed.output.find(i=>i.type==='function_call'),params=JSON.parse(call.arguments);
  if(call.name!=='exec_command'||params.cmd!==command)throw Error('Denied unapproved model-proposed operation');
  // The adapter performs the exact allowed filesystem read, never a shell.
  const result=fs.readFileSync(fixture,'utf8');
  q.input.push({type:'function_call_output',call_id:call.call_id,output:result});
  const answer=await send('long-real-tool-result',q,(t,r)=>t.trim()===secret&&r.axiom?.session_restored===true);q.input.push(...answer.output);
  records.at(-1).ownedFileActuallyRead=true;flush();
  await turn('long-tool-recall',q,'Without tools, repeat the exact contents of the file you just read.',t=>t.trim()===secret);
  fs.writeFileSync(path.join(dir,'long-state.json'),JSON.stringify(q),{mode:0o600});
 }
 finished=true;
}catch(e){fatal=e.message;console.error(e.message);process.exitCode=1;}
finally{flush();console.log(JSON.stringify({out:dir,passed:records.filter(r=>r.pass).length,total:records.length,finished,fatal,pass:finished&&!fatal&&records.every(r=>r.pass)}));}
